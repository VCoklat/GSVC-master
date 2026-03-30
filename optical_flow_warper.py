"""
GSVC-Turbo — Optical Flow Warper
=================================
Implements the complete FR1-FR7 pipeline for Hybrid Eulerian-Lagrangian
tracking of 2D Gaussian Splatting particles.

Pipeline Summary:
    FR1: Adaptive VRAM Protection (downscale large frames for RAFT)
    FR2: Eulerian Flow Extraction (RAFT-Small dense optical flow)
    FR3: Resolution Restoration & Safety Clamping
    FR4: Particle Space Normalization (tanh transform)
    FR5: Sub-Pixel Eulerian-to-Lagrangian Mapping (grid_sample)
    FR6: Kinematic Advection (translate particles by sampled flow)
    FR7: State Injection & Optimizer Prep (atanh inverse + clamp)

Usage:
    warper = OpticalFlowWarper(device='cuda:0')
    warper.warp(prev_frame, curr_frame, gaussian_model)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from turbo_config import TurboConfig

# Lazy-loaded to avoid import overhead when turbo is disabled
_raft_model = None
_raft_device = None


def _get_raft_model(device: torch.device):
    """Lazy-load and cache the RAFT-Small model from torchvision."""
    global _raft_model, _raft_device
    if _raft_model is not None and _raft_device == device:
        return _raft_model

    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    _raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device)
    _raft_model.eval()
    _raft_device = device

    # Freeze all parameters — we never train RAFT
    for p in _raft_model.parameters():
        p.requires_grad_(False)

    return _raft_model


class OpticalFlowWarper:
    """
    Pre-advects Gaussian splat positions using dense optical flow,
    implementing the complete FR1-FR7 Turbo pipeline.

    Args:
        config: TurboConfig instance controlling all pipeline parameters.
        device: torch device for computation (default: cuda:0).
    """

    def __init__(
        self,
        config: TurboConfig = None,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.config = config or TurboConfig()
        self.device = device

        # Pre-warm RAFT on first instantiation
        if self.config.enabled:
            _get_raft_model(self.device)

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def warp(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        gaussian_model: nn.Module,
        is_delta: bool = False,
    ) -> dict:
        """
        Run the full FR1-FR7 pipeline.

        Args:
            prev_frame: Previous frame as numpy HxWx3 uint8 RGB array
                        or PIL Image or torch Tensor [1,3,H,W].
            curr_frame: Current frame (same format as prev_frame).
            gaussian_model: The GaussianVideo_frame or GaussianVideo_delta model.
                           Must have `._xyz` parameter.
            is_delta: If True, the model is a GaussianVideo_delta and
                      positions are stored as deltas relative to p_xyz.

        Returns:
            dict with diagnostic info:
              - 'mean_flow': mean pixel displacement
              - 'clamped_pct': percentage of flow vectors that were clamped
              - 'num_gaussians': number of particles warped
        """
        if not self.config.enabled:
            return {"skipped": True}

        cfg = self.config

        # ── Convert inputs to [1, 3, H, W] float32 tensors ────────────
        prev_t = self._to_tensor(prev_frame)   # [1, 3, H, W]
        curr_t = self._to_tensor(curr_frame)    # [1, 3, H, W]
        original_H, original_W = prev_t.shape[2], prev_t.shape[3]

        # ── FR1: Adaptive VRAM Protection ──────────────────────────────
        prev_ds, curr_ds, scale_factor = self._adaptive_downscale(
            prev_t, curr_t, original_H, original_W
        )

        # ── FR2: Eulerian Flow Extraction ──────────────────────────────
        flow = self._compute_flow(prev_ds, curr_ds)  # [1, 2, H_ds, W_ds]

        # ── FR3: Resolution Restoration & Safety Clamping ──────────────
        flow = self._restore_and_clamp(flow, original_H, original_W, scale_factor)
        # flow is now [1, 2, H_orig, W_orig]

        # ── FR4: Particle Space Normalization ──────────────────────────
        if is_delta:
            # For delta models: actual position = tanh(_xyz + p_xyz)
            # We need to advect the *combined* position
            raw_params = gaussian_model._xyz.data  # [N, 2]  (delta)
            p_xyz = gaussian_model.p_xyz            # [N, 2]  (parent buffer)
            combined_params = raw_params + p_xyz    # [N, 2]  (combined atanh space)
            normalized_pos = torch.tanh(combined_params)  # [N, 2] in [-1, 1]
        else:
            raw_params = gaussian_model._xyz.data   # [N, 2]
            normalized_pos = torch.tanh(raw_params) # [N, 2] in [-1, 1]

        N = normalized_pos.shape[0]

        # ── FR5: Sub-Pixel Eulerian-to-Lagrangian Mapping ──────────────
        per_particle_flow = self._sample_flow_at_particles(
            flow, normalized_pos, original_H, original_W
        )  # [N, 2]

        # ── FR6: Kinematic Advection ───────────────────────────────────
        advected_pos = self._advect(
            normalized_pos, per_particle_flow, original_H, original_W
        )  # [N, 2] in [-1, 1]

        # ── FR7: State Injection & Optimizer Prep ──────────────────────
        diagnostics = self._inject_state(
            gaussian_model, advected_pos, is_delta, per_particle_flow
        )
        diagnostics["num_gaussians"] = N

        if cfg.verbose:
            print(
                f"  [Turbo] N={N}, "
                f"mean_flow={diagnostics['mean_flow']:.2f}px, "
                f"clamped={diagnostics['clamped_pct']:.1f}%"
            )

        return diagnostics

    # ═══════════════════════════════════════════════════════════════════
    # FR1: Adaptive VRAM Protection
    # ═══════════════════════════════════════════════════════════════════

    def _adaptive_downscale(self, prev_t, curr_t, H, W):
        """
        Downsample frames if they exceed max_resolution, ensuring
        dimensions are divisible by 8 (RAFT architecture constraint).

        Returns:
            (prev_ds, curr_ds, scale_factor)
        """
        cfg = self.config
        scale_factor = 1.0

        if cfg.max_resolution > 0 and H > cfg.max_resolution:
            scale_factor = cfg.max_resolution / H
            new_H = int(H * scale_factor)
            new_W = int(W * scale_factor)
            # Round down to nearest multiple of 8
            new_H = (new_H // 8) * 8
            new_W = (new_W // 8) * 8
            # Recompute exact scale factor after rounding
            scale_factor = new_H / H

            prev_t = F.interpolate(
                prev_t, size=(new_H, new_W),
                mode="bilinear", align_corners=False
            )
            curr_t = F.interpolate(
                curr_t, size=(new_H, new_W),
                mode="bilinear", align_corners=False
            )
        else:
            # Still ensure divisible by 8
            new_H = (H // 8) * 8
            new_W = (W // 8) * 8
            if new_H != H or new_W != W:
                prev_t = F.interpolate(
                    prev_t, size=(new_H, new_W),
                    mode="bilinear", align_corners=False
                )
                curr_t = F.interpolate(
                    curr_t, size=(new_H, new_W),
                    mode="bilinear", align_corners=False
                )
                scale_factor = new_H / H

        return prev_t, curr_t, scale_factor

    # ═══════════════════════════════════════════════════════════════════
    # FR2: Eulerian Flow Extraction
    # ═══════════════════════════════════════════════════════════════════

    def _compute_flow(self, prev_ds, curr_ds):
        """
        Run RAFT-Small on the (possibly downscaled) frame pair.

        Args:
            prev_ds: [1, 3, H_ds, W_ds] float32 tensor in [0, 1]
            curr_ds: [1, 3, H_ds, W_ds] float32 tensor in [0, 1]

        Returns:
            flow: [1, 2, H_ds, W_ds] — dense Eulerian flow field (dx, dy)
        """
        cfg = self.config
        raft = _get_raft_model(self.device)

        # RAFT expects [0, 255] range input
        prev_input = (prev_ds * 255.0).to(self.device)
        curr_input = (curr_ds * 255.0).to(self.device)

        if cfg.use_fp16_flow and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                flow_list = raft(prev_input.half(), curr_input.half(),
                                 num_flow_updates=cfg.raft_iterations)
        else:
            flow_list = raft(prev_input, curr_input,
                             num_flow_updates=cfg.raft_iterations)

        # RAFT returns a list of iterative refinements; take the last
        flow = flow_list[-1].float()  # [1, 2, H_ds, W_ds]

        return flow

    # ═══════════════════════════════════════════════════════════════════
    # FR3: Resolution Restoration & Safety Clamping
    # ═══════════════════════════════════════════════════════════════════

    def _restore_and_clamp(self, flow, target_H, target_W, scale_factor):
        """
        Upsample flow back to original resolution and clamp magnitude.

        The flow is in pixel displacement units at the downscaled resolution.
        After upscaling, we must divide by scale_factor to correct the
        magnitude to original-resolution pixel units.
        """
        cfg = self.config

        # Upsample to original resolution
        if flow.shape[2] != target_H or flow.shape[3] != target_W:
            flow = F.interpolate(
                flow, size=(target_H, target_W),
                mode="bilinear", align_corners=False
            )

        # Correct magnitude: flow was computed at downscaled res
        if scale_factor != 1.0 and scale_factor > 0:
            flow = flow / scale_factor

        # Safety clamp
        if cfg.max_flow_magnitude > 0:
            flow_mag = torch.sqrt(flow[:, 0:1] ** 2 + flow[:, 1:2] ** 2)
            # Count how many exceed threshold (for diagnostics)
            self._clamped_count = (flow_mag > cfg.max_flow_magnitude).sum().item()
            self._total_pixels = flow_mag.numel()

            # Clamp by scaling down vectors that exceed max magnitude
            scale_mask = torch.clamp(
                cfg.max_flow_magnitude / (flow_mag + 1e-8), max=1.0
            )
            flow = flow * scale_mask
        else:
            self._clamped_count = 0
            self._total_pixels = flow.shape[2] * flow.shape[3]

        return flow

    # ═══════════════════════════════════════════════════════════════════
    # FR5: Sub-Pixel Eulerian-to-Lagrangian Mapping
    # ═══════════════════════════════════════════════════════════════════

    def _sample_flow_at_particles(self, flow, normalized_pos, H, W):
        """
        Use grid_sample to probe the Eulerian flow field at each
        Gaussian particle's sub-pixel position.

        Args:
            flow:           [1, 2, H, W] flow field in pixel units
            normalized_pos: [N, 2] positions in [-1, 1] (x, y)
            H, W:           original image dimensions

        Returns:
            per_particle_flow: [N, 2] — individualized (dx, dy) in pixel units
        """
        N = normalized_pos.shape[0]

        # grid_sample expects grid in shape [1, 1, N, 2] with (x, y) in [-1,1]
        # Our normalized_pos is already in [-1, 1] (column0=x, column1=y)
        grid = normalized_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

        # grid_sample: input [1, 2, H, W], grid [1, 1, N, 2]
        # Output: [1, 2, 1, N]
        sampled = F.grid_sample(
            flow, grid, mode="bilinear",
            padding_mode="border", align_corners=True
        )  # [1, 2, 1, N]

        # Reshape to [N, 2]
        per_particle_flow = sampled.squeeze(0).squeeze(1).permute(1, 0)  # [N, 2]

        return per_particle_flow

    # ═══════════════════════════════════════════════════════════════════
    # FR6: Kinematic Advection
    # ═══════════════════════════════════════════════════════════════════

    def _advect(self, normalized_pos, per_particle_flow, H, W):
        """
        Translate particles by their sampled flow vectors.

        The flow is in pixel units, but positions are in [-1, 1] normalized
        space. We convert flow to normalized units by dividing by (W/2, H/2).

        Args:
            normalized_pos:    [N, 2] in [-1, 1] (x, y)
            per_particle_flow: [N, 2] in pixel units (dx, dy)
            H, W:              image dimensions

        Returns:
            advected_pos: [N, 2] in [-1, 1]
        """
        # Convert pixel displacement to normalized displacement
        # x-axis: divide by W/2 ;  y-axis: divide by H/2
        norm_flow = per_particle_flow.clone()
        norm_flow[:, 0] = per_particle_flow[:, 0] / (W / 2.0)  # dx
        norm_flow[:, 1] = per_particle_flow[:, 1] / (H / 2.0)  # dy

        advected_pos = normalized_pos + norm_flow

        return advected_pos

    # ═══════════════════════════════════════════════════════════════════
    # FR7: State Injection & Optimizer Prep
    # ═══════════════════════════════════════════════════════════════════

    def _inject_state(self, gaussian_model, advected_pos, is_delta, per_particle_flow):
        """
        Clamp, apply inverse transform (atanh), and overwrite _xyz.

        For delta models, we subtract p_xyz before writing back so that
        the delta semantics are preserved.

        Returns diagnostics dict.
        """
        cfg = self.config

        # Compute diagnostics before clamping
        flow_magnitude = torch.sqrt(
            per_particle_flow[:, 0] ** 2 + per_particle_flow[:, 1] ** 2
        )
        mean_flow = flow_magnitude.mean().item()

        # Strict boundary clamping
        advected_clamped = torch.clamp(
            advected_pos, -cfg.position_clamp, cfg.position_clamp
        )

        # Inverse transform: normalized [-1,1] → unconstrained atanh space
        new_atanh = torch.atanh(advected_clamped)

        if is_delta:
            # For delta model: _xyz stores the *delta*, not the full position
            # full_atanh = delta + p_xyz  →  delta = full_atanh - p_xyz
            new_delta = new_atanh - gaussian_model.p_xyz
            gaussian_model._xyz.data.copy_(new_delta)
        else:
            gaussian_model._xyz.data.copy_(new_atanh)

        # Reset optimizer momentum for _xyz to prevent stale gradients
        # from fighting the advection
        self._reset_optimizer_state(gaussian_model)

        clamped_pct = (
            self._clamped_count / max(self._total_pixels, 1) * 100.0
        )

        return {
            "mean_flow": mean_flow,
            "clamped_pct": clamped_pct,
        }

    def _reset_optimizer_state(self, gaussian_model):
        """
        Reset the optimizer's momentum buffers for _xyz parameter only.
        This prevents stale momentum from the previous frame from
        counteracting the flow-based advection.
        """
        if not hasattr(gaussian_model, 'optimizer'):
            return

        optimizer = gaussian_model.optimizer

        for group in optimizer.param_groups:
            for p in group['params']:
                if p is gaussian_model._xyz:
                    if p in optimizer.state:
                        state = optimizer.state[p]
                        # Reset all momentum buffers
                        for key in list(state.keys()):
                            if isinstance(state[key], torch.Tensor) and key != 'step':
                                state[key].zero_()
                    break

    # ═══════════════════════════════════════════════════════════════════
    # Utility
    # ═══════════════════════════════════════════════════════════════════

    def _to_tensor(self, frame):
        """
        Convert various frame formats to [1, 3, H, W] float32 tensor on device.
        Supports: numpy uint8 HWC, PIL Image, torch Tensor.
        """
        if isinstance(frame, torch.Tensor):
            t = frame.float()
            if t.dim() == 3:
                t = t.unsqueeze(0)  # [3,H,W] → [1,3,H,W]
            if t.max() > 1.0:
                t = t / 255.0
            return t.to(self.device)

        if isinstance(frame, Image.Image):
            frame = np.array(frame)

        if isinstance(frame, np.ndarray):
            # Expect HxWx3 uint8
            t = torch.from_numpy(frame).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            return t.to(self.device)

        raise TypeError(f"Unsupported frame type: {type(frame)}")
