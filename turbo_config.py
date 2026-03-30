"""
GSVC-Turbo Configuration
========================
Dataclass-based configuration for the OpticalFlowWarper module.
Controls VRAM protection, flow clamping, and ablation toggles.
"""

from dataclasses import dataclass, field


@dataclass
class TurboConfig:
    """Configuration for the GSVC-Turbo optical flow warping pipeline."""

    # ── Master toggle ──────────────────────────────────────────────────
    enabled: bool = True

    # ── FR1: Adaptive VRAM Protection ──────────────────────────────────
    max_resolution: int = 540
    """Maximum height (in pixels) fed to RAFT. Frames taller than this
    are bilinearly downsampled before flow estimation. Set to 0 to
    disable downscaling (VRAM Limit Ablation)."""

    # ── FR2: RAFT Model Selection ──────────────────────────────────────
    raft_model: str = "raft_small"
    """Which RAFT variant to use.  Currently only 'raft_small' is
    supported via torchvision."""

    raft_iterations: int = 12
    """Number of RAFT iterative update steps. More = smoother flow,
    but slower.  12 is a good default for 540p."""

    # ── FR3: Flow Clamping ─────────────────────────────────────────────
    max_flow_magnitude: float = 80.0
    """Maximum allowed optical flow displacement in pixels (at the
    *original* resolution).  Prevents explosive advection from noisy
    flow in textureless regions. Set to 0 to disable (Clamping Ablation)."""

    # ── FR7: Output Clamping ───────────────────────────────────────────
    position_clamp: float = 0.999
    """Clamp advected normalized positions to [-clamp, +clamp] before
    applying atanh.  Prevents infinity from boundary saturation.
    Set to 1.0 to disable (Clamping Ablation)."""

    # ── Performance ────────────────────────────────────────────────────
    use_fp16_flow: bool = True
    """Run RAFT inference in float16 to reduce VRAM.  Generally safe
    since flow vectors don't need full float32 precision."""

    # ── Logging ────────────────────────────────────────────────────────
    verbose: bool = False
    """Print per-frame flow statistics (mean displacement, clamped %)."""


# ── Preset Configurations ──────────────────────────────────────────────

TURBO_PRESETS = {
    "default": TurboConfig(),

    "conservative": TurboConfig(
        max_resolution=720,
        max_flow_magnitude=40.0,
        raft_iterations=20,
        position_clamp=0.995,
    ),

    "aggressive": TurboConfig(
        max_resolution=480,
        max_flow_magnitude=120.0,
        raft_iterations=8,
        position_clamp=0.999,
    ),

    # ── Ablation Presets ───────────────────────────────────────────────
    "ablation_no_clamp": TurboConfig(
        max_flow_magnitude=0.0,   # FR3 disabled
        position_clamp=1.0,       # FR7 disabled
    ),

    "ablation_no_downscale": TurboConfig(
        max_resolution=0,         # FR1 disabled — native resolution
    ),

    "disabled": TurboConfig(enabled=False),
}


def get_turbo_config(preset: str = "default") -> TurboConfig:
    """Retrieve a TurboConfig by preset name."""
    if preset not in TURBO_PRESETS:
        raise ValueError(
            f"Unknown turbo preset '{preset}'. "
            f"Available: {list(TURBO_PRESETS.keys())}"
        )
    return TURBO_PRESETS[preset]
