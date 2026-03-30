from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
from torch.distributions import MultivariateNormal

class GaussianVideo_frame(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.max_num_points = kwargs["max_num_points"]
        self.densification_interval=kwargs["densification_interval"]
        self.iterations=kwargs["iterations"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]
        self.removal_rate=kwargs["removal_rate"]
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.isdensity=kwargs["isdensity"]
        self.isremoval=kwargs["isremoval"]
        if self.isremoval:
            self.rgb_W = nn.Parameter(0.01 * torch.ones(self.init_num_points, 1))
        elif self.isdensity:
            self.rgb_W = nn.Parameter(torch.ones(self.init_num_points, 1))
        else:
           self.register_buffer('rgb_W', torch.ones((self.init_num_points, 1))) 
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.rgbW_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.lr = kwargs["lr"]
        self.opt_type =kwargs["opt_type"]
        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc*self.get_rgb_W
    
    @property
    def get_rgb_W(self):
        return self.rgb_W

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    def forward_pos(self,num_points):
        features_dc = torch.ones(num_points, 3).to(self.device)
        cholesky = torch.full((num_points, 3), 1.0).to(self.device)
        _opacity = torch.ones(num_points, 1).to(self.device)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, cholesky+self.cholesky_bound, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                features_dc, _opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render_pos": out_img}
    
    def forward(self):
        _opacity = torch.ones(self._xyz.shape[0], 1).to(self.device)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, _opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def update_optimizer(self):
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = Adan(self.parameters(), lr=self.lr)

    def removal_control(self, iter):
        iter_threshold_remove =4000  
        if iter>iter_threshold_remove:
            return
        rgb_weight = torch.norm(self.rgb_W, dim=1)
        _, sorted_indices = torch.sort(rgb_weight)
        removal_rate_per_step = self.removal_rate/int(iter_threshold_remove/(self.densification_interval))
        if iter < iter_threshold_remove:
            with torch.no_grad():
                remove_count = int(removal_rate_per_step * self.max_num_points)     
                remove_indices = sorted_indices[:remove_count]
                keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
                keep_indices[remove_indices] = False
                self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
                self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
                self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
                self.rgb_W = torch.nn.Parameter(self.rgb_W[keep_indices])
            for param_group in self.optimizer.param_groups:
                param_group['params'] = [p for p in self.parameters() if p.requires_grad]
        elif iter == iter_threshold_remove:
            remove_count = self._xyz.shape[0]-int(self.max_num_points * (1-self.removal_rate))
            if remove_count>0:
                with torch.no_grad():
                    remove_indices = sorted_indices[:remove_count]
                    keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
                    keep_indices[remove_indices] = False
                    self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
                    self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
                    self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
                    self.rgb_W = torch.nn.Parameter(self.rgb_W[keep_indices])  
            self.update_optimizer()

    def adaptive_control(self, iter):
        iter_threshold_remove =500  
        iter_threshold_add = 500
        densification_num = int(self.max_num_points * self.removal_rate)
        if iter>iter_threshold_add+iter_threshold_remove or iter<iter_threshold_add:
            if iter == 1 and densification_num > 0:
                new_xyz = torch.atanh(2 * (torch.rand(densification_num, 2) - 0.5)).to(self._xyz.device)
                new_cholesky = torch.rand(densification_num, 3).to(self._xyz.device)
                new_features_dc = torch.rand(densification_num, 3).to(self._xyz.device)
                new_rgb_W = 0.01 * torch.ones(densification_num, 1).to(self._xyz.device)
                self._xyz = torch.nn.Parameter(torch.cat((self._xyz, new_xyz), dim=0))
                self._cholesky = torch.nn.Parameter(torch.cat((self._cholesky, new_cholesky), dim=0))
                self._features_dc = torch.nn.Parameter(torch.cat((self._features_dc, new_features_dc), dim=0))
                self.rgb_W = torch.nn.Parameter(torch.cat((self.rgb_W, new_rgb_W), dim=0))
                for param_group in self.optimizer.param_groups:
                    param_group['params'] = [p for p in self.parameters() if p.requires_grad]
            return
        rgb_weight = torch.norm(self.rgb_W, dim=1)
        _, sorted_indices = torch.sort(rgb_weight)
        if iter < iter_threshold_add+iter_threshold_remove:
            with torch.no_grad():
                remove_count = int(densification_num/int(iter_threshold_remove/(self.densification_interval)))     
                remove_indices = sorted_indices[:remove_count]
                keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
                keep_indices[remove_indices] = False
                self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
                self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
                self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
                self.rgb_W = torch.nn.Parameter(self.rgb_W[keep_indices])
            for param_group in self.optimizer.param_groups:
                param_group['params'] = [p for p in self.parameters() if p.requires_grad]
        elif iter == iter_threshold_add+iter_threshold_remove:
            remove_count = self._xyz.shape[0]-int(self.max_num_points * (1-self.removal_rate))
            if remove_count>0:
                with torch.no_grad():
                    remove_indices = sorted_indices[:remove_count]
                    keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
                    keep_indices[remove_indices] = False
                    self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
                    self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
                    self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
                    self.rgb_W = torch.nn.Parameter(self.rgb_W[keep_indices])  
            self.update_optimizer()


    def train_iter_trace(self, gt_image,iter):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image.squeeze(0), gt_image.squeeze(0), self.loss_type, lambda_value=0)
        loss.backward()
        if (iter==1 or (iter) % (self.densification_interval) == 0) and self.isdensity:
            self.adaptive_control(iter)
        elif (iter) % (self.densification_interval) == 0 and self.isremoval:
            self.removal_control(iter)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        self.scheduler.step()
        return image
        

    def train_iter(self, gt_image,iter):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image.squeeze(0), gt_image.squeeze(0), self.loss_type, lambda_value=0)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        if (iter==1 or (iter) % (self.densification_interval) == 0) and self.isdensity:
            self.adaptive_control(iter)
        elif (iter) % (self.densification_interval) == 0 and self.isremoval:
            self.removal_control(iter)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        self.scheduler.step()
        return loss, psnr
    
    
    def pre_train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr
    


    
