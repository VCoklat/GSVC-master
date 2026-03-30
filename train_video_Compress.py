import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from sklearn.mixture import GaussianMixture
import copy
from GaussianSplats_Compress import GaussianVideo_frame, GaussianVideo_delta

# ── GSVC-Turbo imports ────────────────────────────────────────────────
from optical_flow_warper import OpticalFlowWarper
from turbo_config import get_turbo_config, TurboConfig
class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image,
        frame_num,
        savdir,
        loss_type,
        num_points: int = 2000,
        model_name:str = "GaussianVideo",
        iterations:int = 30000,
        trained_model = None,
        p_trained_model=None,
        args = None,
        isremoval=False,
        removal_rate=0,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_to_tensor(image).to(self.device)
        self.frame_num=frame_num
        self.num_points = num_points
        if isremoval:
            self.num_points = int(num_points*(1-removal_rate))
        self.model_name=model_name
        self.data_name=args.data_name
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.save_everyimgs = args.save_everyimgs
        self.log_dir = Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
        self.loss_type = loss_type
        if p_trained_model is not None:
            self.gaussian_model = GaussianVideo_delta(loss_type=self.loss_type, opt_type="adan", num_points=self.num_points,iterations=self.iterations, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=args.lr, quantize=True).to(self.device)
            checkpoint = trained_model
            p_checkpoint = p_trained_model
            model_dict = self.gaussian_model.state_dict()
            delta_model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            p_pretrained_dict = {k: v for k, v in p_checkpoint.items() if k in delta_model_dict}
            delta_pretrained_dict = {
                k: pretrained_dict[k] - p_pretrained_dict[k]
                for k in pretrained_dict.keys() & p_pretrained_dict.keys()
            }
            delta_model_dict.update(delta_pretrained_dict)
            for param_name, buffer_name in [
                ("_xyz", "p_xyz"),
                ("_cholesky", "p_cholesky"),
                ("_features_dc", "p_features_dc")
            ]:
                if param_name in pretrained_dict:
                    delta_model_dict[buffer_name] = pretrained_dict[param_name]
            self.gaussian_model.load_state_dict(delta_model_dict)
        else:
            self.gaussian_model = GaussianVideo_frame(loss_type=self.loss_type, opt_type="adan", num_points=self.num_points,iterations=self.iterations, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=args.lr, quantize=True).to(self.device)
            checkpoint = trained_model
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
            

    def train(self): 
        progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        early_stopping = EarlyStopping(patience=100, min_delta=1e-7)
        for iter in range(1, int(self.iterations)+1):
            loss, psnr = self.gaussian_model.train_iter_quantize(self.gt_image)
            if best_psnr < psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
            # if early_stopping(loss.item()):
            #     break
        end_time = time.time() - start_time
        progress_bar.close()
        self.gaussian_model.load_state_dict(best_model_dict)
        psnr_value, ms_ssim_value, bpp, img = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100  

        Gmodel =self.gaussian_model.state_dict()
        filtered_Gmodel = {
            k: v for k, v in Gmodel.items()
            if k in ['_xyz', '_cholesky','_features_dc']
        }
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, bpp, filtered_Gmodel,img   
        
    
    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
        out_image = out["render"]
        mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_image.float(), self.gt_image.float(), data_range=1, size_average=True).item()
        m_bit, s_bit, r_bit, c_bit = out["unit_bit"]
        bpp = (m_bit + s_bit + r_bit + c_bit)/self.H/self.W
        transform = transforms.ToPILImage()
        img = transform(out_image.float().squeeze(0))
        if self.save_everyimgs:
            save_path_img = self.log_dir / "img"
            save_path_img.mkdir(parents=True, exist_ok=True)
            name =str(self.frame_num) + "_fitting.png" 
            img.save(str(save_path_img / name))
        return psnr, ms_ssim_value, bpp, img

def image_to_tensor(img: Image.Image):
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='Beauty', help="Training dataset"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianVideo"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to a checkpoint"
    )
    parser.add_argument(
        "--savdir", type=str, default="result", help="Path to save results"
    )
    parser.add_argument(
        "--savdir_m", type=str, default="models", help="Path to save models"
    )

    parser.add_argument(
        "--fps", type=int, default=120, help="number of frames per second (default: %(default)s)"
    )
    parser.add_argument(
        "--image_length", type=int, default=50, help="number of input frames (default: %(default)s)"
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="width (default: %(default)s)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="height (default: %(default)s)"
    )
    parser.add_argument(
        "--iterations", type=int, default=30000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points", type=int, default=4000, help="2D GS points (default: %(default)s)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: %(default)s)"
    )
    
    parser.add_argument("--loss_type", type=str, default="L2", help="Type of Loss")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--save_everyimgs", action="store_true", help="Save Every Images")
    parser.add_argument("--removal_rate", type=float, default=0.1, help="Removal rate")
    parser.add_argument("--is_rm", action="store_true", help="Removal control of gaussians setup")

    # ── GSVC-Turbo arguments ──────────────────────────────────────────
    parser.add_argument("--turbo", action="store_true",
                        help="Enable GSVC-Turbo optical flow pre-advection")
    parser.add_argument("--turbo_preset", type=str, default="default",
                        choices=["default", "conservative", "aggressive",
                                 "ablation_no_clamp", "ablation_no_downscale"],
                        help="Turbo configuration preset")
    parser.add_argument("--turbo_max_resolution", type=int, default=None,
                        help="Override max RAFT input resolution (FR1)")
    parser.add_argument("--turbo_max_flow", type=float, default=None,
                        help="Override max flow magnitude clamp (FR3)")
    parser.add_argument("--turbo_verbose", action="store_true",
                        help="Print per-frame turbo diagnostics")

    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    args.save_imgs=True
    loss_type=args.loss_type
    savdir=args.savdir
    savdir_m=args.savdir_m
    args.fps=24
    width = args.width
    height = args.height
    is_rm=args.is_rm
    removal_rate=args.removal_rate
    gmodel_save_path = Path(f"./checkpoints_quant/{savdir_m}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
    gmodel_save_path.mkdir(parents=True, exist_ok=True)
    img_list = []

    # ── GSVC-Turbo: Initialize OpticalFlowWarper ──────────────────────
    turbo_warper = None
    if args.turbo:
        turbo_cfg = get_turbo_config(args.turbo_preset)
        if args.turbo_max_resolution is not None:
            turbo_cfg.max_resolution = args.turbo_max_resolution
        if args.turbo_max_flow is not None:
            turbo_cfg.max_flow_magnitude = args.turbo_max_flow
        turbo_cfg.verbose = args.turbo_verbose
        turbo_warper = OpticalFlowWarper(config=turbo_cfg, device=torch.device("cuda:0"))
        print(f"[GSVC-Turbo] Compress mode enabled with preset='{args.turbo_preset}'")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses, bpps = [], [], [], [], [], []
    image_h, image_w = 0, 0
    video_frames = process_yuv_video(args.dataset, width, height)
    frame_length,start=len(video_frames),0
    if args.image_length <= frame_length:
        image_length=args.image_length
    else:
        image_length=frame_length
    Gmodel=None
    Overfit_gmodels_state_dict = torch.load(args.model_path,map_location=torch.device("cuda:0"))
    gmodels_state_dict = {}
    output_path_K_frames = Path(f"./checkpoints/{savdir}/{args.data_name}/K_frames.txt")
    if output_path_K_frames.exists():
        # If exists, read the file and assign values to K_frames
        with open(output_path_K_frames, "r") as f:
            K_frames = [int(line.strip()) for line in f.readlines()]
    else:
        K_frames = [1]
    for i in range(start, start+image_length):
        frame_num=i+1
        modelid=f"frame_{i + 1}"
        Model = Overfit_gmodels_state_dict[modelid]
        if frame_num in K_frames:
        # if frame_num ==1:
            print(f"modelid:frame_{i + 1};")
            trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=args.num_points,
                iterations=args.iterations, model_name=args.model_name, args=args, trained_model=Model,isremoval=is_rm,removal_rate=removal_rate)
        else:
            p_modelid = f"frame_{i}"
            P_Model = Overfit_gmodels_state_dict[p_modelid]
            print(f"modelid:{modelid}; p_modelid:{p_modelid}")
            trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=args.num_points,
                iterations=args.iterations, model_name=args.model_name, args=args, p_trained_model =P_Model, trained_model=Model,isremoval=is_rm,removal_rate=removal_rate)

            # ── GSVC-Turbo: Pre-advect delta Gaussian positions ───────
            if turbo_warper is not None and i > 0:
                turbo_diag = turbo_warper.warp(
                    prev_frame=video_frames[i - 1],
                    curr_frame=video_frames[i],
                    gaussian_model=trainer.gaussian_model,
                    is_delta=True,  # Delta model: _xyz stores residuals
                )
                if args.turbo_verbose:
                    print(f"  [Turbo-Compress] Frame {frame_num}: {turbo_diag}")

        psnr, ms_ssim, training_time, eval_time, eval_fps, bpp, Gmodel,img = trainer.train()
        img_list.append(img)
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        bpps.append(bpp)
        image_h += trainer.H
        image_w += trainer.W
        gmodels_state_dict[f"frame_{frame_num}"] = Gmodel
        torch.cuda.empty_cache()
        if i==0 or (i+1)%1==0:
            logwriter.write("Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            frame_num, trainer.H, trainer.W, psnr, ms_ssim, bpp, training_time, eval_time, eval_fps))
    torch.save(gmodels_state_dict, gmodel_save_path / "gmodels_state_dict.pth")
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_bpp = torch.tensor(bpps).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length
    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_bpp, avg_training_time, avg_eval_time, avg_eval_fps))    
    generate_video(savdir,img_list, args.data_name, args.model_name,args.fps,args.iterations,args.num_points,origin=True)
if __name__ == "__main__":
    
    main(sys.argv[1:])


