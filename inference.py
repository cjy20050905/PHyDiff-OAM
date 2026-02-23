"""
PHyDiff-OAM: Inference & Evaluation (Final Dtype Fix)
-----------------------------------------------------
Benchmarks the trained model against traditional Back-Projection (BP).
"""

import os
# ⚡ Mirror for China Region
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.utils as vutils
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import local modules
from data_engine import AircraftRadarDataset
from models.physics_adapter import modify_unet_input_layer, preprocess_radar_signal

DEVICE = torch.device("cuda")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
CHECKPOINT_PATH = "checkpoints/radar_unet.pth"

def back_projection(S_radar, K):
    """Traditional Physics Baseline: Back-Projection Algorithm"""
    bp_image = torch.sum(S_radar * torch.conj(K.unsqueeze(0)), dim=1).abs()
    bp_image = (bp_image - bp_image.min()) / (bp_image.max() - bp_image.min() + 1e-8)
    return bp_image.unsqueeze(1)

def main():
    print("🚀 Starting Evaluation...")
    
    # 1. Load Model
    print("📦 Loading Model Weights...")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16).to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", variant="fp16", torch_dtype=torch.bfloat16).to(DEVICE)
    
    # Surgery & Dtype Fix
    unet = modify_unet_input_layer(unet, new_channels=12)
    unet = unet.to(device=DEVICE, dtype=torch.bfloat16)
    
    if os.path.exists(CHECKPOINT_PATH):
        unet.load_state_dict(torch.load(CHECKPOINT_PATH))
        print("✅ Loaded trained weights.")
    else:
        print("⚠️ Checkpoint not found! Using random weights.")

    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    
    # Empty Text Embeddings
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", variant="fp16", torch_dtype=torch.bfloat16).to(DEVICE)
    empty_ids = tokenizer([""] * 1, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(DEVICE)
    empty_embeds = text_encoder(empty_ids)[0]
    
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    # 2. Evaluation Loop
    dataset = AircraftRadarDataset(num_samples=50)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    K_matrix = dataset.K.to(DEVICE)
    
    psnr_list, ssim_list = [], []
    results_vis = []

    print("📊 Calculating Metrics...")
    unet.eval()
    
    with torch.no_grad():
        for i, (S_radar, I_gt) in enumerate(loader):
            S_radar = S_radar.to(DEVICE)
            
            # A. Traditional BP Baseline
            I_bp = back_projection(S_radar, K_matrix)
            I_bp = F.interpolate(I_bp, size=(512, 512), mode='bilinear')
            
            # B. AI Inference
            S_phy = preprocess_radar_signal(S_radar, target_dtype=torch.bfloat16)
            latents = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, device=DEVICE)
            scheduler.set_timesteps(20) 
            
            for t in scheduler.timesteps:
                unet_input = torch.cat([latents, S_phy], dim=1)
                noise_pred = unet(unet_input, t, encoder_hidden_states=empty_embeds).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            I_ai = vae.decode(latents / vae.config.scaling_factor).sample
            I_ai = (I_ai / 2 + 0.5).clamp(0, 1).cpu()
            I_gt = (I_gt / 2 + 0.5).clamp(0, 1).cpu()
            I_bp = I_bp.repeat(1, 3, 1, 1).cpu()
            
            # [关键修复] 转 NumPy 前必须先转 float32
            I_ai_np = I_ai.float().numpy()[0].transpose(1, 2, 0)
            I_gt_np = I_gt.float().numpy()[0].transpose(1, 2, 0)
            
            p = psnr(I_gt_np, I_ai_np, data_range=1.0)
            s = ssim(I_gt_np, I_ai_np, data_range=1.0, channel_axis=2, win_size=7)
            psnr_list.append(p)
            ssim_list.append(s)
            
            if i < 4:
                results_vis.append(torch.cat([I_gt, I_bp, I_ai], dim=3))

    # 3. Report & Save
    print(f"\n🏆 Final Results (Avg over 50 samples):")
    print(f"✅ PSNR: {np.mean(psnr_list):.4f} dB")
    print(f"✅ SSIM: {np.mean(ssim_list):.4f}")
    
    os.makedirs("results", exist_ok=True)
    vis_grid = torch.cat(results_vis, dim=2)
    vutils.save_image(vis_grid, "results/final_comparison.png")
    print("🖼️ Comparison image saved to results/final_comparison.png")

if __name__ == "__main__":
    main()
