"""
PHyDiff-OAM: Training Script (Fixed Dtype Issue)
------------------------------------------------
Trains the Physics-Aligned UNet on the H800 GPU.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

# Import local modules
from data_engine import AircraftRadarDataset
from models.physics_adapter import modify_unet_input_layer, preprocess_radar_signal

# --- Configuration ---
DEVICE = torch.device("cuda")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
BATCH_SIZE = 8
NUM_STEPS = 5000  
SAVE_PATH = "checkpoints/radar_unet.pth"

def main():
    print(f"🚀 Starting PHyDiff-OAM Training on {torch.cuda.get_device_name(0)}...")
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Load Pre-trained Components
    print("📦 Loading Stable Diffusion v1.5 components...")
    # Load everything in BFloat16 initially
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16).to(DEVICE)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", variant="fp16", torch_dtype=torch.bfloat16).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", variant="fp16", torch_dtype=torch.bfloat16).to(DEVICE)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 2. Apply Hard-Concatenation Surgery
    print("🔧 Applying Hard-Concatenation surgery to UNet...")
    unet = modify_unet_input_layer(unet, new_channels=12)
    
    # [关键修复点] 手术后，新层是 float32，必须强制转回 bfloat16
    unet = unet.to(device=DEVICE, dtype=torch.bfloat16)
    
    unet.enable_gradient_checkpointing()

    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    # 4. Prepare Empty Text Embeddings
    empty_ids = tokenizer([""] * BATCH_SIZE, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(DEVICE)
    empty_embeds = text_encoder(empty_ids)[0]

    # 5. Data Loader
    dataset = AircraftRadarDataset(num_samples=10000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 6. Training Loop
    print(f"🔥 Start Training ({NUM_STEPS} steps)...")
    unet.train()
    
    for step, (S_radar, I_gt) in enumerate(dataloader):
        if step >= NUM_STEPS: break
        
        S_radar = S_radar.to(DEVICE)
        I_gt = I_gt.to(DEVICE, dtype=torch.bfloat16).clamp(-1.0, 1.0)
        
        # Prepare Physics Features
        S_phy = preprocess_radar_signal(S_radar, target_dtype=torch.bfloat16)

        optimizer.zero_grad()
        
        with torch.no_grad():
            latents = vae.encode(I_gt).latent_dist.sample() * vae.config.scaling_factor
            
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Concatenate: [Noisy Latents, Physics Features]
        # Ensure all inputs are bfloat16
        unet_input = torch.cat([noisy_latents, S_phy], dim=1)
        
        noise_pred = unet(unet_input, timesteps, encoder_hidden_states=empty_embeds[:S_radar.size(0)]).sample
        
        loss = F.mse_loss(noise_pred, noise)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            print(f"   Step {step+1:04d}/{NUM_STEPS} | Loss: {loss.item():.4f}")

    print(f"✅ Saving model to {SAVE_PATH}...")
    torch.save(unet.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    main()
