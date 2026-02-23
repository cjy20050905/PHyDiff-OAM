"""
PHyDiff-OAM: Physics Adapter
----------------------------
Utilities for modifying the Stable Diffusion UNet to accept physical radar signals.
Implements the "Hard-Concatenation" strategy.
"""

import torch
import torch.nn as nn

def preprocess_radar_signal(S_radar, target_dtype=torch.bfloat16):
    """
    Converts complex-valued radar echoes into normalized feature maps.
    
    Args:
        S_radar (Tensor): Complex input [B, Modes, H, W]
    Returns:
        S_phy (Tensor): Normalized magnitude features [B, Modes, H, W]
    """
    # 1. Compute Magnitude (with epsilon for numerical stability)
    S_phy = torch.sqrt(S_radar.real**2 + S_radar.imag**2 + 1e-8)
    
    # 2. Min-Max Normalization per batch
    # This ensures the physical features match the distribution of latent noise
    S_min = S_phy.view(S_phy.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    S_max = S_phy.view(S_phy.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    S_phy = (S_phy - S_min) / (S_max - S_min + 1e-8)
    
    return S_phy.to(dtype=target_dtype)

def modify_unet_input_layer(unet, new_channels=12):
    """
    Performs 'surgery' on the UNet input layer to enable Hard Concatenation.
    
    Original Input: 4 channels (Latent Noise)
    New Input: 12 channels (4 Latent + 8 Physics)
    """
    with torch.no_grad():
        old_conv = unet.conv_in
        
        # Create new convolution layer with expanded input channels
        new_conv = nn.Conv2d(
            in_channels=new_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        
        # Initialize weights:
        # Copy original weights for the first 4 channels (preserve pre-trained knowledge)
        new_conv.weight[:, :4, :, :] = old_conv.weight
        
        # Initialize physics channel weights to ZERO (Zero-Initialization Strategy)
        # This ensures the model starts training without breaking the pre-trained distribution
        nn.init.zeros_(new_conv.weight[:, 4:, :, :])
        
        # Copy bias
        new_conv.bias = old_conv.bias
        
        # Replace the layer in UNet
        unet.conv_in = new_conv
        
    return unet
