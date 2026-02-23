"""
PHyDiff-OAM: Data Generation Engine
-----------------------------------
This module implements the On-the-fly Synthetic Radar Dataset.
It generates random aircraft geometries in RAM and simulates their 
Orbital Angular Momentum (OAM) radar echoes using the Stratton-Chu integral approximation.

Author: dryoung
License: MIT
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math

class AircraftRadarDataset(Dataset):
    """
    A PyTorch Dataset that generates synthetic aircraft targets and their corresponding
    OAM radar echoes on-the-fly. This eliminates the need for HDD storage.
    """
    def __init__(self, num_samples=2000, img_size=512, sim_size=64, num_modes=8, freq=10e9):
        """
        Args:
            num_samples (int): Number of samples per epoch (virtual size).
            img_size (int): Resolution of the Ground Truth target image (e.g., 512).
            sim_size (int): Resolution of the radar simulation grid (e.g., 64).
            num_modes (int): Number of OAM modes (l). Typically 8 for sparse aperture.
            freq (float): Radar center frequency in Hz.
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.sim_size = sim_size
        self.num_modes = num_modes
        
        # Pre-calculate the OAM Sensing Matrix (Green's Function Kernel)
        # Formula: K(l, r) = exp(-j2kρ) * exp(jlφ)
        self.c = 3e8
        self.lambda_c = self.c / freq
        self.k = 2 * torch.pi / self.lambda_c
        
        # OAM modes range: e.g., l = [-3, -2, ..., 3, 4]
        self.modes = torch.arange(-num_modes//2 + 1, num_modes//2 + 1).float()
        
        # Coordinate grids for simulation
        x = torch.linspace(-5, 5, sim_size)
        y = torch.linspace(-5, 5, sim_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        # Polar coordinates
        rho = torch.sqrt(grid_x**2 + grid_y**2) + 1e-5
        phi = torch.atan2(grid_y, grid_x)
        
        # Register the complex-valued sensing matrix [Modes, H, W]
        self.K = torch.zeros(num_modes, sim_size, sim_size, dtype=torch.complex64)
        for i, l in enumerate(self.modes):
            phase_term = -2 * self.k * rho + l * phi
            self.K[i] = torch.exp(1j * phase_term)

    def __len__(self):
        return self.num_samples

    def _draw_random_aircraft(self, img_tensor):
        """
        Procedurally draws a random aircraft geometry (fuselage + wings + tail).
        """
        size = img_tensor.shape[-1]
        
        # Random parameters for geometry
        cx, cy = torch.rand(2) * 0.6 + 0.2  # Center position
        length = torch.rand(1) * 0.4 + 0.3  # Fuselage length
        width = length * (torch.rand(1) * 0.1 + 0.05) # Fuselage width
        wing_span = length * (torch.rand(1) * 0.6 + 0.4)
        wing_pos = torch.rand(1) * 0.2
        angle = torch.rand(1) * 3.14159 # Random rotation
        
        # Generate coordinate grid
        x = torch.linspace(0, 1, size)
        y = torch.linspace(0, 1, size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        # Rotate coordinates
        rel_x = (grid_x - cx) * torch.cos(angle) + (grid_y - cy) * torch.sin(angle)
        rel_y = -(grid_x - cx) * torch.sin(angle) + (grid_y - cy) * torch.cos(angle)
        
        # 1. Fuselage (Ellipse)
        fuselage = (rel_x**2 / (width/2)**2 + rel_y**2 / (length/2)**2) < 1.0
        
        # 2. Wings (Ellipse)
        wings = ((rel_x - wing_pos)**2 / (wing_span/2)**2 + rel_y**2 / (width*1.5)**2) < 1.0
        
        # 3. Tail (Small Ellipse)
        tail = ((rel_x + length/2.5)**2 / (wing_span/3)**2 + rel_y**2 / (width)**2) < 1.0
        
        # Combine parts
        img_tensor[0, fuselage | wings | tail] = 1.0
        return img_tensor

    def __getitem__(self, idx):
        """
        Returns:
            S_radar (Tensor): Complex radar echo [Modes, 64, 64]
            I_gt (Tensor): Ground truth image normalized to [-1, 1] [3, 512, 512]
        """
        # 1. Generate High-Res Ground Truth
        img = torch.zeros(1, self.img_size, self.img_size)
        img = self._draw_random_aircraft(img)
        
        # 2. Downsample for Physics Simulation (simulating limited radar resolution)
        img_sim = F.interpolate(img.unsqueeze(0), size=(self.sim_size, self.sim_size), 
                                mode='bilinear', align_corners=False).squeeze(0)
        
        # 3. Physics Simulation (Forward Scattering)
        # Element-wise multiplication in complex domain
        S_radar = img_sim.to(torch.complex64) * self.K
        
        # 4. Prepare GT for VAE (Repeat to 3 channels RGB, Normalize to [-1, 1])
        I_gt = img.repeat(3, 1, 1) * 2.0 - 1.0
        
        return S_radar, I_gt
        