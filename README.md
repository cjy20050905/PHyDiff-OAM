# PHyDiff-OAM

**Physics-aligned Diffusion for OAM Radar Imaging**

English | [简体中文](README_zh.md)

A physics-constrained diffusion model for Orbital Angular Momentum (OAM) radar imaging, optimized for NVIDIA H800 NVL GPU.

---

## 📋 Overview

PHyDiff-OAM is a radar imaging project that combines physical constraints with deep learning. By directly injecting OAM radar signals into the Stable Diffusion model, it achieves high-quality detection and reconstruction of aircraft targets.

### Key Innovations

- **Hard-Concatenation Strategy**: Concatenates 8-channel OAM radar physical features with 4-channel noisy latent representations to form a 12-channel input
- **Physics-AI Hybrid**: Combines OAM radar simulation based on Straton-Chu integral approximation with Stable Diffusion diffusion model
- **H800 Optimization**: Uses bfloat16 data type to fully leverage the native computing power of H800 GPU

---

## 🔧 Hardware Requirements

- **GPU**: NVIDIA H800 NVL
- **Recommended VRAM**: ≥ 40GB
- **Data Type**: bfloat16 (natively supported by H800)

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

### Main Dependencies

- PyTorch >= 2.4.0
- Diffusers >= 0.30.0
- Transformers >= 4.44.0
- NumPy >= 1.26.0
- scikit-image >= 0.24.0

---

## 🚀 Quick Start

### 1. Training

```bash
python train.py
```

**Training Configuration:**
- Batch Size: 8
- Training Steps: 5000
- Learning Rate: 1e-4
- Optimizer: AdamW
- Data Type: bfloat16

**Training Output Example:**
```
🚀 Starting PHyDiff-OAM Training on NVIDIA H800 NVL...
📦 Loading Stable Diffusion v1.5 components...
🔧 Applying Hard-Concatenation surgery to UNet...
🔥 Start Training (5000 steps)...
   Step 0100/5000 | Loss: 0.0027
   Step 0200/5000 | Loss: 0.0101
   ...
   Step 1200/5000 | Loss: 0.0092
✅ Saving model to checkpoints/radar_unet.pth...
```

After training, model weights will be saved to `checkpoints/radar_unet.pth`.

### 2. Inference and Evaluation

```bash
python inference.py
```

**Evaluation Metrics:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

**Inference Output Example:**
```
🚀 Starting Evaluation...
📦 Loading Model Weights...
✅ Loaded trained weights.
📊 Calculating Metrics...

🏆 Final Results (Avg over 50 samples):
✅ PSNR: 9.3698 dB
✅ SSIM: 0.0694
🖼️ Comparison image saved to results/final_comparison.png
```

Inference comparison images will be saved to `results/final_comparison.png`.

---

## 📁 Project Structure

```
PHyDiff-OAM/
├── data_engine.py              # Synthetic radar data generation engine
├── train.py                    # H800 GPU training script
├── inference.py                # Inference and evaluation script
├── requirements.txt            # Dependency configuration
├── models/
│   ├── __init__.py
│   └── physics_adapter.py      # Physics adapter (radar signal preprocessing + UNet surgery)
├── checkpoints/                # Model weights directory
│   └── radar_unet.pth
└── results/                    # Inference results directory
    └── final_comparison.png
```

---

## 🔬 Technical Details

### Data Generation Engine (`data_engine.py`)

The **AircraftRadarDataset** class generates synthetic aircraft targets and OAM radar echoes in real-time:

- **Image Resolution**: 512×512 (ground truth)
- **Simulation Grid**: 64×64 (physical simulation)
- **Number of OAM Modes**: 8 (mode range: l = -3 to +4)
- **Radar Frequency**: 10 GHz
- **Physical Model**: Green's function kernel based on Straton-Chu integral approximation

**OAM Sensing Matrix:**
```
K(l, r) = exp(-j2kρ) * exp(jlφ)
```

Where:
- `k = 2π/λ` is the wave number
- `ρ` is the radial distance
- `φ` is the azimuth angle
- `l` is the OAM mode index

### Physics Adapter (`models/physics_adapter.py`)

#### 1. Radar Signal Preprocessing

```python
preprocess_radar_signal(S_radar, target_dtype)
```

- Converts complex radar echoes to real-valued feature maps
- Computes magnitude: `|S| = √(real² + imag² + ε)`
- Batch-wise Min-Max normalization
- Output: [B, 8, 64, 64] bfloat16 tensor

#### 2. UNet Input Layer Surgery

```python
modify_unet_input_layer(unet, new_channels=12)
```

- **Original Input**: 4 channels (noisy latent representation)
- **Extended Input**: 12 channels (4-channel noise + 8-channel physical features)
- **Weight Initialization Strategy**:
  - First 4 channels: Copy pretrained weights (preserve knowledge)
  - Last 8 channels: Zero initialization (avoid disrupting pretrained distribution)

### Training Pipeline (`train.py`)

1. Load Stable Diffusion v1.5 components (VAE, UNet, text encoder)
2. Perform "surgery" on UNet input layer, expanding to 12 channels
3. Force conversion to bfloat16 data type (H800 optimization)
4. Enable gradient checkpointing to save VRAM
5. Freeze VAE and text encoder, train only UNet
6. Training loop:
   - VAE encodes ground truth to latent representation
   - Add noise and randomly sample timesteps
   - Concatenate noisy latent representation with physical features
   - UNet predicts noise, compute MSE loss
   - Gradient clipping (max_norm=1.0)

### Inference Pipeline (`inference.py`)

1. Load trained UNet weights
2. Perform inference on 50 test samples
3. Compare with traditional Back-Projection (BP) algorithm
4. Calculate PSNR and SSIM metrics
5. Generate visualization comparison images

---

## 📊 Performance Metrics

Average results based on 50 test samples:

| Metric | Value |
|--------|-------|
| **PSNR** | 9.37 dB |
| **SSIM** | 0.069 |

*Note: These metrics reflect the model's preliminary performance on synthetic data and can be further optimized by increasing training steps, adjusting hyperparameters, or using larger datasets.*

---

## ⚙️ Configuration

### Training Parameters (`train.py`)

```python
BATCH_SIZE = 8          # Batch size
NUM_STEPS = 5000        # Training steps
LEARNING_RATE = 1e-4    # Learning rate
DEVICE = "cuda"         # Device
```

### Data Generation Parameters (`data_engine.py`)

```python
num_samples = 2000      # Samples per epoch
img_size = 512          # Ground truth image resolution
sim_size = 64           # Radar simulation grid resolution
num_modes = 8           # Number of OAM modes
freq = 10e9             # Radar center frequency (10 GHz)
```

### Inference Parameters (`inference.py`)

```python
NUM_SAMPLES = 50        # Number of evaluation samples
NUM_INFERENCE_STEPS = 50  # DDIM inference steps
GUIDANCE_SCALE = 7.5    # Classifier-free guidance strength
```

---

## 🌐 China Mirror Support

The project is configured with Hugging Face China mirror to accelerate model downloads:

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

## 🛠️ Troubleshooting

### 1. Out of Memory

- Reduce batch size (`BATCH_SIZE`)
- Enable gradient checkpointing (enabled by default)
- Use gradient accumulation

### 2. Training Instability

- Ensure bfloat16 data type is used
- Check if gradient clipping is enabled
- Lower learning rate

### 3. Slow Inference

- Reduce inference steps (`NUM_INFERENCE_STEPS`)
- Use faster schedulers (e.g., DPM-Solver++)

---

## 📝 Citation

If this project helps your research, please consider citing:

```bibtex
@software{phydiff_oam_2025,
  title={PHyDiff-OAM: Physics-aligned Diffusion for OAM Radar Imaging},
  author={Dryoung},
  year={2025},
  url={https://github.com/Dryoung/PHyDiff-OAM}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Base diffusion model
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - Diffusion model library
- NVIDIA H800 NVL - Powerful computing support

---

## 📧 Contact

For questions or suggestions, please contact:

- Email: 3241347200@qq.com
- GitHub Issues: [Submit Issue](https://github.com/Dryoung/PHyDiff-OAM/issues)

---

**Last Updated**: 2025-02-23
