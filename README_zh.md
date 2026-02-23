# PHyDiff-OAM

**Physics-aligned Diffusion for OAM Radar Imaging**

[English](README.md) | 简体中文

基于物理约束的扩散模型用于轨道角动量（OAM）雷达成像，专为 NVIDIA H800 NVL GPU 优化。

---

## 📋 项目简介

PHyDiff-OAM 是一个将物理约束与深度学习相结合的雷达成像项目，通过将 OAM 雷达信号直接注入 Stable Diffusion 模型，实现飞机目标的高质量检测和重建。

### 核心创新

- **硬连接策略（Hard-Concatenation）**: 将 8 通道 OAM 雷达物理特征与 4 通道噪声潜在表示连接，形成 12 通道输入
- **物理-AI 混合**: 基于 Straton-Chu 积分近似的 OAM 雷达仿真与 Stable Diffusion 扩散模型结合
- **H800 优化**: 使用 bfloat16 数据类型，充分利用 H800 GPU 的原生计算能力

---

## 🔧 硬件环境

- **GPU**: NVIDIA H800 NVL
- **推荐显存**: ≥ 40GB
- **数据类型**: bfloat16（H800 原生支持）

---

## 📦 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖

- PyTorch >= 2.4.0
- Diffusers >= 0.30.0
- Transformers >= 4.44.0
- NumPy >= 1.26.0
- scikit-image >= 0.24.0

---

## 🚀 快速开始

### 1. 训练模型

```bash
python train.py
```

**训练配置：**
- 批大小: 8
- 训练步数: 5000
- 学习率: 1e-4
- 优化器: AdamW
- 数据类型: bfloat16

**训练输出示例：**
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

训练完成后，模型权重将保存在 `checkpoints/radar_unet.pth`。

### 2. 推理与评估

```bash
python inference.py
```

**评估指标：**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

**推理输出示例：**
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

推理结果对比图将保存在 `results/final_comparison.png`。

---

## 📁 项目结构

```
PHyDiff-OAM/
├── data_engine.py              # 合成雷达数据生成引擎
├── train.py                    # H800 GPU 训练脚本
├── inference.py                # 推理与评估脚本
├── requirements.txt            # 依赖配置
├── models/
│   ├── __init__.py
│   └── physics_adapter.py      # 物理适配器（雷达信号预处理 + UNet 手术）
├── checkpoints/                # 模型权重保存目录
│   └── radar_unet.pth
└── results/                    # 推理结果保存目录
    └── final_comparison.png
```

---

## 🔬 技术细节

### 数据生成引擎 (`data_engine.py`)

**AircraftRadarDataset** 类实时生成合成飞机目标和 OAM 雷达回波：

- **图像分辨率**: 512×512（地面真值）
- **仿真网格**: 64×64（物理仿真）
- **OAM 模式数**: 8（模式范围: l = -3 到 +4）
- **雷达频率**: 10 GHz
- **物理模型**: 基于 Straton-Chu 积分近似的 Green 函数核

**OAM 感知矩阵：**
```
K(l, r) = exp(-j2kρ) * exp(jlφ)
```

其中：
- `k = 2π/λ` 为波数
- `ρ` 为径向距离
- `φ` 为方位角
- `l` 为 OAM 模式索引

### 物理适配器 (`models/physics_adapter.py`)

#### 1. 雷达信号预处理

```python
preprocess_radar_signal(S_radar, target_dtype)
```

- 将复数雷达回波转换为实值特征图
- 计算幅度: `|S| = √(real² + imag² + ε)`
- 逐批次 Min-Max 归一化
- 输出: [B, 8, 64, 64] 的 bfloat16 张量

#### 2. UNet 输入层手术

```python
modify_unet_input_layer(unet, new_channels=12)
```

- **原始输入**: 4 通道（噪声潜在表示）
- **扩展输入**: 12 通道（4 通道噪声 + 8 通道物理特征）
- **权重初始化策略**:
  - 前 4 通道: 复制预训练权重（保留知识）
  - 后 8 通道: 零初始化（避免破坏预训练分布）

### 训练流程 (`train.py`)

1. 加载 Stable Diffusion v1.5 组件（VAE、UNet、文本编码器）
2. 对 UNet 输入层进行"手术"，扩展为 12 通道
3. 强制转换为 bfloat16 数据类型（H800 优化）
4. 启用梯度检查点以节省显存
5. 冻结 VAE 和文本编码器，仅训练 UNet
6. 训练循环：
   - VAE 编码地面真值为潜在表示
   - 添加噪声并随机采样时间步
   - 连接噪声潜在表示与物理特征
   - UNet 预测噪声，计算 MSE 损失
   - 梯度裁剪（max_norm=1.0）

### 推理流程 (`inference.py`)

1. 加载训练好的 UNet 权重
2. 对 50 个测试样本进行推理
3. 与传统 Back-Projection (BP) 算法对比
4. 计算 PSNR 和 SSIM 指标
5. 生成可视化对比图

---

## 📊 性能指标

基于 50 个测试样本的平均结果：

| 指标 | 数值 |
|------|------|
| **PSNR** | 9.37 dB |
| **SSIM** | 0.069 |

*注：这些指标反映了模型在合成数据上的初步性能，可通过增加训练步数、调整超参数或使用更大的数据集进一步优化。*

---

## ⚙️ 配置说明

### 训练参数（`train.py`）

```python
BATCH_SIZE = 8          # 批大小
NUM_STEPS = 5000        # 训练步数
LEARNING_RATE = 1e-4    # 学习率
DEVICE = "cuda"         # 设备
```

### 数据生成参数（`data_engine.py`）

```python
num_samples = 2000      # 每个 epoch 的样本数
img_size = 512          # 地面真值图像分辨率
sim_size = 64           # 雷达仿真网格分辨率
num_modes = 8           # OAM 模式数量
freq = 10e9             # 雷达中心频率（10 GHz）
```

### 推理参数（`inference.py`）

```python
NUM_SAMPLES = 50        # 评估样本数
NUM_INFERENCE_STEPS = 50  # DDIM 推理步数
GUIDANCE_SCALE = 7.5    # 分类器自由引导强度
```

---

## 🌐 中国镜像支持

项目已配置 Hugging Face 中国镜像，加速模型下载：

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

## 🛠️ 常见问题

### 1. 显存不足

- 减小批大小（`BATCH_SIZE`）
- 启用梯度检查点（已默认启用）
- 使用梯度累积

### 2. 训练不稳定

- 确保使用 bfloat16 数据类型
- 检查梯度裁剪是否启用
- 降低学习率

### 3. 推理速度慢

- 减少推理步数（`NUM_INFERENCE_STEPS`）
- 使用更快的调度器（如 DPM-Solver++）

---

## 📝 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@software{phydiff_oam_2025,
  title={PHyDiff-OAM: Physics-aligned Diffusion for OAM Radar Imaging},
  author={Dryoung},
  year={2025},
  url={https://github.com/Dryoung/PHyDiff-OAM}
}
```

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - 基础扩散模型
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - 扩散模型库
- NVIDIA H800 NVL - 强大的计算支持

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- Email: 3241347200@qq.com
- GitHub Issues: [提交问题](https://github.com/Dryoung/PHyDiff-OAM/issues)

---

**最后更新**: 2025-02-23
