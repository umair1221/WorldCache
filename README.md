<p align="center">
  <h1 align="center">🌍 WorldCache: Content-Aware Caching for Accelerated Video World Models</h1>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-worldcache-parameters">Parameters</a> •
  <a href="#-method-overview">Method</a> •
  <a href="#-results">Results</a> •
  <a href="#-acknowledgements">Acknowledgements</a>
</p>

---

## 📖 Overview

**WorldCache** is a **training-free, plug-and-play** inference acceleration framework for diffusion-based video world models. It significantly reduces end-to-end latency by intelligently caching and reusing intermediate computations during the denoising process, without sacrificing generation quality.

Built on top of [NVIDIA Cosmos-Predict2](https://github.com/NVIDIA/Cosmos-Predict2), WorldCache introduces a suite of content-aware caching strategies that adapt to the dynamic nature of video generation — from motion intensity to spectral content — achieving up to **3× speedup** over the baseline while maintaining high visual fidelity.

> **Key Insight:** Not all denoising steps require full forward passes through the diffusion transformer. WorldCache identifies redundant computations via lightweight probing and replaces them with cached approximations, guided by motion, saliency, and spectral signals.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **🔌 Training-Free** | No fine-tuning or retraining required. Applied entirely at inference time. |
| **🎯 Content-Aware Caching (CFC)** | Adaptive threshold that responds to motion intensity — computes more in dynamic scenes, caches aggressively in static ones. |
| **🌊 Flow-Warped Feature Caching (OFA)** | Uses GPU-native optical flow to warp cached features, aligning them with scene motion for higher-quality approximations. |
| **🔬 Saliency-Weighted Drift (SWD)** | Prioritizes perceptually important regions (edges, textures) when deciding whether to reuse cached features. |
| **🧠 Online System Identification (OSI)** | Computes optimal interpolation coefficients via least-squares fitting for more accurate feature extrapolation. |
| **📈 Adaptive Threshold Scheduling (ATS)** | Dynamically relaxes caching thresholds in later denoising steps where features converge naturally. |
| **⚡ Torch Compile Support** | Compatible with `torch.compile` for additional kernel-level speedups. |

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA 12.x compatible GPU (tested on NVIDIA A100 80GB, H100)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/umair1221/WorldCache.git
cd WorldCache

# Create a virtual environment (using uv)
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Set PYTHONPATH
export PYTHONPATH="./:$PYTHONPATH"
```

### Model Checkpoints

WorldCache uses the pre-trained checkpoints from [NVIDIA Cosmos-Predict2](https://github.com/NVIDIA/Cosmos-Predict2). Follow the official instructions to download the model weights (2B or 14B variants) and place them in the `checkpoints/` directory.

---

## 🚀 Quick Start

### Text-to-World Generation

```bash
# Basic WorldCache inference (recommended settings)
CUDA_VISIBLE_DEVICES=0 python examples/inference.py \
  -i path/to/prompt.json \
  -o outputs/worldcache_output \
  --model 2B/post-trained \
  --disable-guardrails \
  --worldcache_enabled \
  --worldcache_motion_sensitivity 2 \
  --worldcache_flow_enabled \
  --worldcache_flow_scale 2.0 \
  --worldcache_rel_l1_thresh 0.08 \
  --worldcache_ret_ratio 0.2 \
  --worldcache_saliency_enabled \
  --worldcache_saliency_weight 1.0 \
  --worldcache_dynamic_decay \
  --worldcache_probe_depth 3 \
  --use_torch_compile
```

### Batch Inference

```bash
# Process multiple prompts from JSON files
CUDA_VISIBLE_DEVICES=0 python examples/inference.py \
  -i cosmos_eval_jsons/*.json \
  -o outputs/batch_output \
  --model 2B/post-trained \
  --disable-guardrails \
  --worldcache_enabled \
  --worldcache_motion_sensitivity 2 \
  --worldcache_flow_enabled \
  --worldcache_dynamic_decay
```

### Input JSON Format

Each input JSON file should follow this format:

```json
{
  "name": "sample_001",
  "prompt": "A robot arm picks up a red cube from a table and places it into a box.",
  "inference_type": "text2world",
  "num_steps": 35,
  "seed": 42,
  "guidance": 7
}
```

For **Image-to-World** or **Video-to-World** generation, set `inference_type` to `"image2world"` or `"video2world"` and include an `input_path` field pointing to the conditioning image/video.

---

## ⚙️ WorldCache Parameters

### Core Parameters

| Parameter | Default | Description |
|---|---|---|
| `--worldcache_enabled` | `False` | Enable WorldCache acceleration. |
| `--worldcache_rel_l1_thresh` | `0.08` | Base relative L1 drift threshold. Lower = higher quality, less caching. |
| `--worldcache_ret_ratio` | `0.2` | Fraction of initial steps to always compute fully (warm-up phase). |
| `--worldcache_probe_depth` | `2` | Number of initial transformer blocks used as a lightweight probe. |
| `--worldcache_motion_sensitivity` | `5.0` | Motion sensitivity (α). Higher = more responsive to motion (less skipping in dynamic scenes). |

### Content-Aware Modules

| Parameter | Default | Description |
|---|---|---|
| `--worldcache_flow_enabled` | `False` | Enable Optical Flow-based Feature Alignment (OFA). |
| `--worldcache_flow_scale` | `0.5` | Optical flow downscale factor. `2.0` = full resolution; `0.5` = 2× downsampled (faster). |
| `--worldcache_saliency_enabled` | `False` | Enable Saliency-Weighted Drift (SWD). |
| `--worldcache_saliency_weight` | `5.0` | Saliency weight (β). Controls how much salient regions influence the caching decision. |
| `--worldcache_osi_enabled` | `False` | Enable Online System Identification (OSI) for optimal gamma computation. |
| `--worldcache_dynamic_decay` | `False` | Enable Adaptive Threshold Scheduling (ATS). Relaxes threshold in later steps. |

### Advanced Parameters

| Parameter | Default | Description |
|---|---|---|
| `--worldcache_hf_enabled` | `False` | Enable Spectral-Adaptive Caching (high-frequency drift monitoring). |
| `--worldcache_hf_thresh` | `0.01` | Threshold for high-frequency drift. Exceeding this aborts caching. |
| `--worldcache_aduc_enabled` | `False` | Enable Adaptive Unconditional Caching (skip unconditional CFG pass in later steps). |
| `--worldcache_aduc_start` | `0.5` | Step ratio after which AdUC activates (e.g., 0.5 = after 50% of denoising). |
| `--use_torch_compile` | `False` | Enable `torch.compile` for additional kernel-level optimizations. |

### Recommended Configurations

```bash
# 🏎️ Balanced (Quality ≈ Baseline, ~2.5× Speedup)
--worldcache_enabled --worldcache_motion_sensitivity 2 \
--worldcache_flow_enabled --worldcache_dynamic_decay \
--worldcache_saliency_enabled --worldcache_saliency_weight 1.0 \
--worldcache_probe_depth 3

# ⚡ Aggressive (Maximum Speed, ~3× Speedup)
--worldcache_enabled --worldcache_motion_sensitivity 2 \
--worldcache_flow_enabled --worldcache_dynamic_decay \
--worldcache_saliency_enabled --worldcache_saliency_weight 1.0 \
--worldcache_rel_l1_thresh 0.12 --worldcache_ret_ratio 0.15 \
--worldcache_probe_depth 3 --use_torch_compile
```

---

## 🔬 Method Overview

WorldCache introduces a multi-signal content-aware caching framework for diffusion transformer inference:

```
┌─────────────────────────────────────────────────────────────┐
│                    Denoising Step t                         │
│                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────────┐  │
│  │  Input    │───▶│ Probe Blocks│───▶│  Drift Analysis   │  │
│  │  x_t      │    │ (1..K)      │    │                   │  │
│  └──────────┘    └─────────────┘    │  • CFC (Motion)   │  │
│                                      │  • SWD (Saliency) │  │
│                                      │  • ATS (Schedule) │  │
│                                      └────────┬──────────┘  │
│                                               │              │
│                              ┌────────────────┴────────┐     │
│                              │                         │     │
│                         drift < θ              drift ≥ θ     │
│                         (Cache Hit)          (Cache Miss)    │
│                              │                         │     │
│                    ┌─────────▼────────┐    ┌──────────▼──┐  │
│                    │  OSI Extrapol.   │    │ Full Forward │  │
│                    │  + OFA Warping   │    │   Pass       │  │
│                    │  (Approximate)   │    │ (Update      │  │
│                    └─────────┬────────┘    │  Cache)      │  │
│                              │             └──────────┬──┘  │
│                              └────────────┬───────────┘     │
│                                           ▼                  │
│                                    Output x_{t-1}            │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Content-Aware Feature Caching (CFC):** A motion-adaptive threshold mechanism that adjusts the caching aggressiveness based on the velocity of latent features between timesteps. Fast-moving scenes receive more compute; static scenes are cached more aggressively.

2. **Optical Flow-based Feature Alignment (OFA):** When reusing cached residuals, the cached features are warped using GPU-native Lucas-Kanade optical flow to compensate for inter-frame motion, reducing artifacts.

3. **Saliency-Weighted Drift (SWD):** Drift is weighted by a spatial saliency map (computed from channel-wise variance), ensuring that perceptually important regions have stronger influence on the cache-or-compute decision.

4. **Online System Identification (OSI):** Instead of heuristic interpolation, OSI computes the optimal scaling factor (γ) for feature extrapolation using a closed-form least-squares solution.

5. **Adaptive Threshold Scheduling (ATS):** The caching threshold is dynamically relaxed in later denoising steps, exploiting the natural convergence of the diffusion process to cache more aggressively without quality loss.

---

## 📊 Results

WorldCache achieves significant speedups across different model sizes and step budgets while maintaining generation quality:

| Model | Method | Latency (s) | Speedup | Quality Score |
|---|---|---|---|---|
| Cosmos 2B | Baseline | 57.2 | 1.0× | 0.7920 |
| Cosmos 2B | DiCache | 39.5 | 1.5× | 0.7890 |
| Cosmos 2B | **WorldCache** | **24.5** | **2.3×** | **0.7920** |
| Cosmos 14B | Baseline | 199.0 | 1.0× | — |
| Cosmos 14B | DiCache | 86.5 | 2.3× | — |
| Cosmos 14B | **WorldCache** | **66.2** | **3.0×** | — |

> Evaluated on the Physical-AI-Bench comprising 438 diverse physical reasoning prompts.

---

## 📁 Project Structure

```
WorldCache/
├── cosmos_predict2/                    # Core model library
│   ├── _src/
│   │   └── predict2/
│   │       └── inference/
│   │           ├── worldcache_utils.py # 🌟 WorldCache core implementation
│   │           ├── dicache_utils.py    # DiCache baseline
│   │           ├── teacache_utils.py   # TeaCache baseline
│   │           ├── easycache_utils.py  # EasyCache baseline
│   │           └── video2world.py      # Video generation pipeline
│   ├── config.py                       # All CLI arguments & model configs
│   └── inference.py                    # High-level inference API
├── examples/
│   └── inference.py                    # Main inference entry point
├── cosmos_eval_jsons/                  # Evaluation prompt JSONs
├── checkpoints/                        # Model weights (not tracked)
├── pyproject.toml                      # Project dependencies
└── README.md                           # This file
```

---

## 🏗️ Built On

This project is built on top of the [**NVIDIA Cosmos-Predict2**](https://github.com/NVIDIA/Cosmos-Predict2) platform. We gratefully acknowledge NVIDIA for open-sourcing their state-of-the-art video world model framework.

---

## 🙏 Acknowledgements

We acknowledge the following works that informed and inspired this project:

- **[Cosmos-Predict2](https://github.com/NVIDIA/Cosmos-Predict2)** — NVIDIA's world foundation model platform that serves as the backbone for this project.
- **[DiCache](https://arxiv.org/abs/2407.02705)** — Ma et al., "Learning to Cache: Accelerating Diffusion Transformer with Layer Caching," *NeurIPS 2024*. The foundational caching framework upon which WorldCache builds.
- **[TeaCache](https://arxiv.org/abs/2411.19150)** — Liu et al., "TeaCache: Timestep-Aware Cache for Accelerating Diffusion Model," *2024*. Timestep-aware caching approach for diffusion models.
- **[FasterCache](https://arxiv.org/abs/2410.05355)** — Li et al., "FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality," *2024*. Training-free acceleration methods for video diffusion.
- **[Physical-AI-Bench](https://arxiv.org/abs/xxxx.xxxxx)** — Comprehensive evaluation benchmark for physical AI world models.

---

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@misc{worldcache2026,
  title={WorldCache: Content-Aware Caching for Accelerated Video World Models},
  author={},
  year={2026},
  note={Built on NVIDIA Cosmos-Predict2}
}
```

---

## 📄 License

This project inherits the [Apache 2.0 License](LICENSE) from the NVIDIA Cosmos-Predict2 codebase.
