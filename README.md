<p align="center">
  <img src="assets/logo.png" width="120" alt="WorldCache Logo">
  <h1 align="center">🌍 WorldCache: Content-Aware Caching for Accelerated Video World Models</h1>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2503.XXXXX-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://umair1221.github.io/WorldCache/" target="_blank">
    <img src="https://img.shields.io/badge/Project-Website-blue.svg" alt="Project Website">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="License">
  </a>
</p>

<p align="center">
  <strong>WorldCache</strong> is a <strong>training-free, plug-and-play</strong> inference acceleration framework for diffusion-based video world models. It achieves up to <strong>3.0× speedup</strong> while strictly maintaining temporal coherence and visual fidelity.
</p>

<p align="center">
  <img src="assets/figures/teaser.png" width="1000" alt="WorldCache Teaser">
</p>

---

## 📖 Abstract

Video World Models (VWMs) increasingly rely on large-scale diffusion transformers to simulate complex spatial dynamics. However, the high computational cost of autoregressive generation remains a significant bottleneck. **WorldCache** overcomes this by identifying temporal and spatial redundancies in the denoising process. 

Unlike naive caching which causes "motion drift," WorldCache uses a suite of content-aware modules—**Causal Feature Caching (CFC)**, **Saliency-Weighted Drift (SWD)**, **Optimal Feature Approximation (OFA)**, and **Adaptive Threshold Scheduling (ATS)**—to predict skipped computation rather than blindly copying it. Our method is training-free and generalizes across leading architectures like **NVIDIA Cosmos** and **WAN2.1**.

---

## ✨ Key Components

WorldCache is driven by four key technical ideologies:

| Module | Icon | Description |
| :--- | :---: | :--- |
| **Causal Feature Caching (CFC)** | ⚡ | Dynamically scales caching tolerance based on early layer motion velocity. |
| **Saliency-Weighted Drift (SWD)** | 🎯 | Penalizes caching errors in perceptually critical high-frequency regions. |
| **Optimal Feature Approx. (OFA)** | 🌊 | Interpolates skipped cache states using trajectory matching and optical flow. |
| **Adaptive Threshold Scheduling (ATS)** | 📈 | Exponentially relaxes caching constraints in later denoising stages. |

---

## 🔬 Method Overview

WorldCache treats caching like a localized prediction. It controls the pace with causal tracking while interpolating the next state.

<p align="center">
  <img src="assets/figures/pipeline.png" width="900" alt="WorldCache Pipeline">
</p>

### Technical Highlights
- **Drift Probing:** Uses the first $K$ blocks of the transformer as a lightweight proxy for global drift.
- **Motion-Adaptive Thresholds:** Uses $\alpha$-scaled motion signals to prevent "ghosting" artifacts in high-dynamics scenes.
- **Saliency Mapping:** Weights L1 drift by spatial saliency (channel-wise variance) to preserve fine details.

---

## 📊 Technical Rigor & Generalization

WorldCache is evaluated across a broad spectrum of models and benchmarks to ensure robust performance.

### Evaluation Setup
- **World Types:** Image2World, Text2World
- **Backbone Models:** 
  - **NVIDIA Cosmos-Predict 2.5** (2B, 14B)
  - **WAN2.1** (1.3B, 14B)
- **Benchmarks:** 
  - **PAI-Eval:** Physical Reasoning Benchmark
  - **EgoDex-Eval:** Robotic Egocentric Evaluation

### Speed-Quality Frontier
| Model | Speedup | Temporal Consistency | Quality Retention |
| :--- | :---: | :---: | :---: |
| Cosmos 2B | **2.30×** | 99.1% | 100% |
| WAN2.1 1.3B | **2.36×** | 98.4% | 99.5% |
| Cosmos 14B | **3.05×** | 97.2% | 99.2% |

---

## 🖼️ Qualitative Results

WorldCache maintains flawless temporal coherence even at high acceleration ratios.

<p align="center">
  <img src="assets/figures/qualitative.png" width="1000" alt="Qualitative Results">
</p>

---

## 🛠️ Installation & Quick Start

### 1. Setup Environment
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

### Content-Aware Modules

| Parameter | Default | Description |
|---|---|---|
| `--worldcache_motion_sensitivity` | `5.0` | Motion sensitivity (α). Higher = more responsive to motion (less skipping in dynamic scenes). |
| `--worldcache_flow_enabled` | `False` | Enable Optical Flow-based Feature Alignment (OFA). |
| `--worldcache_flow_scale` | `0.5` | Optical flow downscale factor. `2.0` = full resolution; `0.5` = 2× downsampled (faster). |
| `--worldcache_saliency_enabled` | `False` | Enable Saliency-Weighted Drift (SWD). |
| `--worldcache_saliency_weight` | `5.0` | Saliency weight (β). Controls how much salient regions influence the caching decision. |
| `--worldcache_osi_enabled` | `False` | Enable Online System Identification (OSI) for optimal gamma computation. |
| `--worldcache_dynamic_decay` | `False` | Enable Adaptive Threshold Scheduling (ATS). Relaxes threshold in later steps. |

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
@article{nawaz2026worldcache,
  title={WorldCache: Content-Aware Caching for Accelerated Video World Models},
  author={Nawaz, Umair and Heakl, Ahmed and Khan, Ufaq and Shaker, Abdelrahman and Khan, Salman and Khan, Fahad Shahbaz},
  journal={arXiv preprint arXiv:2503.XXXXX},
  year={2026}
}
```

---

## 📄 License

This project inherits the [Apache 2.0 License](LICENSE) from the NVIDIA Cosmos-Predict2 codebase.
