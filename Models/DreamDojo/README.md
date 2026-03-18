## Quick Start

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/NVIDIA/DreamDojo .
bash install.sh
```
Download the pretrained checkpoints from [here](https://huggingface.co/nvidia/DreamDojo) and place them in the `checkpoints` directory.

Then you need to copy our implementations into DreamDojo's codebase. You can do this by running the following command:

```bash
cp video2world.py            cosmos_predict2/_src/predict2/inference/video2world.py
cp worldcache_utils.py       cosmos_predict2/_src/predict2/inference/worldcache_utils.py
cp dicache_utils.py          cosmos_predict2/_src/predict2/inference/dicache_utils.py
cp action_conditioned.py     cosmos_predict2/action_conditioned.py
cp action_conditioned_config.py cosmos_predict2/action_conditioned_config.py
```

## Inference

### Baseline
```bash

export HF_TOKEN=<your-token>
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$HOME/ffmpeg_env/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/ffmpeg_env/lib/pkgconfig:$PKG_CONFIG_PATH

python examples/action_conditioned.py \
  -o outputs/action_conditioned/baseline \
  --checkpoints-dir checkpoints/DreamDojo/2B_GR1_post-train\
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir dreamdojo_results/egodex_baseline \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "EgoDex_Eval" \
  --data-split full \
  --deterministic-uniform-sampling \
  --data-split test \
  --checkpoint-interval 5000 \
  --infinite 
```

### DiCache
```bash

export HF_TOKEN=<your-token>
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$HOME/ffmpeg_env/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/ffmpeg_env/lib/pkgconfig:$PKG_CONFIG_PATH


DC_FLAGS="
  --dicache-enabled
  --dicache-rel-l1-thresh 0.08
  --dicache-probe-depth 1
  --dicache-ret-ratio 0.2
"

python examples/action_conditioned.py \
  -o outputs/action_conditioned/dicache \
  --checkpoints-dir checkpoints/DreamDojo/2B_GR1_post-train\
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir dreamdojo_results/egodex_dicache \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "EgoDex_Eval" \
  --data-split full \
  --deterministic-uniform-sampling \
  --data-split test \
  --checkpoint-interval 5000 \
  --infinite \
  ${DC_FLAGS}
```

### WorldCache
```bash
export HF_TOKEN=<your-token>
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$HOME/ffmpeg_env/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/ffmpeg_env/lib/pkgconfig:$PKG_CONFIG_PATH

python examples/action_conditioned.py \
  -o outputs/action_conditioned/worldcache_v2 \
  --checkpoints-dir checkpoints/DreamDojo/2B_GR1_post-train\
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir dreamdojo_results/egodex_worldcache_v2 \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "EgoDex_Eval" \
  --data-split full \
  --deterministic-uniform-sampling \
  --data-split test \
  --checkpoint-interval 5000 \
  --infinite \
  --worldcache-enabled
```