
## PAI-Bench Text2Video Inference & Evaluation

TBA

## PAI-Bench Image2Video Inference & Evaluation

TBA

## EgoDex-Eval Inference & Evaluation


#### Baseline Inference
```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log_${TIMESTAMP}.txt"
python infer_egodexeval.py \
    --dataset_path datasets/EgoDex_Eval \
    --ckpt_dir ckpt_i2v_14b \
    --output_dir outputs/wan_egodex_baseline \
    --num_frames 81 \
    --sample_steps 40 \
    --num_samples 65 2>&1 | tee logs/$LOG_FILE 
```

#### DiCache Inference
```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log_${TIMESTAMP}.txt"
python infer_egodexeval_dicache.py \
    --dataset_path datasets/EgoDex_Eval \
    --ckpt_dir ckpt_i2v_14b \
    --output_dir outputs/wan_egodex_dicache \
    --num_frames 81 \
    --sample_steps 40 \
    --num_samples 65 2>&1 | tee logs/$LOG_FILE 
```


### WorldCache Inference

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log_${TIMESTAMP}.txt"
python infer_egodexeval_worldcache.py \
    --dataset_path datasets/EgoDex_Eval \
    --ckpt_dir ckpt_i2v_14b \
    --output_dir outputs/wan_egodex_worldcache_v2 \
    --num_frames 81 \
    --sample_steps 40 \
    --num_samples 65 \
    --motion_sensitivity 0.5 \
    --flow_enabled \
    --flow_scale 0.5 \
    --motion_sensitivity 0.2 \
    --osi_enabled \
    --rel_l1_thresh 0.04 \
    --ret_ratio 0.2 \
    --saliency_enabled \
    --saliency_weight 0.12 \
    --dynamic_decay \
    --probe_depth 4 2>&1 | tee logs/$LOG_FILE 
```