## Quick Start Guide for WAN2.1 Inference & Evaluation

Install the proper `torch` and `transformers` versions, then install the requirements in `requirements.txt`:
```bash
pip install -r requirements.txt
```


## PAI-Bench Text2Video Inference & Evaluation

### Baseline Inference
```bash
DEVICES=(1 2)
FILES=(cosmos_eval_jsons_text_only/*.json)
NUM_DEVICES=${#DEVICES[@]}
FILES_PER_DEVICE=$(( (${#FILES[@]} + NUM_DEVICES - 1) / $NUM_DEVICES ))
for i in "${!DEVICES[@]}"; do
    DEV=${DEVICES[$((i % NUM_DEVICES))]}
    CUDA_VISIBLE_DEVICES=$DEV python infer.py \
        -i "${FILES[@]:$((i * FILES_PER_DEVICE)):FILES_PER_DEVICE}" \
        --ckpt_dir ckpt \
        -o outputs/wan_baseline
done
wait
```

### Baseline Inference
```bash
DEVICES=(1 2)
FILES=(cosmos_eval_jsons_text_only/*.json)
NUM_DEVICES=${#DEVICES[@]}
FILES_PER_DEVICE=$(( (${#FILES[@]} + NUM_DEVICES - 1) / $NUM_DEVICES ))
for i in "${!DEVICES[@]}"; do
    DEV=${DEVICES[$((i % NUM_DEVICES))]}
    CUDA_VISIBLE_DEVICES=$DEV python infer_dicache.py \
        -i "${FILES[@]:$((i * FILES_PER_DEVICE)):FILES_PER_DEVICE}" \
        --ckpt_dir ckpt \
        -o outputs/wan_dicache 
done
wait
```




### WorldCache Inference

```bash
DEVICES=(1 2)
FILES=(cosmos_eval_jsons_text_only/*.json)
NUM_DEVICES=${#DEVICES[@]}
FILES_PER_DEVICE=$(( (${#FILES[@]} + NUM_DEVICES - 1) / $NUM_DEVICES ))
for i in "${!DEVICES[@]}"; do
    DEV=${DEVICES[$((i % NUM_DEVICES))]}
    CUDA_VISIBLE_DEVICES=$DEV python infer_worldcache.py \
        -i "${FILES[@]:$((i * FILES_PER_DEVICE)):FILES_PER_DEVICE}" \
        --ckpt_dir ckpt_t2v_1_3b \
        -o outputs/wan_worldcache_saliency_v1 \
        --motion_sensitivity 0.2 \
        --osi_enabled \
        --saliency_enabled \
        --dynamic_decay
done
wait
```

For each, change `DEVICES` to the GPU device IDs you want to use.

## PAI-Bench Image2Video Inference & Evaluation


### Baseline Inference
```bash
DEVICES=(1 2)
FILES=(../Evaluation/PAI-Eval/cosmos_eval_jsons/*.json)
FILES=("${FILES[@]:0:100}")
NUM_DEVICES=${#DEVICES[@]}
FILES_PER_DEVICE=$(( (${#FILES[@]} + NUM_DEVICES - 1) / $NUM_DEVICES ))
for i in "${!DEVICES[@]}"; do
    DEV=${DEVICES[$((i % NUM_DEVICES))]}
    CUDA_VISIBLE_DEVICES=$DEV python infer_i2v.py \
        -i "${FILES[@]:$((i * FILES_PER_DEVICE)):FILES_PER_DEVICE}" \
        --ckpt_dir ckpt_i2v_14b \
        -o outputs/wan_i2v_baseline
done
wait
```

### DiCache Inference
```bash
DEVICES=(1 2)
FILES=(../Evaluation/PAI-Eval/cosmos_eval_jsons/*.json)
FILES=("${FILES[@]:0:100}")
NUM_DEVICES=${#DEVICES[@]}
FILES_PER_DEVICE=$(( (${#FILES[@]} + NUM_DEVICES - 1) / $NUM_DEVICES ))
for i in "${!DEVICES[@]}"; do
    DEV=${DEVICES[$((i % NUM_DEVICES))]}
    CUDA_VISIBLE_DEVICES=$DEV python infer_i2v_dicache.py \
        -i "${FILES[@]:$((i * FILES_PER_DEVICE)):FILES_PER_DEVICE}" \
        --ckpt_dir ckpt_i2v_14b \
        -o outputs/wan_i2v_dicache
done
wait
```



### WorldCache Inference
```bash
DEVICES=(1 2)
FILES=(../Evaluation/PAI-Eval/cosmos_eval_jsons/*.json)
FILES=("${FILES[@]:0:100}")
NUM_DEVICES=${#DEVICES[@]}
FILES_PER_DEVICE=$(( (${#FILES[@]} + NUM_DEVICES - 1) / $NUM_DEVICES ))
for i in "${!DEVICES[@]}"; do
    DEV=${DEVICES[$((i % NUM_DEVICES))]}
    CUDA_VISIBLE_DEVICES=$DEV python infer_i2v_worldcache.py \
        -i "${FILES[@]:$((i * FILES_PER_DEVICE)):FILES_PER_DEVICE}" \
        --ckpt_dir ckpt_i2v_14b \
        -o outputs/wan_i2v_worldcache \
        --flow_enabled \
        --motion_sensitivity 0.2 \
        --osi_enabled \
        --saliency_enabled \
        --dynamic_decay
done
wait
```


For each, change `DEVICES` to the GPU device IDs you want to use.

## EgoDex-Eval Inference & Evaluation


#### Baseline Inference
```bash
python infer_egodexeval.py \
    --dataset_path datasets/EgoDex_Eval \
    --ckpt_dir ckpt_i2v_14b \
    --output_dir outputs/wan_egodex_baseline \
    --num_frames 81 \
    --sample_steps 40 \
    --num_samples 65
```

#### DiCache Inference
```bash
python infer_egodexeval_dicache.py \
    --dataset_path datasets/EgoDex_Eval \
    --ckpt_dir ckpt_i2v_14b \
    --output_dir outputs/wan_egodex_dicache \
    --num_frames 81 \
    --sample_steps 40 \
    --num_samples 65
```


### WorldCache Inference

```bash
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
    --probe_depth 4
```