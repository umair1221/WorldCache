# infer_egodex_eval.py
# WAN2.1 DiCache Inference on EgoDex-Eval
import argparse
import gc
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import random

import numpy as np
import piq
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool
import torch.cuda.amp as amp
from wan.modules.model import sinusoidal_embedding_1d

# --------------------------------------- T2V ----------------------------------------
def dicache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    r"""
    Forward pass through the diffusion model
    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x
    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    
    skip_forward = False
    ori_x = x
    if self.cnt >= int(self.num_steps * self.ret_ratio):
        test_x, test_kwargs = x.clone(), kwargs
        anchor_blocks = self.blocks[0:self.probe_depth]
        for anchor_block in anchor_blocks:
            test_x = anchor_block(test_x, **test_kwargs)
        delta_x = (x - self.previous_input[self.cnt%2]).abs().mean() / self.previous_input[self.cnt%2].abs().mean()
        delta_y = (test_x - self.previous_internal_states[self.cnt%2]).abs().mean() / self.previous_internal_states[self.cnt%2].abs().mean()
        
        self.accumulated_rel_l1_distance[self.cnt%2] += delta_y # update error accumulater

        if self.accumulated_rel_l1_distance[self.cnt%2] < self.rel_l1_thresh: # skip this step
            skip_forward = True
            self.resume_flag[self.cnt%2] = False 
            residual_x = self.residual_cache[self.cnt%2]
        else:
            self.resume_flag[self.cnt%2] = True
            self.accumulated_rel_l1_distance[self.cnt%2] = 0


    if skip_forward: # skip this step with cached residual
        ori_x = x.clone()
        if len(self.residual_window[self.cnt%2]) >= 2:
            current_residual_indicator = test_x - x
            gamma = ((current_residual_indicator - self.probe_residual_window[self.cnt%2][-2]).abs().mean() / (self.probe_residual_window[self.cnt%2][-1] - self.probe_residual_window[self.cnt%2][-2]).abs().mean()).clip(1, 2)
            x += self.residual_window[self.cnt%2][-2] + gamma * (self.residual_window[self.cnt%2][-1] - self.residual_window[self.cnt%2][-2])
        else:
            x =  x + residual_x 
        self.previous_internal_states[self.cnt%2] = test_x
        self.previous_input[self.cnt%2] = ori_x
    else:
        if self.resume_flag[self.cnt%2]: # resume from test_x
            x = test_x
            kwargs = test_kwargs
            unpass_blocks = self.blocks[self.probe_depth:]
        else: # pass all blocks
            unpass_blocks = self.blocks
        for ind, block in enumerate(unpass_blocks): # self.blocks
            x = block(x, **kwargs)
            if ind == self.probe_depth - 1:
                if self.cnt >= int(self.num_steps * self.ret_ratio):
                    self.previous_internal_states[self.cnt%2] = test_x # directly use test_x
                else:
                    self.previous_internal_states[self.cnt%2] = x # count for internal states
        residual_x = x - ori_x
        
        self.residual_cache[self.cnt%2] = residual_x # residual from block 0 to block N
        self.probe_residual_cache[self.cnt%2] = self.previous_internal_states[self.cnt%2] - ori_x
        self.previous_input[self.cnt%2] = ori_x
        self.previous_output[self.cnt%2] = x

        if len(self.residual_window[self.cnt%2]) <= 2:
            self.residual_window[self.cnt%2].append(residual_x)
            self.probe_residual_window[self.cnt%2].append(self.probe_residual_cache[self.cnt%2])
        else:
            self.residual_window[self.cnt%2][-2] = self.residual_window[self.cnt%2][-1]
            self.residual_window[self.cnt%2][-1] = residual_x
            self.probe_residual_window[self.cnt%2][-2] = self.probe_residual_window[self.cnt%2][-1]
            self.probe_residual_window[self.cnt%2][-1] = self.probe_residual_cache[self.cnt%2]
    
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps: # clear the history of current video and prepare for generating the next video.
        self.cnt = 0
        self.accumulated_rel_l1_distance = [0.0, 0.0]
        self.residual_cache = [None, None]
        self.probe_residual_cache = [None, None]
        self.residual_window = [[], []]
        self.probe_residual_window = [[], []]
        self.previous_internal_states = [None, None]
        self.previous_input = [None, None]
        self.previous_output = [None, None]
        self.resume_flag = [False, False]
    return [u.float() for u in x]
# -----------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# EgoDex-Eval Dataset
# ---------------------------------------------------------------------------

class EgoDexEvalDataset(torch.utils.data.Dataset):
    """
    Loads episodes from the EgoDex-Eval directory structure:
        <dataset_path>/
            data/chunk-000/episode_XXXXXX.parquet
            videos/chunk-000/observation.images.ego_view_freq20/episode_XXXXXX.mp4
            meta/episodes.jsonl
    """

    def __init__(
        self,
        dataset_path: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        deterministic: bool = True,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.deterministic = deterministic

        # Load episode metadata
        episodes_jsonl = os.path.join(dataset_path, "meta", "episodes.jsonl")
        self.episodes = []
        with open(episodes_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.episodes.append(json.loads(line))

        # Sort by episode index
        self.episodes.sort(key=lambda x: x["episode_index"])
        logging.info(f"Loaded {len(self.episodes)} episodes from {episodes_jsonl}")

    def __len__(self):
        return len(self.episodes)

    def _get_video_path(self, episode_index: int) -> str:
        return os.path.join(
            self.dataset_path,
            "videos",
            "chunk-000",
            "observation.images.ego_view_freq20",
            f"episode_{episode_index:06d}.mp4",
        )

    def _get_parquet_path(self, episode_index: int) -> str:
        return os.path.join(
            self.dataset_path,
            "data",
            "chunk-000",
            f"episode_{episode_index:06d}.parquet",
        )

    def _make_prompt(self, description: str) -> str:
        task = description.replace("_", " ")
        return f"A first-person egocentric view of a robot {task}."

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load video frames using torchcodec, fallback to mediapy."""
        try:
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(video_path, dimension_order="NHWC", num_ffmpeg_threads=4)
            total = len(decoder)
            if total < self.num_frames:
                raise ValueError(f"Video has {total} frames, need {self.num_frames}")

            if self.deterministic:
                start = 0
            else:
                start = random.randint(0, total - self.num_frames)

            batch = decoder.get_frames_in_range(start, start + self.num_frames).data
            if batch.device.type != "cpu":
                batch = batch.cpu()
            # (T, H, W, C) uint8
            return batch
        except ImportError:
            import mediapy

            video = mediapy.read_video(video_path)
            total = len(video)
            if total < self.num_frames:
                raise ValueError(f"Video has {total} frames, need {self.num_frames}")

            start = 0 if self.deterministic else random.randint(0, total - self.num_frames)
            frames = video[start : start + self.num_frames]
            return torch.from_numpy(frames)  # (T, H, W, C) uint8

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        episode_index = ep["episode_index"]
        description = ep.get("description", "performing a task")
        prompt = self._make_prompt(description)

        video_path = self._get_video_path(episode_index)
        frames = self._load_video_frames(video_path)  # (T, H, W, C) uint8

        # Resize / crop to target resolution
        # frames: (T, H, W, C) -> (T, C, H, W) for interpolation
        frames = frames.permute(0, 3, 1, 2).float()  # (T, C, H, W)

        # Center crop to target aspect ratio, then resize
        target_ratio = self.width / self.height
        _, _, h, w = frames.shape
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Wider than target: crop width
            new_w = int(h * target_ratio)
            crop_left = (w - new_w) // 2
            frames = frames[:, :, :, crop_left : crop_left + new_w]
        elif current_ratio < target_ratio:
            # Taller than target: crop height
            new_h = int(w / target_ratio)
            crop_top = (h - new_h) // 2
            frames = frames[:, :, crop_top : crop_top + new_h, :]

        frames = torch.nn.functional.interpolate(
            frames, size=(self.height, self.width), mode="bilinear", align_corners=False
        )
        frames = torch.clamp(frames, 0, 255).to(torch.uint8)

        # First frame as PIL image for WanI2V
        first_frame_pil = Image.fromarray(frames[0].permute(1, 2, 0).numpy())

        return {
            "episode_index": episode_index,
            "prompt": prompt,
            "description": description,
            "first_frame": first_frame_pil,
            "gt_video": frames,  # (T, C, H, W) uint8
            "video_path": video_path,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_video: torch.Tensor, gt_video: torch.Tensor):
    """
    pred_video: (T, C, H, W) uint8
    gt_video:   (T, C, H, W) uint8
    Returns dict with psnr, ssim, lpips.
    """
    x = torch.clamp(pred_video.float() / 255.0, 0, 1)
    y = torch.clamp(gt_video.float() / 255.0, 0, 1)

    # Ensure same length
    min_t = min(x.shape[0], y.shape[0])
    x = x[:min_t]
    y = y[:min_t]

    psnr_val = piq.psnr(x, y).mean().item()
    ssim_val = piq.ssim(x, y).mean().item()
    lpips_val = piq.LPIPS()(x, y).mean().item()

    return {"psnr": float(psnr_val), "ssim": float(ssim_val), "lpips": float(lpips_val)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="WAN2.1 DiCache Inference on EgoDex-Eval")

    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to EgoDex-Eval root directory")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of video frames to generate (4n+1)")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of episodes to evaluate")
    parser.add_argument("--height", type=int, default=480, help="Output video height")
    parser.add_argument("--width", type=int, default=832, help="Output video width")

    # WAN2.1 model
    parser.add_argument("--task", type=str, default="i2v-14B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to WAN2.1 checkpoint directory")
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)

    # Sampling
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=40)
    parser.add_argument("--sample_shift", type=float, default=3.0, help="Shift for 480p")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--base_seed", type=int, default=0)

    # DiCache
    parser.add_argument("--rel_l1_thresh", type=float, default=0.08)
    parser.add_argument("--ret_ratio", type=float, default=0.2)

    # Output
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--save_fps", type=int, default=16)

    args = parser.parse_args()

    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupported size {args.size} for task {args.task}"
    )
    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)

    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model not specified, set to {args.offload_model}")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "FSDP not supported in non-distributed mode"
        assert not (args.ulysses_size > 1 or args.ring_size > 1), "Context parallel not supported in non-distributed mode"

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Args: {args}")
    logging.info(f"Model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # ---- Build dataset ----
    dataset = EgoDexEvalDataset(
        dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        deterministic=True,
    )

    if args.num_samples is not None:
        eval_len = min(args.num_samples, len(dataset))
    else:
        eval_len = len(dataset)

    # ---- Build model ----
    logging.info("Creating WanI2V pipeline.")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
    )

    # DiCache state
    wan_i2v.model.__class__.forward = dicache_forward
    wan_i2v.model.__class__.cnt = 0
    wan_i2v.model.__class__.probe_depth = 1
    wan_i2v.model.__class__.num_steps = args.sample_steps * 2
    wan_i2v.model.__class__.rel_l1_thresh = args.rel_l1_thresh
    wan_i2v.model.__class__.accumulated_rel_l1_distance = [0.0, 0.0]
    wan_i2v.model.__class__.ret_ratio = args.ret_ratio
    wan_i2v.model.__class__.residual_cache = [None, None]
    wan_i2v.model.__class__.probe_residual_cache = [None, None]
    wan_i2v.model.__class__.residual_window = [[], []]
    wan_i2v.model.__class__.probe_residual_window = [[], []]
    wan_i2v.model.__class__.previous_internal_states = [None, None]
    wan_i2v.model.__class__.previous_input = [None, None]
    wan_i2v.model.__class__.previous_output = [None, None]
    wan_i2v.model.__class__.resume_flag = [False, False]

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Inference loop ----
    all_psnr, all_ssim, all_lpips = [], [], []

    for idx in tqdm(range(eval_len), desc="Evaluating", disable=rank != 0):
        data = dataset[idx]
        ep_idx = data["episode_index"]
        save_name = f"{ep_idx:06d}"

        pred_path = os.path.join(args.output_dir, f"{save_name}_pred.mp4")
        metrics_path = os.path.join(args.output_dir, f"{save_name}_metrics.json")

        # Skip if already done
        if os.path.exists(pred_path) and os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                m = json.load(f)
            all_psnr.append(m["psnr"])
            all_ssim.append(m["ssim"])
            all_lpips.append(m["lpips"])
            logging.info(f"[{save_name}] Already exists, skipping.")
            continue

        prompt = data["prompt"]
        first_frame = data["first_frame"]  # PIL Image
        gt_video = data["gt_video"]  # (T, C, H, W) uint8

        logging.info(f"[{save_name}] Prompt: {prompt}")

        # Reset DiCache state per sample
        wan_i2v.model.__class__.cnt = 0
        wan_i2v.model.__class__.accumulated_rel_l1_distance = [0.0, 0.0]
        wan_i2v.model.__class__.residual_cache = [None, None]
        wan_i2v.model.__class__.probe_residual_cache = [None, None]
        wan_i2v.model.__class__.residual_window = [[], []]
        wan_i2v.model.__class__.probe_residual_window = [[], []]
        wan_i2v.model.__class__.previous_internal_states = [None, None]
        wan_i2v.model.__class__.previous_input = [None, None]
        wan_i2v.model.__class__.previous_output = [None, None]
        wan_i2v.model.__class__.resume_flag = [False, False]

        # Generate
        video_out, gen_time = wan_i2v.generate(
            input_prompt=prompt,
            img=first_frame,
            max_area=args.height * args.width,
            frame_num=args.num_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            n_prompt=args.negative_prompt,
            seed=args.base_seed + idx,
            offload_model=args.offload_model,
        )

        if rank == 0 and video_out is not None:
            # Save predicted video
            cache_video(
                tensor=video_out[None],
                save_file=pred_path,
                fps=args.save_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

            # Decode predicted video to uint8 for metrics
            # video_out is (C, T, H, W) in [-1, 1]
            pred_frames = ((video_out + 1) / 2).clamp(0, 1)  # (C, T, H, W) [0, 1]
            pred_frames = (pred_frames * 255).to(torch.uint8)
            pred_frames = pred_frames.permute(1, 0, 2, 3).cpu()  # (T, C, H, W)

            # Resize GT to match pred if needed
            gt = gt_video  # (T, C, H, W) uint8
            if gt.shape[2:] != pred_frames.shape[2:]:
                gt = torch.nn.functional.interpolate(
                    gt.float(), size=pred_frames.shape[2:], mode="bilinear", align_corners=False
                ).to(torch.uint8)

            metrics = compute_metrics(pred_frames, gt)
            metrics["gen_time"] = gen_time

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Save GT video
            gt_save_path = os.path.join(args.output_dir, f"{save_name}_gt.mp4")
            if not os.path.exists(gt_save_path):
                import mediapy
                gt_np = gt.permute(0, 2, 3, 1).numpy()  # (T, H, W, C)
                mediapy.write_video(gt_save_path, gt_np, fps=args.save_fps)

            # Save merged (GT | Pred) side by side
            merged_path = os.path.join(args.output_dir, f"{save_name}_merged.mp4")
            if not os.path.exists(merged_path):
                import mediapy
                gt_np = gt.permute(0, 2, 3, 1).numpy()
                pred_np = pred_frames.permute(0, 2, 3, 1).numpy()
                min_t = min(len(gt_np), len(pred_np))
                merged = np.concatenate([gt_np[:min_t], pred_np[:min_t]], axis=2)
                mediapy.write_video(merged_path, merged, fps=args.save_fps)

            all_psnr.append(metrics["psnr"])
            all_ssim.append(metrics["ssim"])
            all_lpips.append(metrics["lpips"])

            logging.info(
                f"[{save_name}] PSNR={metrics['psnr']:.3f}  SSIM={metrics['ssim']:.3f}  "
                f"LPIPS={metrics['lpips']:.3f}  time={gen_time:.1f}s"
            )

    # ---- Summary ----
    if rank == 0 and all_psnr:
        summary = {
            "psnr": f"{sum(all_psnr) / len(all_psnr):.3f}",
            "ssim": f"{sum(all_ssim) / len(all_ssim):.3f}",
            "lpips": f"{sum(all_lpips) / len(all_lpips):.3f}",
            "num_samples": len(all_psnr),
        }
        print(f"\n{'='*50}")
        print(f"PSNR:  {summary['psnr']}")
        print(f"SSIM:  {summary['ssim']}")
        print(f"LPIPS: {summary['lpips']}")
        print(f"N:     {summary['num_samples']}")
        print(f"{'='*50}")

        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = _parse_args()
    generate(args)