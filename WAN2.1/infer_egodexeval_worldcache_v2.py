# infer_egodex_worldcache.py
# WAN2.1 WorldCache Inference on EgoDex-Eval
# Algorithm synced with Cosmos-Predict2.5 WorldCache utils
import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import random

import numpy as np
import torch
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool
from wan.modules.model import sinusoidal_embedding_1d


# ========================== WorldCache Utility Functions ==========================
# Adapted from Cosmos-Predict2.5 worldcache utils for WAN2.1 1D sequence format.
# Cosmos operates on (B, T, H, W, D); WAN2.1 flattens to (B, SeqLen, D).
# All utilities below work in 1D sequence space accordingly.


def estimate_optical_flow_1d(prev_features, curr_features, window_size=21):
    """
    1D Lucas-Kanade displacement estimation between two flattened feature tensors.
    Mirrors Cosmos estimate_optical_flow but operates on (B, S, D) sequences.

    Args:
        prev_features: (B, S, D) previous step features
        curr_features: (B, S, D) current step features
        window_size: smoothing window for structure tensor

    Returns:
        displacement: (B, S, 1) estimated per-position displacement
    """
    prev_signal = prev_features.mean(dim=-1)
    curr_signal = curr_features.mean(dim=-1)

    prev_padded = F.pad(prev_signal.unsqueeze(1), (1, 1), mode='reflect')
    Ix = (prev_padded[:, :, 2:] - prev_padded[:, :, :-2]) / 2.0

    It = (curr_signal - prev_signal).unsqueeze(1)

    Ix2 = Ix * Ix
    IxIt = Ix * It

    pad = window_size // 2
    avg_pool = torch.nn.AvgPool1d(kernel_size=window_size, stride=1, padding=pad)

    sum_Ix2 = avg_pool(Ix2)
    sum_IxIt = avg_pool(IxIt)

    displacement = -sum_IxIt / (sum_Ix2 + 1e-6)
    return displacement.squeeze(1).unsqueeze(-1)


def warp_features_1d(features, displacement):
    """
    Warps features along the sequence dimension using estimated displacement.
    Mirrors Cosmos warp_feature but for 1D sequences.

    Args:
        features: (B, S, D)
        displacement: (B, S, 1)

    Returns:
        warped: (B, S, D)
    """
    if displacement is None:
        return features

    B, S, D = features.shape
    positions = torch.arange(S, device=features.device, dtype=features.dtype)
    positions = positions.unsqueeze(0).expand(B, -1)
    new_positions = positions + displacement.squeeze(-1)

    grid = 2.0 * new_positions / max(S - 1, 1) - 1.0
    grid = grid.unsqueeze(1)
    grid = torch.stack([grid, torch.zeros_like(grid)], dim=-1)

    feat_2d = features.permute(0, 2, 1).unsqueeze(2)
    warped = F.grid_sample(feat_2d, grid, mode='bilinear',
                           padding_mode='reflection', align_corners=True)
    warped = warped.squeeze(2).permute(0, 2, 1)
    return warped


def compute_hf_drift_1d(prev_features, curr_features):
    """
    High-frequency drift via 1D Laplacian kernel [-1, 2, -1].
    Mirrors Cosmos compute_hf_drift for 1D sequences.

    Args:
        prev_features, curr_features: (B, S, D)

    Returns:
        hf_drift: scalar tensor
    """
    prev = prev_features.permute(0, 2, 1)
    curr = curr_features.permute(0, 2, 1)

    D = prev.shape[1]
    kernel = torch.tensor([-1.0, 2.0, -1.0], device=prev.device, dtype=prev.dtype)
    kernel = kernel.view(1, 1, 3).repeat(D, 1, 1)

    prev_hf = F.conv1d(prev, kernel, padding=1, groups=D)
    curr_hf = F.conv1d(curr, kernel, padding=1, groups=D)

    return (prev_hf - curr_hf).abs().mean()


def compute_saliency_map_1d(features):
    """
    Saliency via channel-wise standard deviation.
    Mirrors Cosmos compute_saliency_map for 1D sequences.

    Args:
        features: (B, S, D)

    Returns:
        saliency: (B, S, 1) normalized to [0, 1]
    """
    saliency = torch.std(features, dim=-1, keepdim=True)

    B = saliency.shape[0]
    flat = saliency.view(B, -1)
    lo = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
    hi = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)

    return (saliency - lo) / (hi - lo + 1e-6)


def compute_optimal_gamma_1d(delta_curr, delta_prev):
    """
    Online System Identification (OSI): least-squares optimal gamma.
    Mirrors Cosmos compute_optimal_gamma for 1D sequences.

    Minimizes || delta_curr - gamma * delta_prev ||^2

    Args:
        delta_curr, delta_prev: (B, S, D) or broadcastable

    Returns:
        gamma: (B, 1, 1) clamped to [0.2, 1.8]
    """
    B = delta_curr.shape[0]
    dc = delta_curr.reshape(B, -1)
    dp = delta_prev.reshape(B, -1)

    numer = (dc * dp).sum(dim=1, keepdim=True)
    denom = (dp * dp).sum(dim=1, keepdim=True)

    gamma = torch.clamp(numer / (denom + 1e-6), 0.2, 1.8)
    return gamma.view(B, 1, 1)


# ========================== WorldCache Forward for WAN2.1 ==========================

def worldcache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    """
    WorldCache-enabled forward pass for WAN2.1 DiT.

    Components (synced with Cosmos-Predict2.5):
      1. Causal Feature Caching (CFC)  -- motion-adaptive thresholding
      2. Flow-Warped Feature Caching   -- 1D adapted for flat sequences
      3. Spectral-Adaptive Caching     -- high-frequency monitoring
      4. Saliency-Weighted Drift (SWD) -- perception-aware probing
      5. Optimal State Interpolation (OSI) -- least-squares optimal gamma
      6. Adaptive Threshold Scheduling (ATS) -- phase-aware decay
      7. Adaptive Unconditional Caching (AdUC)
      8. Parallel CFG consensus

    Args / Returns: same as the original WanModel.forward
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None

    # ---- Standard pre-processing (unchanged from WAN) ----
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

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

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
        freqs=self.freqs, context=context, context_lens=context_lens)

    # ==================== WorldCache Logic ====================
    skip_forward = False
    ori_x = x
    residual_x = None

    is_parallel_cfg = getattr(self, 'worldcache_parallel_cfg', False)
    current_idx = 0 if is_parallel_cfg else self.cnt % 2

    test_x = None
    test_kwargs = kwargs

    # ---- Retention phase: probe + drift check ----
    if self.cnt >= int(self.worldcache_num_steps * self.worldcache_ret_ratio):
        test_x = x.clone()
        for blk in self.blocks[:self.worldcache_probe_depth]:
            test_x = blk(test_x, **test_kwargs)

        if (self.previous_input[current_idx] is not None
                and self.previous_internal_states[current_idx] is not None):

            prev_abs_mean = self.previous_input[current_idx].abs().mean() + 1e-8
            delta_x = (x - self.previous_input[current_idx]).abs().mean() / prev_abs_mean

            prev_state_abs = self.previous_internal_states[current_idx].abs().mean() + 1e-8
            delta_y = (test_x - self.previous_internal_states[current_idx]).abs().mean() / prev_state_abs

            self.accumulated_rel_l1_distance[current_idx] += delta_y

            # ---- [SWD] Saliency-Weighted Drift ----
            if getattr(self, 'worldcache_saliency_enabled', False):
                sal = compute_saliency_map_1d(
                    self.previous_internal_states[current_idx])
                diff = (test_x - self.previous_internal_states[current_idx]).abs()
                diff_reduced = diff.mean(dim=-1, keepdim=True)
                beta = getattr(self, 'worldcache_saliency_weight', 5.0)
                w = 1.0 + beta * sal
                weighted_drift = (diff_reduced * w).mean() / prev_state_abs
                self.accumulated_rel_l1_distance[current_idx] += (weighted_drift - delta_y)
                delta_y = weighted_drift

            # ---- [AdUC] Adaptive Unconditional Caching ----
            if (getattr(self, 'worldcache_aduc_enabled', False)
                    and not is_parallel_cfg
                    and current_idx == 1):
                actual_step = self.cnt // 2
                total_half = max(self.worldcache_num_steps // 2, 1)
                if (actual_step / total_half) > getattr(self, 'worldcache_aduc_start', 0.5):
                    if self.previous_output[1] is not None:
                        self.cnt += 1
                        return self.previous_output[1]

            # ---- [CFC] Causal Feature Caching: motion-adaptive threshold ----
            alpha = getattr(self, 'worldcache_motion_sensitivity', 5.0)
            dynamic_thresh = self.worldcache_rel_l1_thresh / (1.0 + alpha * delta_x)

            # ---- [ATS] Adaptive Threshold Scheduling ----
            if getattr(self, 'worldcache_dynamic_decay', False):
                ratio = self.cnt / max(self.worldcache_num_steps, 1)
                dynamic_thresh *= (1.0 + 4.0 * ratio)

            # ---- Spectral-Adaptive (HF monitoring) ----
            hf_ok = True
            if getattr(self, 'worldcache_hf_enabled', False):
                hf_drift = compute_hf_drift_1d(self.previous_input[current_idx], x)
                if hf_drift > getattr(self, 'worldcache_hf_thresh', 0.01):
                    hf_ok = False

            # ---- Skip decision ----
            want_skip = (self.accumulated_rel_l1_distance[current_idx] < dynamic_thresh
                         and hf_ok)

            # ---- Parallel-CFG consensus ----
            if want_skip and is_parallel_cfg and x.shape[0] > 1:
                diff_full = (test_x - self.previous_internal_states[current_idx]).abs()
                half = x.shape[0] // 2
                norm = self.previous_internal_states[current_idx].abs().mean() + 1e-6
                d_c = diff_full[:half].mean() / norm
                d_u = diff_full[half:].mean() / norm
                if not (d_c < dynamic_thresh and d_u < dynamic_thresh):
                    want_skip = False

            # ---- Saliency collector (visualization) ----
            if hasattr(self, 'saliency_collector') and self.saliency_collector is not None:
                _sal_viz = compute_saliency_map_1d(test_x)
                self.saliency_collector.record(
                    saliency_1d=_sal_viz,
                    drift=self.accumulated_rel_l1_distance[current_idx],
                    skip=want_skip,
                    threshold=dynamic_thresh,
                    velocity=float(delta_x),
                    gamma=None,
                    grid_sizes=grid_sizes[0] if grid_sizes is not None else None,
                    patch_size=getattr(self, 'patch_size', None),
                )

            if want_skip:
                skip_forward = True
                self.worldcache_step_skipped_count += 1
                self.resume_flag[current_idx] = False
                residual_x = self.residual_cache[current_idx]
            else:
                self.resume_flag[current_idx] = True
                self.accumulated_rel_l1_distance[current_idx] = 0
                self.previous_internal_states[current_idx] = test_x.clone()

    # ==================== Execution Branch ====================
    if skip_forward:
        # ---- CACHE HIT: approximate output ----
        ori_x_save = x.clone()

        if len(self.residual_window[current_idx]) >= 2:
            resid_indicator = test_x - x

            # ---- [OSI] Optimal State Interpolation ----
            if getattr(self, 'worldcache_osi_enabled', False):
                d_tgt = resid_indicator - self.probe_residual_window[current_idx][-2]
                d_src = (self.probe_residual_window[current_idx][-1]
                         - self.probe_residual_window[current_idx][-2])
                gamma = compute_optimal_gamma_1d(d_tgt, d_src)
            else:
                num = (resid_indicator
                       - self.probe_residual_window[current_idx][-2]).abs().mean()
                den = (self.probe_residual_window[current_idx][-1]
                       - self.probe_residual_window[current_idx][-2]).abs().mean()
                gamma = 1.0 if den < 1e-6 else (num / den).clip(1, 2)

            # Record gamma for visualization
            if (hasattr(self, 'saliency_collector')
                    and self.saliency_collector is not None
                    and len(self.saliency_collector.gamma_values)
                        < self.saliency_collector.step_count):
                _g = float(gamma) if not torch.is_tensor(gamma) else float(gamma.mean())
                self.saliency_collector.gamma_values.append(_g)

            x = x + (self.residual_window[current_idx][-2]
                      + gamma * (self.residual_window[current_idx][-1]
                                 - self.residual_window[current_idx][-2]))
        else:
            x = x + residual_x

        self.previous_internal_states[current_idx] = test_x
        self.previous_input[current_idx] = ori_x_save

        # ---- [OFA] Flow-Warped Residual Correction ----
        if (getattr(self, 'worldcache_flow_enabled', False)
                and self.previous_input[current_idx] is not None
                and self.residual_cache[current_idx] is not None):
            disp = estimate_optical_flow_1d(
                self.previous_input[current_idx], ori_x_save)
            scale = getattr(self, 'worldcache_flow_scale', 0.5)
            raw_res = self.residual_cache[current_idx]
            warped_res = warp_features_1d(raw_res, disp * scale)
            x = x - raw_res + warped_res

    else:
        # ---- CACHE MISS: run full network ----
        if self.resume_flag[current_idx]:
            x = test_x
            kwargs = test_kwargs
            unpass_blocks = self.blocks[self.worldcache_probe_depth:]
        else:
            unpass_blocks = self.blocks

        for ind, block in enumerate(unpass_blocks):
            x = block(x, **kwargs)

            real_idx = (ind if not self.resume_flag[current_idx]
                        else ind + self.worldcache_probe_depth)
            if real_idx == self.worldcache_probe_depth - 1:
                if self.cnt >= int(self.worldcache_num_steps * self.worldcache_ret_ratio):
                    self.previous_internal_states[current_idx] = (
                        test_x.clone() if test_x is not None else x.clone())
                else:
                    self.previous_internal_states[current_idx] = x.clone()

        # Update caches
        residual_x = x - ori_x
        self.residual_cache[current_idx] = residual_x
        self.probe_residual_cache[current_idx] = (
            self.previous_internal_states[current_idx] - ori_x)
        self.previous_input[current_idx] = ori_x
        self.previous_output[current_idx] = x

        if len(self.residual_window[current_idx]) <= 2:
            self.residual_window[current_idx].append(residual_x)
            self.probe_residual_window[current_idx].append(
                self.probe_residual_cache[current_idx])
        else:
            self.residual_window[current_idx][-2] = self.residual_window[current_idx][-1]
            self.residual_window[current_idx][-1] = residual_x
            self.probe_residual_window[current_idx][-2] = self.probe_residual_window[current_idx][-1]
            self.probe_residual_window[current_idx][-1] = self.probe_residual_cache[current_idx]

    # ==================== Final Projection ====================
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)

    self.previous_output[current_idx] = [u.float() for u in x]

    self.cnt += 1
    if self.cnt >= self.worldcache_num_steps:
        rate = self.worldcache_step_skipped_count / max(self.worldcache_num_steps, 1)
        logging.info(
            f"[WorldCache] Skipped {self.worldcache_step_skipped_count}"
            f"/{self.worldcache_num_steps} ({rate:.1%}) | "
            f"alpha={getattr(self, 'worldcache_motion_sensitivity', 'N/A')} "
            f"flow={getattr(self, 'worldcache_flow_enabled', False)} "
            f"hf={getattr(self, 'worldcache_hf_enabled', False)} "
            f"osi={getattr(self, 'worldcache_osi_enabled', False)} "
            f"aduc={getattr(self, 'worldcache_aduc_enabled', False)}")

        self.cnt = 0
        self.worldcache_step_skipped_count = 0
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


# ========================== WorldCache State Setup ==========================

def _apply_worldcache_state(cls, args):
    """Attach all WorldCache state and config to the model class."""
    cls.cnt = 0
    cls.worldcache_probe_depth = args.probe_depth
    cls.worldcache_num_steps = args.sample_steps * 2
    cls.worldcache_rel_l1_thresh = args.rel_l1_thresh
    cls.worldcache_ret_ratio = args.ret_ratio
    cls.worldcache_step_skipped_count = 0
    cls.worldcache_motion_sensitivity = args.motion_sensitivity
    cls.worldcache_flow_enabled = args.flow_enabled
    cls.worldcache_flow_scale = args.flow_scale
    cls.worldcache_hf_enabled = args.hf_enabled
    cls.worldcache_hf_thresh = args.hf_thresh
    cls.worldcache_saliency_enabled = args.saliency_enabled
    cls.worldcache_saliency_weight = args.saliency_weight
    cls.worldcache_osi_enabled = args.osi_enabled
    cls.worldcache_dynamic_decay = args.dynamic_decay
    cls.worldcache_aduc_enabled = args.aduc_enabled
    cls.worldcache_aduc_start = args.aduc_start
    cls.worldcache_parallel_cfg = args.parallel_cfg
    cls.accumulated_rel_l1_distance = [0.0, 0.0]
    cls.residual_cache = [None, None]
    cls.probe_residual_cache = [None, None]
    cls.residual_window = [[], []]
    cls.probe_residual_window = [[], []]
    cls.previous_internal_states = [None, None]
    cls.previous_input = [None, None]
    cls.previous_output = [None, None]
    cls.resume_flag = [False, False]


def _reset_worldcache_state(cls, args):
    """Reset per-sample state (call before each generation)."""
    cls.cnt = 0
    cls.worldcache_step_skipped_count = 0
    cls.accumulated_rel_l1_distance = [0.0, 0.0]
    cls.residual_cache = [None, None]
    cls.probe_residual_cache = [None, None]
    cls.residual_window = [[], []]
    cls.probe_residual_window = [[], []]
    cls.previous_internal_states = [None, None]
    cls.previous_input = [None, None]
    cls.previous_output = [None, None]
    cls.resume_flag = [False, False]


# ========================== EgoDex-Eval Dataset ==========================

class EgoDexEvalDataset(torch.utils.data.Dataset):
    """
    Loads episodes from the EgoDex-Eval directory structure:
        <dataset_path>/
            data/chunk-000/episode_XXXXXX.parquet
            videos/chunk-000/observation.images.ego_view_freq20/episode_XXXXXX.mp4
            meta/episodes.jsonl
    """

    def __init__(self, dataset_path, num_frames=81, height=480, width=832,
                 deterministic=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.deterministic = deterministic

        episodes_jsonl = os.path.join(dataset_path, "meta", "episodes.jsonl")
        self.episodes = []
        with open(episodes_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.episodes.append(json.loads(line))

        self.episodes.sort(key=lambda x: x["episode_index"])
        logging.info(f"Loaded {len(self.episodes)} episodes from {episodes_jsonl}")

    def __len__(self):
        return len(self.episodes)

    def _get_video_path(self, episode_index):
        return os.path.join(
            self.dataset_path, "videos", "chunk-000",
            "observation.images.ego_view_freq20",
            f"episode_{episode_index:06d}.mp4")

    def _make_prompt(self, description):
        task = description.replace("_", " ")
        return f"A first-person egocentric view of a robot {task}."

    def _load_video_frames(self, video_path):
        try:
            from torchcodec.decoders import VideoDecoder
            decoder = VideoDecoder(video_path, dimension_order="NHWC",
                                   num_ffmpeg_threads=4)
            total = len(decoder)
            if total < self.num_frames:
                raise ValueError(f"Video has {total} frames, need {self.num_frames}")
            start = 0 if self.deterministic else random.randint(0, total - self.num_frames)
            batch = decoder.get_frames_in_range(start, start + self.num_frames).data
            if batch.device.type != "cpu":
                batch = batch.cpu()
            return batch
        except ImportError:
            import mediapy
            video = mediapy.read_video(video_path)
            total = len(video)
            if total < self.num_frames:
                raise ValueError(f"Video has {total} frames, need {self.num_frames}")
            start = 0 if self.deterministic else random.randint(0, total - self.num_frames)
            return torch.from_numpy(video[start:start + self.num_frames])

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        episode_index = ep["episode_index"]
        description = ep.get("description", "performing a task")
        prompt = self._make_prompt(description)
        video_path = self._get_video_path(episode_index)
        frames = self._load_video_frames(video_path)

        frames = frames.permute(0, 3, 1, 2).float()
        target_ratio = self.width / self.height
        _, _, h, w = frames.shape
        current_ratio = w / h

        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            crop_left = (w - new_w) // 2
            frames = frames[:, :, :, crop_left:crop_left + new_w]
        elif current_ratio < target_ratio:
            new_h = int(w / target_ratio)
            crop_top = (h - new_h) // 2
            frames = frames[:, :, crop_top:crop_top + new_h, :]

        frames = F.interpolate(frames, size=(self.height, self.width),
                               mode="bilinear", align_corners=False)
        frames = torch.clamp(frames, 0, 255).to(torch.uint8)
        first_frame_pil = Image.fromarray(frames[0].permute(1, 2, 0).numpy())

        return {
            "episode_index": episode_index,
            "prompt": prompt,
            "description": description,
            "first_frame": first_frame_pil,
            "gt_video": frames,
            "video_path": video_path,
        }


# ========================== Metrics ==========================

def compute_metrics(pred_video, gt_video):
    """
    pred_video, gt_video: (T, C, H, W) uint8
    Returns dict with psnr, ssim, lpips.
    """
    import piq

    x = torch.clamp(pred_video.float() / 255.0, 0, 1)
    y = torch.clamp(gt_video.float() / 255.0, 0, 1)

    min_t = min(x.shape[0], y.shape[0])
    x, y = x[:min_t], y[:min_t]

    psnr_val = piq.psnr(x, y).mean().item()
    ssim_val = piq.ssim(x, y).mean().item()
    lpips_val = piq.LPIPS()(x, y).mean().item()

    return {"psnr": float(psnr_val), "ssim": float(ssim_val), "lpips": float(lpips_val)}


# ========================== CLI ==========================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="WAN2.1 WorldCache Inference on EgoDex-Eval")

    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)

    # WAN2.1 model
    parser.add_argument("--task", type=str, default="i2v-14B",
                        choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="832*480",
                        choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)

    # Sampling
    parser.add_argument("--sample_solver", type=str, default="unipc",
                        choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=40)
    parser.add_argument("--sample_shift", type=float, default=3.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--base_seed", type=int, default=0)

    # WorldCache core
    parser.add_argument("--rel_l1_thresh", type=float, default=0.04)
    parser.add_argument("--ret_ratio", type=float, default=0.2)
    parser.add_argument("--probe_depth", type=int, default=4)

    # WorldCache modules
    parser.add_argument("--motion_sensitivity", type=float, default=0.5)
    parser.add_argument("--flow_enabled", action="store_true", default=False)
    parser.add_argument("--flow_scale", type=float, default=0.5)
    parser.add_argument("--hf_enabled", action="store_true", default=False)
    parser.add_argument("--hf_thresh", type=float, default=0.01)
    parser.add_argument("--saliency_enabled", action="store_true", default=False)
    parser.add_argument("--saliency_weight", type=float, default=0.12)
    parser.add_argument("--osi_enabled", action="store_true", default=False)
    parser.add_argument("--dynamic_decay", action="store_true", default=False)
    parser.add_argument("--aduc_enabled", action="store_true", default=False)
    parser.add_argument("--aduc_start", type=float, default=0.5)
    parser.add_argument("--parallel_cfg", action="store_true", default=False)

    # Output
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--save_fps", type=int, default=16)

    args = parser.parse_args()

    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupported size {args.size} for task {args.task}")
    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)

    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


# ========================== Main ==========================

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
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp)
        assert not (args.ulysses_size > 1 or args.ring_size > 1)

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Args: {args}")
    logging.info(f"Model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # ---- Dataset ----
    dataset = EgoDexEvalDataset(
        dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        deterministic=True)

    eval_len = min(args.num_samples, len(dataset)) if args.num_samples else len(dataset)

    # ---- Model ----
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
        init_on_cpu=True)

    # ---- Monkey-patch WorldCache ----
    cls = wan_i2v.model.__class__
    cls.forward = worldcache_forward
    _apply_worldcache_state(cls, args)

    # Log active features
    feats = []
    if args.motion_sensitivity != 5.0:
        feats.append(f"alpha={args.motion_sensitivity}")
    if args.flow_enabled:
        feats.append(f"flow(s={args.flow_scale})")
    if args.hf_enabled:
        feats.append(f"hf(t={args.hf_thresh})")
    if args.saliency_enabled:
        feats.append(f"sal(beta={args.saliency_weight})")
    if args.osi_enabled:
        feats.append("osi")
    if args.dynamic_decay:
        feats.append("decay")
    if args.aduc_enabled:
        feats.append(f"aduc({args.aduc_start})")
    if args.parallel_cfg:
        feats.append("par_cfg")
    logging.info(
        f"[WorldCache] thresh={args.rel_l1_thresh}  ret={args.ret_ratio}  "
        f"probe={args.probe_depth}  features=[{', '.join(feats) or 'base'}]")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Inference loop ----
    all_psnr, all_ssim, all_lpips = [], [], []

    for idx in tqdm(range(eval_len), desc="Evaluating", disable=rank != 0):
        data = dataset[idx]
        ep_idx = data["episode_index"]
        save_name = f"{ep_idx:06d}"

        pred_path = os.path.join(args.output_dir, f"{save_name}_pred.mp4")
        metrics_path = os.path.join(args.output_dir, f"{save_name}_metrics.json")

        # Skip completed
        if os.path.exists(pred_path) and os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                m = json.load(f)
            all_psnr.append(m["psnr"])
            all_ssim.append(m["ssim"])
            all_lpips.append(m["lpips"])
            logging.info(f"[{save_name}] Already exists, skipping.")
            continue

        prompt = data["prompt"]
        first_frame = data["first_frame"]
        gt_video = data["gt_video"]

        logging.info(f"[{save_name}] Prompt: {prompt}")

        # Reset WorldCache state per sample
        _reset_worldcache_state(cls, args)

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
            offload_model=args.offload_model)

        if rank == 0 and video_out is not None:
            # Save predicted video
            cache_video(
                tensor=video_out[None],
                save_file=pred_path,
                fps=args.save_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

            # Decode to uint8 for metrics
            pred_frames = ((video_out + 1) / 2).clamp(0, 1)
            pred_frames = (pred_frames * 255).to(torch.uint8)
            pred_frames = pred_frames.permute(1, 0, 2, 3).cpu()

            gt = gt_video
            if gt.shape[2:] != pred_frames.shape[2:]:
                gt = F.interpolate(
                    gt.float(), size=pred_frames.shape[2:],
                    mode="bilinear", align_corners=False
                ).to(torch.uint8)

            metrics = compute_metrics(pred_frames, gt)
            metrics["gen_time"] = gen_time

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Save GT video
            gt_save_path = os.path.join(args.output_dir, f"{save_name}_gt.mp4")
            if not os.path.exists(gt_save_path):
                import mediapy
                gt_np = gt.permute(0, 2, 3, 1).numpy()
                mediapy.write_video(gt_save_path, gt_np, fps=args.save_fps)

            # Save side-by-side
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
                f"[{save_name}] PSNR={metrics['psnr']:.3f}  "
                f"SSIM={metrics['ssim']:.3f}  "
                f"LPIPS={metrics['lpips']:.3f}  "
                f"time={gen_time:.1f}s")

    # ---- Summary ----
    if rank == 0 and all_psnr:
        summary = {
            "psnr": f"{sum(all_psnr) / len(all_psnr):.3f}",
            "ssim": f"{sum(all_ssim) / len(all_ssim):.3f}",
            "lpips": f"{sum(all_lpips) / len(all_lpips):.3f}",
            "num_samples": len(all_psnr),
        }
        print(f"\n{'=' * 50}")
        print(f"PSNR:  {summary['psnr']}")
        print(f"SSIM:  {summary['ssim']}")
        print(f"LPIPS: {summary['lpips']}")
        print(f"N:     {summary['num_samples']}")
        print(f"{'=' * 50}")

        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = _parse_args()
    generate(args)