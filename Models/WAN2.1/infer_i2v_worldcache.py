# infer_i2v_worldcache.py
# Adapts WorldCache (motion-adaptive caching) from Cosmos Predict2 to WAN2.1
import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import random
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool
from wan.modules.model import sinusoidal_embedding_1d


# ========================== WorldCache Utility Functions ==========================

def estimate_optical_flow_1d(prev_features, curr_features, window_size=21):
    """
    Estimates a 1D displacement field between two flattened feature tensors.
    WAN operates on (B, SeqLen, D), so we treat SeqLen as a 1D spatial axis and
    compute a simplified Lucas-Kanade style displacement per position.

    Args:
        prev_features: (B, S, D) previous step features
        curr_features: (B, S, D) current step features
        window_size: smoothing window for structure tensor

    Returns:
        displacement: (B, S, 1) estimated per-position displacement
    """
    # Reduce channel dim to scalar signal: (B, S)
    prev_signal = prev_features.mean(dim=-1)
    curr_signal = curr_features.mean(dim=-1)

    # Spatial gradient (central difference along S)
    prev_padded = F.pad(prev_signal.unsqueeze(1), (1, 1), mode='reflect')  # (B,1,S+2)
    Ix = (prev_padded[:, :, 2:] - prev_padded[:, :, :-2]) / 2.0           # (B,1,S)

    # Temporal gradient
    It = (curr_signal - prev_signal).unsqueeze(1)  # (B,1,S)

    # Structure tensor elements with box filter
    Ix2 = Ix * Ix
    IxIt = Ix * It

    pad = window_size // 2
    avg_pool = torch.nn.AvgPool1d(kernel_size=window_size, stride=1, padding=pad)

    sum_Ix2 = avg_pool(Ix2)
    sum_IxIt = avg_pool(IxIt)

    # Solve: u = -sum(IxIt) / (sum(Ix2) + eps)
    displacement = -sum_IxIt / (sum_Ix2 + 1e-6)  # (B,1,S)

    return displacement.squeeze(1).unsqueeze(-1)  # (B,S,1)


def warp_features_1d(features, displacement):
    """
    Warps features along the sequence dimension using estimated displacement.

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
    positions = positions.unsqueeze(0).expand(B, -1)  # (B, S)
    new_positions = positions + displacement.squeeze(-1)

    # Normalize to [-1, 1] for grid_sample
    grid = 2.0 * new_positions / max(S - 1, 1) - 1.0
    grid = grid.unsqueeze(1)  # (B, 1, S)
    grid = torch.stack([grid, torch.zeros_like(grid)], dim=-1)  # (B, 1, S, 2)

    feat_2d = features.permute(0, 2, 1).unsqueeze(2)  # (B, D, 1, S)
    warped = F.grid_sample(feat_2d, grid, mode='bilinear',
                           padding_mode='reflection', align_corners=True)
    warped = warped.squeeze(2).permute(0, 2, 1)  # (B, S, D)
    return warped


def compute_hf_drift_1d(prev_features, curr_features):
    """
    Computes high-frequency drift using a 1D Laplacian kernel [-1, 2, -1].

    Args:
        prev_features: (B, S, D)
        curr_features: (B, S, D)

    Returns:
        hf_drift: scalar tensor
    """
    prev = prev_features.permute(0, 2, 1)  # (B, D, S)
    curr = curr_features.permute(0, 2, 1)

    D = prev.shape[1]
    kernel = torch.tensor([-1.0, 2.0, -1.0], device=prev.device, dtype=prev.dtype)
    kernel = kernel.view(1, 1, 3).repeat(D, 1, 1)

    prev_hf = F.conv1d(prev, kernel, padding=1, groups=D)
    curr_hf = F.conv1d(curr, kernel, padding=1, groups=D)

    return (prev_hf - curr_hf).abs().mean()


def compute_saliency_map_1d(features):
    """
    Computes a saliency map via channel-wise standard deviation.

    Args:
        features: (B, S, D)

    Returns:
        saliency: (B, S, 1) normalized to [0, 1]
    """
    saliency = torch.std(features, dim=-1, keepdim=True)  # (B, S, 1)

    B = saliency.shape[0]
    flat = saliency.view(B, -1)
    lo = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
    hi = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)

    return (saliency - lo) / (hi - lo + 1e-6)


def compute_optimal_gamma_1d(delta_curr, delta_prev):
    """
    Online System Identification (OSI):  minimizes || delta_curr - gamma * delta_prev ||^2.

    Args:
        delta_curr, delta_prev: any broadcastable shapes

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
    r"""
    WorldCache-enabled forward pass for WAN2.1 DiT.

    Extends DiCache with eight novelties:
      1. Causal-DiCache  -- motion-adaptive dynamic thresholding
      2. Flow-Warped Feature Caching (1-D adapted for flat sequences)
      3. Spectral-Adaptive Caching (high-frequency monitoring)
      4. Saliency-Guided Thresholding
      5. Online System Identification (OSI) for optimal gamma
      6. Dynamic Threshold Decay
      7. Adaptive Unconditional Caching (AdUC)
      8. Parallel CFG support

    Args / Returns: same as the original WanModel.forward
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None

    # ---- standard pre-processing (unchanged from WAN) ----
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
    ori_x = x                          # (B, seq_len, dim)
    residual_x = None                  # set only when skipping

    is_parallel_cfg = getattr(self, 'worldcache_parallel_cfg', False)
    current_idx = 0 if is_parallel_cfg else self.cnt % 2

    test_x = None
    test_kwargs = kwargs               # reused when resuming

    # ---- Retention-phase: probe + drift check ----
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

            # ---- [4] Saliency-Guided Thresholding ----
            if getattr(self, 'worldcache_saliency_enabled', False):
                sal = compute_saliency_map_1d(
                    self.previous_internal_states[current_idx])
                diff = (test_x - self.previous_internal_states[current_idx]).abs()
                diff_reduced = diff.mean(dim=-1, keepdim=True)
                beta = getattr(self, 'worldcache_saliency_weight', 5.0)
                w = 1.0 + beta * sal
                weighted_drift = (diff_reduced * w).mean() / prev_state_abs
                # replace plain delta_y
                self.accumulated_rel_l1_distance[current_idx] += (weighted_drift - delta_y)
                delta_y = weighted_drift

            # ---- [7] Adaptive Unconditional Caching (AdUC) ----
            if (getattr(self, 'worldcache_aduc_enabled', False)
                    and not is_parallel_cfg
                    and current_idx == 1):
                actual_step = self.cnt // 2
                total_half = max(self.worldcache_num_steps // 2, 1)
                if (actual_step / total_half) > getattr(self, 'worldcache_aduc_start', 0.5):
                    if self.previous_output[1] is not None:
                        self.cnt += 1
                        return self.previous_output[1]

            # ---- [1] Causal-DiCache: motion-adaptive threshold ----
            alpha = getattr(self, 'worldcache_motion_sensitivity', 5.0)
            dynamic_thresh = self.worldcache_rel_l1_thresh / (1.0 + alpha * delta_x)

            # ---- [6] Dynamic Threshold Decay ----
            if getattr(self, 'worldcache_dynamic_decay', False):
                ratio = self.cnt / max(self.worldcache_num_steps, 1)
                dynamic_thresh *= (1.0 + 4.0 * ratio)

            # ---- [3] Spectral-Adaptive (HF monitoring) ----
            hf_ok = True
            if getattr(self, 'worldcache_hf_enabled', False):
                hf_drift = compute_hf_drift_1d(self.previous_input[current_idx], x)
                if hf_drift > getattr(self, 'worldcache_hf_thresh', 0.01):
                    hf_ok = False

            # ---- skip decision ----
            want_skip = (self.accumulated_rel_l1_distance[current_idx] < dynamic_thresh
                         and hf_ok)

            # ---- [8] Parallel-CFG consensus ----
            if want_skip and is_parallel_cfg and x.shape[0] > 1:
                diff_full = (test_x - self.previous_internal_states[current_idx]).abs()
                half = x.shape[0] // 2
                norm = self.previous_internal_states[current_idx].abs().mean() + 1e-6
                d_c = diff_full[:half].mean() / norm
                d_u = diff_full[half:].mean() / norm
                if not (d_c < dynamic_thresh and d_u < dynamic_thresh):
                    want_skip = False

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
        # ---- CACHE HIT ----
        ori_x_save = x.clone()

        if len(self.residual_window[current_idx]) >= 2:
            resid_indicator = test_x - x

            # ---- [5] Online System Identification (OSI) ----
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

            x = x + (self.residual_window[current_idx][-2]
                      + gamma * (self.residual_window[current_idx][-1]
                                 - self.residual_window[current_idx][-2]))
        else:
            x = x + residual_x

        self.previous_internal_states[current_idx] = test_x
        self.previous_input[current_idx] = ori_x_save

        # ---- [2] Flow-Warped Residual Correction ----
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
        # ---- CACHE MISS ----
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

        # update caches
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

    # store projected output for AdUC reuse
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

        # reset for next video
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
# ========================== CLI & Generation ==========================

def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16

    args.base_seed = (args.base_seed if args.base_seed >= 0
                      else random.randint(0, sys.maxsize))

    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupported size {args.size} for task {args.task}, "
        f"supported: {', '.join(SUPPORTED_SIZES[args.task])}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video from text/image using WAN2.1 + WorldCache"
    )
    # ---- Model ----
    parser.add_argument("--task", type=str, default="i2v-14B",
                        choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="832*480",
                        choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str,
                        default="./ckpt/Wan2.1-T2V-1.3B-Original")
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)

    # ---- I/O ----
    parser.add_argument("--input_files", "-i", nargs="+", default=None)
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    parser.add_argument("--src_mask", type=str, default=None)
    parser.add_argument("--src_ref_images", type=str, default=None)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--first_frame", type=str, default=None)
    parser.add_argument("--last_frame", type=str, default=None)

    # ---- Sampling ----
    parser.add_argument("--sample_solver", type=str, default='unipc',
                        choices=['unipc', 'dpm++'])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)

    # ---- WorldCache core ----
    parser.add_argument("--rel_l1_thresh", type=float, default=0.06,
                        help="Base relative L1 threshold for accumulated error")
    parser.add_argument("--ret_ratio", type=float, default=0.2,
                        help="Fraction of initial steps that always compute fully")
    parser.add_argument("--probe_depth", type=int, default=3,
                        help="Number of initial transformer blocks used as probe")

    # ---- WorldCache novelties ----
    parser.add_argument("--motion_sensitivity", type=float, default=5.0,
                        help="Alpha for Causal-DiCache dynamic threshold")
    parser.add_argument("--flow_enabled", action="store_true", default=False,
                        help="Enable Flow-Warped Feature Caching")
    parser.add_argument("--flow_scale", type=float, default=0.5,
                        help="Displacement scale factor for flow warping")
    parser.add_argument("--hf_enabled", action="store_true", default=False,
                        help="Enable Spectral-Adaptive Caching (HF monitoring)")
    parser.add_argument("--hf_thresh", type=float, default=0.01,
                        help="High-frequency drift threshold")
    parser.add_argument("--saliency_enabled", action="store_true", default=False,
                        help="Enable Saliency-Guided Thresholding")
    parser.add_argument("--saliency_weight", type=float, default=0.12,
                        help="Beta weight for saliency-guided drift")
    parser.add_argument("--osi_enabled", action="store_true", default=False,
                        help="Enable Online System Identification (optimal gamma)")
    parser.add_argument("--dynamic_decay", action="store_true", default=False,
                        help="Linearly relax threshold in later steps")
    parser.add_argument("--aduc_enabled", action="store_true", default=False,
                        help="Enable Adaptive Unconditional Caching")
    parser.add_argument("--aduc_start", type=float, default=0.5,
                        help="Step fraction after which AdUC activates")
    parser.add_argument("--parallel_cfg", action="store_true", default=False,
                        help="Batch cond+uncond (parallel CFG)")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), \
            "t5_fsdp/dit_fsdp not supported without distributed."
        assert not (args.ulysses_size > 1 or args.ring_size > 1), \
            "context parallel not supported without distributed."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size
        from xfuser.core.distributed import (
            init_distributed_environment, initialize_model_parallel)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size)

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0

    logging.info(f"Generation args: {args}")
    logging.info(f"Model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # ---- build pipeline ----
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
        init_on_cpu=True
    )

    # ---- monkey-patch WorldCache onto the model ----
    cls = wan_i2v.model.__class__
    cls.forward = worldcache_forward

    # core state
    cls.cnt = 0
    cls.worldcache_probe_depth = args.probe_depth
    cls.worldcache_num_steps = args.sample_steps * 2   # CFG doubles forward calls
    cls.worldcache_rel_l1_thresh = args.rel_l1_thresh
    cls.worldcache_ret_ratio = args.ret_ratio
    cls.worldcache_step_skipped_count = 0

    # novelty flags / params
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

    # ping-pong buffers
    cls.accumulated_rel_l1_distance = [0.0, 0.0]
    cls.residual_cache = [None, None]
    cls.probe_residual_cache = [None, None]
    cls.residual_window = [[], []]
    cls.probe_residual_window = [[], []]
    cls.previous_internal_states = [None, None]
    cls.previous_input = [None, None]
    cls.previous_output = [None, None]
    cls.resume_flag = [False, False]

    # summary log
    feats = []
    if args.motion_sensitivity != 5.0: feats.append(f"alpha={args.motion_sensitivity}")
    if args.flow_enabled:     feats.append(f"flow(s={args.flow_scale})")
    if args.hf_enabled:       feats.append(f"hf(t={args.hf_thresh})")
    if args.saliency_enabled: feats.append(f"sal(beta={args.saliency_weight})")
    if args.osi_enabled:      feats.append("osi")
    if args.dynamic_decay:    feats.append("decay")
    if args.aduc_enabled:     feats.append(f"aduc({args.aduc_start})")
    if args.parallel_cfg:     feats.append("par_cfg")
    logging.info(
        f"[WorldCache] thresh={args.rel_l1_thresh}  ret={args.ret_ratio}  "
        f"probe={args.probe_depth}  features=[{', '.join(feats) or 'base'}]")

    # ---- generate ----
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for prompt_file in tqdm(args.input_files, desc="Generating",
                            disable=rank != 0, total=len(args.input_files)):
        with open(prompt_file, 'r') as f:
            file_data = json.load(f)

        save_video = os.path.join(args.output_dir, file_data['name'] + ".mp4")
        save_json  = os.path.join(args.output_dir, file_data['name'] + ".json")
        if os.path.exists(save_video) and os.path.exists(save_json):
            continue

        prompt          = file_data['prompt']
        num_frames      = file_data['num_output_frames']
        guidance_scale  = file_data['guidance']
        seed            = file_data['seed']
        negative_prompt = file_data['negative_prompt']
        image_path = file_data['input_path']
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        video, gen_time = wan_i2v.generate(
            prompt,
            img,
            max_area=480*832,
            frame_num=num_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=guidance_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=args.offload_model)

        if rank == 0:
            cache_video(
                tensor=video[None],
                save_file=save_video,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

            with open(save_json, 'w') as f:
                json.dump({
                    'prompt': prompt,
                    'num_frames': num_frames,
                    'guidance_scale': guidance_scale,
                    'seed': seed,
                    'negative_prompt': negative_prompt,
                    'gen_time': gen_time,
                }, f, indent=4)


if __name__ == "__main__":
    args = _parse_args()
    generate(args)