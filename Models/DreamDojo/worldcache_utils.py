# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# WorldCache – training-free caching framework for DiT-based video world models.
#
# Additions for DreamDojo:
#   * ``action`` kwarg extracted from condition.to_dict() and processed
#     through ``self.action_embedder`` / ``self.action_proj`` if present.
#   * The resulting action embedding is *added* to ``t_embedding_B_T_D``
#     so that every cached & computed path sees the correct action signal.

import os
import json
import types
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
from einops import rearrange

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.imaginaire.utils import log


# ======================================================================
# Helper functions (unchanged from base WorldCache)
# ======================================================================

def estimate_optical_flow(prev_img_tensor, curr_img_tensor, scale_factor=0.5):
    """GPU-native Lucas-Kanade optical flow. (B,C,H,W) -> (B,H,W,2)."""
    original_h, original_w = prev_img_tensor.shape[2], prev_img_tensor.shape[3]
    if scale_factor != 1.0:
        prev_img_tensor = F.interpolate(prev_img_tensor, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        curr_img_tensor = F.interpolate(curr_img_tensor, scale_factor=scale_factor, mode="bilinear", align_corners=False)
    I1 = prev_img_tensor.mean(dim=1, keepdim=True)
    I2 = curr_img_tensor.mean(dim=1, keepdim=True)
    k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=I1.dtype, device=I1.device).view(1, 1, 3, 3)
    k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=I1.dtype, device=I1.device).view(1, 1, 3, 3)
    Ix = F.conv2d(I1, k_x, padding=1)
    Iy = F.conv2d(I1, k_y, padding=1)
    It = I2 - I1
    win = 21
    avg = torch.nn.AvgPool2d(kernel_size=win, stride=1, padding=win // 2)
    sIx2 = avg(Ix * Ix); sIy2 = avg(Iy * Iy)
    sIxIy = avg(Ix * Iy); sIxIt = avg(Ix * It); sIyIt = avg(Iy * It)
    det = sIx2 * sIy2 - sIxIy * sIxIy + 1e-6
    u = -(sIy2 * sIxIt - sIxIy * sIyIt) / det
    v = -(sIx2 * sIyIt - sIxIy * sIxIt) / det
    flow = torch.cat((u, v), dim=1)
    if scale_factor != 1.0:
        flow = F.interpolate(flow, size=(original_h, original_w), mode="bilinear", align_corners=False) * (1.0 / scale_factor)
    return flow.permute(0, 2, 3, 1)


def warp_feature(feature_tensor, flow_tensor):
    if flow_tensor is None:
        return feature_tensor
    B, C, H, W = feature_tensor.shape
    if flow_tensor.ndim == 3:
        flow_tensor = flow_tensor.unsqueeze(0).expand(B, -1, -1, -1)
    elif flow_tensor.ndim == 4 and flow_tensor.shape[0] == 1 and B > 1:
        flow_tensor = flow_tensor.expand(B, -1, -1, -1)
    flow_tensor = flow_tensor.to(device=feature_tensor.device, dtype=feature_tensor.dtype)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=feature_tensor.device, dtype=feature_tensor.dtype),
        torch.arange(W, device=feature_tensor.device, dtype=feature_tensor.dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)
    yy = yy.unsqueeze(0).expand(B, -1, -1)
    gx = 2.0 * (xx + flow_tensor[..., 0]) / max(W - 1, 1) - 1.0
    gy = 2.0 * (yy + flow_tensor[..., 1]) / max(H - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=3)
    return F.grid_sample(feature_tensor, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def compute_hf_drift(prev_img, curr_img):
    C = prev_img.shape[1]
    k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=prev_img.dtype, device=prev_img.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    prev_hf = F.conv2d(prev_img, k, padding=1, groups=C)
    curr_hf = F.conv2d(curr_img, k, padding=1, groups=C)
    return (prev_hf - curr_hf).abs().mean()


def compute_saliency_map(features):
    if features.ndim == 5:
        B, T, H, W, D = features.shape
        features = rearrange(features, "b t h w d -> b (t d) h w")
    saliency = torch.std(features, dim=1, keepdim=True)
    B = saliency.shape[0]
    sf = saliency.view(B, -1)
    mn = sf.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    mx = sf.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    return (saliency - mn) / (mx - mn + 1e-6)


def compute_optimal_gamma(delta_curr, delta_prev):
    B = delta_curr.shape[0]
    dc = delta_curr.view(B, -1)
    dp = delta_prev.view(B, -1)
    numer = (dc * dp).sum(dim=1, keepdim=True)
    denom = (dp * dp).sum(dim=1, keepdim=True)
    gamma = torch.clamp(numer / (denom + 1e-6), 0.2, 1.8)
    return gamma.view(B, 1, 1, 1, 1)


# ======================================================================
# DreamDojo action-embedding helper
# ======================================================================

def _maybe_embed_action(model, t_embedding_B_T_D, kwargs):
    """If the DiT has an action embedder, embed action & fuse with t_emb.

    DreamDojo nets (``cosmos_v1_*_action_chunk_conditioned``) expose one or
    more of the following attributes:

    * ``action_embedder``  – nn.Module that maps (B, T_act, action_dim) → (B, T, D)
    * ``action_proj``      – Optional linear projection after embedder

    The action tensor arrives via ``condition.to_dict()`` under the key
    ``"action"``.  If neither attribute exists the function is a no-op.
    """
    action = kwargs.pop("action", None)  # remove from kwargs so blocks don't see it
    if action is None:
        return t_embedding_B_T_D

    # --- Try the model's own action embedder ---
    if hasattr(model, "action_embedder") and model.action_embedder is not None:
        action_emb = model.action_embedder(action)
        if hasattr(model, "action_proj") and model.action_proj is not None:
            action_emb = model.action_proj(action_emb)
        # Fuse: add to timestep embedding (most common design)
        # Shapes must match: both (B, T, D)
        if action_emb.shape == t_embedding_B_T_D.shape:
            t_embedding_B_T_D = t_embedding_B_T_D + action_emb
        elif action_emb.shape[-1] == t_embedding_B_T_D.shape[-1]:
            # Temporal mismatch — interpolate or broadcast
            t_embedding_B_T_D = t_embedding_B_T_D + action_emb[:, : t_embedding_B_T_D.shape[1]]
    return t_embedding_B_T_D


# ======================================================================
# Patched forward
# ======================================================================

def worldcache_mini_train_dit_forward(
    self,
    x_B_C_T_H_W: torch.Tensor,
    timesteps_B_T: torch.Tensor,
    crossattn_emb: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    data_type: Optional[DataType] = DataType.VIDEO,
    intermediate_feature_ids: Optional[List[int]] = None,
    img_context_emb: Optional[torch.Tensor] = None,
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    WorldCache-enabled forward pass for MinimalV4DiT / MinimalV1LVGDiT.
    Now also supports DreamDojo's action-conditioned variants.
    """
    assert isinstance(data_type, DataType), f"Expected DataType, got {type(data_type)}."

    # --- MinimalV1LVGDiT compatibility ---
    is_lvg = hasattr(self, "timestep_scale")
    if is_lvg:
        if kwargs.get("timestep_scale") is None:
            timesteps_B_T = timesteps_B_T * self.timestep_scale
        if data_type == DataType.VIDEO:
            if condition_video_input_mask_B_C_T_H_W is not None:
                x_B_C_T_H_W = torch.cat(
                    [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1
                )
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)],
                dim=1,
            )

    # --- Standard preprocessing ---
    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
        x_B_C_T_H_W, fps=fps, padding_mask=padding_mask,
    )

    if self.use_crossattn_projection:
        crossattn_emb = self.crossattn_proj(crossattn_emb)

    if img_context_emb is not None:
        assert self.extra_image_context_dim is not None
        img_context_emb = self.img_context_proj(img_context_emb)
        context_input = (crossattn_emb, img_context_emb)
    else:
        context_input = crossattn_emb

    with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    # --- DreamDojo: embed action into timestep signal ---
    t_embedding_B_T_D = _maybe_embed_action(self, t_embedding_B_T_D, kwargs)

    # Store for logging
    self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
    self.affline_emb = t_embedding_B_T_D
    self.crossattn_emb = crossattn_emb

    if extra_pos_emb is not None:
        assert x_B_T_H_W_D.shape == extra_pos_emb.shape

    B, T, H, W, D = x_B_T_H_W_D.shape

    block_kwargs = {
        "emb_B_T_D": t_embedding_B_T_D,
        "crossattn_emb": context_input,
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb,
    }

    # ------------------------------------------------------------------
    # WorldCache caching logic
    # ------------------------------------------------------------------
    skip_forward = False
    ori_x = x_B_T_H_W_D
    residual_x = None

    is_parallel_cfg = getattr(self, "worldcache_parallel_cfg", False)
    current_idx = 0 if is_parallel_cfg else self.cnt % 2

    test_x = x_B_T_H_W_D.clone()

    if self.cnt >= int(self.worldcache_num_steps * self.worldcache_ret_ratio):
        # --- PROBE: run first `probe_depth` blocks ---
        for blk in self.blocks[: self.probe_depth]:
            test_x = blk(test_x, **block_kwargs)

        if self.previous_input[current_idx] is not None and self.previous_internal_states[current_idx] is not None:
            delta_x = (x_B_T_H_W_D - self.previous_input[current_idx]).abs().mean() / (
                self.previous_input[current_idx].abs().mean() + 1e-8
            )
            delta_y = (test_x - self.previous_internal_states[current_idx]).abs().mean() / (
                self.previous_internal_states[current_idx].abs().mean() + 1e-8
            )
            self.accumulated_rel_l1_distance[current_idx] += delta_y

            # --- Saliency weighting ---
            if getattr(self, "worldcache_saliency_enabled", False):
                diff_map = rearrange(
                    (test_x - self.previous_internal_states[current_idx]).abs(),
                    "b t h w d -> b (t d) h w",
                ).mean(dim=1, keepdim=True)
                smap = compute_saliency_map(self.previous_internal_states[current_idx])
                beta = getattr(self, "worldcache_saliency_weight", 5.0)
                w_drift = (diff_map * (1.0 + beta * smap)).mean()
                denom = self.previous_internal_states[current_idx].abs().mean() + 1e-6
                w_rel = w_drift / denom
                self.accumulated_rel_l1_distance[current_idx] += w_rel - delta_y
                delta_y = w_rel

            # --- AdUC (sequential CFG only) ---
            if getattr(self, "worldcache_aduc_enabled", False) and not is_parallel_cfg:
                actual_step = self.cnt // 2
                step_ratio = actual_step / max(getattr(self, "worldcache_num_steps", 35), 1)
                if current_idx == 1 and step_ratio > getattr(self, "worldcache_aduc_start", 0.5):
                    if len(self.previous_output) > 1 and self.previous_output[1] is not None:
                        self.cnt += 1
                        return self.previous_output[1]

            # --- Motion-adaptive threshold ---
            input_velocity = delta_x
            alpha = getattr(self, "worldcache_motion_sensitivity", 5.0)
            dynamic_thresh = self.worldcache_rel_l1_thresh / (1.0 + alpha * input_velocity)

            # Dynamic decay
            if getattr(self, "worldcache_dynamic_decay", False):
                ns = max(getattr(self, "worldcache_num_steps", 35), 1)
                u = ns / 35.0
                base_mult = (u ** 2) / 6.0 + u / 2.0 + 10.0 / 3.0
                dynamic_thresh *= 1.0 + base_mult * (self.cnt / ns)

            # --- HF guard ---
            hf_ok = True
            if getattr(self, "worldcache_hf_enabled", False):
                ci = rearrange(x_B_T_H_W_D, "b t h w d -> b (t d) h w")
                pi = rearrange(self.previous_input[current_idx], "b t h w d -> b (t d) h w")
                hf = compute_hf_drift(pi, ci)
                if hf > getattr(self, "worldcache_hf_thresh", 0.01):
                    hf_ok = False

            # --- Decision ---
            if self.accumulated_rel_l1_distance[current_idx] < dynamic_thresh and hf_ok:
                if is_parallel_cfg and B > 1:
                    diff_full = (test_x - self.previous_internal_states[current_idx]).abs()
                    hB = B // 2
                    d_c = diff_full[:hB].mean() / (self.previous_internal_states[current_idx][:hB].abs().mean() + 1e-6)
                    d_u = diff_full[hB:].mean() / (self.previous_internal_states[current_idx][hB:].abs().mean() + 1e-6)
                    if d_c < dynamic_thresh and d_u < dynamic_thresh:
                        skip_forward = True
                        self.worldcache_step_skipped_count += 1
                        self.resume_flag[current_idx] = False
                        residual_x = self.residual_cache[current_idx]
                    else:
                        self.resume_flag[current_idx] = True
                        self.accumulated_rel_l1_distance[current_idx] = 0
                        self.previous_internal_states[current_idx] = test_x.clone()
                else:
                    skip_forward = True
                    self.worldcache_step_skipped_count += 1
                    self.resume_flag[current_idx] = False
                    residual_x = self.residual_cache[current_idx]
            else:
                self.resume_flag[current_idx] = True
                self.accumulated_rel_l1_distance[current_idx] = 0
                self.previous_internal_states[current_idx] = test_x.clone()

    # ------------------------------------------------------------------
    # Execute or reuse
    # ------------------------------------------------------------------
    if skip_forward:
        # --- CACHE HIT ---
        if len(self.residual_window[current_idx]) >= 2:
            cur_ri = test_x - x_B_T_H_W_D
            if getattr(self, "worldcache_osi_enabled", False):
                dt = cur_ri - self.probe_residual_window[current_idx][-2]
                ds = self.probe_residual_window[current_idx][-1] - self.probe_residual_window[current_idx][-2]
                gamma = compute_optimal_gamma(dt, ds)
            else:
                n = (cur_ri - self.probe_residual_window[current_idx][-2]).abs().mean()
                d = (self.probe_residual_window[current_idx][-1] - self.probe_residual_window[current_idx][-2]).abs().mean()
                gamma = (n / d).clip(1, 2) if d > 1e-6 else 1.0
            x_B_T_H_W_D = x_B_T_H_W_D + self.residual_window[current_idx][-2] + gamma * (
                self.residual_window[current_idx][-1] - self.residual_window[current_idx][-2]
            )
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + residual_x

        self.previous_internal_states[current_idx] = test_x
        self.previous_input[current_idx] = ori_x

        # --- Flow warp ---
        if getattr(self, "worldcache_flow_enabled", False) and self.previous_input[current_idx] is not None and residual_x is not None:
            ci = rearrange(x_B_T_H_W_D, "b t h w d -> b (t d) h w")
            pi = rearrange(self.previous_input[current_idx], "b t h w d -> b (t d) h w")
            flow = estimate_optical_flow(pi, ci, scale_factor=getattr(self, "worldcache_flow_scale", 0.5))
            if flow is not None:
                ri = rearrange(residual_x, "b t h w d -> b (t d) h w")
                wr = warp_feature(ri, flow)
                warped = rearrange(wr, "b (t d) h w -> b t h w d", t=T)
                x_B_T_H_W_D = x_B_T_H_W_D - residual_x + warped
    else:
        # --- CACHE MISS ---
        if self.resume_flag[current_idx]:
            x_B_T_H_W_D = test_x
            remaining = self.blocks[self.probe_depth :]
        else:
            remaining = self.blocks

        for i, blk in enumerate(remaining):
            x_B_T_H_W_D = blk(x_B_T_H_W_D, **block_kwargs)
            real_idx = i if not self.resume_flag[current_idx] else i + self.probe_depth
            if real_idx == self.probe_depth - 1:
                self.previous_internal_states[current_idx] = x_B_T_H_W_D.clone()

        residual_x = x_B_T_H_W_D - ori_x
        self.residual_cache[current_idx] = residual_x
        probe_res = self.previous_internal_states[current_idx] - ori_x if self.previous_internal_states[current_idx] is not None else residual_x
        self.probe_residual_cache[current_idx] = probe_res
        self.previous_input[current_idx] = ori_x
        self.previous_output[current_idx] = x_B_T_H_W_D

        if len(self.residual_window[current_idx]) <= 2:
            self.residual_window[current_idx].append(residual_x)
            self.probe_residual_window[current_idx].append(probe_res)
        else:
            self.residual_window[current_idx][-2] = self.residual_window[current_idx][-1]
            self.residual_window[current_idx][-1] = residual_x
            self.probe_residual_window[current_idx][-2] = self.probe_residual_window[current_idx][-1]
            self.probe_residual_window[current_idx][-1] = probe_res

    # ------------------------------------------------------------------
    # Final layer & unpatchify
    # ------------------------------------------------------------------
    x_out = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    x_B_C_Tt_Hp_Wp = self.unpatchify(x_out)

    self.cnt += 1
    if self.cnt >= self.worldcache_num_steps:
        rate = self.worldcache_step_skipped_count / max(self.worldcache_num_steps, 1)
        log.info(
            f"[WorldCache] Skipped {self.worldcache_step_skipped_count}/{self.worldcache_num_steps} "
            f"({rate:.1%}) | alpha={getattr(self, 'worldcache_motion_sensitivity', 'N/A')}"
        )
        # Reset
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

    if current_idx < len(self.previous_output):
        self.previous_output[current_idx] = x_B_C_Tt_Hp_Wp.clone()

    return x_B_C_Tt_Hp_Wp


# ======================================================================
# Public API
# ======================================================================

def apply_worldcache(
    model,
    num_steps: int = 35,
    rel_l1_thresh: float = 0.08,
    ret_ratio: float = 0.2,
    probe_depth: int = 1,
    motion_sensitivity: float = 5.0,
    flow_enabled: bool = False,
    flow_scale: float = 0.5,
    hf_enabled: bool = False,
    hf_thresh: float = 0.01,
    saliency_enabled: bool = False,
    saliency_weight: float = 5.0,
    osi_enabled: bool = False,
    dynamic_decay: bool = False,
    aduc_enabled: bool = False,
    aduc_start: float = 0.5,
    parallel_cfg: bool = False,
):
    """Monkey-patch *model* (the DiT network) with WorldCache caching.

    Works for both standard Cosmos-Predict2.5 and DreamDojo
    action-conditioned DiT variants.
    """
    # --- Config ---
    model.worldcache_enabled = True
    model.worldcache_num_steps = num_steps
    model.worldcache_rel_l1_thresh = rel_l1_thresh
    model.worldcache_ret_ratio = ret_ratio
    model.probe_depth = probe_depth
    model.worldcache_motion_sensitivity = motion_sensitivity
    model.worldcache_flow_enabled = flow_enabled
    model.worldcache_flow_scale = flow_scale
    model.worldcache_hf_enabled = hf_enabled
    model.worldcache_hf_thresh = hf_thresh
    model.worldcache_saliency_enabled = saliency_enabled
    model.worldcache_saliency_weight = saliency_weight
    model.worldcache_osi_enabled = osi_enabled
    model.worldcache_dynamic_decay = dynamic_decay
    model.worldcache_aduc_enabled = aduc_enabled
    model.worldcache_aduc_start = aduc_start
    model.worldcache_parallel_cfg = parallel_cfg

    # --- Buffers ---
    model.cnt = 0
    model.worldcache_step_skipped_count = 0
    model.accumulated_rel_l1_distance = [0.0, 0.0]
    model.residual_cache = [None, None]
    model.probe_residual_cache = [None, None]
    model.residual_window = [[], []]
    model.probe_residual_window = [[], []]
    model.previous_internal_states = [None, None]
    model.previous_input = [None, None]
    model.previous_output = [None, None]
    model.resume_flag = [False, False]

    # --- Patch forward ---
    model.forward = types.MethodType(worldcache_mini_train_dit_forward, model)

    log.info(
        f"[WorldCache] Applied: steps={num_steps} thresh={rel_l1_thresh} "
        f"alpha={motion_sensitivity} flow={flow_enabled}({flow_scale}) "
        f"hf={hf_enabled} saliency={saliency_enabled} osi={osi_enabled} "
        f"decay={dynamic_decay} aduc={aduc_enabled}({aduc_start}) "
        f"parallel_cfg={parallel_cfg}"
    )
    return model