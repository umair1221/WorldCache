# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DiCache – training-free residual caching for DiT inference acceleration.
#
# DreamDojo adaptation:
#   * Imports and calls ``_maybe_embed_action`` from worldcache_utils so that
#     DreamDojo's action-conditioned DiT variants receive the action signal
#     even when blocks are skipped via caching.
#
# The core algorithm is identical to the original DiCache paper:
#   1. Warm-up phase (first ``ret_ratio`` fraction of steps): always compute.
#   2. Probe phase: run the first ``probe_depth`` blocks to estimate drift.
#   3. If accumulated relative-L1 drift < threshold → reuse cached residual
#      with gamma interpolation.  Otherwise → full compute.

import types
from typing import List, Optional, Tuple, Union

import torch
import torch.amp as amp
from einops import rearrange

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.imaginaire.utils import log

# Reuse the action-embedding helper from WorldCache
from cosmos_predict2._src.predict2.inference.worldcache_utils import _maybe_embed_action


# ======================================================================
# Patched forward
# ======================================================================

def dicache_mini_train_dit_forward(
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
    DiCache-enabled forward pass for MinimalV4DiT / MinimalV1LVGDiT.
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
            B_img, _, T_img, H_img, W_img = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B_img, 1, T_img, H_img, W_img), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)],
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

    # =====================================================================
    # DreamDojo: embed action into timestep signal
    # This is the CRITICAL addition for action-conditioned DiTs.
    # Without this, cached steps would lose action conditioning entirely.
    # =====================================================================
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
    # DiCache caching logic
    # ------------------------------------------------------------------
    skip_forward = False
    ori_x = x_B_T_H_W_D
    residual_x = None

    # Ping-pong buffer for CFG (cond=0, uncond=1)
    current_idx = self.cnt % 2
    test_x = x_B_T_H_W_D.clone()

    if self.cnt >= int(self.dicache_num_steps * self.dicache_ret_ratio):
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

            # --- Decision: skip or compute ---
            if self.accumulated_rel_l1_distance[current_idx] < self.dicache_rel_l1_thresh:
                skip_forward = True
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
        # --- CACHE HIT: residual + gamma interpolation ---
        if len(self.residual_window[current_idx]) >= 2:
            cur_ri = test_x - x_B_T_H_W_D
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

    else:
        # --- CACHE MISS: run remaining blocks ---
        if self.resume_flag[current_idx]:
            x_B_T_H_W_D = test_x
            remaining = self.blocks[self.probe_depth:]
        else:
            remaining = self.blocks

        for i, blk in enumerate(remaining):
            x_B_T_H_W_D = blk(x_B_T_H_W_D, **block_kwargs)
            real_idx = i if not self.resume_flag[current_idx] else i + self.probe_depth
            if real_idx == self.probe_depth - 1:
                self.previous_internal_states[current_idx] = x_B_T_H_W_D.clone()

        # Update caches
        residual_x = x_B_T_H_W_D - ori_x
        self.residual_cache[current_idx] = residual_x
        probe_res = (
            self.previous_internal_states[current_idx] - ori_x
            if self.previous_internal_states[current_idx] is not None
            else residual_x
        )

        self.previous_input[current_idx] = ori_x

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

    # Increment step counter
    self.cnt += 1
    if self.cnt >= self.dicache_num_steps:
        # Reset at end of generation
        self.cnt = 0
        self.accumulated_rel_l1_distance = [0.0, 0.0]
        self.residual_cache = [None, None]
        self.residual_window = [[], []]
        self.probe_residual_window = [[], []]
        self.previous_internal_states = [None, None]
        self.previous_input = [None, None]
        self.resume_flag = [False, False]

    return x_B_C_Tt_Hp_Wp


# ======================================================================
# Public API
# ======================================================================

def apply_dicache(
    model,
    num_steps: int = 35,
    rel_l1_thresh: float = 0.5,
    ret_ratio: float = 0.2,
    probe_depth: int = 8,
):
    """
    Monkey-patch *model* (the DiT network) with DiCache caching.

    Works for both standard Cosmos-Predict2.5 and DreamDojo
    action-conditioned DiT variants (via ``_maybe_embed_action``).
    """
    model.dicache_enabled = True
    model.dicache_num_steps = num_steps
    model.dicache_rel_l1_thresh = rel_l1_thresh
    model.dicache_ret_ratio = ret_ratio
    model.probe_depth = probe_depth

    model.cnt = 0
    model.accumulated_rel_l1_distance = [0.0, 0.0]
    model.residual_cache = [None, None]
    model.residual_window = [[], []]
    model.probe_residual_window = [[], []]
    model.previous_internal_states = [None, None]
    model.previous_input = [None, None]
    model.resume_flag = [False, False]

    model.forward = types.MethodType(dicache_mini_train_dit_forward, model)

    log.info(
        f"[DiCache] Applied: steps={num_steps} thresh={rel_l1_thresh} "
        f"ret_ratio={ret_ratio} probe_depth={probe_depth}"
    )
    return model