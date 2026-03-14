# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# EasyCache adaptation for Cosmos-Predict2.5
# Based on: "EasyCache: Runtime-Adaptive Dynamic Caching for Diffusion Transformers"
# Original WAN2.1 code adapted to work with Cosmos DiT architecture.
#
# Key idea: Track transformation rate k_t between consecutive steps,
# accumulate predicted error E_t, and skip computation when E_t < tau.
# Caches transformation vectors Delta = v_t - x_t separately for
# cond/uncond CFG paths.

import torch
import types
from typing import List, Optional, Tuple, Union
from cosmos_predict2._src.predict2.conditioner import DataType
import torch.amp as amp
from cosmos_predict2._src.imaginaire.utils import log


def easycache_mini_train_dit_forward(
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
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    EasyCache-enabled forward pass for MinimalV4DiT and MinimalV1LVGDiT.

    Caches at the FINAL OUTPUT level (x_B_C_Tt_Hp_Wp) to avoid redundant
    preprocessing, block computation, final_layer, and unpatchify on skipped steps.
    """
    assert isinstance(data_type, DataType), (
        f"Expected DataType, got {type(data_type)}."
    )

    # Track even/odd for CFG (even=cond, odd=uncond)
    is_even = (self.easycache_cnt % 2 == 0)

    # --- EasyCache Decision Logic (uses raw 5D input for change detection) ---
    should_compute = True  # Default: compute fully
    raw_input_5d = x_B_C_T_H_W  # Reference to raw input (no clone needed for change detection)

    if is_even:
        # Only make caching decisions on even (conditional) steps
        if self.easycache_cnt < self.easycache_ret_steps or self.easycache_cnt >= self.easycache_cutoff_steps:
            # Warm-up or final steps: always compute
            should_compute = True
            self.easycache_accumulated_error_even = 0.0
        else:
            # Check if we can skip using accumulated error
            if (self.easycache_prev_input_even is not None and
                    self.easycache_prev_output_even is not None):

                # Calculate input change: ||x_t - x_{t-1}||
                input_change = (raw_input_5d - self.easycache_prev_input_even).abs().mean()

                if self.easycache_k is not None and input_change > 1e-8:
                    # Calculate predicted output change (Eq. 4):
                    # eps_t = k_t * ||x_t - x_{t-1}|| / ||v_{t-1}||
                    output_norm = self.easycache_prev_output_even.abs().mean()
                    if output_norm > 1e-8:
                        pred_change = self.easycache_k * (input_change / output_norm)
                    else:
                        pred_change = 0.0

                    # Accumulate error (Eq. 5)
                    self.easycache_accumulated_error_even += pred_change

                    # Decision (Eq. 6): if E_t < tau, reuse cache
                    if self.easycache_accumulated_error_even < self.easycache_thresh:
                        should_compute = False
                    else:
                        should_compute = True
                        self.easycache_accumulated_error_even = 0.0
                else:
                    # First time after warm-up or missing k: must compute
                    should_compute = True
            else:
                should_compute = True

        # Store current raw input for next comparison (before any preprocessing)
        self.easycache_prev_input_even = raw_input_5d.clone()
        # Share decision with the odd (uncond) step
        self.easycache_should_compute = should_compute
    else:
        # Odd (unconditional) step: follow the decision from the even step
        should_compute = self.easycache_should_compute

    # --- Cache Hit: Return cached output immediately (skip ALL computation) ---
    if not should_compute:
        cached_output = None
        if is_even and self.easycache_cache_even is not None:
            cached_output = raw_input_5d + self.easycache_cache_even
        elif not is_even and self.easycache_cache_odd is not None:
            cached_output = raw_input_5d + self.easycache_cache_odd

        if cached_output is not None:
            self.easycache_cnt += 1
            self.easycache_step_skipped_count += 1
            if self.easycache_cnt >= self.easycache_num_steps:
                skip_rate = self.easycache_step_skipped_count / self.easycache_num_steps
                log.info(f"[EasyCache Summary] Skipped {self.easycache_step_skipped_count}/{self.easycache_num_steps} steps ({skip_rate:.1%})")
                self._easycache_reset()
            return cached_output
        else:
            # Fallback: no cache available, must compute
            should_compute = True

    # ========================================================================
    # --- FULL COMPUTATION PATH (Cache Miss or Warm-up) ---
    # ========================================================================

    # --- MinimalV1LVGDiT Compatibility Logic ---
    is_lvg = hasattr(self, "timestep_scale")

    if is_lvg:
        if kwargs.get('timestep_scale') is None:
            timesteps_B_T = timesteps_B_T * self.timestep_scale

        if data_type == DataType.VIDEO:
            if condition_video_input_mask_B_C_T_H_W is not None:
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
            else:
                pass
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

    # --- Standard Pre-processing ---
    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
        x_B_C_T_H_W,
        fps=fps,
        padding_mask=padding_mask,
    )

    if self.use_crossattn_projection:
        crossattn_emb = self.crossattn_proj(crossattn_emb)

    if img_context_emb is not None:
        assert self.extra_image_context_dim is not None, (
            "extra_image_context_dim must be set if img_context_emb is provided"
        )
        img_context_emb = self.img_context_proj(img_context_emb)
        context_input = (crossattn_emb, img_context_emb)
    else:
        context_input = crossattn_emb

    with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    # Logging purpose
    affline_scale_log_info = {}
    affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
    self.affline_scale_log_info = affline_scale_log_info
    self.affline_emb = t_embedding_B_T_D
    self.crossattn_emb = crossattn_emb

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
            f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
        )

    # --- Run all transformer blocks ---
    block_kwargs = {
        "emb_B_T_D": t_embedding_B_T_D,
        "crossattn_emb": context_input,
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
    }

    for block in self.blocks:
        x_B_T_H_W_D = block(x_B_T_H_W_D, **block_kwargs)

    # --- Final Layer & Unpatchify ---
    x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

    # --- Update EasyCache state (cache at the final 5D output level) ---
    if is_even:
        # Calculate transformation rate k (Eq. 2):
        # k_t = ||v_t - v_{t-1}|| / ||x_t - x_{t-1}||
        if self.easycache_prev_output_even is not None and self.easycache_prev_prev_input_even is not None:
            output_change = (x_B_C_Tt_Hp_Wp - self.easycache_prev_output_even).abs().mean()
            input_change_for_k = (self.easycache_prev_input_even - self.easycache_prev_prev_input_even).abs().mean()
            if input_change_for_k > 1e-8:
                self.easycache_k = output_change / input_change_for_k

        # Update history
        self.easycache_prev_prev_input_even = self.easycache_prev_input_even.clone() if self.easycache_prev_input_even is not None else None
        self.easycache_prev_output_even = x_B_C_Tt_Hp_Wp.clone()
        # Cache transformation vector at 5D output level: Delta = output - raw_input
        self.easycache_cache_even = x_B_C_Tt_Hp_Wp - raw_input_5d
    else:
        self.easycache_prev_output_odd = x_B_C_Tt_Hp_Wp.clone()
        self.easycache_cache_odd = x_B_C_Tt_Hp_Wp - raw_input_5d

    # --- Increment step counter ---
    self.easycache_cnt += 1
    if self.easycache_cnt >= self.easycache_num_steps:
        skip_rate = self.easycache_step_skipped_count / self.easycache_num_steps
        log.info(f"[EasyCache Summary] Skipped {self.easycache_step_skipped_count}/{self.easycache_num_steps} steps ({skip_rate:.1%})")
        self._easycache_reset()

    return x_B_C_Tt_Hp_Wp


def _easycache_reset(self):
    """Reset all EasyCache state for the next generation."""
    self.easycache_cnt = 0
    self.easycache_accumulated_error_even = 0.0
    self.easycache_k = None
    self.easycache_should_compute = True
    self.easycache_prev_input_even = None
    self.easycache_prev_output_even = None
    self.easycache_prev_output_odd = None
    self.easycache_prev_prev_input_even = None
    self.easycache_cache_even = None
    self.easycache_cache_odd = None
    self.easycache_step_skipped_count = 0


def apply_easycache(
    model,
    num_steps: int = 35,
    thresh: float = 0.05,
    ret_steps: int = 5,
):
    """
    Applies EasyCache patching to the model.

    Args:
        model: The DiT model instance to patch.
        num_steps: Total number of forward passes (steps * 2 for CFG).
        thresh: Accumulated error threshold tau. Higher = more skipping (faster).
                Typical values: 0.05 (~1.5x), 0.1 (~2.0x), 0.2 (~3.0x).
        ret_steps: Number of initial warm-up diffusion steps to always compute fully.
                   Specified in diffusion steps (will be multiplied by 2 for CFG).
    """
    # Attach State Variables
    model.easycache_enabled = True
    model.easycache_num_steps = num_steps
    model.easycache_thresh = thresh
    model.easycache_ret_steps = ret_steps * 2  # Convert from diffusion steps to forward calls (CFG)
    model.easycache_cutoff_steps = num_steps - 2  # Always compute last step pair

    model.easycache_cnt = 0
    model.easycache_accumulated_error_even = 0.0
    model.easycache_k = None
    model.easycache_should_compute = True

    model.easycache_prev_input_even = None
    model.easycache_prev_output_even = None
    model.easycache_prev_output_odd = None
    model.easycache_prev_prev_input_even = None
    model.easycache_cache_even = None
    model.easycache_cache_odd = None

    model.easycache_step_skipped_count = 0

    # Patch the forward method and reset helper
    model.forward = types.MethodType(easycache_mini_train_dit_forward, model)
    model._easycache_reset = types.MethodType(_easycache_reset, model)

    log.info(f"[EasyCache] Applied to model with steps={num_steps}, thresh={thresh}, ret_steps={ret_steps}")
    return model
