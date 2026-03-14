# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TeaCache adaptation for Cosmos-Predict2.5
# Based on: "TeaCache: Timestep Embedding Aware Cache for Diffusion Transformers"
#
# Key idea: Use the timestep-embedding MODULATED noisy input (adaLN output)
# as the change indicator, apply polynomial rescaling to the relative L1
# distance, accumulate rescaled distances, and skip computation when below
# threshold delta.  Caches block residuals separately for cond/uncond CFG.

import torch
import numpy as np
import types
from typing import List, Optional, Tuple, Union
from cosmos_predict2._src.predict2.conditioner import DataType
import torch.amp as amp
from einops import rearrange
from cosmos_predict2._src.imaginaire.utils import log


def teacache_mini_train_dit_forward(
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
    TeaCache-enabled forward pass for MinimalV4DiT and MinimalV1LVGDiT.

    Uses timestep-embedding modulated input (adaLN output from block0) as the
    change indicator, polynomial rescaling of relative L1, and accumulated
    distance thresholding.  Caches block residuals separately for cond/uncond.
    """
    assert isinstance(data_type, DataType), (
        f"Expected DataType, got {type(data_type)}."
    )

    # --- MinimalV1LVGDiT Compatibility Logic ---
    is_lvg = hasattr(self, "timestep_scale")
    if is_lvg:
        if kwargs.get('timestep_scale') is None:
            timesteps_B_T = timesteps_B_T * self.timestep_scale
        if data_type == DataType.VIDEO:
            if condition_video_input_mask_B_C_T_H_W is not None:
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
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

    # Logging purpose
    affline_scale_log_info = {}
    affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
    self.affline_scale_log_info = affline_scale_log_info
    self.affline_emb = t_embedding_B_T_D
    self.crossattn_emb = crossattn_emb

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape

    # ==============================================================
    # TeaCache: Compute timestep-embedding modulated input (Eq. 6-7)
    # Uses block0's adaLN self-attention modulation as the indicator
    # ==============================================================
    block0 = self.blocks[0]

    # Compute adaLN modulation for block0 (self-attention branch)
    with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if block0.use_adaln_lora:
            shift_B_T_D, scale_B_T_D, gate_B_T_D = (
                block0.adaln_modulation_self_attn(t_embedding_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D, gate_B_T_D = block0.adaln_modulation_self_attn(
                t_embedding_B_T_D
            ).chunk(3, dim=-1)

    # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
    scale_B_T_1_1_D = rearrange(scale_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    shift_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

    # modulated_inp = LayerNorm(x) * (1 + scale) + shift  (same as Cosmos 1.0 adaln_norm_state)
    modulated_inp = block0.layer_norm_self_attn(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D

    # ==============================================================
    # TeaCache: Even/Odd decision logic (Eq. 5 + poly rescaling Eq. 6)
    # ==============================================================
    is_even = (self.teacache_cnt % 2 == 0)

    # Polynomial coefficients from the original TeaCache paper for Cosmos
    # These rescale L1_rel(F, t) to better predict output differences
    coefficients = self.teacache_coefficients

    if is_even:
        # --- Even (conditional) step ---
        if self.teacache_cnt == 0 or self.teacache_cnt == self.teacache_num_steps:
            # First step or last-step pair: always compute
            should_calc = True
            self.teacache_accum_even = 0.0
        else:
            # Compute rescaled relative L1 distance
            rel_l1 = ((modulated_inp - self.teacache_prev_mod_even).abs().mean()
                       / self.teacache_prev_mod_even.abs().mean()).cpu().item()
            rescale_func = np.poly1d(coefficients)
            rescaled = rescale_func(rel_l1)
            self.teacache_accum_even += rescaled

            if self.teacache_accum_even < self.teacache_thresh:
                should_calc = False
            else:
                should_calc = True
                self.teacache_accum_even = 0.0

        self.teacache_prev_mod_even = modulated_inp.clone()
        self.teacache_cnt += 1
        if self.teacache_cnt == self.teacache_num_steps + 2:
            self.teacache_cnt = 0

    else:
        # --- Odd (unconditional) step ---
        if self.teacache_cnt == 1 or self.teacache_cnt == self.teacache_num_steps + 1:
            should_calc = True
            self.teacache_accum_odd = 0.0
        else:
            rel_l1 = ((modulated_inp - self.teacache_prev_mod_odd).abs().mean()
                       / self.teacache_prev_mod_odd.abs().mean()).cpu().item()
            rescale_func = np.poly1d(coefficients)
            rescaled = rescale_func(rel_l1)
            self.teacache_accum_odd += rescaled

            if self.teacache_accum_odd < self.teacache_thresh:
                should_calc = False
            else:
                should_calc = True
                self.teacache_accum_odd = 0.0

        self.teacache_prev_mod_odd = modulated_inp.clone()
        self.teacache_cnt += 1
        if self.teacache_cnt == self.teacache_num_steps + 2:
            self.teacache_cnt = 0

    # ==============================================================
    # Execute or reuse cached residual
    # ==============================================================
    block_kwargs = {
        "emb_B_T_D": t_embedding_B_T_D,
        "crossattn_emb": context_input,
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
    }

    if not should_calc:
        # CACHE HIT: Reuse cached block residual
        if is_even and self.teacache_residual_even is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + self.teacache_residual_even
            self.teacache_step_skipped_count += 1
        elif not is_even and self.teacache_residual_odd is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + self.teacache_residual_odd
            self.teacache_step_skipped_count += 1
        else:
            # Fallback: no cache available, compute
            should_calc = True

    if should_calc:
        # CACHE MISS: Run all blocks
        ori_x = x_B_T_H_W_D.clone()

        for block in self.blocks:
            x_B_T_H_W_D = block(x_B_T_H_W_D, **block_kwargs)

        # Store block residual
        if is_even:
            self.teacache_residual_even = x_B_T_H_W_D - ori_x
        else:
            self.teacache_residual_odd = x_B_T_H_W_D - ori_x

    # --- Final Layer & Unpatchify ---
    x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

    # Log summary at end of generation
    total_steps = self.teacache_num_steps + 2
    if self.teacache_cnt == 0:
        skip_rate = self.teacache_step_skipped_count / total_steps if total_steps > 0 else 0
        log.info(f"[TeaCache Summary] Skipped {self.teacache_step_skipped_count}/{total_steps} steps ({skip_rate:.1%})")
        self.teacache_step_skipped_count = 0

    return x_B_C_Tt_Hp_Wp


def apply_teacache(
    model,
    num_steps: int = 35,
    rel_l1_thresh: float = 0.3,
    coefficients: list = None,
):
    """
    Applies TeaCache patching to the model.

    Args:
        model: The DiT model instance to patch.
        num_steps: Number of diffusion steps (will be multiplied by 2 for CFG counting).
        rel_l1_thresh: Accumulated rescaled L1 threshold delta.
                       Higher = more skipping (faster, lower quality).
                       Guidelines: 0.1 (~1.3x), 0.2 (~1.8x), 0.3 (~2.1x).
        coefficients: Polynomial coefficients for rescaling relative L1 distances.
                      Defaults to Cosmos 1.0 coefficients from the paper.
    """
    if coefficients is None:
        # Default: 4th-degree polynomial coefficients from TeaCache paper (Cosmos model)
        coefficients = [2.71156237e+02, -9.19775607e+01, 2.24437250e+00, 2.08355751e+00, 1.41776330e-01]

    # Attach State Variables
    model.teacache_enabled = True
    model.teacache_num_steps = num_steps * 2  # CFG doubles the call count
    model.teacache_thresh = rel_l1_thresh
    model.teacache_coefficients = coefficients

    model.teacache_cnt = 0
    model.teacache_accum_even = 0.0
    model.teacache_accum_odd = 0.0
    model.teacache_prev_mod_even = None
    model.teacache_prev_mod_odd = None
    model.teacache_residual_even = None
    model.teacache_residual_odd = None
    model.teacache_step_skipped_count = 0

    # Patch the forward method
    model.forward = types.MethodType(teacache_mini_train_dit_forward, model)

    log.info(f"[TeaCache] Applied to model with steps={num_steps}, thresh={rel_l1_thresh}, poly_degree={len(coefficients)-1}")
    return model
