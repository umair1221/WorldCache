# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from cosmos_predict2._src.predict2.conditioner import DataType
import torch.amp as amp
from einops import rearrange
from cosmos_predict2._src.imaginaire.utils import log
import torchvision.transforms as transforms

from cosmos_predict2._src.predict2.inference.debug_utils import visualize_latent

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
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    DiCache-enabled forward pass for MinimalV4DiT and MinimalV1LVGDiT.
    """
    assert isinstance(data_type, DataType), (
        f"Expected DataType, got {type(data_type)}."
    )

    # --- MinimalV1LVGDiT Compatibility Logic ---
    # Check if we need to apply LVG-specific pre-processing
    is_lvg = hasattr(self, "timestep_scale")
    
    if is_lvg:
        if kwargs.get('timestep_scale') is None:
             # Apply raw scaling if it hasn't been applied (though usually applied in forward args call? No, it's a property)
             timesteps_B_T = timesteps_B_T * self.timestep_scale

        if data_type == DataType.VIDEO:
            if condition_video_input_mask_B_C_T_H_W is not None:
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
            else:
                # Should not happen for VIDEO if mask is expected?
                # But strict replication says:
                pass 
        else:
             # Image case padding
             B, _, T, H, W = x_B_C_T_H_W.shape
             x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
             )
    
    # --- Standard Pre-processing (from MinimalV4DiT.forward) ---
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

    # Logging purpose (kept from original)
    affline_scale_log_info = {}
    affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
    self.affline_scale_log_info = affline_scale_log_info
    self.affline_emb = t_embedding_B_T_D
    self.crossattn_emb = crossattn_emb

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
            f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
        )

    B, T, H, W, D = x_B_T_H_W_D.shape

    # Prepare block kwargs to simplify calls
    block_kwargs = {
        "emb_B_T_D": t_embedding_B_T_D,
        "crossattn_emb": context_input,
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
    }

    # --- DiCache Logic Start ---
    skip_forward = False
    ori_x = x_B_T_H_W_D  # Keep reference to input
    
    # We maintain state in self.cnt, self.num_steps, self.ret_ratio, etc.
    # self.cnt is incremented at the end.
    
    # Determine if we check for skipping
    # WAN logic: if self.cnt >= int(self.num_steps * self.ret_ratio):
    
    # Use modulo 2 for ping-pong buffer index
    current_idx = self.cnt % 2
    
    test_x = x_B_T_H_W_D.clone()

    if self.cnt >= int(self.dicache_num_steps * self.dicache_ret_ratio):
        # PROBE: Run the first 'probe_depth' blocks
        anchor_blocks = self.blocks[0:self.probe_depth]
        for block in anchor_blocks:
            test_x = block(test_x, **block_kwargs)
            
        # Calculate Drifts
        # delta_x: Input drift (current input vs previous input)
        # delta_y: Internal feature drift (current probe output vs previous probe output)
        
        # Note: self.previous_input and self.previous_internal_states must have been set in previous steps.
        # Since we only start checking after ret_ratio, they should be populated.
        
        # Safety check for first run after ret_ratio if history is somehow missing (shouldn't happen if initialized)
        if self.previous_input[current_idx] is not None and self.previous_internal_states[current_idx] is not None:
             delta_x = (x_B_T_H_W_D - self.previous_input[current_idx]).abs().mean() / self.previous_input[current_idx].abs().mean()
             delta_y = (test_x - self.previous_internal_states[current_idx]).abs().mean() / self.previous_internal_states[current_idx].abs().mean()
             
             self.accumulated_rel_l1_distance[current_idx] += delta_y # update error accumulator

             if self.accumulated_rel_l1_distance[current_idx] < self.dicache_rel_l1_thresh: # skip this step
                 skip_forward = True
                 self.resume_flag[current_idx] = False 
                 residual_x = self.residual_cache[current_idx]
             else:
                 self.resume_flag[current_idx] = True
                 self.accumulated_rel_l1_distance[current_idx] = 0
        else:
             # Fallback if history missing (first step or issue)
             pass

    # --- Execution Branching ---
    if skip_forward:
        # CACHE HIT: Approximate the rest of the network using cached residual + Gamma interpolation
        ori_x_clone = x_B_T_H_W_D.clone() # Needed? ori_x is already a tensor ref, but we don't modify it in place until addition
        
        # Gamma Interpolation
        if len(self.residual_window[current_idx]) >= 2:
            current_residual_indicator = test_x - x_B_T_H_W_D
            
            # (current_resid_ind - prev_probe_resid) / (prev_prev_probe_resid - prev_probe_resid) ??
            # WAN: gamma = ((current_residual_indicator - self.probe_residual_window[self.cnt%2][-2]).abs().mean() / (self.probe_residual_window[self.cnt%2][-1] - self.probe_residual_window[self.cnt%2][-2]).abs().mean()).clip(1, 2)
            # Wait, WAN indices: [-1] is latest? 
            # In WAN code:
            # self.probe_residual_window[self.cnt%2][-2] = self.probe_residual_window[self.cnt%2][-1]
            # self.probe_residual_window[self.cnt%2][-1] = self.probe_residual_cache[self.cnt%2]
            # So [-1] is the most recent (t-1), [-2] is (t-2).
            # But in the formula line 173:
            # denom = (window[-1] - window[-2])
            # num = (current - window[-2])
            # This looks like linear extrapolation/interpolation based on recent trend.
            
            numer = (current_residual_indicator - self.probe_residual_window[current_idx][-2]).abs().mean()
            denom = (self.probe_residual_window[current_idx][-1] - self.probe_residual_window[current_idx][-2]).abs().mean()
            # Avoid div zero
            if denom < 1e-6:
                gamma = 1.0
            else:
                gamma = (numer / denom).clip(1, 2)
                
            # x += window[-2] + gamma * (window[-1] - window[-2])
            x_B_T_H_W_D = x_B_T_H_W_D + self.residual_window[current_idx][-2] + gamma * (self.residual_window[current_idx][-1] - self.residual_window[current_idx][-2])
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + residual_x

        self.previous_internal_states[current_idx] = test_x
        self.previous_input[current_idx] = ori_x # Store original input
        
    else:
        # CACHE MISS or PROBE PHASE: Run the blocks
        
        if self.resume_flag[current_idx]: # resume from test_x (which ran first 'probe_depth' blocks)
            x_B_T_H_W_D = test_x
            unpass_blocks = self.blocks[self.probe_depth:]
            # We already ran blocks 0..probe_depth-1
        else: # pass all blocks
            unpass_blocks = self.blocks
            
        # Run remaining blocks
        # Need to adjust index if reusing test_x? 
        # Yes, enumerate from correct index if needed for logging/intermediate (omitted here for simplicity unless requested)
        for i, block in enumerate(unpass_blocks):
             x_B_T_H_W_D = block(x_B_T_H_W_D, **block_kwargs)
             
             # If we are in the "full run" mode (not resume), we need to capture the probe state 
             # at the correct layer for history.
             # Only if we started from scratch (not resume).
             # WAN logic: 
             # if ind == self.probe_depth - 1:
             #    if self.cnt >= int(self.num_steps * self.ret_ratio):
             #         self.previous_internal_states[self.cnt%2] = test_x # directly use test_x
             #    else:
             #         self.previous_internal_states[self.cnt%2] = x # count for internal states
             
             # The index 'i' here is relative to unpass_blocks.
             # If resume=True, we started after probe_depth. So we are past the probe point.
             # If resume=False, we started at 0.
             
             real_layer_idx = i if not self.resume_flag[current_idx] else i + self.probe_depth
             
             if real_layer_idx == self.probe_depth - 1:
                 # We are at the probe output
                 if self.cnt >= int(self.dicache_num_steps * self.dicache_ret_ratio):
                      # If we are in retention phase but chose to run full (resume=False or just didn't skip?)
                      # If resume=True, we skip this check because we started AFTER probe.
                      # If resume=False (before retention phase), we capture x.
                      pass
                 
                 self.previous_internal_states[current_idx] = x_B_T_H_W_D.clone()

        # Update Cache
        # residual_x = output - input
        residual_x = x_B_T_H_W_D - ori_x
        
        self.residual_cache[current_idx] = residual_x
        # probe_cache = probe_output - input
        self.probe_residual_cache[current_idx] = self.previous_internal_states[current_idx] - ori_x
        
        self.previous_input[current_idx] = ori_x
        self.previous_output[current_idx] = x_B_T_H_W_D
        
        # Update Windows
        if len(self.residual_window[current_idx]) <= 2:
            self.residual_window[current_idx].append(residual_x)
            self.probe_residual_window[current_idx].append(self.probe_residual_cache[current_idx])
        else:
            # Shift buffer
            self.residual_window[current_idx][-2] = self.residual_window[current_idx][-1]
            self.residual_window[current_idx][-1] = residual_x
            
            self.probe_residual_window[current_idx][-2] = self.probe_residual_window[current_idx][-1]
            self.probe_residual_window[current_idx][-1] = self.probe_residual_cache[current_idx]

    # --- Final Layer & Unpatchify ---
    x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

    # Increment Step Counter
    self.cnt += 1
    if self.cnt >= self.dicache_num_steps: 
        # Reset at end of generation
        # WAN resets self.cnt = 0 and clears cache
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


    # Visualize (DiCache Enabled)
    if self.cnt < self.dicache_num_steps:
        # Determine if this step was skipped (cached)
        suffix = ""
        if skip_forward:
             suffix = "_cached"
        
        # Determine CFG part (Even=Cond, Odd=Uncond)
        is_cond = (self.cnt % 2 == 0)
        subfolder = "cond" if is_cond else "uncond"
        step_idx = self.cnt // 2

        # Save to outputs/debug_vis_dicache
        visualize_latent(x_B_C_Tt_Hp_Wp, step_idx, save_dir="outputs/debug_vis_dicache", suffix=suffix, subfolder_name=subfolder)

    return x_B_C_Tt_Hp_Wp


def apply_dicache(
    model,
    num_steps: int = 35,
    rel_l1_thresh: float = 0.08,
    ret_ratio: float = 0.2,
    probe_depth: int = 1
):
    """
    Applies DiCache patching to the model.
    """
    # Attach State Variables
    model.dicache_enabled = True
    model.dicache_num_steps = num_steps
    model.dicache_rel_l1_thresh = rel_l1_thresh
    model.dicache_ret_ratio = ret_ratio
    model.probe_depth = probe_depth
    
    model.cnt = 0
    model.accumulated_rel_l1_distance = [0.0, 0.0]
    model.residual_cache = [None, None]
    model.probe_residual_cache = [None, None]
    model.residual_window = [[], []] # List of lists
    model.probe_residual_window = [[], []]
    model.previous_internal_states = [None, None]
    model.previous_input = [None, None]
    model.previous_output = [None, None]
    model.resume_flag = [False, False]
    
    # Patch the forward method
    # Use __func__ to get the unbound function if it's a bound method, 
    # but since we are replacing the instance method, we can bind it or replace on class.
    # Typically replace on instance if possible, or use types.MethodType.
    
    import types
    model.forward = types.MethodType(dicache_mini_train_dit_forward, model)
    
    log.info(f"DiCache applied to model with steps={num_steps}, thresh={rel_l1_thresh}, ratio={ret_ratio}, probe={probe_depth}")
    return model
