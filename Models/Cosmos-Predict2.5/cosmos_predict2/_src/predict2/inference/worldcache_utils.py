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
import torch.nn.functional as F
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

def estimate_optical_flow(prev_img_tensor, curr_img_tensor, scale_factor=0.5):
    """
    Estimates optical flow using PyTorch-native Lucas-Kanade (GPU).
    prev_img_tensor: (B, C, H, W), normalized to [0,1] or similar scale.
    Input images are expected to be grayscale or will be converted.
    """
    # 1. Preprocessing (Grayscale + Scaling)
    # prev_img_tensor shape: (B, C, H, W)
    original_h, original_w = prev_img_tensor.shape[2], prev_img_tensor.shape[3]
    
    if scale_factor != 1.0:
        prev_img_tensor = F.interpolate(prev_img_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        curr_img_tensor = F.interpolate(curr_img_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    # Convert to grayscale (mean over C) -> (B, 1, H, W)
    I1 = prev_img_tensor.mean(dim=1, keepdim=True)
    I2 = curr_img_tensor.mean(dim=1, keepdim=True)
    
    # 2. Compute Gradients (Sobel)
    # Dx kernel
    k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=I1.dtype, device=I1.device).view(1, 1, 3, 3)
    k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=I1.dtype, device=I1.device).view(1, 1, 3, 3)
    
    # Padding for 'same' convolution
    pad = 1
    Ix = F.conv2d(I1, k_x, padding=pad)
    Iy = F.conv2d(I1, k_y, padding=pad)
    It = I2 - I1
    
    # 3. Compute Structure Tensor Elements (Window Integration)
    # Lucas-Kanade solves:
    # [ sum(Ix^2)  sum(IxIy) ] [ u ] = [ -sum(IxIt) ]
    # [ sum(IxIy)  sum(Iy^2) ] [ v ]   [ -sum(IyIt) ]
    
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It
    
    # Window summation (Mean or Sum is equivalent for linear system solution up to scale)
    # Using AvgPool as a box filter. Window size 21x21 is standard for smooth flow.
    win_size = 21
    avg_pool = torch.nn.AvgPool2d(kernel_size=win_size, stride=1, padding=win_size//2)
    
    sum_Ix2 = avg_pool(Ix2)
    sum_Iy2 = avg_pool(Iy2)
    sum_IxIy = avg_pool(IxIy)
    sum_IxIt = avg_pool(IxIt)
    sum_IyIt = avg_pool(IyIt)
    
    # 4. Solve 2x2 System (Cramer's Rule / Closed Form Inverse)
    # A = [a b; b c], det = ac - b^2
    # inv(A) = 1/det * [c -b; -b a]
    
    det = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy
    
    # Add epsilon for numerical stability
    det = det + 1e-6
    
    # u = (c * (-sum_IxIt) - (-b) * (-sum_IyIt)) / det
    # v = ((-b) * (-sum_IxIt) + a * (-sum_IyIt)) / det
    
    # Simplify signs:
    # RHS = [-Sxt, -Syt]
    # u = (sum_Iy2 * (-sum_IxIt) - (-sum_IxIy) * (-sum_IyIt)) / det
    #   = -(sum_Iy2 * sum_IxIt - sum_IxIy * sum_IyIt) / det
    
    # v = (-(-sum_IxIy) * (-sum_IxIt) + sum_Ix2 * (-sum_IyIt)) / det
    #   = -(sum_Ix2 * sum_IyIt - sum_IxIy * sum_IxIt) / det
    
    u = -(sum_Iy2 * sum_IxIt - sum_IxIy * sum_IyIt) / det
    v = -(sum_Ix2 * sum_IyIt - sum_IxIy * sum_IxIt) / det
    
    # Flow: (B, 2, H, W)
    flow = torch.cat((u, v), dim=1)
    
    # 5. Upscale Flow if needed
    if scale_factor != 1.0:
        flow = F.interpolate(flow, size=(original_h, original_w), mode='bilinear', align_corners=False)
        # Scale flow magnitudes
        flow = flow * (1.0 / scale_factor)
        
    # Return as (B, H, W, 2) for batched warping support.
    return flow.permute(0, 2, 3, 1)

def warp_feature(feature_tensor, flow_tensor):
    """
    Warps a feature tensor (B, C, H, W) using flow tensor of shape (H, W, 2) or (B, H, W, 2).
    """
    if flow_tensor is None:
        return feature_tensor
        
    B, C, H, W = feature_tensor.shape

    if flow_tensor.ndim == 3:
        flow_tensor = flow_tensor.unsqueeze(0).expand(B, -1, -1, -1)
    elif flow_tensor.ndim == 4:
        if flow_tensor.shape[0] == 1 and B > 1:
            flow_tensor = flow_tensor.expand(B, -1, -1, -1)
        elif flow_tensor.shape[0] != B:
            raise ValueError(f"Flow batch ({flow_tensor.shape[0]}) must match feature batch ({B}).")
    else:
        raise ValueError(f"Unsupported flow shape: {tuple(flow_tensor.shape)}")

    flow_tensor = flow_tensor.to(device=feature_tensor.device, dtype=feature_tensor.dtype)

    # Grid sample expects flow in range [-1, 1], but flow is in pixels.
    yy, xx = torch.meshgrid(
        torch.arange(H, device=feature_tensor.device, dtype=feature_tensor.dtype),
        torch.arange(W, device=feature_tensor.device, dtype=feature_tensor.dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)
    yy = yy.unsqueeze(0).expand(B, -1, -1)

    grid_x = xx + flow_tensor[..., 0]
    grid_y = yy + flow_tensor[..., 1]

    grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=3)
    
    # Align corners=True matches cv2 better usually
    warped_feature = F.grid_sample(feature_tensor, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    return warped_feature




def compute_saliency_map(features):
    """
    Computes a saliency map based on channel-wise standard deviation.
    features: (B, C, H, W) or (B, T, H, W, D)
    Returns: (B, 1, H, W) normalized to [0, 1] range per sample.
    """
    # If 5D (B, T, H, W, D), rearrange to (B, T*D, H, W)
    if features.ndim == 5:
        B, T, H, W, D = features.shape
        features = rearrange(features, 'b t h w d -> b (t d) h w')
    
    # Calculate std across channels (dim 1)
    # High variance across channels implies complex features/edges
    saliency = torch.std(features, dim=1, keepdim=True) # (B, 1, H, W)
    
    # Normalize per sample to [0, 1] for weighting
    # Avoid div by zero
    B = saliency.shape[0]
    saliency_flat = saliency.view(B, -1)
    min_val = saliency_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    max_val = saliency_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    saliency_norm = (saliency - min_val) / (max_val - min_val + 1e-6)
    
    return saliency_norm


def compute_optimal_gamma(delta_curr, delta_prev):
    """
    Computes the optimal scalar gamma for Online System Identification (OSI).
    Minimizes || delta_curr - gamma * delta_prev ||^2
    Solution: gamma* = dot(delta_curr, delta_prev) / (dot(delta_prev, delta_prev) + epsilon)
    
    delta_curr: (B, C, H, W) or flattened
    delta_prev: (B, C, H, W) or flattened
    Returns: gamma (B, 1, 1, 1) or scalar
    """
    # Flatten features (B, -1)
    B = delta_curr.shape[0]
    dc = delta_curr.view(B, -1)
    dp = delta_prev.view(B, -1)
    
    # Dot products
    numer = (dc * dp).sum(dim=1, keepdim=True)
    denom = (dp * dp).sum(dim=1, keepdim=True)
    
    # Solve gamma
    gamma = numer / (denom + 1e-6)
    
    # Clamp for stability (e.g., [0.2, 1.8] to avoid exploding predictions)
    gamma = torch.clamp(gamma, 0.2, 1.8)
    # Reshape for broadcasting with 5D tensors [B, T, H, W, D]
    return gamma.view(B, 1, 1, 1, 1)


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
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    WorldCache-enabled forward pass for MinimalV4DiT and MinimalV1LVGDiT.
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

    # --- WorldCache Logic Start ---
    skip_forward = False
    ori_x = x_B_T_H_W_D  # Keep reference to input
    
    # We maintain state in self.cnt, self.num_steps, self.ret_ratio, etc.
    # self.cnt is incremented at the end.
    
    current_idx = self.cnt % 2  # Ping-pong for sequential CFG (cond/uncond)
    
    test_x = x_B_T_H_W_D.clone()

    if self.cnt >= int(self.worldcache_num_steps * self.worldcache_ret_ratio):
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
             
             # --- Saliency-Guided Thresholding ---
             # Weight the drift based on saliency
             if getattr(self, 'worldcache_saliency_enabled', False):
                 # Compute Saliency of the PREVIOUS internal state (which we are comparing against)
                 # self.previous_internal_states[current_idx] is (B, T, H, W, D)
                 saliency_map = compute_saliency_map(self.previous_internal_states[current_idx])
                 
                 # Delta Y Map (per pixel drift)
                 # Recompute element-wise delta: (test_x - prev).abs()
                 # First standardizing shapes
                 diff = (test_x - self.previous_internal_states[current_idx]).abs()
                 # Reduce to (B, 1, H, W) by mean/sum over T/D
                 # Let's align with saliency map dimensions (H, W).
                 diff_map = rearrange(diff, 'b t h w d -> b (t d) h w').mean(dim=1, keepdim=True)
                 
                 # Weighted Drift Calculation
                 # Weight = 1 + beta * Saliency
                 beta = getattr(self, 'worldcache_saliency_weight', 5.0)
                 weight_map = 1.0 + beta * saliency_map
                 
                 # Weighted Mean Distance
                 weighted_drift = (diff_map * weight_map).mean()
                 
                 # Normalize? 
                 # Base delta_y was normalized by prev.abs().mean().
                 # Let's keep consistency.
                 denom = self.previous_internal_states[current_idx].abs().mean() + 1e-6
                 weighted_rel_drift = weighted_drift / denom
                 
                 # Update drift with weighted version
                 # Replacing makes sense: we are refining the error metric.
                 # OVERWRITE the accumulation for this step if enabled (assuming prompt reset)
                 # OR just subtract old and add new.
                 # Simpler: Calculate weighted drift INSTEAD of plain delta_y if enabled.
                 
                 self.accumulated_rel_l1_distance[current_idx] -= delta_y 
                 self.accumulated_rel_l1_distance[current_idx] += weighted_rel_drift
                 
                 # Update for logging
                 delta_y = weighted_rel_drift


             # --- Causal-DiCache (Motion-Adaptive Thresholding) ---
             # Calculate input velocity (drift of x_t vs x_{t-1})
             # delta_x is roughly the velocity of the latent features over the interval.
             # delta_x = (x - prev).abs().mean() / prev.abs().mean()
             # This is "normalized velocity".
             
             input_velocity = delta_x
             alpha = getattr(self, 'worldcache_motion_sensitivity', 5.0) # Default sensitivity
             
             # Dynamic Threshold Formula
             # fast motion -> high velocity -> high denominator -> low threshold -> harder to skip (compute more)
             dynamic_thresh = self.worldcache_rel_l1_thresh / (1.0 + alpha * input_velocity)

             # --- Dynamic Threshold Decay ---
             if getattr(self, 'worldcache_dynamic_decay', False):
                 num_steps = getattr(self, 'worldcache_num_steps', 35)
                 
                 # Using a quadratic fit
                 u = num_steps / 35.0
                 base_mult = (u**2) / 6.0 + (u / 2.0) + (10.0 / 3.0)
                 step_ratio = self.cnt / num_steps
                 decay_factor = 1.0 + base_mult * step_ratio
                 dynamic_thresh *= decay_factor

             log.info(f"[WorldCache] Step {self.cnt}: Drift {self.accumulated_rel_l1_distance[current_idx]:.4f} / Thresh {dynamic_thresh:.4f} (Base {self.worldcache_rel_l1_thresh}, Vel {input_velocity:.4f}, Alpha {alpha})")
             if self.accumulated_rel_l1_distance[current_idx] < dynamic_thresh: # skip this step
                 skip_forward = True
                 self.worldcache_step_skipped_count += 1
                 self.resume_flag[current_idx] = False 
                 residual_x = self.residual_cache[current_idx]
             else:
                 self.resume_flag[current_idx] = True
                 self.accumulated_rel_l1_distance[current_idx] = 0
                 # Update history with the fresh probe output!
                 self.previous_internal_states[current_idx] = test_x.clone()
        else:
             # Fallback if history missing (first step or issue)
             pass

    # --- Execution Branching ---
    if skip_forward:
        # CACHE HIT: Approximate the rest of the network using cached residual + Gamma interpolation
        ori_x_clone = x_B_T_H_W_D.clone() # ori_x is already a tensor ref, but we don't modify it in place until addition
        
        # Gamma Interpolation
        if len(self.residual_window[current_idx]) >= 2:
            current_residual_indicator = test_x - x_B_T_H_W_D
            
            # --- Online System Identification (OSI) ---
            if getattr(self, 'worldcache_osi_enabled', False):
                # Calculate delta_prev (Probe_{t-1} - Probe_{t-2})
                # window[-1] is latest (t-1), -2 is (t-2)
                delta_prev = self.probe_residual_window[current_idx][-1] - self.probe_residual_window[current_idx][-2]
                
                # Calculate delta_curr (Probe_{t} - Probe_{t-1}) ~ current_residual_indicator?
                # current_residual_indicator = test_x - x == Probe_t - (Probe_{t-1})?
                # current_residual_indicator is effectively (Probe_t - Input_t) 
                
                # OSI Logic: 
                # We want to match the DIRECTION of feature evolution.
                # Minimize || (CurrResid - PrevPrevResid) - gamma * (PrevResid - PrevPrevResid) ||^2
                
                delta_target = current_residual_indicator - self.probe_residual_window[current_idx][-2]
                delta_source = self.probe_residual_window[current_idx][-1] - self.probe_residual_window[current_idx][-2]
                
                gamma = compute_optimal_gamma(delta_target, delta_source)
                
                # Log average gamma for debugging
                # avg_gamma = gamma.mean().item()
                # print(f"[OSI ACTIVE] Step {self.cnt}: Optimal Gamma {avg_gamma:.3f}")



            else:
                # --- Original Heuristic Logic ---
                numer = (current_residual_indicator - self.probe_residual_window[current_idx][-2]).abs().mean()
                denom = (self.probe_residual_window[current_idx][-1] - self.probe_residual_window[current_idx][-2]).abs().mean()
                if denom < 1e-6:
                    gamma = 1.0
                else:
                    gamma = (numer / denom).clip(1, 2)

            # Apply Gamma: x += prev_prev_res + gamma * (prev_res - prev_prev_res)
            # This extrapolates the residual trend.
            x_B_T_H_W_D = x_B_T_H_W_D + self.residual_window[current_idx][-2] + gamma * (self.residual_window[current_idx][-1] - self.residual_window[current_idx][-2])
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + residual_x

        self.previous_internal_states[current_idx] = test_x
        self.previous_input[current_idx] = ori_x # Store original input
        
        # --- Flow-Warped Scaling ---
        if getattr(self, 'worldcache_flow_enabled', False) and self.previous_input[current_idx] is not None:
             # Warp the internal state to align with current motion
             # Feature: x_B_T_H_W_D -> rearrange to (B, D, H, W) for image-based flow
             # Or (B, T*D, H, W)
             # Treat T*D as channels for warping.
             
             # 1. Estimate Flow on Inputs (x vs prev_x)
             # x_B_T_H_W_D -> (B, T*D, H, W)
             curr_input_img = rearrange(x_B_T_H_W_D, 'b t h w d -> b (t d) h w')
             prev_input_img = rearrange(self.previous_input[current_idx], 'b t h w d -> b (t d) h w')
             
             flow_scale = getattr(self, 'worldcache_flow_scale', 0.5)
             flow = estimate_optical_flow(prev_input_img, curr_input_img, scale_factor=flow_scale)
             
             # 2. Warp Cached Internal State
             # prev_state: (B, T, H, W, D)
             prev_state_img = rearrange(self.previous_internal_states[current_idx], 'b t h w d -> b (t d) h w')
             warped_state_img = warp_feature(prev_state_img, flow)
             warped_state = rearrange(warped_state_img, 'b (t d) h w -> b t h w d', t=T)
             
             # 3. Use Warped State as Base
             # Diff = WarpedState - PrevState
             # Standard Cache: x = x + Residual
             # Flow Cache: x = WarpedState + Residual
             # Ensure we add WARPED residual instead of raw residual.
             
             if flow is not None:
                 # Warp the residual cache
                 residual_img = rearrange(residual_x, 'b t h w d -> b (t d) h w')
                 warped_residual_img = warp_feature(residual_img, flow)
                 warped_residual = rearrange(warped_residual_img, 'b (t d) h w -> b t h w d', t=T)
                 
                 # Re-calculate x using warped residual
                 # Undo previous addition (x = x + residual)
                 # Re-do with warped
                 x_B_T_H_W_D = x_B_T_H_W_D - residual_x + warped_residual
        
    else:
        # CACHE MISS or PROBE PHASE: Run the blocks
        
        if self.resume_flag[current_idx]: # resume from test_x (which ran first 'probe_depth' blocks)
            x_B_T_H_W_D = test_x
            unpass_blocks = self.blocks[self.probe_depth:]
            # We already ran blocks 0..probe_depth-1
        else: # pass all blocks
            unpass_blocks = self.blocks
            
        # Run remaining blocks
        for i, block in enumerate(unpass_blocks):
             x_B_T_H_W_D = block(x_B_T_H_W_D, **block_kwargs)
             
             # If we are in the "full run" mode (not resume), we need to capture the probe state 
             # at the correct layer for history.
             real_layer_idx = i if not self.resume_flag[current_idx] else i + self.probe_depth
             
             if real_layer_idx == self.probe_depth - 1:
                 # We are at the probe output
                 if self.cnt >= int(self.worldcache_num_steps * self.worldcache_ret_ratio):
                      # If we are in retention phase but chose to run full (resume=False or just didn't skip)
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
    if self.cnt >= self.worldcache_num_steps: 
        # Log Summary before reset
        cache_rate = self.worldcache_step_skipped_count / self.worldcache_num_steps
        log.info(f"[WorldCache Summary] Skipped {self.worldcache_step_skipped_count}/{self.worldcache_num_steps} steps ({cache_rate:.1%}) | MotionSens={getattr(self, 'worldcache_motion_sensitivity', 'N/A')}")
        
        # Reset at end of generation
        # WAN resets self.cnt = 0 and clears cache
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


    # Visualize (WorldCache Enabled)
    # if self.cnt < self.worldcache_num_steps:
    #     # Determine if this step was skipped (cached)
    #     suffix = ""
    #     if skip_forward:
    #          suffix = "_cached"
        
    #     # Determine CFG part (Even=Cond, Odd=Uncond)
    #     is_cond = (self.cnt % 2 == 0)
    #     subfolder = "cond" if is_cond else "uncond"
    #     step_idx = self.cnt // 2

    #     # Save to outputs/debug_vis_worldcache
    #     visualize_latent(x_B_C_Tt_Hp_Wp, step_idx, save_dir="outputs/debug_vis_worldcache", suffix=suffix, subfolder_name=subfolder)

    # Store Final Output
    if current_idx < len(self.previous_output):
        self.previous_output[current_idx] = x_B_C_Tt_Hp_Wp.clone()

    return x_B_C_Tt_Hp_Wp


def apply_worldcache(
    model,
    num_steps: int = 35,
    rel_l1_thresh: float = 0.08,
    ret_ratio: float = 0.2,
    probe_depth: int = 1,
    motion_sensitivity: float = 5.0,
    flow_enabled: bool = False,
    flow_scale: float = 0.5,
    saliency_enabled: bool = False,
    saliency_weight: float = 5.0,
    osi_enabled: bool = False,
    dynamic_decay: bool = False,
):
    """
    Applies WorldCache patching to the model.
    """
    # Attach State Variables
    model.worldcache_enabled = True
    model.worldcache_num_steps = num_steps
    model.worldcache_rel_l1_thresh = rel_l1_thresh
    model.worldcache_ret_ratio = ret_ratio
    model.probe_depth = probe_depth
    model.worldcache_motion_sensitivity = motion_sensitivity
    model.worldcache_flow_enabled = flow_enabled
    model.worldcache_flow_scale = flow_scale
    model.worldcache_saliency_enabled = saliency_enabled
    model.worldcache_saliency_weight = saliency_weight
    model.worldcache_osi_enabled = osi_enabled
    model.worldcache_dynamic_decay = dynamic_decay

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
    
    # Patch the forward method
    import types
    model.forward = types.MethodType(worldcache_mini_train_dit_forward, model)
    
    log.info(f"WorldCache applied: steps={num_steps}, thresh={rel_l1_thresh}, alpha={motion_sensitivity}, flow={flow_enabled}, scale={flow_scale}, saliency={saliency_enabled}, osi={osi_enabled}, decay={dynamic_decay}")

    return model

