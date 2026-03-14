
import math
import torch
import torch.nn.functional as F
import torch.fft
from einops import rearrange
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType

@torch.no_grad()
def fft(tensor):
    """
    Apply FFT to the input tensor (B, C, H, W) or (B, C, T, H, W) treated as 2D spatial images 
    (flattening T into B or handling T separately if needed, but here we expect (B*T, C, H, W)).
    """
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    B, C, H, W = tensor.size()
    radius = min(H, W) // 5
            
    Y, X = torch.meshgrid(torch.arange(H, device=tensor.device), torch.arange(W, device=tensor.device), indexing='ij')
    center_x, center_y = W // 2, H // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask
            
    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft

@torch.no_grad()
def fastercache_mini_train_dit_forward(
    self,
    x_B_C_T_H_W: torch.Tensor,
    timesteps_B_T: torch.Tensor,
    crossattn_emb: torch.Tensor,
    fps: torch.Tensor = None,
    padding_mask: torch.Tensor = None,
    data_type = None, 
    intermediate_feature_ids = None,
    img_context_emb: torch.Tensor = None,
    condition_video_input_mask_B_C_T_H_W: torch.Tensor = None,
    **kwargs,
):
    """
    Patched forward method for MiniTrainDIT (and MinimalV1LVGDiT) to support FasterCache.
    Includes logic from MinimalV1LVGDiT to handle condition mask and scaling.
    """
    # --- MinimalV1LVGDiT Logic ---
    if hasattr(self, 'timestep_scale'):
        timesteps_B_T = timesteps_B_T * self.timestep_scale

    if condition_video_input_mask_B_C_T_H_W is not None:
        if data_type == DataType.VIDEO:
             x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
            
    # 1. Prepare Inputs
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

    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)
        
    with torch.amp.autocast('cuda', enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    self.fastercache_counter += 1
    counter = self.fastercache_counter
    
    # Defaults if not set
    fc_start = getattr(self, 'fastercache_start_step', 18)
    fc_model_int = getattr(self, 'fastercache_model_interval', 5)
    
    # Sync counters to blocks
    for block in self.blocks:
        block.current_counter = counter
    
    B_total = x_B_T_H_W_D.shape[0]
    
    apply_cache = (
        getattr(self, 'fastercache_enabled', False)
        and B_total >= 2 
        and B_total % 2 == 0 
        and counter >= fc_start 
        and counter % fc_model_int != 0
    )

    # If applying cache, we run only the first half (Conditional)
    if apply_cache:
        half_B = B_total // 2
        
        # Slicing inputs
        x_half = x_B_T_H_W_D[:half_B]
        t_emb_half = t_embedding_B_T_D[:half_B]
        
        if isinstance(context_input, tuple):
            ctx_half = tuple(c[:half_B] for c in context_input)
        else:
            ctx_half = context_input[:half_B]
            
        adaln_lora_half = None
        if adaln_lora_B_T_3D is not None:
            adaln_lora_half = adaln_lora_B_T_3D[:half_B]
            
        extra_pos_half = None
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape[0] == B_total:
                extra_pos_half = extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D[:half_B]
            else:
                extra_pos_half = extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D

        # --- Forward Pass (Conditional Only) ---
        x_out_half = x_half
        for block in self.blocks:
            x_out_half = block(
                x_B_T_H_W_D=x_out_half,
                emb_B_T_D=t_emb_half,
                crossattn_emb=ctx_half,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_half,
                extra_per_block_pos_emb=extra_pos_half,
            )
            
        # Final Layer
        x_out_half = self.final_layer(x_out_half, t_emb_half, adaln_lora_B_T_3D=adaln_lora_half)
        
        # --- FFT Approximation ---
        single_output_unpatched = self.unpatchify(x_out_half) # (B/2, C, T, H, W)
        
        (bb, cc, tt, hh, ww) = single_output_unpatched.shape
        cond_flat = rearrange(single_output_unpatched, "b c t h w -> (b t) c h w")
        
        lf_c, hf_c = fft(cond_flat.float())
        
        # Scale factors hardcoded in CogVideoX, but we can tune them too or leave them for now.
        # Let's keep them hardcoded unless user asks.
        if counter <= 40:
            self.delta_lf = self.delta_lf * 1.1
        if counter >= 30:
            self.delta_hf = self.delta_hf * 1.1
            
        new_hf_uc = self.delta_hf + hf_c
        new_lf_uc = self.delta_lf + lf_c
        
        combine_uc = new_lf_uc + new_hf_uc
        combined_fft = torch.fft.ifftshift(combine_uc)
        recovered_uncond_flat = torch.fft.ifft2(combined_fft).real
        
        recovered_uncond = rearrange(
            recovered_uncond_flat.to(single_output_unpatched.dtype), 
            "(b t) c h w -> b c t h w", b=bb, t=tt
        )
        
        x_B_C_Tt_Hp_Wp = torch.cat([single_output_unpatched, recovered_uncond], dim=0)
        
    else:
        # --- Full Forward Pass (No Skip) ---
        x_out = x_B_T_H_W_D
        for block in self.blocks:
            x_out = block(
                x_B_T_H_W_D=x_out,
                emb_B_T_D=t_embedding_B_T_D,
                crossattn_emb=context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
        x_out = self.final_layer(x_out, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_out)
        
        # Update Cache Logic
        if getattr(self, 'fastercache_enabled', False) and B_total >= 2 and B_total % 2 == 0:
            (bb, cc, tt, hh, ww) = x_B_C_Tt_Hp_Wp.shape
            half_b = bb // 2
            
            cond = x_B_C_Tt_Hp_Wp[:half_b]
            uncond = x_B_C_Tt_Hp_Wp[half_b:]
            
            cond_flat = rearrange(cond, "b c t h w -> (b t) c h w").float()
            uncond_flat = rearrange(uncond, "b c t h w -> (b t) c h w").float()
            
            lf_c, hf_c = fft(cond_flat)
            lf_uc, hf_uc = fft(uncond_flat)
            
            self.delta_lf = lf_uc - lf_c
            self.delta_hf = hf_uc - hf_c

    return x_B_C_Tt_Hp_Wp


def fastercache_block_forward(
    self,
    x_B_T_H_W_D: torch.Tensor,
    emb_B_T_D: torch.Tensor,
    crossattn_emb: torch.Tensor,
    rope_emb_L_1_1_D: torch.Tensor = None,
    adaln_lora_B_T_3D: torch.Tensor = None,
    extra_per_block_pos_emb: torch.Tensor = None,
) -> torch.Tensor:
    """
    Patched forward method for Block to support FasterCache.
    """
    if extra_per_block_pos_emb is not None:
        x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

    with torch.amp.autocast('cuda', enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

    # Reshape helpers
    def r(t): return rearrange(t, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    
    shift_self_attn_B_T_1_1_D = r(shift_self_attn_B_T_D)
    scale_self_attn_B_T_1_1_D = r(scale_self_attn_B_T_D)
    gate_self_attn_B_T_1_1_D = r(gate_self_attn_B_T_D)
    
    shift_cross_attn_B_T_1_1_D = r(shift_cross_attn_B_T_D)
    scale_cross_attn_B_T_1_1_D = r(scale_cross_attn_B_T_D)
    gate_cross_attn_B_T_1_1_D = r(gate_cross_attn_B_T_D)
    
    shift_mlp_B_T_1_1_D = r(shift_mlp_B_T_D)
    scale_mlp_B_T_1_1_D = r(scale_mlp_B_T_D)
    gate_mlp_B_T_1_1_D = r(gate_mlp_B_T_D)

    B, T, H, W, D = x_B_T_H_W_D.shape
    
    def _fn(_x, _norm, _scale, _shift):
        return _norm(_x) * (1 + _scale) + _shift

    normalized_x_B_T_H_W_D = _fn(
        x_B_T_H_W_D,
        self.layer_norm_self_attn,
        scale_self_attn_B_T_1_1_D,
        shift_self_attn_B_T_1_1_D,
    )
    
    # Mock video size if needed
    class MockVideoSize:
         def __init__(self, t, h, w): self.T, self.H, self.W = t, h, w
    video_size = MockVideoSize(T * self.cp_size if self.cp_size and self.cp_size > 1 else T, H, W)

    # --- FASTERCACHE LOGIC ---
    counter = getattr(self, 'current_counter', 0)
    
    # Defaults if not set
    fc_start = getattr(self, 'fastercache_start_step', 18)
    fc_interval = getattr(self, 'fastercache_block_interval', 3)
    
    use_cache = (
        getattr(self, 'fastercache_enabled', False)
        and counter >= fc_start 
        and counter % fc_interval != 0 
        and hasattr(self, 'attn_cache')
        and len(self.attn_cache) >= 2
    )

    if use_cache:
        # Use cached attention output (Interpolated)
        cached_prev = self.attn_cache[0]
        cached_curr = self.attn_cache[1]
        
        # result_B_T_H_W_D = cached_curr + (cached_curr - cached_prev) * 0.3
        
        # Dynamic interpolation factor from OpenSora reference
        # factor = (counter - start_step) / 20.0 * 0.9
        # Capped at reasonable bounds if needed, but following reference logic:
        factor = (counter - fc_start) / 20.0 * 0.9
        
        result_B_T_H_W_D = cached_curr + (cached_curr - cached_prev) * factor
    else:
        # Compute Attention
        attn_in = rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        attn_out = self.self_attn(
            attn_in,
            None,
            rope_emb=rope_emb_L_1_1_D,
            video_size=video_size,
        )
        result_B_T_H_W_D = rearrange(attn_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        
        # Update Cache
        if getattr(self, 'fastercache_enabled', False):
            # We start caching slightly before the active usage to build history
            cache_start_build = max(1, fc_start - 3)
            
            if counter == cache_start_build:
                self.attn_cache = [result_B_T_H_W_D.detach(), result_B_T_H_W_D.detach()]
            elif counter > cache_start_build:
                if isinstance(self.attn_cache, list) and len(self.attn_cache)>0:
                    self.attn_cache = [self.attn_cache[-1], result_B_T_H_W_D.detach()]
                else:
                    self.attn_cache = [result_B_T_H_W_D.detach(), result_B_T_H_W_D.detach()]

    x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

    # Cross Attention (Unchanged)
    def _x_fn(_x_B_T_H_W_D):
        _normalized_x_B_T_H_W_D = _fn(
            _x_B_T_H_W_D, self.layer_norm_cross_attn, scale_cross_attn_B_T_1_1_D, shift_cross_attn_B_T_1_1_D
        )
        _result_B_T_H_W_D = rearrange(
            self.cross_attn(
                rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        return _result_B_T_H_W_D

    result_B_T_H_W_D = _x_fn(x_B_T_H_W_D)
    x_B_T_H_W_D = x_B_T_H_W_D + gate_cross_attn_B_T_1_1_D * result_B_T_H_W_D 

    # MLP (Unchanged)
    normalized_x_B_T_H_W_D = _fn(
        x_B_T_H_W_D,
        self.layer_norm_mlp,
        scale_mlp_B_T_1_1_D,
        shift_mlp_B_T_1_1_D,
    )
    result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
    x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
    
    return x_B_T_H_W_D


def apply_fastercache(
    model, 
    start_step: int = 18, 
    model_interval: int = 5, 
    block_interval: int = 3
):
    """
    Apply FasterCache patches to the MiniTrainDIT model instance.
    args:
        start_step: Step to start applying caching optimizations.
        model_interval: Interval for unconditional branch skipping (FFT approx).
        block_interval: Interval for self-attention caching.
    """
    import types
    
    log.info(f"Applying FasterCache patches to MiniTrainDIT. Start={start_step}, ModInt={model_interval}, BlkInt={block_interval}")
    
    model.fastercache_counter = 0
    model.delta_lf = 0
    model.delta_hf = 0
    model.fastercache_enabled = True
    
    # Store Configs
    model.fastercache_start_step = start_step
    model.fastercache_model_interval = model_interval
    model.fastercache_block_interval = block_interval
    
    model.forward = types.MethodType(fastercache_mini_train_dit_forward, model)
    
    for block in model.blocks:
        block.fastercache_enabled = True
        block.current_counter = 0
        block.fastercache_start_step = start_step
        block.fastercache_block_interval = block_interval
        
        block.attn_cache = [] 
        block.forward = types.MethodType(fastercache_block_forward, block)
    
    return model
