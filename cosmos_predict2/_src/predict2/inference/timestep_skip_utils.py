import torch
import types

def block_forward_tabs(
    self,
    x_B_T_H_W_D: torch.Tensor,
    emb_B_T_D: torch.Tensor,
    crossattn_emb: torch.Tensor,
    rope_emb_L_1_1_D: torch.Tensor = None,
    adaln_lora_B_T_3D: torch.Tensor = None,
    extra_per_block_pos_emb: torch.Tensor = None,
) -> torch.Tensor:
    """
    Patched forward method for TransformerBlock to support Timestep-Aware Block Skipping (TABS).
    """
    # 1. Access the parent model's step counter and configuration
    parent = self.parent_model
    counter = getattr(parent, 'tabs_cnt', 0)
    num_steps = getattr(parent, 'tabs_num_steps', 40)
    early_ratio = getattr(parent, 'tabs_early_ratio', 0.2)
    late_ratio = getattr(parent, 'tabs_late_ratio', 0.2)
    block_idx = self.block_idx
    num_blocks = self.num_blocks

    # 2. Determine if this block should be skipped based on progress
    progress = counter / max(1.0, float(num_steps))
    skip = False
    
    if progress < 0.3:
        # High noise (early stages): Model is focusing on global structure.
        # Skip the last X% of blocks (fine detail blocks).
        blocks_to_skip = int(num_blocks * early_ratio)
        if blocks_to_skip > 0 and block_idx >= (num_blocks - blocks_to_skip):
            skip = True
    elif progress > 0.7:
        # Low noise (late stages): Model is focusing on fine details.
        # Skip the first Y% of blocks (global structure blocks).
        blocks_to_skip = int(num_blocks * late_ratio)
        if blocks_to_skip > 0 and block_idx < blocks_to_skip:
            skip = True
            
    # Allow other caching mechanisms (like FasterCache) to override skip if needed,
    # but TABS is meant to be a hard architectural skip to save compute.

    # 3. Execute or Skip
    if skip:
        # If skipped, the block acts as an identity function + positional embeddings
        if extra_per_block_pos_emb is not None:
            return x_B_T_H_W_D + extra_per_block_pos_emb
        return x_B_T_H_W_D
    else:
        # Execute normal forward pass (calling the original forward method we saved)
        return self.original_forward(
            x_B_T_H_W_D=x_B_T_H_W_D,
            emb_B_T_D=emb_B_T_D,
            crossattn_emb=crossattn_emb,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            extra_per_block_pos_emb=extra_per_block_pos_emb,
        )

def model_forward_tabs_wrapper(self, *args, **kwargs):
    """
    Wrapper around the model's forward to track the timestep count.
    """
    self.tabs_cnt += 1
    return self.original_forward(*args, **kwargs)

def apply_timestep_skip(model, early_ratio: float = 0.2, late_ratio: float = 0.2):
    """
    Apply Timestep-Aware Block Skipping (TABS) patches to the model.
    """
    import types
    from cosmos_predict2._src.imaginaire.utils import log
    
    log.info(f"Applying TABS patches to {model.__class__.__name__}. Early Ratio={early_ratio}, Late Ratio={late_ratio}")
    
    # Store settings on the model
    model.tabs_enabled = True
    model.tabs_cnt = 0
    model.tabs_early_ratio = early_ratio
    model.tabs_late_ratio = late_ratio
    
    # We need num_steps for the progress calculation. This will be updated per generation by Video2WorldInference.
    # Defaulting to 40, but will be overwritten dynamically if possible.
    model.tabs_num_steps = 40 
    
    if not hasattr(model, 'original_forward'):
        model.original_forward = getattr(model, 'forward')
        model.forward = types.MethodType(model_forward_tabs_wrapper, model)
    
    num_blocks = len(model.blocks)
    for idx, block in enumerate(model.blocks):
        if not hasattr(block, 'original_forward'):
            block.original_forward = block.forward
            block.block_idx = idx
            block.num_blocks = num_blocks
            block.parent_model = model
            block.forward = types.MethodType(block_forward_tabs, block)
            
    return model
