
import torch
import torchvision
import os
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import types

def visualize_latent(x_tensor, step_idx, save_dir="debug_vis", suffix="", subfolder_name=None):
    """
    Visualizes a 16-channel latent tensor by projecting to RGB using PCA.
    """
    return # Disable visualization globally
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle if tensor is on GPU
    if x_tensor.device.type != 'cpu':
        x_tensor = x_tensor.cpu()
        
    # 1. Select Middle Frame (Temporal Slice)
    T = x_tensor.shape[2]
    mid_t = T // 2
    
    # Shape: (1, 16, H, W)
    feat = x_tensor[0, :, mid_t, :, :].detach().float()
    C, H, W = feat.shape
    
    # 2. PCA Projection (16ch -> 3ch)
    flat_feat = feat.permute(1, 2, 0).reshape(-1, C).numpy()
    flat_feat = (flat_feat - flat_feat.mean(0)) / (flat_feat.std(0) + 1e-5)
    
    try:
        pca = PCA(n_components=3)
        rgb_flat = pca.fit_transform(flat_feat)
    except Exception as e:
        print(f"[DebugVis] PCA Failed: {e}")
        return None
    
    # Normalize to 0-255
    rgb_flat = (rgb_flat - rgb_flat.min()) / (rgb_flat.max() - rgb_flat.min())
    rgb_flat = (rgb_flat * 255).astype(np.uint8)
    
    # Reshape
    rgb_img = rgb_flat.reshape(H, W, 3)
    
    # 3. Save
    img = Image.fromarray(rgb_img)
    img = img.resize((W*4, H*4), Image.NEAREST)
    
    # Create a unique subdirectory for this specific run instance (video generation)
    # Using timestamp to prevent overwriting across multiple videos
    import time
    if not hasattr(visualize_latent, "run_id"):
        visualize_latent.run_id = int(time.time())
    
    unique_dir = os.path.join(save_dir, f"video_{visualize_latent.run_id}")
    if subfolder_name:
        unique_dir = os.path.join(unique_dir, subfolder_name)
    os.makedirs(unique_dir, exist_ok=True)
    
    filename = f"step_{step_idx:03d}{suffix}.png"
    save_path = os.path.join(unique_dir, filename)
    img.save(save_path)
    # print(f"[DebugVis] Saved latent visualization to {save_path}")
    return save_path

def apply_visual_trace(model, save_dir="outputs/debug_vis_baseline"):
    """
    Wraps a model's forward method to visualize the output at every step.
    Used for the 'No DiCache' baseline comparison.
    """
    # Initialize state
    model.trace_cnt = 0
    model.trace_save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Save original forward
    original_forward = model.forward
    
    def traced_forward(self, *args, **kwargs):
        # 1. Run Original
        output = original_forward(*args, **kwargs)
        
        # 2. Visualize
        try:
            # Logic: Even = Cond, Odd = Uncond
            is_cond = (self.trace_cnt % 2 == 0)
            subfolder = "cond" if is_cond else "uncond"
            step_idx = self.trace_cnt // 2
            
            if isinstance(output, torch.Tensor):
                visualize_latent(output, step_idx, save_dir=self.trace_save_dir, subfolder_name=subfolder)
            elif isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor):
                visualize_latent(output[0], step_idx, save_dir=self.trace_save_dir, subfolder_name=subfolder)
        except Exception as e:
            print(f"[VisualTrace] Failed to visualize step {self.trace_cnt}: {e}")
            
        # 3. Increment
        self.trace_cnt += 1
        return output

    model.forward = types.MethodType(traced_forward, model)
    print(f"[VisualTrace] Tracing enabled. Saving to {save_dir}")
    return model

def reset_run_id():
    """Resets the run_id timestamp for visualizing a new video."""
    import time
    visualize_latent.run_id = int(time.time())
    print(f"[DebugVis] Reset Run ID to {visualize_latent.run_id}")
