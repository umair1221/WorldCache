# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Script for generating I2W videos in s3
PYTHONPATH=. python cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root /project/cosmos/ybalaji/data/internal_val_set_clean

# Script for text2world generation
export EXPERIMENT=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/inference/video2world.py \
--experiment=${EXPERIMENT} \
--ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/${EXPERIMENT}/checkpoints/iter_000025000 \
--save_root results/base_model/${EXPERIMENT}_025k_seed0_t2w \
--num_latent_conditional_frames=0 --seed=0 \
--input_root /project/cosmos/fangyinw/data/pbench/v0

# I2W with context parallel with 8 GPUs:
PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root /project/cosmos/ybalaji/data/internal_val_set_clean --context_parallel_size 8

# V2W with context parallel with 8 GPUs:
PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root pbench_upsampled_prompts --num_latent_conditional_frames=2 --context_parallel_size=8


Folder structure:
We assume the input root contains images and prompts in the following format:
input_root/
 ├── image_1.jpg
 ├── image_1.txt
 ├── image_2.jpg
 └── image_2.txt
 └── ...

or videos and prompts in the following format:
input_root/
 ├── video_1.mp4
 ├── video_1.txt
 ├── video_2.mp4
 └── video_2.txt
 └── ...
"""

import math
import os
from typing import TYPE_CHECKING

import torch
import torchvision
from megatron.core import parallel_state
from PIL import Image

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2.inference.dicache_utils import apply_dicache
from cosmos_predict2._src.predict2.inference.worldcache_utils import apply_worldcache
from cosmos_predict2._src.predict2.inference.easycache_utils import apply_easycache
from cosmos_predict2._src.predict2.inference.teacache_utils import apply_teacache
from cosmos_predict2._src.predict2.inference.debug_utils import apply_visual_trace, reset_run_id

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
_VIDEO_EXTENSIONS = [".mp4"]

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def resize_input(video: torch.Tensor, resolution: list[int]):
    r"""
    Resizes and crops the input video tensor while preserving aspect ratio.

    The video is first resized so that the smaller dimension matches the target resolution,
    preserving the aspect ratio. Then, it's center-cropped to the target resolution.

    Args:
        video (torch.Tensor): Input video tensor of shape (T, C, H, W).
        resolution (list[int]): Target resolution [H, W].

    Returns:
        torch.Tensor: Resized and cropped video tensor of shape (T, C, target_H, target_W).
    """

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, resolution)
    return video_cropped


def read_and_process_image(img_path: str, resolution: list[int], num_video_frames: int, resize: bool = True):
    """
    Reads an image, converts it to a video tensor, and processes it for model input.

    The image is loaded, converted to a tensor, and replicated to match the
    `num_video_frames`. It's then optionally resized and permuted to the
    standard video format (B, C, T, H, W).

    Args:
        img_path (str): Path to the input image file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): The number of frames the output video tensor should have.
        resize (bool, optional): Whether to resize the image to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W).

    Raises:
        ValueError: If the image extension is not one of the supported types.
    """
    ext = os.path.splitext(img_path)[1]
    if ext not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {ext}")

    # Read the image
    img = Image.open(img_path)

    # Convert to tensor
    img = torchvision.transforms.functional.to_tensor(img)
    # Create a video tensor by repeating the first frame
    vid_input = img.unsqueeze(0)  # Add temporal dimension T=1

    # Repeat the first frame to match the desired number of video frames
    # Note: The actual content for frames > 0 will be generated by the model.
    vid_input = torch.cat([vid_input, torch.zeros_like(vid_input).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
    vid_input = (vid_input * 255.0).to(torch.uint8)  # Convert to uint8 range if needed (might depend on model)
    if resize:
        # Resize and crop to the target resolution
        vid_input = resize_input(vid_input, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return vid_input


def read_and_process_video(
    video_path: str,
    resolution: list[int],
    num_video_frames: int,
    num_latent_conditional_frames: int = 2,
    resize: bool = True,
):
    """
    Reads a video, processes it for model input.

    The video is loaded using easy_io, and uses the last 4x(num_latent_conditional_frames - 1) + 1 from the video.
    If the video is shorter than num_video_frames, it pads with the last frame repeated.
    The first num_latent_conditional_frames are marked as conditioning frames.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): Number of frames needed by the model (should equal model.tokenizer.get_pixel_num_frames(model.config.state_t)).
        num_latent_conditional_frames (int): Number of latent conditional frames from the input video (1 or 2).
        resize (bool, optional): Whether to resize the video to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W) where T equals num_video_frames.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.

    Note:
        Uses the last 4x(num_latent_conditional_frames - 1) + 1 frames from the video. If video is shorter, pads with last frame repeated.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    # Load video using easy_io
    try:
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
        log.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert numpy array to tensor and rearrange dimensions
    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1] range
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    available_frames = video_tensor.shape[1]

    # Calculate how many frames to extract from input video
    frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
    log.info(f"Will extract {frames_to_extract} frames from input video and pad to {num_video_frames}")

    # Validate num_latent_conditional_frames
    if num_latent_conditional_frames not in [1, 2]:
        raise ValueError(f"num_latent_conditional_frames must be 1 or 2, but got {num_latent_conditional_frames}")

    # Create output tensor with exact num_video_frames
    C, _, H, W = video_tensor.shape
    full_video = torch.zeros(C, num_video_frames, H, W)

    if available_frames < frames_to_extract:
        raise ValueError(
            f"Video has only {available_frames} frames but needs at least {frames_to_extract} frames for num_latent_conditional_frames={num_latent_conditional_frames}"
        )

    # Extract the last frames_to_extract from input video
    start_idx = available_frames - frames_to_extract
    extracted_frames = video_tensor[:, start_idx:, :, :]
    full_video[:, :frames_to_extract, :, :] = extracted_frames
    log.info(f"Extracted last {frames_to_extract} frames from video (frames {start_idx} to {available_frames - 1})")

    # Pad remaining frames with the last extracted frame
    if frames_to_extract < num_video_frames:
        last_frame = extracted_frames[:, -1:, :, :]  # (C, 1, H, W)
        padding_frames = num_video_frames - frames_to_extract
        last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)  # (C, padding_frames, H, W)
        full_video[:, frames_to_extract:, :, :] = last_frame_repeated
        log.info(f"Padded {padding_frames} frames with last extracted frame")

    # Convert to the format expected by the rest of the pipeline
    full_video = full_video.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
    full_video = (full_video * 255.0).to(torch.uint8)  # Convert to uint8 range

    if resize:
        # Resize and crop to the target resolution
        full_video = resize_input(full_video, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return full_video


class Video2WorldInference:
    """
    Handles the Video2World inference process, including model loading, data preparation,
    and video generation from an image/video and text prompt. Now supports context parallelism.
    """

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str,
        context_parallel_size: int = 1,
        config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py",
        offload_diffusion_model: bool = False,
        offload_text_encoder: bool = False,
        offload_tokenizer: bool = False,
        dicache_enabled: bool = False,
        dicache_num_steps: int = 35,
        dicache_rel_l1_thresh: float = 0.5,
        dicache_ret_ratio: float = 0.2,
        dicache_probe_depth: int = 8,
        fastercache_enabled: bool = False,
        fastercache_start_step: int = 1,
        fastercache_model_interval: int = 2,
        fastercache_block_interval: int = 3,
        # EasyCache Arguments
        easycache_enabled: bool = False,
        easycache_num_steps: int = 35,
        easycache_thresh: float = 0.05,
        easycache_ret_steps: int = 5,
        # TeaCache Arguments
        teacache_enabled: bool = False,
        teacache_num_steps: int = 35,
        teacache_thresh: float = 0.3,
        worldcache_enabled: bool = False,
        worldcache_num_steps: int = 35,
        worldcache_rel_l1_thresh: float = 0.5,
        worldcache_ret_ratio: float = 0.2,
        worldcache_probe_depth: int = 8,
        worldcache_motion_sensitivity: float = 5.0,
        worldcache_flow_enabled: bool = False,
        worldcache_flow_scale: float = 0.5,
        worldcache_hf_enabled: bool = False,
        worldcache_hf_thresh: float = 0.01,
        worldcache_saliency_enabled: bool = False,
        worldcache_saliency_weight: float = 5.0,
        worldcache_osi_enabled: bool = False, # Online System Identification
        worldcache_dynamic_decay: bool = False, # Dynamic Threshold Decay
        worldcache_aduc_enabled: bool = False, # Adaptive Unconditional Caching
        worldcache_aduc_start: float = 0.5,
        worldcache_parallel_cfg: bool = False, # Parallel CFG (batch size 2)
        timestep_skip_enabled: bool = False,
        timestep_skip_early_ratio: float = 0.2,
        timestep_skip_late_ratio: float = 0.2,
        # Progressive Denoising — Early Exit
        early_exit_enabled: bool = False,
        early_exit_min_ratio: float = 0.5,
        early_exit_threshold: float = 0.02,
        use_torch_compile: bool = False, # Compile DiT natively
        
    ):
        """
        Initializes the Video2WorldInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path. Sets up distributed processing if needed.

        Args:
            experiment_name (str): Name of the experiment configuration.
            ckpt_path (str): Path to the model checkpoint (local or S3).
            s3_credential_path (str): Path to S3 credentials file (if loading from S3).
            context_parallel_size (int): Number of GPUs for context parallelism.
        """
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        self.offload_diffusion_model = offload_diffusion_model
        self.offload_text_encoder = offload_text_encoder
        self.offload_tokenizer = offload_tokenizer
        
        # DiCache parameters
        self.dicache_enabled = dicache_enabled
        self.dicache_num_steps = dicache_num_steps
        self.dicache_rel_l1_thresh = dicache_rel_l1_thresh
        self.dicache_ret_ratio = dicache_ret_ratio
        self.dicache_probe_depth = dicache_probe_depth
        
        if self.dicache_enabled and fastercache_enabled:
             raise ValueError("dicache and fastercache cannot be enabled at the same time.")

        self.fastercache_enabled = fastercache_enabled
        self.fastercache_start_step = fastercache_start_step
        self.fastercache_model_interval = fastercache_model_interval
        self.fastercache_block_interval = fastercache_block_interval
        
        # EasyCache parameters
        self.easycache_enabled = easycache_enabled
        self.easycache_num_steps = easycache_num_steps
        self.easycache_thresh = easycache_thresh
        self.easycache_ret_steps = easycache_ret_steps
        
        # TeaCache parameters
        self.teacache_enabled = teacache_enabled
        self.teacache_num_steps = teacache_num_steps
        self.teacache_thresh = teacache_thresh
        
        self.use_torch_compile = use_torch_compile
        
        # WorldCache parameters
        self.worldcache_enabled = worldcache_enabled
        self.worldcache_num_steps = worldcache_num_steps
        self.worldcache_rel_l1_thresh = worldcache_rel_l1_thresh
        self.worldcache_ret_ratio = worldcache_ret_ratio
        self.worldcache_probe_depth = worldcache_probe_depth
        self.worldcache_motion_sensitivity = worldcache_motion_sensitivity
        self.worldcache_flow_enabled = worldcache_flow_enabled
        self.worldcache_flow_scale = worldcache_flow_scale
        self.worldcache_hf_enabled = worldcache_hf_enabled
        self.worldcache_hf_thresh = worldcache_hf_thresh
        self.worldcache_saliency_enabled = worldcache_saliency_enabled
        self.worldcache_saliency_weight = worldcache_saliency_weight
        self.worldcache_osi_enabled = worldcache_osi_enabled
        self.worldcache_dynamic_decay = worldcache_dynamic_decay
        self.worldcache_aduc_enabled = worldcache_aduc_enabled
        self.worldcache_aduc_start = worldcache_aduc_start
        self.worldcache_parallel_cfg = worldcache_parallel_cfg

        # Progressive Denoising — Early Exit
        self.early_exit_enabled = early_exit_enabled
        self.early_exit_min_ratio = early_exit_min_ratio
        self.early_exit_threshold = early_exit_threshold
        if sum([self.dicache_enabled, self.fastercache_enabled, self.worldcache_enabled, self.easycache_enabled, self.teacache_enabled]) > 1:
             raise ValueError("Only one of dicache, fastercache, worldcache, easycache, or teacache can be enabled at a time.")

        # If no offloading is specified, instruct model loader to move the model to GPU
        model_device = None if offload_diffusion_model else "cuda"
 
        # Initialize distributed processing if context parallel size > 1
        if self.context_parallel_size > 1:
            self._init_distributed()

        # Load the model and config
        experiment_opts = []
        if not INTERNAL:
            experiment_opts.append("~data_train")

        # LazyConfig interference is not available yet
        # Use envvar to control whether DiT should be offloaded immediately after ctor
        if self.offload_diffusion_model:
            os.environ["COSMOS_PREDICT2_OFFLOAD_DIT"] = "1"

        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=config_file,
            load_ema_to_reg=True,
            experiment_opts=experiment_opts,
            to_device=model_device,
        )

        # By default, everything will be constructed directly on the GPU (except DiT)
        # Handle offloading options at inference entry

        # [On-entry offloading part 1]: DiT was offloaded as default by the lazy ctor
        # Offload or reload according to setup
        if self.offload_diffusion_model:
            log.info("[Memory Optimization] Offloading DiT conditioner to CPU")
            if hasattr(model, "conditioner") and model.conditioner is not None:
                model.conditioner = model.conditioner.to("cpu")
        else:
            # Move everything to the GPU (marginal overhead)
            model.net.to("cuda")
            
        # Optional native compile of DiT for faster inference (if not already compiled by config)
        if self.use_torch_compile:
            if not getattr(model.config, "use_torch_compile", False):
                log.info("[Optimization] Compiling DiT with torch.compile(mode='reduce-overhead')")
                # Using reduce-overhead for max performance. DiT shapes are static here.
                model.net = torch.compile(model.net, mode="reduce-overhead")
            else:
                log.info("[Optimization] torch.compile is already enabled via model config.")

        # [On-entry offloading part 2]: Tokenizer
        if self.offload_tokenizer:
            log.info("[Memory Optimization] Offloading tokenizer encoder & decoder to CPU")
            if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
                model.tokenizer.encoder = model.tokenizer.encoder.to("cpu")
            if hasattr(model.tokenizer, "decoder") and model.tokenizer.decoder is not None:
                model.tokenizer.decoder = model.tokenizer.decoder.to("cpu")
            torch.cuda.empty_cache()

        # [On-entry offloading part 3]: Text encoder
        if self.offload_text_encoder:
            # Text encoder is the first module in the pipeline.
            # Rather offload it **during** DiT run.
            pass

        if TYPE_CHECKING:
            from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
                Video2WorldModelRectifiedFlow,
            )

            model: Video2WorldModelRectifiedFlow = model

        # Enable context parallel on the model if using context parallelism

        # Enable context parallel on the model if using context parallelism
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model
        
        # Apply DiCache if enabled
        if self.dicache_enabled:
            patch_target = self.model.net if hasattr(self.model, 'net') else self.model
            print(f"[DiCache] Applying DiCache")
            # log.info("[Memory Optimization] Offloading DiT conditioner to CPU")
            patch_target = apply_dicache(
                patch_target,
                num_steps=self.dicache_num_steps,
                rel_l1_thresh=self.dicache_rel_l1_thresh,
                ret_ratio=self.dicache_ret_ratio,
                probe_depth=self.dicache_probe_depth
            )
            
            if hasattr(self.model, 'net'):
                self.model.net = patch_target
            else:
                self.model = patch_target

        # Apply FasterCache if enabled
        if self.fastercache_enabled:
            from cosmos_predict2._src.predict2.inference.fastercache_utils import apply_fastercache
            log.info(f"[FasterCache] Enabling FasterCache. Start={self.fastercache_start_step}, intervals={self.fastercache_model_interval}/{self.fastercache_block_interval}")
            
            # Patch the underlying 'net' which is the MiniTrainDIT instance
            patch_target = self.model.net if hasattr(self.model, 'net') else self.model
            
            patch_target = apply_fastercache(
                patch_target, 
                start_step=self.fastercache_start_step,
                model_interval=self.fastercache_model_interval,
                block_interval=self.fastercache_block_interval
            )
            
            if hasattr(self.model, 'net'):
                self.model.net = patch_target
            else:
                self.model = patch_target

        # Apply EasyCache if enabled
        if self.easycache_enabled:
            log.info(f"[EasyCache] Enabling EasyCache. thresh={self.easycache_thresh}, ret_steps={self.easycache_ret_steps}")
            patch_target = self.model.net if hasattr(self.model, 'net') else self.model
            patch_target = apply_easycache(
                patch_target,
                num_steps=self.easycache_num_steps,
                thresh=self.easycache_thresh,
                ret_steps=self.easycache_ret_steps,
            )
            if hasattr(self.model, 'net'):
                self.model.net = patch_target
            else:
                self.model = patch_target

        # Apply TeaCache if enabled
        if self.teacache_enabled:
            log.info(f"[TeaCache] Enabling TeaCache. thresh={self.teacache_thresh}")
            patch_target = self.model.net if hasattr(self.model, 'net') else self.model
            patch_target = apply_teacache(
                patch_target,
                num_steps=self.teacache_num_steps,
                rel_l1_thresh=self.teacache_thresh,
            )
            if hasattr(self.model, 'net'):
                self.model.net = patch_target
            else:
                self.model = patch_target

        # Apply WorldCache if enabled
        if self.worldcache_enabled:
            patch_target = self.model.net if hasattr(self.model, 'net') else self.model
            print(f"[WorldCache] Applying WorldCache")
            patch_target = apply_worldcache(
                patch_target,
                num_steps=self.worldcache_num_steps,
                rel_l1_thresh=self.worldcache_rel_l1_thresh,
                ret_ratio=self.worldcache_ret_ratio,
                probe_depth=self.worldcache_probe_depth,
                motion_sensitivity=self.worldcache_motion_sensitivity,
                flow_enabled=self.worldcache_flow_enabled,
                flow_scale=self.worldcache_flow_scale,
                hf_enabled=self.worldcache_hf_enabled,
                hf_thresh=self.worldcache_hf_thresh,
                saliency_enabled=self.worldcache_saliency_enabled,
                saliency_weight=self.worldcache_saliency_weight,
                osi_enabled=self.worldcache_osi_enabled,
                dynamic_decay=self.worldcache_dynamic_decay,
                aduc_enabled=self.worldcache_aduc_enabled,
                aduc_start=self.worldcache_aduc_start,
                parallel_cfg=self.worldcache_parallel_cfg
            )




            
            if hasattr(self.model, 'net'):
                self.model.net = patch_target
            else:
                self.model = patch_target
                
        # Apply Timestep-Aware Block Skipping (TABS)
        if hasattr(self, 'timestep_skip_enabled') and self.timestep_skip_enabled:
            from cosmos_predict2._src.predict2.inference.timestep_skip_utils import apply_timestep_skip
            log.info(f"[TABS] Applying Timestep-Aware Block Skipping (Early: {self.timestep_skip_early_ratio}, Late: {self.timestep_skip_late_ratio})")
            patch_target = self.model.net if hasattr(self.model, 'net') else self.model
            
            patch_target = apply_timestep_skip(
                patch_target,
                early_ratio=self.timestep_skip_early_ratio,
                late_ratio=self.timestep_skip_late_ratio
            )
            
            if hasattr(self.model, 'net'):
                self.model.net = patch_target
            else:
                self.model = patch_target
                
        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None

    def _init_distributed(self):
        """Initialize distributed processing for context parallelism."""

        # Initialize distributed environment
        distributed.init()

        # Initialize model parallel states
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )

        # Get the process group for context parallel
        self.process_group = parallel_state.get_context_parallel_group()

        log.info(f"Initialized context parallel with size {self.context_parallel_size}")
        log.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prompt: str,
        num_conditional_frames: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        use_neg_prompt: bool = True,
        camera: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ):
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.
            num_conditional_frames (int): Number of conditional frames to use.
            negative_prompt (str, optional): Custom negative prompt.
            use_neg_prompt (bool, optional): Whether to include negative prompt embeddings. Defaults to True.
            camera: (torch.Tensor, optional) Target camera extrinsics and intrinsics for the K output videos, must be provided for camera conditioned model.
            action: (torch.Tensor, optional) Target robot action for the K output videos, must be provided for action conditioned model.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape

        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "camera": camera,
            "action": action.unsqueeze(0) if action is not None else None,
            "fps": torch.randint(16, 32, (self.batch_size,)).float(),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),  # Padding mask (assumed no padding here)
            "num_conditional_frames": num_conditional_frames,  # Specify number of conditional frames
        }

        if use_neg_prompt:
            assert negative_prompt is not None, "Negative prompt is required when use_neg_prompt is True"

        # Compute text embeddings
        if self.model.text_encoder is not None:
            data_batch["ai_caption"] = [prompt]
            data_batch["t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [prompt], "images": None},
                input_caption_key="ai_caption",
            )
            if use_neg_prompt:
                data_batch["neg_t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [negative_prompt], "images": None},
                    input_caption_key="ai_caption",
                )
        else:
            data_batch["t5_text_embeddings"] = get_text_embedding(prompt)
            if use_neg_prompt:
                data_batch["neg_t5_text_embeddings"] = get_text_embedding(negative_prompt)

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    def generate_vid2world(
        self,
        prompt: str,
        input_path: str | torch.Tensor | None,
        guidance: int = 7,
        num_video_frames: int = 77,
        num_latent_conditional_frames: int = 1,
        num_input_video: int = 1,
        num_output_video: int = 1,
        resolution: str = "192,320",
        seed: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        camera: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        num_steps: int = 35,
    ):
        """
        Generates a video based on an input image or video and text prompt.

        Processes the input, prepares the data batch, runs the diffusion
        model sampling, and decodes the result into a video tensor.

        Args:
            prompt: The text prompt describing the desired video content/style.
            input_path: Path to the input image or video file or a torch.Tensor.
            guidance: Classifier-free guidance scale. Defaults to 7.
            num_video_frames: Number of video frames to generate. Defaults to 77.
            num_latent_conditional_frames : Number of latent conditional frames. Defaults to 1.
            resolution: Target video resolution in "H,W" format. Defaults to "192,320".
            seed: Random seed for reproducibility. Defaults to 1.
            negative_prompt: Custom negative prompt. Defaults to the predefined default negative prompt.
            camera: Target camera extrinsics and intrinsics for the K output videos. Must be provided if model is camera conditioned.
            action: Target robot action for the K output videos. Must be provided if model is action conditioned.
            num_steps: Number of generation steps. Defaults to 35.
            offload_diffusion_model: If True, offload diffusion model to CPU to save GPU memory. Defaults to False.
            offload_text_encoder: If True, offload text encoder to CPU to save GPU memory. Defaults to False.
            offload_tokenizer: If True, offload tokenizer to CPU to save GPU memory. Defaults to False.

        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
        """
        assert camera is not None or action is not None or num_input_video == 1 and num_output_video == 1, (
            "expected num_output_video==1 and num_output_video==1 for no camera conditioning or action conditioning"
        )

        # Parse resolution string into tuple of integers
        if resolution == "none":
            h, w = self.model.get_video_height_width()
            video_resolution = (h, w)
        else:
            video_resolution = resolution.split(",")
            video_resolution = tuple([int(x) for x in video_resolution])
            assert len(video_resolution) == 2, "Resolution must be in 'H,W' format"

        # Get the correct number of frames needed by the model
        model_required_frames = self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t)

        # Determine if input is image or video and process accordingly
        if input_path is None or num_latent_conditional_frames == 0:
            vid_input = torch.zeros(1, 3, model_required_frames, video_resolution[0], video_resolution[1]).to(
                torch.uint8
            )
        elif isinstance(input_path, str):
            ext = os.path.splitext(input_path)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                log.info(f"Processing image input: {input_path}")
                vid_input = read_and_process_image(
                    img_path=input_path,
                    resolution=video_resolution,
                    num_video_frames=model_required_frames,
                    resize=True,
                )
            elif ext in _VIDEO_EXTENSIONS:
                log.info(f"Processing video input: {input_path}")
                vid_input = read_and_process_video(
                    video_path=input_path,
                    resolution=video_resolution,
                    num_video_frames=model_required_frames,
                    num_latent_conditional_frames=num_latent_conditional_frames,
                    resize=True,
                )
            else:
                raise ValueError(
                    f"Unsupported file extension: {ext}. Supported extensions: {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS}"
                )
        elif isinstance(input_path, torch.Tensor):
            vid_input = input_path
        else:
            raise ValueError(f"Unsupported input_path type: {type(input_path)}")

        # Prepare the data batch with text embeddings
        # Note: TextEncoder.compute_text_embeddings_online() will automatically move its model to GPU
        data_batch = self._get_data_batch_input(
            video=vid_input,
            prompt=prompt,
            camera=camera,
            action=action,
            num_conditional_frames=num_latent_conditional_frames,
            negative_prompt=negative_prompt,
            use_neg_prompt=True,
        )

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        log.info(f"GPU memory usage after getting data_batch: {mem_bytes / (1024**3):.2f} GB")

        # Memory Optimization Step 1: Offload Text Encoder
        # Offload text encoder after computing embeddings to free memory
        if self.offload_text_encoder and self.model.text_encoder is not None:
            log.info("[Memory Optimization] Offloading text encoder to CPU")
            # TextEncoder is a wrapper class with self.model (the actual neural network)
            if hasattr(self.model.text_encoder, "model") and self.model.text_encoder.model is not None:
                self.model.text_encoder.model = self.model.text_encoder.model.to("cpu")
            torch.cuda.empty_cache()

        # Memory Optimization Step 2: Tokenizer Encoder
        # Load tokenizer encoder to GPU for encoding input video
        if self.offload_tokenizer:
            log.info("[Memory Optimization] Loading tokenizer encoder to GPU")
            if hasattr(self.model.tokenizer, "encoder") and self.model.tokenizer.encoder is not None:
                self.model.tokenizer.encoder = self.model.tokenizer.encoder.to("cuda")
            torch.cuda.empty_cache()

        # Memory Optimization Step 3: Diffusion Network
        # Load the main diffusion network to GPU for sampling
        if self.offload_diffusion_model:
            log.info("[Memory Optimization] Loading diffusion network to GPU")
            self.model.net = self.model.net.to("cuda")
            # Also load conditioner if it exists
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                self.model.conditioner = self.model.conditioner.to("cuda")
            torch.cuda.empty_cache()

        extra_kwargs = {}
        if camera is not None:
            extra_kwargs = {
                "num_input_video": num_input_video,
                "num_output_video": num_output_video,
            }

        # Generate latent samples using the diffusion model
        # Video should be of shape torch.Size([1, 3, 93, 192, 320]) # Note: Shape check comment
        log.info("[Memory Optimization] Starting latent sample generation")
        if self.model.config.use_lora:
            generate_samples = self.model.generate_samples_from_batch_lora
        else:
            generate_samples = self.model.generate_samples_from_batch
            
        # reset_run_id() -> Removed (no visualization)
        if self.dicache_enabled:
             # Update num_steps for DiCache (accounting for CFG implicitly if we count every forward pass)
             # If CFG, we have 2 calls per step.
             target_steps = num_steps * 2
             if hasattr(self.model.net, "dicache_num_steps"):
                 self.model.net.dicache_num_steps = target_steps
                 # Explicitly reset state to safe defaults before every generation
                 self.model.net.cnt = 0
                 self.model.net.accumulated_rel_l1_distance = [0.0, 0.0]
                 self.model.net.residual_cache = [None, None]
                 self.model.net.residual_window = [[], []]
                 self.model.net.probe_residual_window = [[], []]
                 self.model.net.previous_internal_states = [None, None]
                 self.model.net.previous_input = [None, None]
                 self.model.net.resume_flag = [False, False]
             elif hasattr(self.model, "dicache_num_steps"): # If net is model
                 self.model.dicache_num_steps = target_steps
                 # Explicitly reset state to safe defaults before every generation
                 self.model.cnt = 0
                 self.model.accumulated_rel_l1_distance = [0.0, 0.0]
                 self.model.residual_cache = [None, None]
                 self.model.residual_window = [[], []]
                 self.model.probe_residual_window = [[], []]
                 self.model.previous_internal_states = [None, None]
                 self.model.previous_input = [None, None]
                 self.model.resume_flag = [False, False]
        elif self.worldcache_enabled:
            # WorldCache Reset Logic
            # If parallel_cfg: B=2 batched, cnt increments once per step
            # If sequential: B=1 ping-pong, cnt increments twice per step (cond + uncond)
            if self.worldcache_parallel_cfg:
                target_steps = num_steps
            else:
                target_steps = num_steps * 2
            if hasattr(self.model.net, "worldcache_num_steps"):
                self.model.net.worldcache_num_steps = target_steps
                self.model.net.cnt = 0
                self.model.net.accumulated_rel_l1_distance = [0.0, 0.0]
                self.model.net.residual_cache = [None, None]
                self.model.net.residual_window = [[], []]
                self.model.net.probe_residual_window = [[], []]
                self.model.net.previous_internal_states = [None, None]
                self.model.net.previous_input = [None, None]
                self.model.net.resume_flag = [False, False]
            elif hasattr(self.model, "worldcache_num_steps"):
                self.model.worldcache_num_steps = target_steps
                self.model.cnt = 0
                self.model.accumulated_rel_l1_distance = [0.0, 0.0]
                self.model.residual_cache = [None, None]
                self.model.residual_window = [[], []]
                self.model.probe_residual_window = [[], []]
                self.model.previous_internal_states = [None, None]
                self.model.previous_input = [None, None]
                self.model.resume_flag = [False, False]
        elif self.easycache_enabled:
            # EasyCache reset
            target_steps = num_steps * 2
            patch_target = self.model.net if hasattr(self.model, 'net') else self.model
            if hasattr(patch_target, 'easycache_num_steps'):
                patch_target.easycache_num_steps = target_steps
                patch_target.easycache_cnt = 0
                patch_target.easycache_accumulated_error_even = 0.0
                patch_target.easycache_k = None
                patch_target.easycache_should_compute = True
                patch_target.easycache_prev_input_even = None
                patch_target.easycache_prev_output_even = None
                patch_target.easycache_prev_output_odd = None
                patch_target.easycache_prev_prev_input_even = None
                patch_target.easycache_cache_even = None
                patch_target.easycache_cache_odd = None
                patch_target.easycache_cutoff_steps = target_steps - 2
                patch_target.easycache_ret_steps = self.easycache_ret_steps * 2
        elif self.teacache_enabled:
            # TeaCache reset
            patch_target = self.model.net if hasattr(self.model, 'net') else self.model
            if hasattr(patch_target, 'teacache_enabled'):
                patch_target.teacache_num_steps = num_steps * 2
                patch_target.teacache_cnt = 0
                patch_target.teacache_accum_even = 0.0
                patch_target.teacache_accum_odd = 0.0
                patch_target.teacache_prev_mod_even = None
                patch_target.teacache_prev_mod_odd = None
                patch_target.teacache_residual_even = None
                patch_target.teacache_residual_odd = None
                patch_target.teacache_step_skipped_count = 0
        else: # DiCache Disabled -> Apply Visual Trace for Baseline Comparison
            log.info("Applying Visual Trace for Baseline (No DiCache)")
            if hasattr(self.model, "net"):
                apply_visual_trace(self.model.net, save_dir="outputs/debug_vis_baseline")
            else:
                apply_visual_trace(self.model, save_dir="outputs/debug_vis_baseline")
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**3)
        
        # Set parallel_cfg on model for velocity_fn auto-detection
        self.model._worldcache_parallel_cfg = self.worldcache_parallel_cfg
        
        # Set early exit config on model for the sampling loop
        self.model._early_exit_enabled = self.early_exit_enabled
        self.model._early_exit_min_ratio = self.early_exit_min_ratio
        self.model._early_exit_threshold = self.early_exit_threshold
        
        # Timestep-Aware Block Skipping (TABS) reset
        if hasattr(self.model, 'net') and getattr(self.model.net, 'tabs_enabled', False):
            self.model.net.tabs_cnt = 0
            self.model.net.tabs_num_steps = num_steps if self.worldcache_parallel_cfg else num_steps * 2
        elif hasattr(self.model, 'tabs_enabled') and getattr(self.model, 'tabs_enabled', False):
            self.model.tabs_cnt = 0
            self.model.tabs_num_steps = num_steps if self.worldcache_parallel_cfg else num_steps * 2
        
        import time
        method_name = "Baseline"
        if self.worldcache_enabled:
            method_name = "WorldCache"
        elif self.dicache_enabled:
            method_name = "DiCache"
        elif self.fastercache_enabled:
            method_name = "FasterCache"
        elif self.easycache_enabled:
            method_name = "EasyCache"
        elif self.teacache_enabled:
            method_name = "TeaCache"
            
        if getattr(self, 'timestep_skip_enabled', False):
            method_name += " + TABS" if method_name != "Baseline" else "TABS"
        
        if self.early_exit_enabled:
            method_name += " + EarlyExit" if method_name != "Baseline" else "EarlyExit"
            
        print(f"\nRunning inference with {method_name} ({num_steps} steps)...")
        
        sample = generate_samples(
            data_batch,
            n_sample=1,  # Generate one sample
            guidance=guidance,
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=True,  # Use classifier-free guidance
            num_steps=num_steps,
            **extra_kwargs,
        )

        # Memory Optimization Step 4: Offload Diffusion Network
        # Offload diffusion network after sampling to make room for decoder
        if self.offload_diffusion_model:
            log.info("[Memory Optimization] Offloading diffusion network to CPU")
            self.model.net = self.model.net.to("cpu")
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                self.model.conditioner = self.model.conditioner.to("cpu")

        if self.offload_tokenizer:
            # Also offload encoder since we only need decoder now
            if hasattr(self.model.tokenizer, "encoder") and self.model.tokenizer.encoder is not None:
                self.model.tokenizer.encoder = self.model.tokenizer.encoder.to("cpu")
            torch.cuda.empty_cache()

        # Memory Optimization Step 5: Load Decoder
        # Load tokenizer decoder to GPU for decoding latents
        if self.offload_tokenizer:
            log.info("[Memory Optimization] Loading tokenizer decoder to GPU")
            if hasattr(self.model.tokenizer, "decoder") and self.model.tokenizer.decoder is not None:
                self.model.tokenizer.decoder = self.model.tokenizer.decoder.to("cuda")
            torch.cuda.empty_cache()

        # Decode the latent samples
        if isinstance(sample, list):
            # Decode the latent sample into a video tensor
            video_list = []
            for sample_chunk in sample:
                video_chunk = self.model.decode(sample_chunk)
                video_list.append(video_chunk)
            video = torch.cat(video_list, dim=3)
        else:
            # Decode the latent sample into a video tensor
            video = self.model.decode(sample)

        # Memory Optimization Step 6: Final Cleanup
        # Offload decoder after decoding & reload the tokenizer for the next inference call
        if self.offload_tokenizer:
            log.info("[Memory Optimization] Offloading tokenizer decoder to CPU")
            if hasattr(self.model.tokenizer, "decoder") and self.model.tokenizer.decoder is not None:
                self.model.tokenizer.decoder = self.model.tokenizer.decoder.to("cpu")
            torch.cuda.empty_cache()

        if self.offload_text_encoder and self.model.text_encoder is not None:
            log.info("[Memory Optimization] Load text encoder to GPU")
            # TextEncoder is a wrapper class with self.model (the actual neural network)
            if hasattr(self.model.text_encoder, "model") and self.model.text_encoder.model is not None:
                self.model.text_encoder.model = self.model.text_encoder.model.to("cuda")
            torch.cuda.empty_cache()

        return video

    def generate_autoregressive_from_batch(
        self,
        prompt: str,
        input_path: str | torch.Tensor | None,
        num_output_frames: int,
        chunk_size: int,
        chunk_overlap: int,
        guidance: int = 7,
        num_latent_conditional_frames: int = 1,
        resolution: str = "192,320",
        seed: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        camera: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        num_steps: int = 35,
    ) -> torch.Tensor:
        """
        Generate video using autoregressive sliding window approach.

        Args:
            prompt: The text prompt describing the desired video content/style.
            input_path: Path to the input image or video file or a torch.Tensor.
            num_output_frames: Total number of frames to generate in the final output.
            chunk_size: Number of frames per chunk (model's native capacity).
            chunk_overlap: Number of overlapping frames between chunks.
            guidance: Classifier-free guidance scale.
            num_latent_conditional_frames: Number of latent conditional frames.
            resolution: Target video resolution in "H,W" format.
            seed: Random seed for reproducibility.
            negative_prompt: Custom negative prompt.
            camera: Target camera extrinsics and intrinsics for the K output videos.
            action: Target robot action for the K output videos.
            num_steps: Number of generation steps.

        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
        """
        # Parse resolution string into tuple of integers
        if resolution == "none":
            h, w = self.model.get_video_height_width()
            video_resolution = (h, w)
        else:
            video_resolution = resolution.split(",")
            video_resolution = tuple([int(x) for x in video_resolution])
            assert len(video_resolution) == 2, "Resolution must be in 'H,W' format"

        # Get the correct number of frames needed by the model
        model_required_frames = self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t)

        # Load and process the full input video/image
        if input_path is None or num_latent_conditional_frames == 0:
            # For text2world, create a full length zero video
            full_input_video = torch.zeros(1, 3, num_output_frames, video_resolution[0], video_resolution[1]).to(
                torch.uint8
            )
        elif isinstance(input_path, str):
            ext = os.path.splitext(input_path)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                log.info(f"Processing image input for autoregressive: {input_path}")
                # For image input, create full video with first frame as image, rest zeros
                img = Image.open(input_path)
                img = torchvision.transforms.functional.to_tensor(img)
                img = img.unsqueeze(0)  # Add temporal dimension T=1
                img = (img * 255.0).to(torch.uint8)
                if video_resolution:
                    img = resize_input(img, video_resolution)
                # Create full length video with first frame as image
                full_input_video = torch.cat([img, torch.zeros_like(img).repeat(num_output_frames - 1, 1, 1, 1)], dim=0)
                full_input_video = full_input_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
            elif ext in _VIDEO_EXTENSIONS:
                log.info(f"Processing video input for autoregressive: {input_path}")
                # Load video and extend to full length if needed
                video_frames, _ = easy_io.load(input_path)
                video_tensor = torch.from_numpy(video_frames).float() / 255.0
                video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
                available_frames = video_tensor.shape[1]

                # Calculate frames to extract
                frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
                if available_frames < frames_to_extract:
                    raise ValueError(f"Video has only {available_frames} frames but needs at least {frames_to_extract}")

                # Extract last frames_to_extract
                start_idx = available_frames - frames_to_extract
                extracted_frames = video_tensor[:, start_idx:, :, :]

                # Create full length tensor
                C, _, H, W = video_tensor.shape
                full_video = torch.zeros(C, num_output_frames, H, W)
                full_video[:, :frames_to_extract, :, :] = extracted_frames

                # Pad with last frame
                if frames_to_extract < num_output_frames:
                    last_frame = extracted_frames[:, -1:, :, :]
                    padding_frames = num_output_frames - frames_to_extract
                    last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)
                    full_video[:, frames_to_extract:, :, :] = last_frame_repeated

                full_video = full_video.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
                full_video = (full_video * 255.0).to(torch.uint8)
                if video_resolution:
                    full_video = resize_input(full_video, video_resolution)
                full_input_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        elif isinstance(input_path, torch.Tensor):
            # If tensor, extend to full length
            full_input_video = input_path
            if full_input_video.shape[2] < num_output_frames:
                # Pad with zeros
                padding_frames = num_output_frames - full_input_video.shape[2]
                padding = torch.zeros(
                    full_input_video.shape[0],
                    full_input_video.shape[1],
                    padding_frames,
                    full_input_video.shape[3],
                    full_input_video.shape[4],
                ).to(full_input_video.dtype)
                full_input_video = torch.cat([full_input_video, padding], dim=2)
        else:
            raise ValueError(f"Unsupported input_path type: {type(input_path)}")

        # Initialize output
        generated_chunks = []

        # Calculate number of chunks
        # Note: All chunks generate chunk_size frames, we store all of chunk 0 and (chunk_size - chunk_overlap) from others
        # Total stored = chunk_size + (num_chunks - 1) * (chunk_size - chunk_overlap) >= num_output_frames
        effective_chunk_size = chunk_size - chunk_overlap

        # Solve for num_chunks: chunk_size + (num_chunks - 1) * effective_chunk_size >= num_output_frames
        remaining_after_first = num_output_frames - chunk_size
        if remaining_after_first <= 0:
            num_chunks = 1
        else:
            # Ceiling division to ensure we have enough frames for the last chunk.
            num_chunks = 1 + (remaining_after_first + effective_chunk_size - 1) // effective_chunk_size

        log.info(
            f"Generating {num_chunks} chunks with chunk_size={chunk_size}, chunk_overlap={chunk_overlap} "
            f"for {num_output_frames} total frames"
        )

        # Generate chunks
        current_input_video = full_input_video.clone()

        for chunk_idx in range(num_chunks):
            # Calculate frame range for this chunk
            # All chunks are positioned with stride (chunk_size - chunk_overlap)
            start_frame = chunk_idx * effective_chunk_size
            end_frame = min(start_frame + chunk_size, num_output_frames)
            actual_chunk_size = end_frame - start_frame

            if start_frame >= num_output_frames:
                break

            log.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}, frames {start_frame}-{end_frame}")

            # Extract chunk from current input
            chunk_input = current_input_video[:, :, start_frame:end_frame, :, :]

            # Pad to model_required_frames if needed
            if actual_chunk_size < model_required_frames:
                padding_frames = model_required_frames - actual_chunk_size
                padding = torch.zeros(
                    chunk_input.shape[0],
                    chunk_input.shape[1],
                    padding_frames,
                    chunk_input.shape[3],
                    chunk_input.shape[4],
                ).to(chunk_input.dtype)
                chunk_input = torch.cat([chunk_input, padding], dim=2)

            # Determine num_conditional_frames for this chunk
            if chunk_idx == 0:
                chunk_num_conditional = num_latent_conditional_frames
            else:
                chunk_num_conditional = chunk_overlap

            # Generate chunk
            chunk_video = self.generate_vid2world(
                prompt=prompt,
                input_path=chunk_input,
                guidance=guidance,
                num_video_frames=model_required_frames,
                num_latent_conditional_frames=chunk_num_conditional,
                resolution=resolution,
                seed=seed + chunk_idx,
                negative_prompt=negative_prompt,
                camera=camera,
                action=action,
                num_steps=num_steps,
            )  # Returns (1, C, T, H, W)

            # Extract only the actual generated frames (remove padding)
            chunk_video = chunk_video[:, :, :actual_chunk_size, :, :]

            # Store generated chunk
            if chunk_idx == 0:
                generated_chunks.append(chunk_video)
            else:
                # Remove overlap frames from the beginning
                generated_chunks.append(chunk_video[:, :, chunk_overlap:, :, :])

            # Update input for next iteration using generated frames
            if chunk_idx < num_chunks - 1:
                # Convert generated chunk from [-1, 1] to [0, 255] uint8 range
                chunk_video_uint8 = ((chunk_video / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
                # Update the input video with generated frames for conditioning next chunk
                update_start = start_frame + chunk_num_conditional
                update_end = end_frame
                current_input_video[:, :, update_start:update_end, :, :] = chunk_video_uint8[
                    :, :, chunk_num_conditional:, :, :
                ]

        # Concatenate all chunks along time dimension
        final_video = torch.cat(generated_chunks, dim=2)

        log.info(f"Generated final video with shape {final_video.shape}")
        return final_video

    def cleanup(self):
        """Clean up distributed resources."""
        if self.context_parallel_size > 1:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
