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
DreamDojo Video2World inference with WorldCache support.

Adds:
  - lam_video passthrough for DreamDojo's Latent Action Model (LAM)
  - WorldCache / DiCache / FasterCache acceleration
  - Parallel-CFG batching (B=2 cond+uncond in a single forward)
"""

import math
import os
import time
from typing import TYPE_CHECKING

import torch
import torchvision
from einops import rearrange
from megatron.core import parallel_state
from PIL import Image

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
_VIDEO_EXTENSIONS = [".mp4"]

_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated "
    "images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed "
    "out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, "
    "unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly "
    "edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
)


# ---------------------------------------------------------------------------
# Utility helpers (unchanged from upstream)
# ---------------------------------------------------------------------------

def resize_input(video: torch.Tensor, resolution: list[int]):
    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, resolution)
    return video_cropped


def read_and_process_image(img_path, resolution, num_video_frames, resize=True):
    ext = os.path.splitext(img_path)[1]
    if ext not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {ext}")
    img = Image.open(img_path)
    img = torchvision.transforms.functional.to_tensor(img)
    vid_input = img.unsqueeze(0)
    vid_input = torch.cat([vid_input, torch.zeros_like(vid_input).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
    vid_input = (vid_input * 255.0).to(torch.uint8)
    if resize:
        vid_input = resize_input(vid_input, resolution)
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return vid_input


def read_and_process_video(video_path, resolution, num_video_frames, num_latent_conditional_frames=2, resize=True):
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")
    try:
        video_frames, video_metadata = easy_io.load(video_path)
        log.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    video_tensor = torch.from_numpy(video_frames).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2)
    available_frames = video_tensor.shape[1]
    frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
    if num_latent_conditional_frames not in [1, 2]:
        raise ValueError(f"num_latent_conditional_frames must be 1 or 2, got {num_latent_conditional_frames}")
    C, _, H, W = video_tensor.shape
    full_video = torch.zeros(C, num_video_frames, H, W)
    if available_frames < frames_to_extract:
        raise ValueError(
            f"Video has only {available_frames} frames but needs at least {frames_to_extract}"
        )
    start_idx = available_frames - frames_to_extract
    extracted_frames = video_tensor[:, start_idx:, :, :]
    full_video[:, :frames_to_extract, :, :] = extracted_frames
    if frames_to_extract < num_video_frames:
        last_frame = extracted_frames[:, -1:, :, :]
        padding_frames = num_video_frames - frames_to_extract
        full_video[:, frames_to_extract:, :, :] = last_frame.repeat(1, padding_frames, 1, 1)
    full_video = full_video.permute(1, 0, 2, 3)
    full_video = (full_video * 255.0).to(torch.uint8)
    if resize:
        full_video = resize_input(full_video, resolution)
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return full_video


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class Video2WorldInference:
    """
    DreamDojo Video2World inference handler.

    Key additions over upstream Cosmos-Predict2.5:
      * ``lam_video`` passthrough so the Latent Action Model (LAM) can
        modulate actions during inference.
      * WorldCache / DiCache / FasterCache acceleration hooks.
      * Parallel-CFG flag that lets WorldCache batch cond+uncond in B=2.
    """

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str,
        context_parallel_size: int = 1,
        config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py",
        experiment_opts: list[str] | None = None,
        offload_diffusion_model: bool = False,
        offload_text_encoder: bool = False,
        offload_tokenizer: bool = False,
        # ----- WorldCache -----
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
        worldcache_osi_enabled: bool = False,
        worldcache_dynamic_decay: bool = False,
        worldcache_aduc_enabled: bool = False,
        worldcache_aduc_start: float = 0.5,
        worldcache_parallel_cfg: bool = False,
        # ----- DiCache -----
        dicache_enabled: bool = False,
        dicache_num_steps: int = 35,
        dicache_rel_l1_thresh: float = 0.5,
        dicache_ret_ratio: float = 0.2,
        dicache_probe_depth: int = 8,
        # ----- FasterCache -----
        fastercache_enabled: bool = False,
        fastercache_start_step: int = 1,
        fastercache_model_interval: int = 2,
        fastercache_block_interval: int = 3,
        # ----- Misc -----
        use_torch_compile: bool = False,
    ):
        # ---- store params ----
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        self.offload_diffusion_model = offload_diffusion_model
        self.offload_text_encoder = offload_text_encoder
        self.offload_tokenizer = offload_tokenizer

        # WorldCache
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

        # DiCache
        self.dicache_enabled = dicache_enabled
        self.dicache_num_steps = dicache_num_steps
        self.dicache_rel_l1_thresh = dicache_rel_l1_thresh
        self.dicache_ret_ratio = dicache_ret_ratio
        self.dicache_probe_depth = dicache_probe_depth

        # FasterCache
        self.fastercache_enabled = fastercache_enabled
        self.fastercache_start_step = fastercache_start_step
        self.fastercache_model_interval = fastercache_model_interval
        self.fastercache_block_interval = fastercache_block_interval

        self.use_torch_compile = use_torch_compile

        # Validate mutual exclusivity
        enabled_count = sum([self.dicache_enabled, self.fastercache_enabled, self.worldcache_enabled])
        if enabled_count > 1:
            raise ValueError("Only one of dicache, fastercache, or worldcache can be enabled at a time.")

        model_device = None if offload_diffusion_model else "cuda"

        if self.context_parallel_size > 1:
            self._init_distributed()

        # ---- Load model ----
        if experiment_opts is None:
            experiment_opts = []
        if not INTERNAL:
            experiment_opts.append("~data_train")

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

        # ---- Offloading / placement ----
        if self.offload_diffusion_model:
            log.info("[Memory] Offloading DiT conditioner to CPU")
            if hasattr(model, "conditioner") and model.conditioner is not None:
                model.conditioner = model.conditioner.to("cpu")
        else:
            model.net.to("cuda")

        if self.use_torch_compile and not getattr(model.config, "use_torch_compile", False):
            log.info("[Opt] Compiling DiT with torch.compile(mode='reduce-overhead')")
            model.net = torch.compile(model.net, mode="reduce-overhead")

        if self.offload_tokenizer:
            log.info("[Memory] Offloading tokenizer to CPU")
            if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
                model.tokenizer.encoder = model.tokenizer.encoder.to("cpu")
            if hasattr(model.tokenizer, "decoder") and model.tokenizer.decoder is not None:
                model.tokenizer.decoder = model.tokenizer.decoder.to("cpu")
            torch.cuda.empty_cache()

        if TYPE_CHECKING:
            from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
                Text2WorldModelRectifiedFlow,
            )
            model: Text2WorldModelRectifiedFlow = model

        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model

        # ---- Apply caching patches to self.model.net ----
        patch_target = self.model.net if hasattr(self.model, "net") else self.model

        if self.dicache_enabled:
            from cosmos_predict2._src.predict2.inference.dicache_utils import apply_dicache
            log.info("[DiCache] Applying DiCache")
            patch_target = apply_dicache(
                patch_target,
                num_steps=self.dicache_num_steps,
                rel_l1_thresh=self.dicache_rel_l1_thresh,
                ret_ratio=self.dicache_ret_ratio,
                probe_depth=self.dicache_probe_depth,
            )
        elif self.fastercache_enabled:
            from cosmos_predict2._src.predict2.inference.fastercache_utils import apply_fastercache
            log.info("[FasterCache] Applying FasterCache")
            patch_target = apply_fastercache(
                patch_target,
                start_step=self.fastercache_start_step,
                model_interval=self.fastercache_model_interval,
                block_interval=self.fastercache_block_interval,
            )
        elif self.worldcache_enabled:
            from cosmos_predict2._src.predict2.inference.worldcache_utils import apply_worldcache
            log.info("[WorldCache] Applying WorldCache")
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
                parallel_cfg=self.worldcache_parallel_cfg,
            )

        if hasattr(self.model, "net"):
            self.model.net = patch_target
        else:
            self.model = patch_target

        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None

    # ------------------------------------------------------------------
    def _init_distributed(self):
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=self.context_parallel_size)
        self.process_group = parallel_state.get_context_parallel_group()
        log.info(f"CP init: size={self.context_parallel_size}, rank={distributed.get_rank()}")

    # ------------------------------------------------------------------
    def _reset_worldcache_state(self, num_steps: int):
        """Reset WorldCache buffers before each generation call."""
        if self.worldcache_parallel_cfg:
            target_steps = num_steps          # B=2 batched, cnt++ once per step
        else:
            target_steps = num_steps * 2      # sequential cond+uncond, cnt++ twice per step

        net = self.model.net if hasattr(self.model, "net") else self.model
        if not hasattr(net, "worldcache_num_steps"):
            return
        net.worldcache_num_steps = target_steps
        net.cnt = 0
        net.worldcache_step_skipped_count = 0
        net.accumulated_rel_l1_distance = [0.0, 0.0]
        net.residual_cache = [None, None]
        net.probe_residual_cache = [None, None]
        net.residual_window = [[], []]
        net.probe_residual_window = [[], []]
        net.previous_internal_states = [None, None]
        net.previous_input = [None, None]
        net.previous_output = [None, None]
        net.resume_flag = [False, False]

    def _reset_dicache_state(self, num_steps: int):
        """Reset DiCache buffers before each generation call."""
        target_steps = num_steps * 2  # CFG = 2 passes per step
        net = self.model.net if hasattr(self.model, "net") else self.model
        if not hasattr(net, "dicache_num_steps"):
            return
        net.dicache_num_steps = target_steps
        net.cnt = 0
        net.accumulated_rel_l1_distance = [0.0, 0.0]
        net.residual_cache = [None, None]
        net.residual_window = [[], []]
        net.probe_residual_window = [[], []]
        net.previous_internal_states = [None, None]
        net.previous_input = [None, None]
        net.resume_flag = [False, False]

    # ------------------------------------------------------------------
    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prompt: str,
        num_conditional_frames: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        use_neg_prompt: bool = True,
        camera: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        lam_video: torch.Tensor | None = None,
    ):
        """Build the data_batch dict consumed by the model.

        DreamDojo-specific: includes ``lam_video`` so the Latent Action
        Model can modulate actions inside the model's forward / velocity_fn.
        """
        B, C, T, H, W = video.shape

        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "camera": camera,
            "action": action.unsqueeze(0) if action is not None else None,
            "fps": torch.randint(16, 32, (self.batch_size,)).float(),
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),
            "num_conditional_frames": num_conditional_frames,
            # DreamDojo LAM input
            "lam_video": lam_video.unsqueeze(0) if lam_video is not None else None,
        }

        if use_neg_prompt:
            assert negative_prompt is not None

        # Text embeddings
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

        # Move float tensors to GPU / bf16
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    # ------------------------------------------------------------------
    def _process_lam(self, data_batch: dict) -> dict:
        """Run the Latent Action Model on data_batch *in-place* (DreamDojo only).

        Mimics the LAM block inside ``Text2WorldModelRectifiedFlow.forward()``:
            data_batch["action"][:, :, -32:] *= latent_action
        """
        if not hasattr(self.model, "lam") or self.model.lam is None:
            return data_batch
        if data_batch.get("lam_video") is None or data_batch.get("action") is None:
            return data_batch

        lam_video = rearrange(data_batch["lam_video"], "b (p t) h w c -> (b p) t h w c", t=2)
        lam_input = {"videos": lam_video}
        with torch.no_grad():
            outputs = self.model.lam.lam(lam_input)
        latent_action = outputs["z_rep"].squeeze().to(data_batch["action"].dtype).detach()
        latent_action = rearrange(latent_action, "(t b) d -> b t d", t=data_batch["action"].shape[1])
        data_batch["action"][:, :, -32:] = data_batch["action"][:, :, -32:] * latent_action
        return data_batch

    # ------------------------------------------------------------------
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
        lam_video: torch.Tensor | None = None,
    ):
        """Generate a video clip from image/video + text + action.

        ``lam_video`` is the DreamDojo-specific input consumed by the
        Latent Action Model that modulates ``action`` before diffusion.
        """
        assert camera is not None or action is not None or (num_input_video == 1 and num_output_video == 1)

        # Resolution
        if resolution == "none":
            h, w = self.model.get_video_height_width()
            video_resolution = (h, w)
        else:
            video_resolution = tuple(int(x) for x in resolution.split(","))
            assert len(video_resolution) == 2

        model_required_frames = self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t)

        # ---- Build vid_input ----
        if input_path is None or num_latent_conditional_frames == 0:
            vid_input = torch.zeros(1, 3, model_required_frames, *video_resolution, dtype=torch.uint8)
        elif isinstance(input_path, str):
            ext = os.path.splitext(input_path)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                vid_input = read_and_process_image(input_path, video_resolution, model_required_frames)
            elif ext in _VIDEO_EXTENSIONS:
                vid_input = read_and_process_video(
                    input_path, video_resolution, model_required_frames, num_latent_conditional_frames
                )
            else:
                raise ValueError(f"Unsupported extension: {ext}")
        elif isinstance(input_path, torch.Tensor):
            vid_input = input_path
        else:
            raise ValueError(f"Unsupported input_path type: {type(input_path)}")

        # ---- Data batch (includes lam_video!) ----
        data_batch = self._get_data_batch_input(
            video=vid_input,
            prompt=prompt,
            camera=camera,
            action=action,
            num_conditional_frames=num_latent_conditional_frames,
            negative_prompt=negative_prompt,
            use_neg_prompt=True,
            lam_video=lam_video,
        )

        # ---- DreamDojo LAM: modulate action with latent actions ----
        data_batch = self._process_lam(data_batch)

        mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        log.info(f"GPU memory after data_batch: {mem_gb:.2f} GB")

        # ---- Offload text encoder after embedding computation ----
        if self.offload_text_encoder and self.model.text_encoder is not None:
            log.info("[Memory] Offloading text encoder to CPU")
            if hasattr(self.model.text_encoder, "model") and self.model.text_encoder.model is not None:
                self.model.text_encoder.model = self.model.text_encoder.model.to("cpu")
            torch.cuda.empty_cache()

        if self.offload_tokenizer:
            log.info("[Memory] Loading tokenizer encoder to GPU")
            if hasattr(self.model.tokenizer, "encoder") and self.model.tokenizer.encoder is not None:
                self.model.tokenizer.encoder = self.model.tokenizer.encoder.to("cuda")
            torch.cuda.empty_cache()

        if self.offload_diffusion_model:
            log.info("[Memory] Loading DiT to GPU")
            self.model.net = self.model.net.to("cuda")
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                self.model.conditioner = self.model.conditioner.to("cuda")
            torch.cuda.empty_cache()

        # ---- Reset caching state ----
        if self.worldcache_enabled:
            self._reset_worldcache_state(num_steps)
        elif self.dicache_enabled:
            self._reset_dicache_state(num_steps)

        # Expose parallel_cfg flag to the model so velocity_fn can detect it
        self.model._worldcache_parallel_cfg = self.worldcache_parallel_cfg

        # ---- Select generation method ----
        extra_kwargs = {}
        if camera is not None:
            extra_kwargs = {"num_input_video": num_input_video, "num_output_video": num_output_video}

        if self.model.config.use_lora:
            generate_fn = self.model.generate_samples_from_batch_lora
        else:
            generate_fn = self.model.generate_samples_from_batch

        method = "Baseline"
        if self.worldcache_enabled:
            method = "WorldCache"
        elif self.dicache_enabled:
            method = "DiCache"
        elif self.fastercache_enabled:
            method = "FasterCache"
        log.info(f"Running inference: {method} ({num_steps} steps)")

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        sample = generate_fn(
            data_batch,
            n_sample=1,
            guidance=guidance,
            seed=seed,
            is_negative_prompt=True,
            num_steps=num_steps,
            **extra_kwargs,
        )

        t1 = time.perf_counter()
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        log.info(f"Generation done in {t1 - t0:.1f}s | peak GPU {peak_gb:.2f} GB")

        # ---- Offload DiT ----
        if self.offload_diffusion_model:
            self.model.net = self.model.net.to("cpu")
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                self.model.conditioner = self.model.conditioner.to("cpu")

        if self.offload_tokenizer:
            if hasattr(self.model.tokenizer, "encoder") and self.model.tokenizer.encoder is not None:
                self.model.tokenizer.encoder = self.model.tokenizer.encoder.to("cpu")
            torch.cuda.empty_cache()
            log.info("[Memory] Loading tokenizer decoder to GPU")
            if hasattr(self.model.tokenizer, "decoder") and self.model.tokenizer.decoder is not None:
                self.model.tokenizer.decoder = self.model.tokenizer.decoder.to("cuda")
            torch.cuda.empty_cache()

        # ---- Decode latent -> pixels ----
        if isinstance(sample, list):
            video = torch.cat([self.model.decode(s) for s in sample], dim=3)
        else:
            video = self.model.decode(sample)

        # ---- Final cleanup ----
        if self.offload_tokenizer:
            if hasattr(self.model.tokenizer, "decoder") and self.model.tokenizer.decoder is not None:
                self.model.tokenizer.decoder = self.model.tokenizer.decoder.to("cpu")
            torch.cuda.empty_cache()
        if self.offload_text_encoder and self.model.text_encoder is not None:
            if hasattr(self.model.text_encoder, "model") and self.model.text_encoder.model is not None:
                self.model.text_encoder.model = self.model.text_encoder.model.to("cuda")
            torch.cuda.empty_cache()

        return video

    # ------------------------------------------------------------------
    def cleanup(self):
        if self.context_parallel_size > 1:
            import torch.distributed as dist
            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()