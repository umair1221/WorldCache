# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cosmos_predict2.config import (
    DEFAULT_NEGATIVE_PROMPT,
    CommonSetupArguments,
    Guidance,
    ModelKey,
    ModelVariant,
    get_model_literal,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.ROBOT_ACTION_COND)


class ActionLoadFn(Protocol):
    def __call__(self, json_data: dict, video_path: str, args: "ActionConditionedInferenceArguments") -> dict: ...


class ActionConditionedSetupArguments(CommonSetupArguments):
    """Setup arguments for action-conditioned inference (DreamDojo)."""

    config_file: str = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"

    # pyrefly: ignore
    model: get_model_literal(ModelVariant.ROBOT_ACTION_COND) = DEFAULT_MODEL_KEY.name

    checkpoints_dir: str = "checkpoints/exp1201/gr1/checkpoints"
    save_dir: str = "results/action2world"
    num_frames: int = 37
    num_samples: int = 100
    dataset_path: str = "datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot"
    data_split: str = "test"
    single_base_index: bool = False
    deterministic_uniform_sampling: bool = False
    checkpoint_interval: int = 2000
    infinite: bool = False

    # ==================== WorldCache ====================
    worldcache_enabled: bool = False
    """Enable WorldCache acceleration for DiT inference."""
    worldcache_num_steps: int = 35
    """Total denoising steps (must match --num-steps in inference)."""
    worldcache_rel_l1_thresh: float = 0.5
    """Relative L1 threshold for cache hit decision."""
    worldcache_ret_ratio: float = 0.2
    """Fraction of initial steps to always compute (warm-up)."""
    worldcache_probe_depth: int = 8
    """Number of leading DiT blocks used as the probe."""
    worldcache_motion_sensitivity: float = 5.0
    """Alpha in dynamic threshold: thresh / (1 + alpha * velocity)."""
    worldcache_flow_enabled: bool = False
    """Warp cached features using optical-flow before reuse."""
    worldcache_flow_scale: float = 0.5
    """Down-scale factor for flow estimation (0-1)."""
    worldcache_hf_enabled: bool = False
    """Enable spectral (high-frequency) drift guard."""
    worldcache_hf_thresh: float = 0.01
    """HF drift above this forces a full compute step."""
    worldcache_saliency_enabled: bool = False
    """Weight drift by channel-variance saliency map."""
    worldcache_saliency_weight: float = 5.0
    """Beta multiplier for saliency-guided thresholding."""
    worldcache_osi_enabled: bool = False
    """Online System Identification for optimal gamma."""
    worldcache_dynamic_decay: bool = False
    """Increase threshold over time (late steps are easier to cache)."""
    worldcache_aduc_enabled: bool = False
    """Adaptive Unconditional Caching (skip uncond at late steps)."""
    worldcache_aduc_start: float = 0.5
    """Step ratio after which AdUC activates."""
    worldcache_parallel_cfg: bool = False
    """Batch cond+uncond in B=2 (halves wall-time, needs 2x memory)."""

    # ==================== DiCache ====================
    dicache_enabled: bool = False
    """Enable DiCache acceleration (mutually exclusive with WorldCache)."""
    dicache_num_steps: int = 35
    dicache_rel_l1_thresh: float = 0.5
    dicache_ret_ratio: float = 0.2
    dicache_probe_depth: int = 8

    # ==================== FasterCache ====================
    fastercache_enabled: bool = False
    fastercache_start_step: int = 1
    fastercache_model_interval: int = 2
    fastercache_block_interval: int = 3


@dataclass
class ActionConditionedInferenceArguments:
    """Per-run inference arguments for DreamDojo action-conditioned generation."""

    save_root: Path = Path("results/action2world")
    chunk_size: int = 12
    guidance: Guidance = 0
    resolution: str = "none"

    # Dataset
    camera_id: int = 0
    start: int = 0
    end: int = 100
    fps_downsample_ratio: int = 1
    gripper_scale: float = 1.0
    gripper_key: str = "continuous_gripper_state"
    state_key: str = "state"
    action_scaler: float = 20.0
    use_quat: bool = False

    # Inference knobs
    reverse: bool = False
    single_chunk: bool = False
    start_frame_idx: int = 0
    save_fps: int = 10
    zero_actions: bool = False
    num_latent_conditional_frames: int = 1
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT