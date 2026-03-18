# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DreamDojo action-conditioned inference with WorldCache support.

Changes vs upstream:
  * Reads WorldCache / DiCache / FasterCache flags from ``setup_args``
    and forwards them to ``Video2WorldInference``.
  * Passes ``lam_video`` through for each chunk so the Latent Action
    Model can modulate actions.
"""

import json
import os
from glob import glob
from pathlib import Path

import mediapy
import numpy as np
import piq
import torch
import torch.nn.functional as F
import torchvision
from loguru import logger

from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.predict2.action.datasets.dataset_utils import euler2rotm, rotm2euler, rotm2quat
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedSetupArguments,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, VIDEO_EXTENSIONS

from groot_dreams.dataloader import MultiVideoActionDataset


# ------------------------------------------------------------------
# Action helpers (unchanged from upstream)
# ------------------------------------------------------------------

def _get_robot_states(label, state_key="state", gripper_key="continuous_gripper_state"):
    all_states = np.array(label[state_key])
    all_cont_gripper_states = np.array(label[gripper_key])
    return all_states, all_cont_gripper_states


def _get_actions(arm_states, gripper_states, sequence_length, use_quat=False):
    action = np.zeros((sequence_length - 1, 8 if use_quat else 7))
    for k in range(1, sequence_length):
        prev_xyz = arm_states[k - 1, 0:3]
        prev_rpy = arm_states[k - 1, 3:6]
        prev_rotm = euler2rotm(prev_rpy)
        curr_xyz = arm_states[k, 0:3]
        curr_rpy = arm_states[k, 3:6]
        curr_gripper = gripper_states[k]
        curr_rotm = euler2rotm(curr_rpy)
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm
        if use_quat:
            rel_rot = rotm2quat(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:7] = rel_rot
            action[k - 1, 7] = curr_gripper
        else:
            rel_rot = rotm2euler(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:6] = rel_rot
            action[k - 1, 6] = curr_gripper
    return action


def get_action_sequence_from_states(
    data, fps_downsample_ratio=1, use_quat=False,
    state_key="state", gripper_scale=1.0,
    gripper_key="continuous_gripper_state", action_scaler=20.0,
):
    arm_states, cont_gripper_states = _get_robot_states(data, state_key, gripper_key)
    actions = _get_actions(
        arm_states[::fps_downsample_ratio],
        cont_gripper_states[::fps_downsample_ratio],
        len(data[state_key][::fps_downsample_ratio]),
        use_quat=use_quat,
    )
    actions *= np.array(
        [action_scaler] * 6 + [gripper_scale] if not use_quat
        else [action_scaler] * 6 + [action_scaler, gripper_scale]
    )
    return actions


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def inference(
    setup_args: ActionConditionedSetupArguments,
    inference_args,
    checkpoint_path,
):
    """Run action-conditioned video generation with optional WorldCache."""
    torch.enable_grad(False)

    if inference_args.num_latent_conditional_frames not in [0, 1, 2]:
        raise ValueError(f"num_latent_conditional_frames must be 0, 1 or 2, got {inference_args.num_latent_conditional_frames}")

    # ---- Build Video2WorldInference with caching flags ----
    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    if experiment is None:
        raise ValueError("Experiment name must be provided")

    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=str(checkpoint_path),
        s3_credential_path="",
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
        # WorldCache
        worldcache_enabled=setup_args.worldcache_enabled,
        worldcache_num_steps=setup_args.worldcache_num_steps,
        worldcache_rel_l1_thresh=setup_args.worldcache_rel_l1_thresh,
        worldcache_ret_ratio=setup_args.worldcache_ret_ratio,
        worldcache_probe_depth=setup_args.worldcache_probe_depth,
        worldcache_motion_sensitivity=setup_args.worldcache_motion_sensitivity,
        worldcache_flow_enabled=setup_args.worldcache_flow_enabled,
        worldcache_flow_scale=setup_args.worldcache_flow_scale,
        worldcache_hf_enabled=setup_args.worldcache_hf_enabled,
        worldcache_hf_thresh=setup_args.worldcache_hf_thresh,
        worldcache_saliency_enabled=setup_args.worldcache_saliency_enabled,
        worldcache_saliency_weight=setup_args.worldcache_saliency_weight,
        worldcache_osi_enabled=setup_args.worldcache_osi_enabled,
        worldcache_dynamic_decay=setup_args.worldcache_dynamic_decay,
        worldcache_aduc_enabled=setup_args.worldcache_aduc_enabled,
        worldcache_aduc_start=setup_args.worldcache_aduc_start,
        worldcache_parallel_cfg=setup_args.worldcache_parallel_cfg,
        # DiCache
        dicache_enabled=setup_args.dicache_enabled,
        dicache_num_steps=setup_args.dicache_num_steps,
        dicache_rel_l1_thresh=setup_args.dicache_rel_l1_thresh,
        dicache_ret_ratio=setup_args.dicache_ret_ratio,
        dicache_probe_depth=setup_args.dicache_probe_depth,
        # FasterCache
        fastercache_enabled=setup_args.fastercache_enabled,
        fastercache_start_step=setup_args.fastercache_start_step,
        fastercache_model_interval=setup_args.fastercache_model_interval,
        fastercache_block_interval=setup_args.fastercache_block_interval,
    )

    mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    logger.info(f"GPU memory after model load: {mem_gb:.2f} GB")

    rank0 = True
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    inference_args.save_root = Path(setup_args.save_dir) / checkpoint_path.parent.name
    inference_args.save_root.mkdir(parents=True, exist_ok=True)

    num_frames = setup_args.num_frames
    num_samples = setup_args.num_samples
    dataset = MultiVideoActionDataset(
        num_frames=num_frames,
        dataset_path=setup_args.dataset_path,
        data_split=setup_args.data_split,
        single_base_index=setup_args.single_base_index,
        restrict_len=num_samples,
        deterministic_uniform_sampling=setup_args.deterministic_uniform_sampling,
    )
    eval_indices = [
        idx for idx in range(len(dataset))
        if not (inference_args.save_root / f"{idx:04d}_psnr.json").exists()
    ]

    all_psnr, all_ssim, all_lpips = [], [], []

    for idx in eval_indices:
        data = dataset[idx]
        gt_video = data["video"].permute(1, 2, 3, 0)
        img_array = data["video"].transpose(0, 1)[:1]
        actions = data["action"][:num_frames - 1].numpy()
        lam_video = data["lam_video"]

        if inference_args.zero_actions:
            actions = np.zeros_like(actions)

        chunk_video = []
        save_name = f"{idx:04d}"
        video_name = str(inference_args.save_root / save_name)
        chunk_video_name = str(inference_args.save_root / f"{save_name}_pred.mp4")
        logger.info(f"Saving video to {video_name}")
        if os.path.exists(chunk_video_name):
            logger.info(f"Exists, skipping: {chunk_video_name}")
            continue

        first_round = True
        for i in range(inference_args.start_frame_idx, len(actions), inference_args.chunk_size):
            actions_chunk = actions[i : i + inference_args.chunk_size]
            assert actions_chunk.shape[0] == inference_args.chunk_size

            # DreamDojo: slice lam_video for this chunk
            current_lam_video = lam_video[i * 2 : (i + inference_args.chunk_size) * 2]

            if not first_round:
                img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0) * 255.0
            else:
                img_tensor = img_array
            first_round = False

            num_video_frames = actions_chunk.shape[0] + 1
            vid_input = torch.cat(
                [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
            )
            vid_input = vid_input.to(torch.uint8)
            vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)

            video = video2world_cli.generate_vid2world(
                prompt="",
                input_path=vid_input,
                action=torch.from_numpy(actions_chunk).float()
                if isinstance(actions_chunk, np.ndarray)
                else actions_chunk,
                guidance=inference_args.guidance,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=inference_args.num_latent_conditional_frames,
                resolution="480,640",
                seed=i,
                negative_prompt=inference_args.negative_prompt,
                lam_video=current_lam_video,  # <-- DreamDojo LAM
            )

            video_normalized = (video - (-1)) / (1 - (-1))
            video_clamped = (
                (torch.clamp(video_normalized[0], 0, 1) * 255)
                .to(torch.uint8)
                .permute(1, 2, 3, 0)
                .cpu()
                .numpy()
            )
            next_img_array = video_clamped[-1]
            img_array = next_img_array
            chunk_video.append(video_clamped)

            if inference_args.single_chunk:
                break

        chunk_list = [chunk_video[0]] + [
            chunk_video[i][: inference_args.chunk_size] for i in range(1, len(chunk_video))
        ]
        chunk_video = np.concatenate(chunk_list, axis=0)
        chunk_video_name = str(inference_args.save_root / f"{save_name}_pred.mp4")

        if rank0:
            mediapy.write_video(chunk_video_name, chunk_video, fps=inference_args.save_fps)
            mediapy.write_video(
                str(inference_args.save_root / f"{save_name}_gt.mp4"),
                gt_video.numpy(), fps=inference_args.save_fps,
            )
            concat_video = np.concatenate([gt_video.numpy(), chunk_video], axis=2)
            mediapy.write_video(
                str(inference_args.save_root / f"{save_name}_merged.mp4"),
                concat_video, fps=inference_args.save_fps,
            )
            np.save(str(inference_args.save_root / f"{save_name}_actions.npy"), actions)
            logger.info(f"Saved video to {chunk_video_name}")

            x_batch = torch.clamp(torch.from_numpy(chunk_video) / 255.0, 0, 1).permute(0, 3, 1, 2)
            y_batch = torch.clamp(gt_video / 255.0, 0, 1).permute(0, 3, 1, 2)
            psnr = piq.psnr(x_batch, y_batch).mean().item()
            ssim = piq.ssim(x_batch, y_batch).mean().item()
            lpips = piq.LPIPS()(x_batch, y_batch).mean().item()
            with open(inference_args.save_root / f"{save_name}_metrics.json", "w") as f:
                json.dump({"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lpips)}, f)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lpips)

    if rank0 and all_psnr:
        print(f"PSNR: {sum(all_psnr) / len(all_psnr):.3f}")
        print(f"SSIM: {sum(all_ssim) / len(all_ssim):.3f}")
        print(f"LPIPS: {sum(all_lpips) / len(all_lpips):.3f}")
        with open(inference_args.save_root / "all_summary.json", "w") as f:
            json.dump({
                "psnr": f"{sum(all_psnr) / len(all_psnr):.3f}",
                "ssim": f"{sum(all_ssim) / len(all_ssim):.3f}",
                "lpips": f"{sum(all_lpips) / len(all_lpips):.3f}",
            }, f)

    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()

    video2world_cli.cleanup()