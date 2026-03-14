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

from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import Callable, Dict, Literal, Optional, Tuple

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldCondition,
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class Video2WorldModelRectifiedFlowConfig(Text2WorldModelRectifiedFlowConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    conditional_frame_timestep: float = (
        -1.0
    )  # Noise level used for conditional frames; default is -1 which will not take effective
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]


class Video2WorldModelRectifiedFlow(Text2WorldModelRectifiedFlow):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        # generate random number of conditional frames for training
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
    ) -> DenoisePrediction:
        """
        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            velocity prediction
        """
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            # Make the first few frames of x_t be the ground truth frames
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )

                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )  # add dimension for batch

        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=timesteps_B_T,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        ).float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W

    def _prepare_xt_for_condition(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
    ):
        """
        Prepares the xt tensor and timesteps for a single condition (cond or uncond).
        Returns (prepared_xt, prepared_timesteps, condition_video_mask_or_None).
        This extracts the shared logic from denoise() so it can be reused for batching.
        """
        condition_video_mask = None
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )
                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )
                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )

        return xt_B_C_T_H_W, timesteps_B_T, condition_video_mask

    def _postprocess_output(
        self,
        noise: torch.Tensor,
        net_output: torch.Tensor,
        condition: Text2WorldCondition,
        condition_video_mask: torch.Tensor,
    ):
        """
        Applies GT frame velocity replacement on the network output if needed.
        """
        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output)
            gt_frames_velocity = noise - gt_frames_x0
            net_output = gt_frames_velocity * condition_video_mask + net_output * (
                1 - condition_video_mask
            )
        return net_output

    @staticmethod
    def _cat_conditions(cond_a: Text2WorldCondition, cond_b: Text2WorldCondition):
        """
        Concatenates two condition dataclass instances along the batch dimension.
        Tensors are torch.cat'd along dim=0. Non-tensor fields (e.g., data_type, bools)
        are taken from cond_a (they must match between cond and uncond).
        """
        kwargs = {}
        for f in dataclass_fields(cond_a):
            val_a = getattr(cond_a, f.name)
            val_b = getattr(cond_b, f.name)
            if isinstance(val_a, torch.Tensor) and isinstance(val_b, torch.Tensor):
                kwargs[f.name] = torch.cat([val_a, val_b], dim=0)
            else:
                # For non-tensor fields (DataType, bool, None, etc.), use cond_a's value
                kwargs[f.name] = val_a
        return type(cond_a)(**kwargs)

    def denoise_batched(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
        uncondition: Text2WorldCondition,
    ):
        """
        Batched CFG denoise: prepares both cond/uncond xt tensors, concatenates along
        batch dim (B=2), runs self.net() once, then splits the output.

        Returns:
            (cond_v, uncond_v): Tuple of velocity predictions, each with original batch size.
        """
        # Prepare xt for both conditions independently
        xt_cond, ts_cond, mask_cond = self._prepare_xt_for_condition(
            noise, xt_B_C_T_H_W.clone(), timesteps_B_T.clone(), condition
        )
        xt_uncond, ts_uncond, mask_uncond = self._prepare_xt_for_condition(
            noise, xt_B_C_T_H_W.clone(), timesteps_B_T.clone(), uncondition
        )

        # Concatenate inputs along batch dim
        xt_batched = torch.cat([xt_cond, xt_uncond], dim=0)
        ts_batched = torch.cat([ts_cond, ts_uncond], dim=0)

        # Concatenate condition dicts along batch dim
        batched_condition = self._cat_conditions(condition, uncondition)

        # Single forward pass with B=2
        net_output = self.net(
            x_B_C_T_H_W=xt_batched.to(**self.tensor_kwargs),
            timesteps_B_T=ts_batched,
            **batched_condition.to_dict(),
        ).float()

        # Split output back
        B_orig = xt_B_C_T_H_W.shape[0]
        cond_output, uncond_output = net_output[:B_orig], net_output[B_orig:]

        # Postprocess (GT frame velocity replacement)
        noise_cond = noise
        noise_uncond = noise
        cond_v = self._postprocess_output(noise_cond, cond_output, condition, mask_cond)
        uncond_v = self._postprocess_output(noise_uncond, uncond_output, uncondition, mask_uncond)

        return cond_v, uncond_v

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        parallel_cfg: bool = False,
    ) -> Callable:
        """
        Generates a callable function `velocity_fn` based on the provided data batch and guidance factor.

        Args:
        - data_batch (Dict): A batch of data used for conditioning.
        - guidance (float, optional): Guidance scale. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true
        - parallel_cfg (bool): If True, batch cond+uncond into a single forward pass (B=2).

        Returns:
        - Callable: A function `velocity_fn(noise_x, sigma)` that returns velocity prediction.
        """
        # Auto-detect parallel_cfg from model attribute if not explicitly passed
        if not parallel_cfg:
            parallel_cfg = getattr(self.net, 'worldcache_parallel_cfg', False) or getattr(self, '_worldcache_parallel_cfg', False)

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        if parallel_cfg:
            def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                cond_v, uncond_v = self.denoise_batched(noise, noise_x, timestep, condition, uncondition)
                velocity_pred = cond_v + guidance * (cond_v - uncond_v)
                return velocity_pred
        else:
            def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                cond_v = self.denoise(noise, noise_x, timestep, condition)
                uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
                velocity_pred = cond_v + guidance * (cond_v - uncond_v)
                return velocity_pred

        return velocity_fn

    def denoise_edm(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        time: torch.Tensor,
        condition: Video2WorldCondition,
        net_type: Literal["teacher", "fake_score", "student"] = "teacher",
    ) -> DenoisePrediction:
        """
        Network forward to denoise the input noised data given noise level, and condition.

        Assumes EDM-scaling parameterization.

        Compared to base class denoise function, this function supports different net types:
        - fake_score: the fake score net on student generator's outputs
        - student: the student net (few-step generator)

        Args:
            xt (torch.Tensor): The input noise data.
            time (torch.Tensor): The noise level under TrigFlow parameterization.
            condition (Video2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        if time.ndim == 1:
            time_B_T = rearrange(time, "b -> b 1")
        elif time.ndim == 2:
            time_B_T = time
        else:
            raise ValueError(f"time shape {time.shape} is not supported")
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")

        if condition.is_video:
            # replace the noise level of the cond frames to be the pre-defined conditional noise level (very low)
            # the scaling coefficients computed later will inherit the setting.
            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )
            condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True).type_as(
                time_B_1_T_1_1
            )  # (B,1,T,1,1)
            t_cond = torch.atan(torch.ones_like(time_B_1_T_1_1) * (self.sigma_conditional / self.sigma_data))
            time_B_1_T_1_1 = t_cond * condition_video_mask_B_1_T_1_1 + time_B_1_T_1_1 * (
                1 - condition_video_mask_B_1_T_1_1
            )

        # convert noise level time to EDM-formulation coefficients
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling_from_time(time_B_1_T_1_1)

        # EDM preconditioning
        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        if net_type == "student" and self.change_time_embed:
            # Use c_noise(t)=t to improve numerical stability
            c_noise_B_1_T_1_1 = time_B_1_T_1_1

        net = self.net

        # Apply vid2vid conditioning
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.sigma_data
            # during training we temporarily concat some variables (e.g. x0 and G_x0)
            # into a batch. They are from the same data sample, so their use_video_condition
            # is a boolean tensor with batch dim; their bool value should be the same.
            use_video_cond = condition.use_video_condition
            if isinstance(use_video_cond, torch.Tensor):
                assert bool((use_video_cond == use_video_cond[0]).all().item()), (
                    "inconsistent use_video_condition in concatenated batch"
                )
                use_video_cond = bool(use_video_cond[0].item())
            if not use_video_cond:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                net_state_in_B_C_T_H_W
            )

            # Replace the first few frames of the video with the conditional frames
            # Update the c_noise as the conditional frames are clean and have very low noise

            # x_in = mask*GT + (1-mask)*x; tangent passes only through the (1-mask) branch
            net_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + net_state_in_B_C_T_H_W * (
                1 - condition_video_mask
            )

        call_kwargs = dict(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )
        if net_type == "fake_score" and getattr(self, "intermediate_feature_ids", None):
            call_kwargs["intermediate_feature_ids"] = self.intermediate_feature_ids

        # forward pass through the network
        net_out = net(**call_kwargs)

        if net_type == "fake_score" and getattr(self, "intermediate_feature_ids", None):
            net_output_B_C_T_H_W, intermediate_features_outputs = net_out
            net_output_B_C_T_H_W = net_output_B_C_T_H_W.float()
        else:
            net_output_B_C_T_H_W = net_out.float()
            intermediate_features_outputs = []

        net_output_B_C_T_H_W = net_output_B_C_T_H_W.to(dtype=xt_B_C_T_H_W.dtype)

        # EDM reconstruction of x0
        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W

        # Replace GT on conditioned frames to avoid training on pinned frames (parity with base Video2WorldModel)
        if getattr(self.config, "denoise_replace_gt_frames", False) and condition.is_video:
            # Replace condition frames to be gt frames to zero out loss on these frames
            gt_frames = condition.gt_frames.type_as(x0_pred_B_C_T_H_W)
            x0_pred_B_C_T_H_W = gt_frames * condition_video_mask.type_as(x0_pred_B_C_T_H_W) + x0_pred_B_C_T_H_W * (
                1 - condition_video_mask
            )

        if net_type == "fake_score":
            return DenoisePrediction(x0=x0_pred_B_C_T_H_W, intermediate_features=intermediate_features_outputs)
        else:  # student and teacher need F
            F_pred_B_C_T_H_W = (torch.cos(time_B_1_T_1_1) * xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / (
                torch.sin(time_B_1_T_1_1) * self.sigma_data
            )
            return DenoisePrediction(
                x0=x0_pred_B_C_T_H_W, F=F_pred_B_C_T_H_W, intermediate_features=intermediate_features_outputs
            )
