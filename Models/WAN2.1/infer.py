
import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool



def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16


    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-1.3B", # "t2v-1.3B"
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./ckpt/Wan2.1-T2V-1.3B-Original",
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument("--input_files", "-i", nargs="+", default=None)
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="The directory to save the generated image or video to.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.")
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (first frame) to generate the video from."
    )
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (last frame) to generate the video from."
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--rel_l1_thresh",
        type=float,
        default=0.08, 
        help="the upper bound of accumulated err")
    parser.add_argument(
        "--ret_ratio",
        type=float,
        default=0.2,
        help="Retention ratio of unchanged steps")
        
    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    
    wan_t2v.model.__class__.cnt = 0
    wan_t2v.model.__class__.probe_depth = 1 
    wan_t2v.model.__class__.num_steps = args.sample_steps * 2
    wan_t2v.model.__class__.rel_l1_thresh = args.rel_l1_thresh
    wan_t2v.model.__class__.accumulated_rel_l1_distance = [0.0, 0.0]
    wan_t2v.model.__class__.ret_ratio = args.ret_ratio
    wan_t2v.model.__class__.residual_cache = [None, None]
    wan_t2v.model.__class__.probe_residual_cache = [None, None]
    wan_t2v.model.__class__.residual_window = [[], []]
    wan_t2v.model.__class__.probe_residual_window = [[], []]
    wan_t2v.model.__class__.previous_internal_states = [None, None]
    wan_t2v.model.__class__.previous_input = [None, None]
    wan_t2v.model.__class__.previous_output = [None, None]
    wan_t2v.model.__class__.resume_flag = [False, False]
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    import json
    from tqdm import tqdm
    for prompt_file in tqdm(args.input_files, desc="Generating", disable=rank != 0, total=len(args.input_files)):
        with open(prompt_file, 'r') as f:
            file_data = json.load(f)
        save_video = os.path.join(args.output_dir, file_data['name'] + ".mp4")
        save_json = os.path.join(args.output_dir, file_data['name'] + ".json")
        if os.path.exists(save_video) and os.path.exists(save_json):
            continue
        prompt = file_data['prompt']
        num_frames = file_data['num_output_frames']
        guidance_scale = file_data['guidance']
        seed = file_data['seed']
        negative_prompt = file_data['negative_prompt']
        video, gen_time = wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=num_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=guidance_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=args.offload_model)

        if rank == 0:
            
            cache_video(
                tensor=video[None],
                save_file=save_video,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
            
            with open(save_json, 'w') as f:
                json.dump({
                    'prompt': prompt,
                    'num_frames': num_frames,
                    'guidance_scale': guidance_scale,
                    'seed': seed,
                    'negative_prompt': negative_prompt,
                    'gen_time': gen_time
                }, f, indent=4)


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
