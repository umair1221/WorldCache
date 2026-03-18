#!/usr/bin/env python3
"""
Compute PSNR, SSIM, and LPIPS for predicted vs ground-truth videos
stored in TWO separate folders with naming convention:

Pred folder:
    {name}.mp4

GT folder:
    {name}_gt.mp4   OR   {name}.mp4

Outputs per-video metrics and a summary.json file with averages.
"""

import os
import json
import argparse
from typing import Dict, List

import torch
import piq
import mediapy
from tqdm import tqdm


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def load_video_uint8(path: str) -> torch.Tensor:
    """
    Load video as (T, C, H, W) uint8 tensor.
    """
    video = mediapy.read_video(path)  # (T, H, W, C) uint8
    video = torch.from_numpy(video)   # (T, H, W, C)
    video = video.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
    return video


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """
    pred, gt: (T, C, H, W) uint8
    Returns dict with psnr, ssim, lpips.
    """
    # Match length
    min_t = min(pred.shape[0], gt.shape[0])
    pred = pred[:min_t]
    gt = gt[:min_t]

    # Resize GT if needed
    if pred.shape[2:] != gt.shape[2:]:
        gt = torch.nn.functional.interpolate(
            gt.float(), size=pred.shape[2:], mode="bilinear", align_corners=False
        ).to(torch.uint8)

    x = torch.clamp(pred.float() / 255.0, 0, 1)
    y = torch.clamp(gt.float() / 255.0, 0, 1)

    psnr_val = piq.psnr(x, y).mean().item()
    ssim_val = piq.ssim(x, y).mean().item()
    lpips_val = piq.LPIPS()(x, y).mean().item()

    return {
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "lpips": float(lpips_val),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Compute video metrics from two folders.")
    parser.add_argument("--pred_dir", "-d", type=str, required=True,
                        help="Folder containing predicted videos: {name}.mp4")
    parser.add_argument("--gt_dir", type=str, default="/share_2/users/ahmed_heakl/DiCache/WAN2.1/outputs/wan_egodex_worldcache",
                        help="Folder containing GT videos: {name}_gt.mp4 or {name}.mp4")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional output summary JSON path (default: pred_dir/summary.json)")
    return parser.parse_args()


def find_gt_file(gt_dir: str, name: str) -> str:
    """
    Try matching GT filename patterns.
    """
    candidates = [
        os.path.join(gt_dir, f"{name}_gt.mp4"),
        os.path.join(gt_dir, f"{name}.mp4"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    args = parse_args()

    pred_dir = args.pred_dir
    gt_dir = args.gt_dir

    assert os.path.isdir(pred_dir), f"Not a directory: {pred_dir}"
    assert os.path.isdir(gt_dir), f"Not a directory: {gt_dir}"

    if args.output_json is None:
        output_json = os.path.join(pred_dir, "summary.json")
    else:
        output_json = args.output_json

    pred_files = [f for f in os.listdir(pred_dir)
                  if f.endswith(".mp4")]

    all_metrics: List[Dict] = []
    all_psnr, all_ssim, all_lpips = [], [], []

    for pred_file in tqdm(pred_files, desc="Evaluating"):
        name = pred_file[:-4]  # remove .mp4
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = find_gt_file(gt_dir, name)

        if gt_path is None:
            print(f"[Warning] Missing GT for {name}, skipping.")
            continue

        pred_video = load_video_uint8(pred_path)
        gt_video = load_video_uint8(gt_path)

        metrics = compute_metrics(pred_video, gt_video)
        metrics["name"] = name

        all_metrics.append(metrics)
        all_psnr.append(metrics["psnr"])
        all_ssim.append(metrics["ssim"])
        all_lpips.append(metrics["lpips"])

        print(f"[{name}] PSNR={metrics['psnr']:.3f}  "
              f"SSIM={metrics['ssim']:.3f}  "
              f"LPIPS={metrics['lpips']:.3f}")

    if len(all_metrics) == 0:
        print("No valid video pairs found.")
        return

    summary = {
        "psnr": float(sum(all_psnr) / len(all_psnr)),
        "ssim": float(sum(all_ssim) / len(all_ssim)),
        "lpips": float(sum(all_lpips) / len(all_lpips)),
        "num_samples": len(all_metrics),
        "per_video": all_metrics,
    }

    print("\n" + "=" * 50)
    print(f"PSNR:  {summary['psnr']:.3f}")
    print(f"SSIM:  {summary['ssim']:.3f}")
    print(f"LPIPS: {summary['lpips']:.3f}")
    print(f"N:     {summary['num_samples']}")
    print("=" * 50)

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {output_json}")


if __name__ == "__main__":
    main()
