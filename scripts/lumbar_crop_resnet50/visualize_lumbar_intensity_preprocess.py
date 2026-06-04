#!/usr/bin/env python3
"""
Visualize intensity preprocessing for one lumbar_crop image.

Given one lumbar crop image, this script saves:
  1. original grayscale image
  2. CLAHE-processed grayscale image
  3. percentile-normalized grayscale image
  4. side-by-side comparison figure
  5. optional histogram comparison figure

Example:
  python visualize_lumbar_intensity_preprocess.py \
    --image_path ./dataset/train/lumbar_crop/0001-Lat_lumbar.png \
    --outdir ./debug_intensity_vis \
    --percentile_low 1.0 \
    --percentile_high 99.0 \
    --clahe_clip_limit 2.0 \
    --clahe_tile_grid_size 8 8
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply CLAHE and percentile normalization to one lumbar crop and save visualizations."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to one lumbar_crop image, e.g. ./dataset/train/lumbar_crop/0001-Lat_lumbar.png",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./debug_intensity_preprocess",
        help="Output directory for processed images and comparison figures.",
    )
    parser.add_argument(
        "--percentile_low",
        type=float,
        default=1.0,
        help="Low percentile for percentile normalization. Default: 1.0",
    )
    parser.add_argument(
        "--percentile_high",
        type=float,
        default=99.0,
        help="High percentile for percentile normalization. Default: 99.0",
    )
    parser.add_argument(
        "--clahe_clip_limit",
        type=float,
        default=2.0,
        help="CLAHE clipLimit. Default: 2.0",
    )
    parser.add_argument(
        "--clahe_tile_grid_size",
        type=int,
        nargs=2,
        default=[8, 8],
        metavar=("W", "H"),
        help="CLAHE tileGridSize. Default: 8 8",
    )
    parser.add_argument(
        "--save_histogram",
        action="store_true",
        help="If set, also save histogram comparison figure.",
    )
    return parser.parse_args()


def percentile_normalize(gray: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Per-image percentile normalization.

    Values below p_low are clipped to 0.
    Values above p_high are clipped to 255.
    Values between are linearly mapped to [0, 255].
    """
    gray_f = gray.astype(np.float32)
    lo = float(np.percentile(gray_f, p_low))
    hi = float(np.percentile(gray_f, p_high))

    if hi <= lo + 1e-6:
        out = np.clip(gray_f, 0, 255)
    else:
        out = (gray_f - lo) / (hi - lo + 1e-6)
        out = np.clip(out, 0.0, 1.0) * 255.0

    return out.astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )
    return clahe.apply(gray.astype(np.uint8))


def save_side_by_side(original: np.ndarray, clahe_img: np.ndarray, percentile_img: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [original, clahe_img, percentile_img]
    titles = ["Original grayscale", "CLAHE", "Percentile norm"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_histograms(original: np.ndarray, clahe_img: np.ndarray, percentile_img: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(original.ravel(), bins=256, alpha=0.45, label="Original")
    ax.hist(clahe_img.ravel(), bins=256, alpha=0.45, label="CLAHE")
    ax.hist(percentile_img.ravel(), bins=256, alpha=0.45, label="Percentile norm")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Count")
    ax.set_title("Intensity histogram comparison")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    image_path = Path(args.image_path).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to read image as grayscale: {image_path}")

    clahe_img = apply_clahe(
        gray,
        clip_limit=args.clahe_clip_limit,
        tile_grid_size=tuple(args.clahe_tile_grid_size),
    )
    percentile_img = percentile_normalize(
        gray,
        p_low=args.percentile_low,
        p_high=args.percentile_high,
    )

    stem = image_path.stem
    original_out = outdir / f"{stem}_original_gray.png"
    clahe_out = outdir / f"{stem}_clahe_clip{args.clahe_clip_limit}_tile{args.clahe_tile_grid_size[0]}x{args.clahe_tile_grid_size[1]}.png"
    percentile_out = outdir / f"{stem}_percentile_p{args.percentile_low:g}_{args.percentile_high:g}.png"
    compare_out = outdir / f"{stem}_comparison_original_clahe_percentile.png"

    cv2.imwrite(str(original_out), gray)
    cv2.imwrite(str(clahe_out), clahe_img)
    cv2.imwrite(str(percentile_out), percentile_img)
    save_side_by_side(gray, clahe_img, percentile_img, compare_out)

    print("[DONE] Saved intensity preprocessing visualizations")
    print(f"  original_gray     : {original_out}")
    print(f"  clahe             : {clahe_out}")
    print(f"  percentile_norm   : {percentile_out}")
    print(f"  side_by_side      : {compare_out}")

    if args.save_histogram:
        hist_out = outdir / f"{stem}_histogram_original_clahe_percentile.png"
        save_histograms(gray, clahe_img, percentile_img, hist_out)
        print(f"  histogram         : {hist_out}")


if __name__ == "__main__":
    main()
