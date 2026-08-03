#!/usr/bin/env python3
"""
visualize_pair_geometry_features.py

Visualize 15 default pair-crop geometric features for lumbar spondylolisthesis analysis.

Designed for your pair_crop_features.csv columns:

  image_id, base_image_id, augmentation, is_augmented, position_image_id,
  crop_image_name, pair_name, upper_level, lower_level,
  abs_dx_slip_px, abs_dx_slip_ratio_wavg,
  estimated_slip_percentage_raw, estimated_slip_percentage,
  estimated_slip_percentage_outlier,
  abs_left_edge_dx_px, abs_left_edge_dx_ratio_wavg,
  abs_right_edge_dx_px, abs_right_edge_dx_ratio_wavg,
  dx_slip_px, left_edge_dx_px, right_edge_dx_px,
  dy_slip_px, abs_dy_slip_px,
  pair_interface_width, pair_reference_width,
  upper_bottom_length, lower_top_length,
  upper_average_width, lower_average_width

Default behavior:
  - Case grouping column: image_id
  - Pair level column: pair_name
  - Image path column: crop_image_name
  - Visualize first 5 case-level images
  - Each case contains up to 5 pair crops:
      L1-L2, L2-L3, L3-L4, L4-L5, L5-S1

Example:
  python ./presence/scripts/pair_feature_aug/visualize_pair_geometry_features.py \
    --csv dataset_half_sacrum/train/pair_crop_features.csv \
    --image_root dataset_half_sacrum/train/pair_crop \
    --output_dir ./presence/experiments/geometry_feature_visualizations \
    --num_cases 5
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PAIR_ORDER = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]

DEFAULT_GEOMETRY_FEATURES_15 = [
    "estimated_slip_percentage",
    "estimated_slip_percentage_outlier",
    "abs_dx_slip_px",
    "abs_dx_slip_ratio_wavg",
    "abs_left_edge_dx_px",
    "abs_left_edge_dx_ratio_wavg",
    "abs_right_edge_dx_px",
    "abs_right_edge_dx_ratio_wavg",
    "dx_slip_px",
    "left_edge_dx_px",
    "right_edge_dx_px",
    "dy_slip_px",
    "abs_dy_slip_px",
    "pair_interface_width",
    "pair_reference_width",
]

OPTIONAL_SUPPORT_FEATURES = [
    "estimated_slip_percentage_raw",
    "upper_bottom_length",
    "lower_top_length",
    "upper_average_width",
    "lower_average_width",
]

# Text-label background transparency.
# Lower alpha makes the white label boxes more transparent, so endpoint dots and
# anatomy behind the text remain visible.
TEXT_BBOX_ALPHA = 0.48
SUMMARY_BBOX_ALPHA = 0.52

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_IMAGE_INDEX_CACHE: Dict[str, Tuple[Dict[str, List[Path]], List[Path]]] = {}

FORMULAS = {
    "estimated_slip_percentage": (
        "Slip percentage is computed upstream from the slip displacement divided by a "
        "reference vertebral width or a corrected/projected width. In your shown row, "
        "estimated_slip_percentage is not exactly 100 * abs_dx_slip_px / pair_reference_width, "
        "so the feature extractor likely used continuous corner coordinates or another "
        "denominator before rounding the exported scalar features."
    ),
    "estimated_slip_percentage_raw": (
        "Raw slip percentage before clipping, smoothing, or outlier handling."
    ),
    "estimated_slip_percentage_outlier": (
        "Quality-control flag. Usually 1 means the estimated slip percentage is implausible "
        "or failed a preprocessing validity criterion; 0 means accepted."
    ),
    "abs_dx_slip_px": "abs(dx_slip_px).",
    "abs_dx_slip_ratio_wavg": (
        "abs_dx_slip_px / max(pair_interface_width, eps) in the provided example. "
        "The name 'wavg' may refer to the interface/weighted width used by the extractor."
    ),
    "abs_left_edge_dx_px": "abs(left_edge_dx_px).",
    "abs_left_edge_dx_ratio_wavg": (
        "abs_left_edge_dx_px / max(pair_interface_width, eps) in the provided example."
    ),
    "abs_right_edge_dx_px": "abs(right_edge_dx_px).",
    "abs_right_edge_dx_ratio_wavg": (
        "abs_right_edge_dx_px / max(pair_interface_width, eps) in the provided example."
    ),
    "dx_slip_px": (
        "Signed horizontal slip displacement. Positive/negative direction depends on the "
        "coordinate convention used during feature extraction."
    ),
    "left_edge_dx_px": (
        "Signed horizontal displacement measured at one edge of the pair interface."
    ),
    "right_edge_dx_px": (
        "Signed horizontal displacement measured at the opposite edge of the pair interface."
    ),
    "dy_slip_px": (
        "Signed vertical displacement between paired anatomical reference/interface points."
    ),
    "abs_dy_slip_px": "abs(dy_slip_px).",
    "pair_interface_width": (
        "Local interface width of the adjacent vertebral pair, often derived from "
        "upper_bottom_length and/or lower_top_length."
    ),
    "pair_reference_width": (
        "Reference width used as the denominator for normalized displacement features and/or "
        "slip-percentage estimation. In the provided feature-generation script, the recommended "
        "width_mode='interface' sets pair_reference_width = mean(upper_bottom_length, lower_top_length), "
        "i.e., the local disc-interface width. This avoids using S1 average width for L5-S1, because "
        "the sacrum's triangular shape can distort the denominator. If another width_mode was used, "
        "pair_reference_width may instead be lower_top_length, lower_average_width, or an S1-specific "
        "top-endplate width."
    ),
    "upper_bottom_length": "Length of the upper vertebra bottom/endplate segment.",
    "lower_top_length": "Length of the lower vertebra top/endplate segment.",
    "upper_average_width": "Average width estimate of the upper vertebra.",
    "lower_average_width": "Average width estimate of the lower vertebra.",
}


def safe_float(x, default: float = 0.0) -> float:
    if x is None or pd.isna(x):
        return default
    try:
        return float(x)
    except Exception:
        return default


def format_value(v, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "NA"
    try:
        fv = float(v)
        if abs(fv - int(fv)) < 1e-9:
            return str(int(fv))
        return f"{fv:.{digits}f}"
    except Exception:
        return str(v)


def normalize_pair_level(x: object) -> str:
    s = str(x).strip().upper().replace("_", "-").replace(" ", "")
    replacements = {
        "L1L2": "L1-L2",
        "L2L3": "L2-L3",
        "L3L4": "L3-L4",
        "L4L5": "L4-L5",
        "L5S1": "L5-S1",
    }
    return replacements.get(s, s)

def norm_key(x: object) -> str:
    """Normalize a string for fuzzy filename matching."""
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def image_index(image_root: Path) -> Tuple[Dict[str, List[Path]], List[Path]]:
    """
    Build a small recursive filename index once per image_root.

    This fixes cases where crop_image_name stores the lumbar crop name, while
    the actual pair crop image is saved under a pair-specific folder or filename.
    """
    key = str(image_root.resolve())
    if key in _IMAGE_INDEX_CACHE:
        return _IMAGE_INDEX_CACHE[key]

    by_name: Dict[str, List[Path]] = defaultdict(list)
    all_images: List[Path] = []
    if image_root.exists():
        for p in image_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                by_name[p.name].append(p)
                all_images.append(p)

    _IMAGE_INDEX_CACHE[key] = (by_name, all_images)
    return by_name, all_images


def row_id_variants(row: Optional[pd.Series]) -> List[str]:
    """Generate possible image-id stems from CSV metadata."""
    if row is None:
        return []

    vals = []
    for col in ["image_id", "base_image_id", "position_image_id"]:
        if col in row and not pd.isna(row[col]):
            vals.append(str(row[col]))

    aug = str(row.get("augmentation", "")).strip()
    base = str(row.get("base_image_id", "")).strip()
    pos = str(row.get("position_image_id", "")).strip()

    if aug and aug.lower() not in {"original", "nan", "none"}:
        if base:
            vals.append(f"{base}_{aug}")
        if pos:
            vals.append(f"{pos}_{aug}")

    # De-duplicate while preserving order.
    out = []
    seen = set()
    for v in vals:
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def pair_variants_from_row(row: Optional[pd.Series]) -> List[str]:
    """Generate possible pair-name strings used in filenames/folders."""
    vals = []
    if row is not None:
        if "pair_name" in row and not pd.isna(row["pair_name"]):
            pair = normalize_pair_level(row["pair_name"])
            vals.extend([pair, pair.replace("-", "_"), pair.replace("-", "")])

        if "upper_level" in row and "lower_level" in row:
            if not pd.isna(row["upper_level"]) and not pd.isna(row["lower_level"]):
                u = str(row["upper_level"]).strip().upper()
                l = str(row["lower_level"]).strip().upper()
                vals.extend([f"{u}-{l}", f"{u}_{l}", f"{u}{l}"])

    out = []
    seen = set()
    for v in vals:
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def filename_variants(raw_path: object, row: Optional[pd.Series]) -> List[str]:
    """
    Generate likely pair-crop filenames.

    This handles the common mismatch:
      crop_image_name = 0001-Lat_hflip_lumbar.png
    but actual pair crop may be:
      0001-Lat_hflip_L4-L5.png
      0001-Lat_hflip_L4_L5.png
      0001-Lat_hflip_lumbar_L4-L5.png
      or be stored in pair-specific folders.
    """
    raw = Path(str(raw_path))
    suffix = raw.suffix if raw.suffix else ".png"
    raw_name = raw.name
    stem = raw.stem

    ids = row_id_variants(row)
    pairs = pair_variants_from_row(row)

    names = [raw_name]

    # Replace the lumbar marker with pair name.
    for pair in pairs:
        for marker in ["_lumbar", "-lumbar", "lumbar"]:
            if marker in stem:
                names.append(stem.replace(marker, f"_{pair}") + suffix)
        names.append(f"{stem}_{pair}{suffix}")

    # Construct from image ids.
    for img_id in ids:
        for pair in pairs:
            names.extend([
                f"{img_id}_{pair}{suffix}",
                f"{img_id}-{pair}{suffix}",
                f"{img_id}_pair_{pair}{suffix}",
                f"{img_id}_lumbar_{pair}{suffix}",
            ])

    # If raw filename includes an extension but actual image uses another extension,
    # generate same stems with common image suffixes.
    expanded = []
    for n in names:
        p = Path(n)
        expanded.append(n)
        for ext in _IMAGE_SUFFIXES:
            expanded.append(p.with_suffix(ext).name)

    out = []
    seen = set()
    for n in expanded:
        if n and n not in seen:
            out.append(n)
            seen.add(n)
    return out


def candidate_image_paths(
    raw_path: object,
    image_root: Optional[str | Path],
    row: Optional[pd.Series] = None,
) -> List[Path]:
    """
    Robust path resolver.

    The CSV may store crop_image_name as a lumbar crop name, for example:
      0001-Lat_hflip_lumbar.png

    However, pair-crop images are often saved as pair-specific files/folders, for example:
      pair_crop/L4-L5/0001-Lat_hflip_L4-L5.png
      pair_crop/L4_L5/0001-Lat_hflip_L4_L5.png
      pair_crop/0001-Lat_hflip_L4-L5.png
      pair_crop/0001-Lat_hflip_lumbar_L4-L5.png

    This function generates direct candidate paths. load_image() also performs
    indexed exact/fuzzy recursive matching under image_root if direct matching fails.
    """
    raw = Path(str(raw_path))
    candidates = [raw]

    if image_root is not None:
        root = Path(image_root)
        names = filename_variants(raw_path, row)
        pairs = pair_variants_from_row(row)
        ids = row_id_variants(row)

        for name in names:
            candidates.append(root / name)

            for pair in pairs:
                candidates.append(root / pair / name)

            for img_id in ids:
                candidates.append(root / img_id / name)
                for pair in pairs:
                    candidates.append(root / img_id / pair / name)
                    candidates.append(root / pair / img_id / name)

        if row is not None:
            for col in ["pair_name", "base_image_id", "image_id", "position_image_id"]:
                if col in row and not pd.isna(row[col]):
                    for name in names:
                        candidates.append(root / str(row[col]) / name)

            if "upper_level" in row and "lower_level" in row:
                pair = f"{row['upper_level']}-{row['lower_level']}"
                for name in names:
                    candidates.append(root / pair / name)

    seen = set()
    unique = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def load_image(raw_path: object, image_root: Optional[str | Path], row: Optional[pd.Series] = None) -> np.ndarray:
    tried = candidate_image_paths(raw_path, image_root, row)

    # 1) Direct candidate paths.
    for p in tried:
        if p.exists():
            return np.asarray(Image.open(p).convert("RGB"))

    # 2) Recursive exact filename matching under image_root.
    if image_root is not None:
        root = Path(image_root)
        by_name, all_images = image_index(root)
        for name in filename_variants(raw_path, row):
            if name in by_name:
                return np.asarray(Image.open(by_name[name][0]).convert("RGB"))

        # 3) Fuzzy matching by image-id and pair-name tokens.
        id_keys = [norm_key(x) for x in row_id_variants(row)]
        pair_keys = [norm_key(x) for x in pair_variants_from_row(row)]
        id_keys = [x for x in id_keys if x]
        pair_keys = [x for x in pair_keys if x]

        scored = []
        for p in all_images:
            n = norm_key(p.stem)
            parent = norm_key(p.parent.name)
            id_match = any(k in n for k in id_keys)
            pair_match = any(k in n or k in parent for k in pair_keys)
            if id_match and pair_match:
                score = 0
                # Prefer pair name in filename over only parent folder.
                score += 2 if any(k in n for k in pair_keys) else 0
                # Prefer exact image id string in filename.
                score += 2 if any(k == n or k in n for k in id_keys) else 0
                scored.append((score, p))

        if scored:
            scored.sort(key=lambda x: (-x[0], len(str(x[1]))))
            return np.asarray(Image.open(scored[0][1]).convert("RGB"))

    msg = "Cannot find pair crop image. This usually means crop_image_name is a lumbar-crop filename, not a pair-crop filename. Tried direct paths such as:\n"
    msg += "\n".join(f"  - {p}" for p in tried[:18])
    if len(tried) > 18:
        msg += f"\n  ... and {len(tried) - 18} more"
    if image_root is not None:
        msg += (
            "\n\nAlso searched recursively under image_root using image_id/base_image_id/"
            "position_image_id + pair_name fuzzy matching."
        )
    raise FileNotFoundError(msg)


def draw_endpoint_dots(ax, points, size: float = 70.0) -> None:
    """
    Draw large endpoint dots for geometry vectors/rulers.

    size=70 is moderately large but smaller than the previous 120 setting.
    Increase to 90 or 110 if you need stronger visibility for slides.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=size, marker="o", edgecolors="black", linewidths=0.8, zorder=6)


def draw_arrow(ax, x0, y0, x1, y1, label: str, fontsize: int = 7, lw: float = 1.8) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=lw, shrinkA=0, shrinkB=0),
        zorder=5,
    )
    draw_endpoint_dots(ax, [(x0, y0), (x1, y1)], size=70.0)
    ax.text(
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        label,
        fontsize=fontsize,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.45", alpha=TEXT_BBOX_ALPHA),
        zorder=7,
    )


def draw_ruler(ax, x0, y0, x1, y1, label: str, fontsize: int = 7, lw: float = 2.0) -> None:
    ax.plot([x0, x1], [y0, y1], lw=lw, zorder=5)
    tick = 7
    ax.plot([x0, x0], [y0 - tick, y0 + tick], lw=1.5, zorder=5)
    ax.plot([x1, x1], [y1 - tick, y1 + tick], lw=1.5, zorder=5)
    draw_endpoint_dots(ax, [(x0, y0), (x1, y1)], size=70.0)
    ax.text(
        (x0 + x1) / 2,
        y0 - 10,
        label,
        fontsize=fontsize,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.45", alpha=TEXT_BBOX_ALPHA),
        zorder=7,
    )


def make_feature_table_text(row: pd.Series) -> str:
    lines = []
    for feat in DEFAULT_GEOMETRY_FEATURES_15:
        lines.append(f"{feat}: {format_value(row.get(feat, np.nan))}")

    optional_existing = [c for c in OPTIONAL_SUPPORT_FEATURES if c in row.index]
    if optional_existing:
        lines.append("-" * 34)
        for feat in optional_existing:
            lines.append(f"{feat}: {format_value(row.get(feat, np.nan))}")
    return "\n".join(lines)


def draw_pair_overlay(ax, img: np.ndarray, row: pd.Series, pair_level: str) -> None:
    """
    Draw scalar geometry features on a pair crop.

    The overlay is schematic because the CSV contains scalar features, not the exact
    source/target corner coordinates for each vector. Arrow lengths reflect feature values.
    """
    ax.imshow(img)
    ax.set_title(pair_level, fontsize=11, fontweight="bold")
    ax.axis("off")

    h, w = img.shape[:2]

    dx = safe_float(row.get("dx_slip_px", 0.0))
    left_dx = safe_float(row.get("left_edge_dx_px", 0.0))
    right_dx = safe_float(row.get("right_edge_dx_px", 0.0))
    dy = safe_float(row.get("dy_slip_px", 0.0))
    abs_dy = safe_float(row.get("abs_dy_slip_px", abs(dy)))
    interface_w = safe_float(row.get("pair_interface_width", w * 0.55))
    reference_w = safe_float(row.get("pair_reference_width", w * 0.60))
    slip_pct = safe_float(row.get("estimated_slip_percentage", np.nan), np.nan)
    slip_raw = safe_float(row.get("estimated_slip_percentage_raw", np.nan), np.nan)

    max_disp = max(1.0, abs(dx), abs(left_dx), abs(right_dx), abs(dy), abs_dy)
    scale = min(w * 0.28 / max_disp, 2.0)

    x_mid = w * 0.50
    y_mid = h * 0.50

    # Keep the main dx_slip vector around the center because it represents the
    # overall slip estimate of the pair.
    draw_arrow(
        ax,
        x_mid - dx * scale / 2,
        y_mid,
        x_mid + dx * scale / 2,
        y_mid,
        f"dx_slip={format_value(dx)} px",
    )

    # Draw left_edge_dx_px and right_edge_dx_px near their corresponding
    # edge/corner regions instead of centering them. These are schematic
    # anchors because the CSV stores scalar features, not exact corner points.
    left_anchor_x = w * 0.22
    right_anchor_x = w * 0.70
    edge_y = h * 0.40

    draw_arrow(
        ax,
        left_anchor_x,
        edge_y,
        left_anchor_x + left_dx * scale,
        edge_y,
        f"left_dx={format_value(left_dx)}",
    )
    draw_arrow(
        ax,
        right_anchor_x,
        edge_y,
        right_anchor_x + right_dx * scale,
        edge_y,
        f"right_dx={format_value(right_dx)}",
    )

    draw_arrow(
        ax,
        w * 0.83,
        y_mid - dy * scale / 2,
        w * 0.83,
        y_mid + dy * scale / 2,
        f"dy={format_value(dy)}",
    )
    draw_arrow(
        ax,
        w * 0.91,
        y_mid - abs_dy * scale / 2,
        w * 0.91,
        y_mid + abs_dy * scale / 2,
        f"|dy|={format_value(abs_dy)}",
    )

    ref_len = min(max(reference_w, 1.0), w * 0.82)
    int_len = min(max(interface_w, 1.0), w * 0.82)
    draw_ruler(
        ax,
        w * 0.50 - ref_len / 2,
        h * 0.15,
        w * 0.50 + ref_len / 2,
        h * 0.15,
        f"reference_width={format_value(reference_w)}",
    )
    draw_ruler(
        ax,
        w * 0.50 - int_len / 2,
        h * 0.85,
        w * 0.50 + int_len / 2,
        h * 0.85,
        f"interface_width={format_value(interface_w)}",
    )

    outlier = row.get("estimated_slip_percentage_outlier", "NA")
    summary = (
        f"slip%={format_value(slip_pct)}"
        + (f" | raw={format_value(slip_raw)}" if not pd.isna(slip_raw) else "")
        + f" | outlier={format_value(outlier)}"
    )
    ax.text(
        0.02,
        0.02,
        summary,
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.45", alpha=SUMMARY_BBOX_ALPHA),
    )


def get_first_case_ids(df: pd.DataFrame, case_col: str, num_cases: int) -> List[object]:
    seen = []
    seen_set = set()
    for cid in df[case_col].tolist():
        if cid not in seen_set:
            seen.append(cid)
            seen_set.add(cid)
        if len(seen) >= num_cases:
            break
    return seen


def plot_case(
    case_df: pd.DataFrame,
    case_id: object,
    pair_col: str,
    path_col: str,
    output_dir: Path,
    image_root: Optional[str | Path],
) -> Path:
    case_df = case_df.copy()
    case_df["_pair_norm"] = case_df[pair_col].map(normalize_pair_level)
    rank = {p: i for i, p in enumerate(PAIR_ORDER)}
    case_df["_rank"] = case_df["_pair_norm"].map(lambda x: rank.get(x, 999))
    case_df = case_df.sort_values("_rank")
    pair_to_row = {r["_pair_norm"]: r for _, r in case_df.iterrows()}

    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(23, 9.5),
        gridspec_kw={"height_ratios": [3.1, 1.45]},
    )
    fig.suptitle(
        f"Pair-Crop Geometry Feature Visualization | case_id = {case_id}",
        fontsize=16,
        fontweight="bold",
    )

    for i, pair in enumerate(PAIR_ORDER):
        ax_img = axes[0, i]
        ax_txt = axes[1, i]
        ax_txt.axis("off")

        if pair not in pair_to_row:
            ax_img.axis("off")
            ax_img.set_title(f"{pair}\nMISSING", fontsize=11, fontweight="bold")
            ax_txt.text(0.5, 0.5, "No row found", ha="center", va="center", fontsize=10)
            continue

        row = pair_to_row[pair]
        try:
            img = load_image(row[path_col], image_root=image_root, row=row)
            draw_pair_overlay(ax_img, img, row, pair)
        except Exception as exc:
            ax_img.axis("off")
            ax_img.set_title(f"{pair}\nImage load failed", fontsize=11, fontweight="bold")
            ax_img.text(
                0.0,
                0.5,
                str(exc),
                transform=ax_img.transAxes,
                ha="left",
                va="center",
                fontsize=7,
                wrap=True,
            )

        ax_txt.text(
            0.0,
            1.0,
            make_feature_table_text(row),
            ha="left",
            va="top",
            fontsize=7.1,
            family="monospace",
            transform=ax_txt.transAxes,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id))
    out_path = output_dir / f"case_{safe_case_id}_geometry_features.png"
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    return out_path


def write_formula_markdown(output_dir: Path) -> Path:
    lines = [
        "# Pair-Crop Geometry Feature Formulas",
        "",
        "## Directly visualized features",
        "",
        "The following features are drawn as arrows or rulers in each pair-crop panel:",
        "",
        "- `dx_slip_px`: signed horizontal slip vector.",
        "- `left_edge_dx_px`: signed horizontal edge displacement.",
        "- `right_edge_dx_px`: signed horizontal edge displacement.",
        "- `dy_slip_px`: signed vertical displacement.",
        "- `abs_dy_slip_px`: absolute vertical displacement.",
        "- `pair_interface_width`: local interface width ruler.",
        "- `pair_reference_width`: reference width ruler.",
        "",
        "## Formula / interpretation list",
        "",
    ]

    for feat in DEFAULT_GEOMETRY_FEATURES_15 + OPTIONAL_SUPPORT_FEATURES:
        lines.append(f"### `{feat}`")
        lines.append("")
        lines.append(FORMULAS.get(feat, "No formula registered."))
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- The overlay is schematic unless your CSV includes exact corner coordinates for arrow anchors.",
        "- If exact vertebral corner coordinates are available, replace the schematic anchors in `draw_pair_overlay()`.",
        "- In your provided example, `*_ratio_wavg` equals displacement divided by `pair_interface_width`.",
        "- `pair_reference_width` may be a case-level normalization width if it is constant for all pairs.",
        "  Check the original feature extraction script to identify whether it comes from max/median vertebral width,",
        "  lower-level width, upper-level width, or a projected corner-based width.",
        "",
    ]

    out_path = output_dir / "geometry_feature_formulas.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def validate_columns(df: pd.DataFrame, case_col: str, pair_col: str, path_col: str) -> None:
    required = [case_col, pair_col, path_col]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(
            "Missing required columns: "
            f"{missing_required}\nAvailable columns: {list(df.columns)}"
        )

    missing_features = [c for c in DEFAULT_GEOMETRY_FEATURES_15 if c not in df.columns]
    if missing_features:
        print("[Warning] Missing geometry feature columns:")
        for c in missing_features:
            print(f"  - {c}")
        print("The script will still run and mark missing values as MISSING/0 where needed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize 15 geometric features for first N case-level pair-crop images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, type=str, help="pair_crop_features.csv path.")
    parser.add_argument("--output_dir", default="geometry_feature_visualizations", type=str)
    parser.add_argument("--image_root", default=None, type=str, help="Root folder for crop_image_name.")
    parser.add_argument("--num_cases", default=5, type=int)
    parser.add_argument("--case_col", default="image_id", type=str)
    parser.add_argument("--pair_col", default="pair_name", type=str)
    parser.add_argument("--path_col", default="crop_image_name", type=str)
    parser.add_argument(
        "--text_alpha",
        default=TEXT_BBOX_ALPHA,
        type=float,
        help="Transparency of arrow/ruler text label background. Lower = more transparent.",
    )
    parser.add_argument(
        "--summary_alpha",
        default=SUMMARY_BBOX_ALPHA,
        type=float,
        help="Transparency of the slip summary text background. Lower = more transparent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global TEXT_BBOX_ALPHA, SUMMARY_BBOX_ALPHA
    TEXT_BBOX_ALPHA = max(0.0, min(1.0, float(args.text_alpha)))
    SUMMARY_BBOX_ALPHA = max(0.0, min(1.0, float(args.summary_alpha)))

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    validate_columns(df, args.case_col, args.pair_col, args.path_col)

    case_ids = get_first_case_ids(df, args.case_col, args.num_cases)
    if args.image_root is not None:
        root = Path(args.image_root)
        if root.exists():
            _, indexed = image_index(root)
            print(f"[Info] Indexed {len(indexed)} image files under image_root='{root}' for robust path matching.")
        else:
            print(f"[Warning] image_root does not exist: {root}")

    print(f"[Info] Visualizing first {len(case_ids)} case(s) using case_col='{args.case_col}':")
    for cid in case_ids:
        print(f"  - {cid}")

    saved = []
    for case_id in case_ids:
        case_df = df[df[args.case_col] == case_id]
        out_path = plot_case(
            case_df=case_df,
            case_id=case_id,
            pair_col=args.pair_col,
            path_col=args.path_col,
            output_dir=output_dir,
            image_root=args.image_root,
        )
        saved.append(out_path)
        print(f"[Saved] {out_path}")

    formula_path = write_formula_markdown(output_dir)
    print(f"[Saved] {formula_path}")

    print("\nDone. Generated files:")
    for p in saved:
        print(f"  - {p}")
    print(f"  - {formula_path}")


if __name__ == "__main__":
    main()