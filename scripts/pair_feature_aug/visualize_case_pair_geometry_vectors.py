#!/usr/bin/env python3
"""Visualize selected pair-level geometry vectors for one case across five pair crops."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PAIR_ORDER = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
LEVEL_ORDER = {"S1": 0, "L5": 1, "L4": 2, "L3": 3, "L2": 4, "L1": 5}
ID_TO_LEVEL = {0: "S1", 1: "L5", 2: "L4", 3: "L3", 4: "L2", 5: "L1"}

COLORS = {
    "upper_avg": (172, 112, 78),      # brown
    "lower_avg": (172, 112, 78),
    "interface": (220, 52, 135),      # magenta
    "dx_mid": (28, 160, 228),         # blue
    "dx_left": (255, 126, 36),        # orange
    "dx_right": (35, 170, 72),        # green
    "dy": (156, 86, 210),             # purple
    "guide": (0, 0, 0),               # black construction guides
    "point": (245, 245, 245),
    "text": (25, 25, 25),
}

FEATURE_CATEGORIES = {
    "dy": ["dy_slip_px", "abs_dy_slip_px"],
    "width": ["pair_interface_width", "pair_reference_width"],
    "slip_dx": [
        "dx_slip_px", "abs_dx_slip_px",
        "left_edge_dx_px", "abs_left_edge_dx_px",
        "right_edge_dx_px", "abs_right_edge_dx_px",
        "estimated_slip_percentage", "estimated_slip_percentage_outlier",
    ],
    "ratio": [
        "abs_dx_slip_ratio_wavg",
        "abs_left_edge_dx_ratio_wavg",
        "abs_right_edge_dx_ratio_wavg",
    ],
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", default="0001-Lat_clahe")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dataset-root", default="dataset_half_sacrum")
    ap.add_argument("--feature-csv", default=None)
    ap.add_argument("--position-csv", default=None)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--outdir", default="presence/experiments/geometry_feature_visualizations/case_0001_Lat_clahe_pair_vector_features")
    return ap.parse_args()


def safe_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def level_name(v) -> str:
    if isinstance(v, str):
        s = v.strip().upper()
        if s in LEVEL_ORDER:
            return s
    try:
        return ID_TO_LEVEL[int(float(v))]
    except Exception:
        return ""


def pt(row, prefix: str) -> np.ndarray:
    return np.array([safe_float(row[f"{prefix}_x"]), safe_float(row[f"{prefix}_y"])], dtype=float)


def unit(v: np.ndarray, fallback=(1.0, 0.0)) -> np.ndarray:
    if not np.isfinite(v).all():
        return np.array(fallback, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 1e-6:
        return np.array(fallback, dtype=float)
    return v / n


def draw_line(img, a, b, color, thickness=4, dots=True):
    a = tuple(np.round(a).astype(int))
    b = tuple(np.round(b).astype(int))
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    if dots:
        cv2.circle(img, a, max(thickness + 1, 4), color, -1, cv2.LINE_AA)
        cv2.circle(img, b, max(thickness + 1, 4), color, -1, cv2.LINE_AA)


def draw_dotted_line(img, a, b, color, thickness=3, gap=9):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length <= 1e-6:
        return
    direction = vec / length
    dist = 0.0
    while dist < length:
        start = a + direction * dist
        end = a + direction * min(dist + gap * 0.55, length)
        cv2.line(
            img,
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        dist += gap


def draw_arrow(img, a, b, color, thickness=4):
    a = tuple(np.round(a).astype(int))
    b = tuple(np.round(b).astype(int))
    cv2.arrowedLine(img, a, b, color, thickness, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(img, a, max(thickness + 1, 4), color, -1, cv2.LINE_AA)


def draw_label(img, text, xy, scale=0.48):
    x, y = map(int, np.round(xy))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    x = max(0, min(x, img.shape[1] - tw - 2 * pad - 1))
    y = max(th + pad, min(y, img.shape[0] - base - pad - 1))
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - th - pad), (x + tw + 2 * pad, y + base + pad), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, dst=img)
    cv2.putText(img, text, (x + pad, y), font, scale, COLORS["text"], thickness, cv2.LINE_AA)


def segment_centered(center, axis, length):
    axis = unit(axis)
    half = float(length) / 2.0
    return center - axis * half, center + axis * half


def draw_average_width(img, row, label, color, y_offset=0):
    left_mid = pt(row, "left_mid")
    right_mid = pt(row, "right_mid")
    center = (left_mid + right_mid) / 2.0 + np.array([0.0, y_offset])
    axis = right_mid - left_mid
    length = safe_float(row.get("average_width"), np.linalg.norm(axis))
    a, b = segment_centered(center, axis, length)
    draw_line(img, a, b, color, thickness=4)
    draw_label(img, f"{label} avg length={length:.1f}", (a + b) / 2 + np.array([-95, -8]), scale=0.43)


def draw_slip_parallel_dx(img, lower_point, upper_point, axis, color, offset_px):
    lower_point = np.asarray(lower_point, dtype=float)
    upper_point = np.asarray(upper_point, dtype=float)
    axis = unit(axis)
    perp = np.array([-axis[1], axis[0]])
    start = lower_point + perp * float(offset_px)
    dx = float(np.dot(upper_point - lower_point, axis))
    end = start + axis * dx
    dot_r = 6
    draw_dotted_line(img, lower_point, start, COLORS["guide"], thickness=2, gap=9)
    draw_dotted_line(img, upper_point, end, COLORS["guide"], thickness=2, gap=9)
    draw_line(img, start, end, color, thickness=4, dots=True)
    cv2.circle(img, tuple(np.round(lower_point).astype(int)), dot_r, color, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(upper_point).astype(int)), dot_r, color, -1, cv2.LINE_AA)


def segment_inside_image(img, start, end, margin=10):
    h, w = img.shape[:2]
    pts = np.vstack([start, end])
    return bool(
        np.all(pts[:, 0] >= margin)
        and np.all(pts[:, 0] <= w - margin)
        and np.all(pts[:, 1] >= margin)
        and np.all(pts[:, 1] <= h - margin)
    )


def clamp_segment_inside_image(img, start, end, margin=10):
    h, w = img.shape[:2]
    pts = np.vstack([start, end])
    shift = np.zeros(2, dtype=float)
    if pts[:, 0].min() < margin:
        shift[0] += margin - pts[:, 0].min()
    if pts[:, 0].max() > w - margin:
        shift[0] -= pts[:, 0].max() - (w - margin)
    if pts[:, 1].min() < margin:
        shift[1] += margin - pts[:, 1].min()
    if pts[:, 1].max() > h - margin:
        shift[1] -= pts[:, 1].max() - (h - margin)
    return start + shift, end + shift


def draw_dy(img, anchor, axis, dy_value):
    dy = float(dy_value)
    if not np.isfinite(dy):
        return
    axis = unit(axis)
    perp = np.array([-axis[1], axis[0]])
    anchor = np.asarray(anchor, dtype=float)

    start = end = None
    for offset in (55.0, 40.0, 25.0, 12.0, 0.0, -12.0, -25.0):
        side = anchor + axis * offset
        cand_start = side - perp * dy / 2.0
        cand_end = side + perp * dy / 2.0
        if segment_inside_image(img, cand_start, cand_end):
            start, end = cand_start, cand_end
            break

    if start is None or end is None:
        side = anchor
        start = side - perp * dy / 2.0
        end = side + perp * dy / 2.0
        start, end = clamp_segment_inside_image(img, start, end)

    draw_line(img, start, end, COLORS["dy"], thickness=4, dots=True)


def pair_rows(pos_df, crop_name):
    rows = pos_df[pos_df["crop_image_name"].eq(crop_name)].copy()
    rows["level_name"] = rows["lumbar_id"].apply(level_name)
    return {r["level_name"]: r for _, r in rows.iterrows()}


def visualize_pair(case_id, pair_name, img_path, pos_lookup, feat_row, out_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    upper_level = str(feat_row["upper_level"])
    lower_level = str(feat_row["lower_level"])
    upper = pos_lookup[upper_level]
    lower = pos_lookup[lower_level]

    upper_bottom_left = pt(upper, "bottom_left")
    upper_bottom_right = pt(upper, "bottom_right")
    upper_bottom_mid = pt(upper, "bottom_mid")
    lower_top_left = pt(lower, "top_left")
    lower_top_right = pt(lower, "top_right")
    lower_top_mid = pt(lower, "top_mid")
    axis = unit((upper_bottom_right - upper_bottom_left) + (lower_top_right - lower_top_left))
    interface_center = (upper_bottom_mid + lower_top_mid) / 2.0

    # Width feature: local pair interface only. Average-width context is intentionally omitted.
    interface_w = safe_float(feat_row["pair_interface_width"])
    ia, ib = segment_centered(interface_center, axis, interface_w)
    draw_line(img, ia, ib, COLORS["interface"], thickness=4)

    # Slip/dx features: black construction guides connect each endpoint to an
    # offset, interface-parallel colored segment that shows the AP slip amount.
    draw_slip_parallel_dx(img, lower_top_mid, upper_bottom_mid, axis, COLORS["dx_mid"], offset_px=-22)
    draw_slip_parallel_dx(img, lower_top_left, upper_bottom_left, axis, COLORS["dx_left"], offset_px=18)
    draw_slip_parallel_dx(img, lower_top_right, upper_bottom_right, axis, COLORS["dx_right"], offset_px=34)

    # dy feature: shifted to the right of the right-edge slip marker.
    right_anchor = (lower_top_right + upper_bottom_right) / 2.0
    draw_dy(img, right_anchor, axis, safe_float(feat_row["dy_slip_px"]))

    Image.fromarray(img).save(out_path)
    return img


def write_category_doc(outdir: Path):
    lines = ["# Default Geometry Feature Categories", ""]
    for cat, feats in FEATURE_CATEGORIES.items():
        lines.append(f"## {cat}")
        for f in feats:
            lines.append(f"- `{f}`")
        lines.append("")
    lines.extend([
        "Notes:",
        "- `slip_dx` contains signed displacement, absolute displacement, and slip percentage/outlier features because these all describe AP translation/slip magnitude or quality.",
        "- `ratio` contains normalized dx features where pixel displacement is divided by the local width denominator.",
        "- `width` contains raw width denominators/context features.",
        "- `dy` contains vertical displacement features.",
    ])
    (outdir / "default_geometry_feature_categories.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    root = Path(args.dataset_root) / args.split
    feature_csv = Path(args.feature_csv) if args.feature_csv else root / "pair_crop_features.csv"
    position_csv = Path(args.position_csv) if args.position_csv else root / "pair_crop_position.csv"
    image_root = Path(args.image_root) if args.image_root else root / "pair_crop"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(feature_csv)
    pos = pd.read_csv(position_csv, dtype={"crop_image_name": str})
    rows = feat[feat["image_id"].eq(args.case_id)].copy()
    if rows.empty:
        raise ValueError(f"No feature rows for case_id={args.case_id} in {feature_csv}")
    write_category_doc(outdir)

    panels = []
    manifest = []
    for pair_name in PAIR_ORDER:
        match = rows[rows["pair_name"].eq(pair_name)]
        if match.empty:
            continue
        feat_row = match.iloc[0]
        img_path = image_root / f"{args.case_id}_{pair_name}.png"
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        pos_lookup = pair_rows(pos, img_path.name)
        out_path = outdir / f"{args.case_id}_{pair_name}_geometry_vectors.png"
        panel = visualize_pair(args.case_id, pair_name, img_path, pos_lookup, feat_row, out_path)
        panels.append(panel)
        manifest.append({"case_id": args.case_id, "pair_name": pair_name, "output_png": str(out_path)})

    # Combined vertical overview.
    if panels:
        max_w = max(p.shape[1] for p in panels)
        gap = 16
        total_h = sum(p.shape[0] for p in panels) + gap * (len(panels) - 1)
        combined = np.full((total_h, max_w, 3), 255, dtype=np.uint8)
        y = 0
        for p_img in panels:
            x = (max_w - p_img.shape[1]) // 2
            combined[y:y + p_img.shape[0], x:x + p_img.shape[1]] = p_img
            y += p_img.shape[0] + gap
        combined_path = outdir / f"{args.case_id}_all_pairs_geometry_vectors.png"
        Image.fromarray(combined).save(combined_path)
        manifest.append({"case_id": args.case_id, "pair_name": "ALL", "output_png": str(combined_path)})

    pd.DataFrame(manifest).to_csv(outdir / f"{args.case_id}_geometry_vector_manifest.csv", index=False)
    print(f"[OK] wrote {outdir}")


if __name__ == "__main__":
    main()
