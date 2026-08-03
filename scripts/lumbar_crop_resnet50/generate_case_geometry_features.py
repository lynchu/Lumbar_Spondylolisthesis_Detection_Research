#!/usr/bin/env python3
"""Aggregate pair-crop geometry features into case-level lumbar-crop features.

The case-level vector includes slip/outlier, horizontal displacement ratio features,
and optional vertebral corner-angle features when they are present in the pair CSV.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAIR_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
BASE_FEATURES = [
    "estimated_slip_percentage",
    "estimated_slip_percentage_outlier",
    "abs_dx_slip_ratio_wavg",
    "abs_left_edge_dx_ratio_wavg",
    "abs_right_edge_dx_ratio_wavg",
    "min_top_left_corner_angle_deg",
    "min_top_right_corner_angle_deg",
    "min_bottom_right_corner_angle_deg",
    "min_bottom_left_corner_angle_deg",
]
ANGLE_FEATURES = [
    "min_top_left_corner_angle_deg",
    "min_top_right_corner_angle_deg",
    "min_bottom_right_corner_angle_deg",
    "min_bottom_left_corner_angle_deg",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset_half_sacrum")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--input-name", default="pair_crop_features.csv")
    parser.add_argument("--output-name", default="case_geometry_features.csv")
    return parser.parse_args()


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def aggregate_case(group: pd.DataFrame) -> dict:
    row = {"image_id": str(group["image_id"].iloc[0])}

    slip = safe_num(group["estimated_slip_percentage"]).fillna(0.0)
    outlier = safe_num(group["estimated_slip_percentage_outlier"]).fillna(0.0)
    dx_ratio = safe_num(group["abs_dx_slip_ratio_wavg"]).fillna(0.0)
    left_ratio = safe_num(group["abs_left_edge_dx_ratio_wavg"]).fillna(0.0)
    right_ratio = safe_num(group["abs_right_edge_dx_ratio_wavg"]).fillna(0.0)
    angle_values = []
    for feature_name in ANGLE_FEATURES:
        if feature_name in group.columns:
            angle_values.append(safe_num(group[feature_name]))
    if angle_values:
        all_angles = pd.concat(angle_values, ignore_index=True).dropna()
    else:
        all_angles = pd.Series(dtype=float)

    row.update({
        "case_max_slip_pct": float(slip.max()),
        "case_mean_slip_pct": float(slip.mean()),
        "case_std_slip_pct": float(slip.std(ddof=0)),
        "case_outlier_pair_count": float(outlier.sum()),
        "case_has_slip_outlier": float(outlier.max()),
        "case_max_abs_dx_ratio": float(dx_ratio.max()),
        "case_mean_abs_dx_ratio": float(dx_ratio.mean()),
        "case_max_left_edge_dx_ratio": float(left_ratio.max()),
        "case_max_right_edge_dx_ratio": float(right_ratio.max()),
        "case_mean_edge_dx_ratio": float(pd.concat([left_ratio, right_ratio], ignore_index=True).mean()),
        "case_min_corner_angle_deg": float(all_angles.min()) if len(all_angles) else 0.0,
        "case_mean_corner_angle_deg": float(all_angles.mean()) if len(all_angles) else 0.0,
        "case_std_corner_angle_deg": float(all_angles.std(ddof=0)) if len(all_angles) else 0.0,
    })

    by_pair = {str(r["pair_name"]): r for _, r in group.iterrows()}
    for pair_name in PAIR_NAMES:
        prefix = pair_name.replace("-", "_")
        pair_row = by_pair.get(pair_name)
        if pair_row is None:
            for feature_name in BASE_FEATURES:
                row[f"{prefix}_{feature_name}"] = 0.0
            row[f"{prefix}_present"] = 0.0
            continue

        for feature_name in BASE_FEATURES:
            row[f"{prefix}_{feature_name}"] = float(pd.to_numeric(pd.Series([pair_row.get(feature_name)]), errors="coerce").fillna(0.0).iloc[0])
        row[f"{prefix}_present"] = 1.0

    return row


def process_split(dataset_root: Path, split: str, input_name: str, output_name: str):
    in_path = dataset_root / split / input_name
    out_path = dataset_root / split / output_name
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path)
    required = {"image_id", "pair_name", *BASE_FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{in_path} is missing required columns: {missing}")

    rows = [aggregate_case(g.sort_values("pair_name")) for _, g in df.groupby("image_id", sort=False)]
    out = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] {split}: {len(out)} rows -> {out_path}")
    print(f"[OK] {split}: {len(out.columns) - 1} geometry features")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    for split in args.splits:
        process_split(dataset_root, split, args.input_name, args.output_name)


if __name__ == "__main__":
    main()
