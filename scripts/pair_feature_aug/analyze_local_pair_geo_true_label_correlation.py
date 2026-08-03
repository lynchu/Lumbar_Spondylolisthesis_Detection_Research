#!/usr/bin/env python3
"""Rank local pair geometry features by correlation with true labels on test data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_EXPERIMENTS = [
    Path("presence/experiments/local_pair_diagnostic/resnet50_local_pair_slip_aux_balacc_half_sacrum_cv5_th02"),
    Path("presence/experiments/local_pair_diagnostic/resnet50_local_pair_slip_aux_geo19_corner_angles_balacc_half_sacrum_cv5_th02"),
]

PAIR_FEATURES_15 = [
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

CORNER_ANGLE_FEATURES = [
    "min_top_left_corner_angle_deg",
    "min_top_right_corner_angle_deg",
    "min_bottom_right_corner_angle_deg",
    "min_bottom_left_corner_angle_deg",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset_half_sacrum")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--out-root",
        default="presence/experiments/local_pair_geo_true_label_feature_importance",
    )
    parser.add_argument("--local-experiments", nargs="*", default=[str(p) for p in LOCAL_EXPERIMENTS])
    return parser.parse_args()


def has_geo19_name(exp_dir: Path) -> bool:
    return "geo19" in exp_dir.name or "corner_angles" in exp_dir.name


def finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def corr_pair(x: pd.Series, y: pd.Series, method: str) -> float:
    frame = pd.DataFrame({"x": finite_numeric(x), "y": finite_numeric(y)}).dropna()
    if len(frame) < 3:
        return np.nan
    if frame["x"].std(ddof=0) <= 1e-12 or frame["y"].std(ddof=0) <= 1e-12:
        return np.nan
    return float(frame["x"].corr(frame["y"], method=method))


def summarize(df: pd.DataFrame, features: list[str], targets: list[str], group: str) -> pd.DataFrame:
    rows = []
    for target in targets:
        target_values = finite_numeric(df[target])
        for feature in features:
            values = finite_numeric(df[feature])
            valid = values.notna() & target_values.notna()
            pearson = corr_pair(values, target_values, "pearson")
            spearman = corr_pair(values, target_values, "spearman")
            pos_mean = neg_mean = np.nan
            if target == "pair_y_true":
                pos_mean = float(values[target_values == 1].mean())
                neg_mean = float(values[target_values == 0].mean())
            rows.append({
                "group": group,
                "target": target,
                "feature": feature,
                "n": int(valid.sum()),
                "positive_pair_n": int((target_values == 1).sum()) if target == "pair_y_true" else np.nan,
                "negative_pair_n": int((target_values == 0).sum()) if target == "pair_y_true" else np.nan,
                "pearson_corr_with_true_label": pearson,
                "abs_pearson_corr": abs(pearson) if math.isfinite(pearson) else np.nan,
                "spearman_corr_with_true_label": spearman,
                "abs_spearman_corr": abs(spearman) if math.isfinite(spearman) else np.nan,
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "mean_difference_pos_minus_neg": pos_mean - neg_mean if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan,
            })
    out = pd.DataFrame(rows)
    out = out.sort_values(["target", "abs_pearson_corr", "feature"], ascending=[True, False, True]).reset_index(drop=True)
    out.insert(0, "rank_by_abs_pearson_for_target", out.groupby("target").cumcount() + 1)
    return out


def analyze(exp_dir: Path, dataset_root: Path, split: str, out_root: Path) -> pd.DataFrame:
    pred_path = exp_dir / "test_analysis" / "local_pair_diagnostic_pair_predictions.csv"
    feature_name = "pair_crop_features_with_corner_angles.csv" if has_geo19_name(exp_dir) else "pair_crop_features.csv"
    feature_path = dataset_root / split / feature_name
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    pred = pd.read_csv(pred_path, dtype={"image_id": str, "pair_name": str})
    feat = pd.read_csv(feature_path, dtype={"image_id": str, "pair_name": str})
    features = PAIR_FEATURES_15 + (CORNER_ANGLE_FEATURES if has_geo19_name(exp_dir) else [])
    missing = [c for c in features if c not in feat.columns]
    if missing:
        raise ValueError(f"{feature_path} missing columns: {missing}")

    merged = pred.merge(
        feat[["image_id", "pair_name", *features]],
        on=["image_id", "pair_name"],
        how="inner",
        validate="one_to_one",
    )
    targets = [c for c in ["pair_y_true", "slip_target", "slip_grade_target"] if c in merged.columns]
    summary = summarize(merged, features, targets, "all_test_pairs")
    by_pair = pd.concat(
        [summarize(group, features, targets, pair_name) for pair_name, group in merged.groupby("pair_name", sort=True)],
        ignore_index=True,
    )
    summary.insert(0, "experiment", exp_dir.name)
    by_pair.insert(0, "experiment", exp_dir.name)

    outdir = out_root / exp_dir.name
    outdir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(outdir / "local_pair_true_label_geometry_rows.csv", index=False)
    summary.to_csv(outdir / "local_pair_geometry_true_label_correlation.csv", index=False)
    by_pair.to_csv(outdir / "local_pair_geometry_true_label_correlation_by_pair.csv", index=False)
    write_report(outdir, exp_dir, summary, len(merged))
    return summary


def write_report(outdir: Path, exp_dir: Path, summary: pd.DataFrame, rows_used: int):
    lines = []
    lines.append("Local Pair Geometry-True Label Correlation")
    lines.append("=" * 42)
    lines.append("")
    lines.append(f"Experiment: {exp_dir}")
    lines.append(f"Rows used: {rows_used}")
    lines.append("Main target: pair_y_true, the true pair-level presence label.")
    lines.append("Secondary targets: slip_target and slip_grade_target where available.")
    lines.append("")
    for target in summary["target"].drop_duplicates().tolist():
        sub = summary[summary["target"] == target].sort_values("abs_pearson_corr", ascending=False).head(10)
        lines.append(f"Top features for {target}:")
        for row in sub.itertuples(index=False):
            lines.append(
                f"- {row.feature}: pearson={row.pearson_corr_with_true_label:.6f}, "
                f"spearman={row.spearman_corr_with_true_label:.6f}, n={int(row.n)}"
            )
        lines.append("")
    lines.append("Files:")
    for path in sorted(outdir.glob("*.csv")):
        lines.append(f"- {path.name}")
    (outdir / "local_pair_geo_true_label_correlation_report.txt").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tables = []
    for raw in args.local_experiments:
        table = analyze(Path(raw), dataset_root, args.split, out_root)
        tables.append(table)
        print(f"[OK] {raw}")
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(out_root / "all_local_pair_geometry_true_label_correlations.csv", index=False)
    pair_true = combined[combined["target"] == "pair_y_true"].sort_values(
        ["experiment", "abs_pearson_corr"], ascending=[True, False]
    )
    pair_true.to_csv(out_root / "primary_pair_y_true_geometry_correlations.csv", index=False)
    print(f"[OK] wrote {out_root}")


if __name__ == "__main__":
    main()
