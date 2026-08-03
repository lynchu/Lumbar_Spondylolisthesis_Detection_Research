#!/usr/bin/env python3
"""Separate local-pair geometry importance by model head and true target.

Outputs correlation rankings for:
- location head output: pair_prob
- slip auxiliary head output: slip_pred
- true location label: pair_y_true
- true/geometry slip target: slip_target

The ranking is correlation-based, matching the existing pair_geo19 feature
analysis style. It is not causal attribution.
"""

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
        default="presence/experiments/model_geo_prediction_feature_importance/local_pair_head_separated",
    )
    parser.add_argument("--experiments", nargs="*", default=[str(p) for p in LOCAL_EXPERIMENTS])
    return parser.parse_args()


def finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def corr_pair(x: pd.Series, y: pd.Series, method: str) -> float:
    frame = pd.DataFrame({"x": finite_numeric(x), "y": finite_numeric(y)}).dropna()
    if len(frame) < 3:
        return np.nan
    if frame["x"].std(ddof=0) <= 1e-12 or frame["y"].std(ddof=0) <= 1e-12:
        return np.nan
    return float(frame["x"].corr(frame["y"], method=method))


def summarize(df: pd.DataFrame, features: list[str], target: str, analysis_name: str) -> pd.DataFrame:
    target_values = finite_numeric(df[target])
    rows = []
    for feature in features:
        values = finite_numeric(df[feature])
        valid = values.notna() & target_values.notna()
        pearson = corr_pair(values, target_values, "pearson")
        spearman = corr_pair(values, target_values, "spearman")
        rows.append({
            "analysis": analysis_name,
            "target": target,
            "feature": feature,
            "n": int(valid.sum()),
            "pearson_corr": pearson,
            "abs_pearson_corr": abs(pearson) if math.isfinite(pearson) else np.nan,
            "spearman_corr": spearman,
            "abs_spearman_corr": abs(spearman) if math.isfinite(spearman) else np.nan,
            "feature_mean": float(values[valid].mean()) if valid.any() else np.nan,
            "target_mean": float(target_values[valid].mean()) if valid.any() else np.nan,
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["abs_pearson_corr", "feature"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank_by_abs_pearson", np.arange(1, len(out) + 1))
    return out


def has_geo19_name(exp_dir: Path) -> bool:
    return "geo19" in exp_dir.name or "corner_angles" in exp_dir.name


def analyze_experiment(exp_dir: Path, dataset_root: Path, split: str, out_root: Path) -> pd.DataFrame:
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
    merged = pred.merge(
        feat[["image_id", "pair_name", *features]],
        on=["image_id", "pair_name"],
        how="inner",
        validate="one_to_one",
    )

    tables = []
    definitions = [
        ("location_head_prediction_pair_prob", merged, "pair_prob"),
        ("location_true_label_pair_y_true", merged, "pair_y_true"),
        ("slip_aux_head_prediction_slip_pred_all_pairs", merged, "slip_pred"),
    ]
    slip_mask = finite_numeric(merged["slip_loss_mask"]).eq(1) if "slip_loss_mask" in merged.columns else pd.Series(False, index=merged.index)
    nonoutlier_mask = finite_numeric(merged["slip_outlier"]).eq(0) if "slip_outlier" in merged.columns else pd.Series(True, index=merged.index)
    definitions.extend([
        ("slip_aux_head_prediction_slip_pred_positive_nonoutlier", merged[slip_mask].copy(), "slip_pred"),
        ("slip_true_target_positive_nonoutlier", merged[slip_mask].copy(), "slip_target"),
        ("slip_true_target_all_nonoutlier", merged[nonoutlier_mask].copy(), "slip_target"),
    ])

    outdir = out_root / exp_dir.name
    outdir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(outdir / "local_pair_prediction_geometry_rows.csv", index=False)

    for analysis_name, frame, target in definitions:
        if target not in frame.columns:
            continue
        table = summarize(frame, features, target, analysis_name)
        table.insert(0, "experiment", exp_dir.name)
        table.to_csv(outdir / f"{analysis_name}.csv", index=False)
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(outdir / "local_pair_head_separated_feature_importance_combined.csv", index=False)
    write_report(outdir, exp_dir, combined)
    return combined


def write_report(outdir: Path, exp_dir: Path, combined: pd.DataFrame):
    lines = []
    lines.append("Local Pair Head-Separated Geometry Feature Importance")
    lines.append("=" * 56)
    lines.append("")
    lines.append(f"Experiment: {exp_dir}")
    lines.append("")
    lines.append("Definition: importance = absolute Pearson correlation between feature value and target/output.")
    lines.append("Location model output target: pair_prob.")
    lines.append("Slip auxiliary model output target: slip_pred.")
    lines.append("True location target: pair_y_true.")
    lines.append("True slip target: slip_target, shown for positive non-outlier pairs and all non-outlier pairs.")
    lines.append("Note: estimated_slip_percentage is also the geometry-derived slip target, so true-slip correlation with that feature is expected to be near/perfectly 1.")
    lines.append("")
    for analysis_name in combined["analysis"].drop_duplicates().tolist():
        sub = combined[combined["analysis"] == analysis_name].sort_values("abs_pearson_corr", ascending=False).head(10)
        n = int(sub["n"].iloc[0]) if len(sub) else 0
        lines.append(f"{analysis_name} (n={n}):")
        for row in sub.itertuples(index=False):
            lines.append(
                f"- {row.feature}: pearson={row.pearson_corr:.6f}, "
                f"spearman={row.spearman_corr:.6f}"
            )
        lines.append("")
    lines.append("Files:")
    for path in sorted(outdir.glob("*.csv")):
        lines.append(f"- {path.name}")
    (outdir / "local_pair_head_separated_feature_importance_report.txt").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_tables = []
    for raw in args.experiments:
        exp_dir = Path(raw)
        table = analyze_experiment(exp_dir, dataset_root, args.split, out_root)
        all_tables.append(table)
        print(f"[OK] {exp_dir} -> {out_root / exp_dir.name}")
    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(out_root / "all_local_pair_head_separated_feature_importance.csv", index=False)
    print(f"[OK] combined -> {out_root}")


if __name__ == "__main__":
    main()
