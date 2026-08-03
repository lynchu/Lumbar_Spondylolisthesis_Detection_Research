#!/usr/bin/env python3
"""Rank geometry features by correlation with trained model predictions.

This is correlation analysis, not causal feature attribution. It answers:
"When this trained model assigns a higher probability/output, which measured
geometry features tend to move with that output?"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


GLOBAL_EXPERIMENTS = [
    Path("presence/experiments/global_screening_stage2/resnet50_lumbarCropHalfSacrum_geo_th03_b32_bce_balacc_cv5"),
    Path("presence/experiments/global_screening_stage2/resnet50_lumbarCropHalfSacrum_geo19_corner_angles_th03_b32_bce_balacc_cv5"),
]

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
        default="presence/experiments/model_geo_prediction_feature_importance",
    )
    parser.add_argument("--global-experiments", nargs="*", default=[str(p) for p in GLOBAL_EXPERIMENTS])
    parser.add_argument("--local-experiments", nargs="*", default=[str(p) for p in LOCAL_EXPERIMENTS])
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


def summarize_correlations(df: pd.DataFrame, features: list[str], targets: list[str], group: str) -> pd.DataFrame:
    rows = []
    for target in targets:
        if target not in df.columns:
            continue
        target_values = finite_numeric(df[target])
        for feature in features:
            values = finite_numeric(df[feature])
            valid = values.notna() & target_values.notna()
            pearson = corr_pair(values, target_values, "pearson")
            spearman = corr_pair(values, target_values, "spearman")
            rows.append({
                "group": group,
                "target": target,
                "feature": feature,
                "n": int(valid.sum()),
                "pearson_corr_with_prediction": pearson,
                "abs_pearson_corr": abs(pearson) if math.isfinite(pearson) else np.nan,
                "spearman_corr_with_prediction": spearman,
                "abs_spearman_corr": abs(spearman) if math.isfinite(spearman) else np.nan,
                "feature_mean": float(values[valid].mean()) if valid.any() else np.nan,
                "feature_std": float(values[valid].std(ddof=0)) if valid.any() else np.nan,
                "prediction_mean": float(target_values[valid].mean()) if valid.any() else np.nan,
                "prediction_std": float(target_values[valid].std(ddof=0)) if valid.any() else np.nan,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["target", "abs_pearson_corr", "feature"], ascending=[True, False, True]).reset_index(drop=True)
    out.insert(0, "rank_by_abs_pearson_for_target", out.groupby("target").cumcount() + 1)
    return out


def numeric_feature_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        values = finite_numeric(df[col])
        if values.notna().any():
            cols.append(col)
    return cols


def has_geo19_name(exp_dir: Path) -> bool:
    return "geo19" in exp_dir.name or "corner_angles" in exp_dir.name


def analyze_global(exp_dir: Path, dataset_root: Path, split: str, out_root: Path) -> tuple[pd.DataFrame, Path]:
    pred_path = exp_dir / "test_analysis" / "basic_cv_ensemble_case_predictions.csv"
    feature_name = "case_geometry_features_with_corner_angles.csv" if has_geo19_name(exp_dir) else "case_geometry_features.csv"
    feature_path = dataset_root / split / feature_name
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    pred = pd.read_csv(pred_path, dtype={"image_id": str})
    feat = pd.read_csv(feature_path, dtype={"image_id": str})
    merged = pred.merge(feat, on="image_id", how="inner", validate="one_to_one")
    features = numeric_feature_columns(feat, {"image_id"})
    targets = [t for t in ["y_prob", "y_logit"] if t in merged.columns]
    summary = summarize_correlations(merged, features, targets, "all_test_cases")
    summary.insert(0, "experiment_type", "global_screening")
    summary.insert(1, "experiment", exp_dir.name)

    outdir = out_root / "global_screening" / exp_dir.name
    outdir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(outdir / "global_prediction_geometry_rows.csv", index=False)
    summary.to_csv(outdir / "global_geometry_prediction_correlation.csv", index=False)
    write_report(outdir, exp_dir, "global_screening", summary, primary_target="y_prob", rows_used=len(merged))
    return summary, outdir


def analyze_local(exp_dir: Path, dataset_root: Path, split: str, out_root: Path) -> tuple[pd.DataFrame, Path]:
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
        raise ValueError(f"{feature_path} missing local geometry columns: {missing}")
    merged = pred.merge(
        feat[["image_id", "pair_name", *features]],
        on=["image_id", "pair_name"],
        how="inner",
        validate="one_to_one",
    )
    if {"pair_prob", "slip_pred"}.issubset(merged.columns):
        slip_norm = (finite_numeric(merged["slip_pred"]) / 25.0).clip(lower=0.0, upper=1.0)
        merged["slip_pred_norm_grade1_cap"] = slip_norm
        merged["joint_pair_prob_slip_score_0p6_0p4"] = 0.6 * finite_numeric(merged["pair_prob"]) + 0.4 * slip_norm
    targets = [
        t for t in [
            "pair_prob",
            "slip_pred",
            "slip_pred_norm_grade1_cap",
            "joint_pair_prob_slip_score_0p6_0p4",
        ]
        if t in merged.columns
    ]
    summary = summarize_correlations(merged, features, targets, "all_test_pairs")
    by_pair = pd.concat(
        [summarize_correlations(group, features, targets, pair_name) for pair_name, group in merged.groupby("pair_name", sort=True)],
        ignore_index=True,
    )
    summary.insert(0, "experiment_type", "local_pair_diagnostic")
    summary.insert(1, "experiment", exp_dir.name)
    by_pair.insert(0, "experiment_type", "local_pair_diagnostic")
    by_pair.insert(1, "experiment", exp_dir.name)

    outdir = out_root / "local_pair_diagnostic" / exp_dir.name
    outdir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(outdir / "local_pair_prediction_geometry_rows.csv", index=False)
    summary.to_csv(outdir / "local_pair_geometry_prediction_correlation.csv", index=False)
    by_pair.to_csv(outdir / "local_pair_geometry_prediction_correlation_by_pair.csv", index=False)
    write_report(
        outdir,
        exp_dir,
        "local_pair_diagnostic",
        summary,
        primary_target="joint_pair_prob_slip_score_0p6_0p4",
        rows_used=len(merged),
    )
    return summary, outdir


def write_report(outdir: Path, exp_dir: Path, experiment_type: str, summary: pd.DataFrame, primary_target: str, rows_used: int):
    lines = []
    lines.append("Model Geometry-Prediction Correlation")
    lines.append("=" * 37)
    lines.append("")
    lines.append(f"Experiment type: {experiment_type}")
    lines.append(f"Experiment: {exp_dir}")
    lines.append(f"Rows used: {rows_used}")
    lines.append("")
    lines.append("Definition: importance = absolute Pearson correlation between feature value and trained model output.")
    lines.append("Also reported: Spearman correlation for monotonic association.")
    lines.append("Interpretation note: this is model-output association, not causal attribution.")
    lines.append("")
    if summary.empty:
        lines.append("No correlation rows were produced.")
    else:
        for target in summary["target"].drop_duplicates().tolist():
            sub = summary[summary["target"] == target].sort_values("abs_pearson_corr", ascending=False).head(10)
            label = "primary" if target == primary_target else "secondary"
            lines.append(f"Top features for {target} ({label}):")
            for row in sub.itertuples(index=False):
                lines.append(
                    f"- {row.feature}: pearson={row.pearson_corr_with_prediction:.6f}, "
                    f"spearman={row.spearman_corr_with_prediction:.6f}, n={int(row.n)}"
                )
            lines.append("")
    lines.append("Files:")
    for path in sorted(outdir.glob("*.csv")):
        lines.append(f"- {path.name}")
    (outdir / "model_geo_prediction_correlation_report.txt").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_tables = []
    outdirs = []
    for raw in args.global_experiments:
        table, outdir = analyze_global(Path(raw), dataset_root, args.split, out_root)
        all_tables.append(table)
        outdirs.append(outdir)
        print(f"[OK] global: {raw} -> {outdir}")
    for raw in args.local_experiments:
        table, outdir = analyze_local(Path(raw), dataset_root, args.split, out_root)
        all_tables.append(table)
        outdirs.append(outdir)
        print(f"[OK] local: {raw} -> {outdir}")

    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(out_root / "all_model_geometry_prediction_correlations.csv", index=False)

    primary = combined[
        ((combined["experiment_type"] == "global_screening") & (combined["target"] == "y_prob"))
        | ((combined["experiment_type"] == "local_pair_diagnostic") & (combined["target"] == "joint_pair_prob_slip_score_0p6_0p4"))
    ].copy()
    primary = primary.sort_values(["experiment_type", "experiment", "abs_pearson_corr"], ascending=[True, True, False])
    primary.to_csv(out_root / "primary_output_geometry_prediction_correlations.csv", index=False)
    print(f"[OK] combined -> {out_root}")


if __name__ == "__main__":
    main()
