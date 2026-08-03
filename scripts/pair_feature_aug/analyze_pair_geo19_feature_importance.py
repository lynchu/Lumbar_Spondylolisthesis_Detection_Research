#!/usr/bin/env python3
"""Rank 19 pair-level geometry features by correlation with positive-pair labels."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.resnet50.dataset_local_pair_diagnostic import DEFAULT_GEOMETRY_FEATURES, location_to_pair_set


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset_half_sacrum")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--feature-csv-name", default="pair_crop_features_with_corner_angles.csv")
    parser.add_argument("--diag-csv-name", default="lumbar_diagnosis.csv")
    parser.add_argument(
        "--outdir",
        default="presence/experiments/pair_geo19_feature_importance",
        help="Directory for CSV summaries and text report.",
    )
    parser.add_argument(
        "--feature-names",
        nargs="*",
        default=None,
        help="Optional explicit feature list. Default uses DEFAULT_GEOMETRY_FEATURES.",
    )
    parser.add_argument(
        "--exclude-pairs",
        nargs="*",
        default=[],
        help="Pair names to exclude before correlation analysis, e.g. L5-S1.",
    )
    return parser.parse_args()


def load_labeled_pairs(dataset_root: Path, split: str, feature_csv_name: str, diag_csv_name: str, features: list[str]) -> pd.DataFrame:
    feature_path = dataset_root / split / feature_csv_name
    diag_path = dataset_root / split / diag_csv_name
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    if not diag_path.exists():
        raise FileNotFoundError(diag_path)

    feat = pd.read_csv(feature_path, dtype={"image_id": str, "pair_name": str})
    diag = pd.read_csv(diag_path, dtype={"image_id": str})
    missing = [c for c in ["image_id", "pair_name", *features] if c not in feat.columns]
    if missing:
        raise ValueError(f"{feature_path} missing required columns: {missing}")
    if "location" not in diag.columns:
        raise ValueError(f"{diag_path} missing location column")

    diag = diag.copy()
    diag["image_id"] = diag["image_id"].astype(str).str.strip()
    diag["positive_pair_set"] = diag["location"].apply(location_to_pair_set)
    lookup = dict(zip(diag["image_id"], diag["positive_pair_set"]))

    out = feat[["image_id", "pair_name", *features]].copy()
    out["image_id"] = out["image_id"].astype(str).str.strip()
    out["pair_name"] = out["pair_name"].astype(str).str.strip()
    out["split"] = split
    out["pair_y_true"] = [
        int(pair_name in lookup.get(image_id, set()))
        for image_id, pair_name in zip(out["image_id"], out["pair_name"])
    ]
    out["has_diag_row"] = out["image_id"].isin(lookup).astype(int)
    out = out[out["has_diag_row"] == 1].copy()
    return out


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    arr_x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr_y = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    if int(mask.sum()) < 3:
        return np.nan
    arr_x = arr_x[mask]
    arr_y = arr_y[mask]
    if np.nanstd(arr_x) <= 1e-12 or np.nanstd(arr_y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(arr_x, arr_y)[0, 1])


def summarize_feature_importance(df: pd.DataFrame, features: list[str], group_name: str) -> pd.DataFrame:
    y = df["pair_y_true"].astype(float)
    rows = []
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        pos = values[df["pair_y_true"] == 1]
        neg = values[df["pair_y_true"] == 0]
        corr = pearson_corr(values, y)
        rows.append({
            "group": group_name,
            "feature": feature,
            "n": int(values.notna().sum()),
            "positive_pair_n": int((df["pair_y_true"] == 1).sum()),
            "negative_pair_n": int((df["pair_y_true"] == 0).sum()),
            "pearson_corr_with_positive_pair": corr,
            "abs_pearson_corr": abs(corr) if np.isfinite(corr) else np.nan,
            "positive_mean": float(pos.mean()) if len(pos) else np.nan,
            "negative_mean": float(neg.mean()) if len(neg) else np.nan,
            "mean_difference_pos_minus_neg": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else np.nan,
            "positive_median": float(pos.median()) if len(pos) else np.nan,
            "negative_median": float(neg.median()) if len(neg) else np.nan,
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["abs_pearson_corr", "feature"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank_by_abs_corr", np.arange(1, len(out) + 1))
    return out


def write_report(outdir: Path, combined: pd.DataFrame, split_summary: pd.DataFrame, labeled: pd.DataFrame):
    lines = []
    lines.append("Pair-Level Geo19 Feature Importance")
    lines.append("=" * 38)
    lines.append("")
    lines.append("Definition: importance = absolute Pearson correlation between pair_y_true and feature value.")
    lines.append("pair_y_true = 1 when the pair crop's pair_name appears in the diagnosis location for that image.")
    lines.append("")
    lines.append(f"Rows used: {len(labeled)}")
    lines.append(f"Positive pairs: {int(labeled['pair_y_true'].sum())}")
    lines.append(f"Negative pairs: {int((labeled['pair_y_true'] == 0).sum())}")
    lines.append("")
    lines.append("Combined ranking:")
    for row in combined.itertuples(index=False):
        lines.append(
            f"{int(row.rank_by_abs_corr):02d}. {row.feature}: "
            f"corr={row.pearson_corr_with_positive_pair:.6f}, "
            f"abs={row.abs_pearson_corr:.6f}, "
            f"pos_mean={row.positive_mean:.4f}, neg_mean={row.negative_mean:.4f}"
        )
    lines.append("")
    lines.append("Files:")
    lines.append("- pair_geo19_feature_importance_combined.csv")
    lines.append("- pair_geo19_feature_importance_by_split.csv")
    lines.append("- pair_geo19_labeled_feature_rows.csv")
    (outdir / "pair_geo19_feature_importance_report.txt").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    features = list(args.feature_names or DEFAULT_GEOMETRY_FEATURES)

    frames = [
        load_labeled_pairs(dataset_root, split, args.feature_csv_name, args.diag_csv_name, features)
        for split in args.splits
    ]
    labeled = pd.concat(frames, ignore_index=True)
    exclude_pairs = {str(x).strip() for x in args.exclude_pairs if str(x).strip()}
    if exclude_pairs:
        labeled = labeled[~labeled["pair_name"].isin(exclude_pairs)].copy()
    labeled.to_csv(outdir / "pair_geo19_labeled_feature_rows.csv", index=False)
    pd.DataFrame([{
        "dataset_root": str(dataset_root),
        "splits": ";".join(args.splits),
        "feature_csv_name": args.feature_csv_name,
        "diag_csv_name": args.diag_csv_name,
        "excluded_pairs": ";".join(sorted(exclude_pairs)),
        "rows_used": int(len(labeled)),
        "positive_pairs": int(labeled["pair_y_true"].sum()),
        "negative_pairs": int((labeled["pair_y_true"] == 0).sum()),
    }]).to_csv(outdir / "pair_geo19_feature_importance_meta.csv", index=False)

    combined = summarize_feature_importance(labeled, features, "all_splits")
    combined.to_csv(outdir / "pair_geo19_feature_importance_combined.csv", index=False)

    split_tables = [
        summarize_feature_importance(group, features, split)
        for split, group in labeled.groupby("split", sort=False)
    ]
    by_split = pd.concat(split_tables, ignore_index=True)
    by_split.to_csv(outdir / "pair_geo19_feature_importance_by_split.csv", index=False)

    write_report(outdir, combined, by_split, labeled)
    print(f"[OK] wrote {outdir}")
    print(combined[["rank_by_abs_corr", "feature", "pearson_corr_with_positive_pair", "abs_pearson_corr", "positive_mean", "negative_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
