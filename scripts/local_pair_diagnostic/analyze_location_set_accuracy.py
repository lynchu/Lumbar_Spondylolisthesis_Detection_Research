#!/usr/bin/env python3
"""Compute detailed case-level location set accuracy from local pair predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


DEFAULT_RUNS = [
    {
        "branch": "location_only",
        "threshold": 0.2,
        "predictions": "presence/experiments/local_pair_diagnostic/resnet50_local_pair_location_only_balacc_half_sacrum_cv5_th02/test_analysis_th02/local_pair_diagnostic_pair_predictions.csv",
    },
    {
        "branch": "location_only",
        "threshold": 0.3,
        "predictions": "presence/experiments/local_pair_diagnostic/resnet50_local_pair_location_only_balacc_half_sacrum_cv5_th03/test_analysis_th03/local_pair_diagnostic_pair_predictions.csv",
    },
    {
        "branch": "location_only",
        "threshold": 0.4,
        "predictions": "presence/experiments/local_pair_diagnostic/resnet50_local_pair_location_only_balacc_half_sacrum_cv5_th04/test_analysis_th04/local_pair_diagnostic_pair_predictions.csv",
    },
    {
        "branch": "slip_aux",
        "threshold": 0.2,
        "predictions": "presence/experiments/local_pair_diagnostic/resnet50_local_pair_slip_aux_balacc_half_sacrum_cv5_th02/test_analysis_th02/local_pair_diagnostic_pair_predictions.csv",
    },
    {
        "branch": "slip_aux",
        "threshold": 0.3,
        "predictions": "presence/experiments/local_pair_diagnostic/resnet50_local_pair_slip_aux_balacc_half_sacrum_cv5_th03/test_analysis_th03/local_pair_diagnostic_pair_predictions.csv",
    },
    {
        "branch": "slip_aux",
        "threshold": 0.4,
        "predictions": "presence/experiments/local_pair_diagnostic/resnet50_local_pair_slip_aux_balacc_half_sacrum_cv5_th04/test_analysis_th04/local_pair_diagnostic_pair_predictions.csv",
    },
    {
        "branch": "slip_ordinal",
        "threshold": 0.2,
        "predictions": "presence/experiments/local_pair_diagnostic/resnet50_local_pair_slip_ordinal_balacc_half_sacrum_cv5_th02/test_analysis_th02/local_pair_diagnostic_pair_predictions.csv",
    },
    {
        "branch": "slip_ordinal",
        "threshold": 0.3,
        "predictions": "presence/experiments/local_pair_diagnostic/resnet50_local_pair_slip_ordinal_balacc_half_sacrum_cv5_th03/test_analysis_th03/local_pair_diagnostic_pair_predictions.csv",
    },
    {
        "branch": "slip_ordinal",
        "threshold": 0.4,
        "predictions": "presence/experiments/local_pair_diagnostic/resnet50_local_pair_slip_ordinal_balacc_half_sacrum_cv5_th04/test_analysis_th04/local_pair_diagnostic_pair_predictions.csv",
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_csv",
        default="presence/experiments/local_pair_diagnostic/location_set_accuracy_summary.csv",
    )
    return p.parse_args()


def set_to_string(indices):
    if not indices:
        return "none"
    return "|".join(PAIR_NAMES[i] for i in indices)


def compute_metrics(pair_df, threshold):
    exact_all = []
    exact_pos = []
    pred_set_hit_pos = []
    top1_hit_pos = []
    rows = []

    for image_id, group in pair_df.groupby("image_id", sort=False):
        group = group.sort_values("pair_idx")
        true_mask = group["pair_y_true"].to_numpy(int) == 1
        prob = group["pair_prob"].to_numpy(float)
        pred_mask = prob >= float(threshold)

        true_set = set(np.flatnonzero(true_mask).tolist())
        pred_set = set(np.flatnonzero(pred_mask).tolist())
        top_idx = int(np.argmax(prob))
        is_positive = len(true_set) > 0

        exact = pred_set == true_set
        exact_all.append(exact)
        if is_positive:
            exact_pos.append(exact)
            pred_set_hit_pos.append(len(pred_set & true_set) > 0)
            top1_hit_pos.append(top_idx in true_set)

        rows.append(
            {
                "image_id": image_id,
                "study_y_true": int(group["study_y_true"].iloc[0]),
                "true_location_set": set_to_string(sorted(true_set)),
                "pred_location_set": set_to_string(sorted(pred_set)),
                "top1_location": PAIR_NAMES[top_idx],
                "top1_prob": float(prob[top_idx]),
                "exact_match": int(exact),
                "pred_set_hit": int(len(pred_set & true_set) > 0) if is_positive else np.nan,
                "top1_hit": int(top_idx in true_set) if is_positive else np.nan,
            }
        )

    n_all = len(exact_all)
    n_pos = len(exact_pos)
    return {
        "n_cases": int(n_all),
        "n_positive_cases": int(n_pos),
        "exact_accuracy_all_cases": float(np.mean(exact_all)) if n_all else np.nan,
        "exact_accuracy_positive_cases": float(np.mean(exact_pos)) if n_pos else np.nan,
        "pred_set_hit_positive_cases": float(np.mean(pred_set_hit_pos)) if n_pos else np.nan,
        "top1_hit_positive_cases": float(np.mean(top1_hit_pos)) if n_pos else np.nan,
    }, pd.DataFrame(rows)


def main():
    args = parse_args()
    summary_rows = []
    out_csv = Path(args.out_csv)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in DEFAULT_RUNS:
        pred_path = Path(run["predictions"])
        row = {
            "branch": run["branch"],
            "threshold": run["threshold"],
            "predictions": str(pred_path),
            "status": "missing",
        }
        if pred_path.exists():
            pair_df = pd.read_csv(pred_path)
            metrics, case_df = compute_metrics(pair_df, run["threshold"])
            row.update(metrics)
            row["status"] = "ok"
            case_out = out_dir / f"{run['branch']}_th{int(run['threshold'] * 100):02d}_location_set_case_details.csv"
            case_df.to_csv(case_out, index=False)
            row["case_details_csv"] = str(case_out)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_csv, index=False)
    print(summary_df.to_string(index=False))
    print(f"[INFO] Saved: {out_csv}")


if __name__ == "__main__":
    main()
