#!/usr/bin/env python3
"""Fuse global screening and local pair-diagnostic predictions at case level."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PAIR_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-case-csv", required=True)
    parser.add_argument("--local-case-csv", required=True)
    parser.add_argument("--local-pair-csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--local-threshold", type=float, default=0.4)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Final case-level thresholds to evaluate.",
    )
    return parser.parse_args()


def logit_np(prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(prob / (1.0 - prob))


def binary_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(len(y_true), 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    balanced_acc = 0.5 * (sensitivity + specificity)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_acc": float(balanced_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(sensitivity),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sens_spec_gap": float(abs(sensitivity - specificity)),
    }


def add_ranking_metrics(row):
    # Screening-oriented score: balanced accuracy first, then sensitivity, then
    # smaller sensitivity/specificity gap. Accuracy is included but not dominant.
    return (
        0.50 * float(row["balanced_acc"])
        + 0.20 * float(row["sensitivity"])
        + 0.20 * float(row["accuracy"])
        - 0.10 * float(row["sens_spec_gap"])
    )


def fusion_rule_probs(df):
    global_prob = df["global_prob"].to_numpy(float)
    local_prob = df["local_mil_prob"].to_numpy(float)
    out = {
        "global_only": global_prob,
        "local_only": local_prob,
        "max_prob": np.maximum(global_prob, local_prob),
        "mean_prob": 0.5 * global_prob + 0.5 * local_prob,
        "weighted_global_025": 0.25 * global_prob + 0.75 * local_prob,
        "weighted_global_050": 0.50 * global_prob + 0.50 * local_prob,
        "weighted_global_075": 0.75 * global_prob + 0.25 * local_prob,
        "noisy_or": 1.0 - (1.0 - global_prob) * (1.0 - local_prob),
    }
    return {k: np.clip(v, 1e-6, 1.0 - 1e-6) for k, v in out.items()}


def build_local_pair_support(local_pair_df, local_threshold):
    rows = []
    for image_id, group in local_pair_df.groupby("image_id", sort=False):
        group = group.sort_values("pair_idx")
        probs = dict(zip(group["pair_name"].astype(str), group["pair_prob"].astype(float)))
        true_pairs = group.loc[group["pair_y_true"].astype(int) == 1, "pair_name"].astype(str).tolist()
        pred_pairs = group.loc[group["pair_prob"].astype(float) >= float(local_threshold), "pair_name"].astype(str).tolist()
        top_idx = group["pair_prob"].astype(float).idxmax()
        top = group.loc[top_idx]
        row = {
            "image_id": image_id,
            "local_true_pair_set": ";".join(true_pairs) if true_pairs else "none",
            "local_pred_pair_set": ";".join(pred_pairs) if pred_pairs else "none",
            "local_pred_pair_count": int(len(pred_pairs)),
            "local_top_pair_name": str(top["pair_name"]),
            "local_top_pair_prob": float(top["pair_prob"]),
        }
        for pair_name in PAIR_NAMES:
            row[f"local_pair_prob_{pair_name}"] = float(probs.get(pair_name, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    global_df = pd.read_csv(args.global_case_csv)
    local_case_df = pd.read_csv(args.local_case_csv)
    local_pair_df = pd.read_csv(args.local_pair_csv)

    required_global = {"image_id", "y_true", "y_prob", "y_logit"}
    required_local_case = {"image_id", "y_true", "y_prob_pair_mil", "top_pair_name", "top_pair_prob"}
    required_local_pair = {"image_id", "pair_name", "pair_idx", "pair_y_true", "pair_prob"}
    for name, df, required in [
        ("global_case", global_df, required_global),
        ("local_case", local_case_df, required_local_case),
        ("local_pair", local_pair_df, required_local_pair),
    ]:
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    global_keep = global_df[["image_id", "y_true", "y_prob", "y_logit"]].copy()
    global_keep = global_keep.rename(
        columns={"y_prob": "global_prob", "y_logit": "global_logit", "y_true": "global_y_true"}
    )
    local_keep = local_case_df[["image_id", "y_true", "y_prob_pair_mil", "top_pair_name", "top_pair_prob"]].copy()
    local_keep = local_keep.rename(
        columns={
            "y_prob_pair_mil": "local_mil_prob",
            "top_pair_name": "local_case_top_pair_name",
            "top_pair_prob": "local_case_top_pair_prob",
            "y_true": "local_y_true",
        }
    )
    local_keep["local_mil_logit"] = logit_np(local_keep["local_mil_prob"])

    local_support = build_local_pair_support(local_pair_df, args.local_threshold)
    fusion_input = global_keep.merge(local_keep, on="image_id", how="inner")
    fusion_input = fusion_input.merge(local_support, on="image_id", how="left")
    if len(fusion_input) != len(global_keep) or len(fusion_input) != len(local_keep):
        raise RuntimeError(
            "Global/local prediction rows did not align exactly by image_id: "
            f"global={len(global_keep)}, local={len(local_keep)}, merged={len(fusion_input)}"
        )
    if not (fusion_input["global_y_true"].astype(int) == fusion_input["local_y_true"].astype(int)).all():
        bad = fusion_input[fusion_input["global_y_true"].astype(int) != fusion_input["local_y_true"].astype(int)]
        raise RuntimeError(f"Global/local y_true mismatch for {len(bad)} rows.")

    fusion_input["y_true"] = fusion_input["global_y_true"].astype(int)
    fusion_input.to_csv(outdir / "case_fusion_input.csv", index=False)

    rule_probs = fusion_rule_probs(fusion_input)
    metric_rows = []
    y_true = fusion_input["y_true"].to_numpy(int)
    for rule_name, probs in rule_probs.items():
        auroc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else math.nan
        ap = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else math.nan
        for threshold in args.thresholds:
            row = {"fusion_rule": rule_name}
            row.update(binary_metrics(y_true, probs, threshold))
            row["auroc"] = auroc
            row["ap"] = ap
            row["selection_score"] = add_ranking_metrics(row)
            metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df.sort_values(
        ["selection_score", "balanced_acc", "sensitivity", "specificity", "accuracy"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    metrics_df.to_csv(outdir / "case_fusion_rule_sweep_metrics.csv", index=False)

    best = metrics_df.iloc[0].to_dict()
    best_rule = str(best["fusion_rule"])
    best_threshold = float(best["threshold"])
    final = fusion_input.copy()
    final["final_fusion_rule"] = best_rule
    final["final_threshold"] = best_threshold
    final["final_presence_prob"] = rule_probs[best_rule]
    final["final_presence_pred"] = (final["final_presence_prob"] >= best_threshold).astype(int)
    final["final_confusion_type"] = "TN"
    final.loc[(final["y_true"] == 0) & (final["final_presence_pred"] == 1), "final_confusion_type"] = "FP"
    final.loc[(final["y_true"] == 1) & (final["final_presence_pred"] == 0), "final_confusion_type"] = "FN"
    final.loc[(final["y_true"] == 1) & (final["final_presence_pred"] == 1), "final_confusion_type"] = "TP"
    final["final_pred_pair_set"] = np.where(
        final["final_presence_pred"] == 1,
        final["local_pred_pair_set"].fillna("none"),
        "none",
    )
    final["final_location_has_thresholded_pair"] = (
        (final["final_presence_pred"] == 1)
        & final["final_pred_pair_set"].fillna("none").ne("none")
    ).astype(int)
    final["final_top_pair_name"] = final["local_top_pair_name"]
    final["final_top_pair_prob"] = final["local_top_pair_prob"]
    final.to_csv(outdir / "case_fusion_predictions.csv", index=False)

    location_cols = [
        "image_id",
        "y_true",
        "final_presence_prob",
        "final_presence_pred",
        "final_pred_pair_set",
        "final_top_pair_name",
        "final_top_pair_prob",
        "local_true_pair_set",
        "local_pred_pair_set",
        "local_pred_pair_count",
    ] + [f"local_pair_prob_{p}" for p in PAIR_NAMES]
    final[location_cols].to_csv(outdir / "location_support_predictions.csv", index=False)

    summary = {
        "global_case_csv": str(args.global_case_csv),
        "local_case_csv": str(args.local_case_csv),
        "local_pair_csv": str(args.local_pair_csv),
        "local_threshold": float(args.local_threshold),
        "n_cases": int(len(final)),
        "best": best,
    }
    with (outdir / "case_fusion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_lines = [
        "Stage 4 Case-Level Fusion Summary",
        "=" * 36,
        f"Cases: {len(final)}",
        f"Local location threshold: {args.local_threshold:.3f}",
        "",
        "Best fusion setting:",
        f"  rule: {best_rule}",
        f"  threshold: {best_threshold:.3f}",
        f"  accuracy: {best['accuracy']:.4f}",
        f"  balanced_acc: {best['balanced_acc']:.4f}",
        f"  sensitivity: {best['sensitivity']:.4f}",
        f"  specificity: {best['specificity']:.4f}",
        f"  precision: {best['precision']:.4f}",
        f"  f1: {best['f1']:.4f}",
        f"  auroc: {best['auroc']:.4f}",
        f"  ap: {best['ap']:.4f}",
        f"  tn/fp/fn/tp: {int(best['tn'])}/{int(best['fp'])}/{int(best['fn'])}/{int(best['tp'])}",
        "",
        "Top 10 fusion settings:",
        metrics_df.head(10).to_string(index=False),
        "",
        "Output files:",
        "  case_fusion_input.csv",
        "  case_fusion_rule_sweep_metrics.csv",
        "  case_fusion_predictions.csv",
        "  location_support_predictions.csv",
        "  case_fusion_summary.json",
    ]
    (outdir / "case_fusion_summary_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines[:20]))


if __name__ == "__main__":
    main()
