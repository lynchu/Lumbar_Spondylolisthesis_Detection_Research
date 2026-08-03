#!/usr/bin/env python3
"""Evaluate hard AND/OR case-level fusion rules."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--rule", required=True, choices=["and", "or"])
    parser.add_argument("--global-threshold", type=float, default=0.3)
    parser.add_argument("--local-threshold", type=float, default=0.2)
    return parser.parse_args()


def logit_np(prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(prob / (1.0 - prob))


def metrics_from_pred(y_true, y_pred, score):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(score, dtype=float)
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
        "accuracy": float(accuracy),
        "balanced_acc": float(balanced_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(sensitivity),
        "f1": float(f1),
        "auroc_score_proxy": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else np.nan,
        "ap_score_proxy": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) > 1 else np.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def build_local_pair_support(local_pair_df, local_threshold):
    rows = []
    for image_id, group in local_pair_df.groupby("image_id", sort=False):
        group = group.sort_values("pair_idx")
        true_pairs = group.loc[group["pair_y_true"].astype(int) == 1, "pair_name"].astype(str).tolist()
        pred_pairs = group.loc[group["pair_prob"].astype(float) >= float(local_threshold), "pair_name"].astype(str).tolist()
        top = group.loc[group["pair_prob"].astype(float).idxmax()]
        row = {
            "image_id": image_id,
            "local_true_pair_set": ";".join(true_pairs) if true_pairs else "none",
            "local_pred_pair_set": ";".join(pred_pairs) if pred_pairs else "none",
            "local_pred_pair_count": int(len(pred_pairs)),
            "local_top_pair_name": str(top["pair_name"]),
            "local_top_pair_prob": float(top["pair_prob"]),
            "local_top_pair_slip_pred": float(top.get("slip_pred", np.nan)),
        }
        probs = dict(zip(group["pair_name"].astype(str), group["pair_prob"].astype(float)))
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

    global_keep = global_df[["image_id", "y_true", "y_prob", "y_logit"]].rename(
        columns={"y_true": "global_y_true", "y_prob": "global_prob", "y_logit": "global_logit"}
    )
    local_keep = local_case_df[["image_id", "y_true", "y_prob_pair_mil", "top_pair_name", "top_pair_prob"]].rename(
        columns={
            "y_true": "local_y_true",
            "y_prob_pair_mil": "local_mil_prob",
            "top_pair_name": "local_case_top_pair_name",
            "top_pair_prob": "local_case_top_pair_prob",
        }
    )
    local_keep["local_mil_logit"] = logit_np(local_keep["local_mil_prob"])
    support = build_local_pair_support(local_pair_df, args.local_threshold)
    merged = global_keep.merge(local_keep, on="image_id", how="inner").merge(support, on="image_id", how="left")
    if len(merged) != len(global_keep) or len(merged) != len(local_keep):
        raise RuntimeError(f"Prediction row mismatch: global={len(global_keep)}, local={len(local_keep)}, merged={len(merged)}")
    if not (merged["global_y_true"].astype(int) == merged["local_y_true"].astype(int)).all():
        raise RuntimeError("global/local y_true mismatch")

    merged["y_true"] = merged["global_y_true"].astype(int)
    merged["global_pred"] = (merged["global_prob"].astype(float) >= float(args.global_threshold)).astype(int)
    merged["local_pred"] = (merged["local_mil_prob"].astype(float) >= float(args.local_threshold)).astype(int)
    if args.rule == "and":
        merged["final_presence_pred"] = (merged["global_pred"].eq(1) & merged["local_pred"].eq(1)).astype(int)
        merged["final_presence_score"] = np.minimum(merged["global_prob"].astype(float), merged["local_mil_prob"].astype(float))
    else:
        merged["final_presence_pred"] = (merged["global_pred"].eq(1) | merged["local_pred"].eq(1)).astype(int)
        merged["final_presence_score"] = np.maximum(merged["global_prob"].astype(float), merged["local_mil_prob"].astype(float))

    merged["final_fusion_rule"] = args.rule.upper()
    merged["global_threshold"] = float(args.global_threshold)
    merged["local_threshold"] = float(args.local_threshold)
    merged["final_confusion_type"] = "TN"
    merged.loc[(merged["y_true"] == 0) & (merged["final_presence_pred"] == 1), "final_confusion_type"] = "FP"
    merged.loc[(merged["y_true"] == 1) & (merged["final_presence_pred"] == 0), "final_confusion_type"] = "FN"
    merged.loc[(merged["y_true"] == 1) & (merged["final_presence_pred"] == 1), "final_confusion_type"] = "TP"
    merged["final_pred_pair_set"] = np.where(merged["final_presence_pred"].eq(1), merged["local_pred_pair_set"].fillna("none"), "none")
    merged["final_top_pair_name"] = merged["local_top_pair_name"]
    merged["final_top_pair_prob"] = merged["local_top_pair_prob"]
    merged["final_top_pair_slip_pred"] = merged["local_top_pair_slip_pred"]

    metric = {
        "fusion_rule": args.rule.upper(),
        "global_threshold": float(args.global_threshold),
        "local_threshold": float(args.local_threshold),
        **metrics_from_pred(merged["y_true"], merged["final_presence_pred"], merged["final_presence_score"]),
        "global_positive_cases": int(merged["global_pred"].sum()),
        "local_positive_cases": int(merged["local_pred"].sum()),
        "final_positive_cases": int(merged["final_presence_pred"].sum()),
        "n_cases": int(len(merged)),
    }
    metrics_df = pd.DataFrame([metric])
    merged.to_csv(outdir / "case_fusion_predictions.csv", index=False)
    metrics_df.to_csv(outdir / "case_fusion_metrics.csv", index=False)
    support_cols = [
        "image_id", "y_true", "final_presence_pred", "final_pred_pair_set",
        "final_top_pair_name", "final_top_pair_prob", "final_top_pair_slip_pred",
        "local_true_pair_set", "local_pred_pair_set", "local_pred_pair_count",
    ] + [f"local_pair_prob_{p}" for p in PAIR_NAMES]
    merged[support_cols].to_csv(outdir / "location_support_predictions.csv", index=False)
    with (outdir / "case_fusion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metric, f, indent=2, ensure_ascii=False)
    report = [
        f"Hard {args.rule.upper()} Fusion Summary",
        "=" * 32,
        f"global_threshold: {args.global_threshold:.3f}",
        f"local_threshold: {args.local_threshold:.3f}",
        f"n_cases: {len(merged)}",
        f"accuracy: {metric['accuracy']:.4f}",
        f"balanced_acc: {metric['balanced_acc']:.4f}",
        f"sensitivity: {metric['sensitivity']:.4f}",
        f"specificity: {metric['specificity']:.4f}",
        f"precision: {metric['precision']:.4f}",
        f"f1: {metric['f1']:.4f}",
        f"tn/fp/fn/tp: {metric['tn']}/{metric['fp']}/{metric['fn']}/{metric['tp']}",
    ]
    (outdir / "case_fusion_summary_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
