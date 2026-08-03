#!/usr/bin/env python3
"""Train learned case-level fusion models from OOF branch predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PAIR_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, choices=["logreg", "mlp"])
    parser.add_argument("--global-oof-csv", required=True)
    parser.add_argument("--local-oof-case-glob", required=True)
    parser.add_argument("--local-oof-pair-glob", required=True)
    parser.add_argument("--global-test-case-csv", required=True)
    parser.add_argument("--local-test-case-csv", required=True)
    parser.add_argument("--local-test-pair-csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--local-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def logit_np(prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(prob / (1.0 - prob))


def metrics(y_true, y_prob, threshold):
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
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
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


def selection_score(row):
    return (
        0.50 * float(row["balanced_acc"])
        + 0.20 * float(row["sensitivity"])
        + 0.20 * float(row["accuracy"])
        - 0.10 * float(row["sens_spec_gap"])
    )


def threshold_sweep(y_true, y_prob):
    rows = []
    for th in np.arange(0.1, 0.91, 0.05):
        row = metrics(y_true, y_prob, float(round(th, 2)))
        row["selection_score"] = selection_score(row)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["selection_score", "balanced_acc", "sensitivity"], ascending=False).reset_index(drop=True)


def load_glob(pattern):
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    out = pd.concat([pd.read_csv(p).assign(source_file=str(p)) for p in files], ignore_index=True)
    return out


def normalize_global(df):
    if {"global_prob", "global_logit"}.issubset(df.columns):
        out = df[["image_id", "y_true", "global_prob", "global_logit"]].copy()
    else:
        out = df[["image_id", "y_true", "y_prob", "y_logit"]].rename(
            columns={"y_prob": "global_prob", "y_logit": "global_logit"}
        )
    return out


def build_pair_features(pair_df, local_threshold):
    rows = []
    for image_id, group in pair_df.groupby("image_id", sort=False):
        group = group.sort_values("pair_idx")
        probs = dict(zip(group["pair_name"].astype(str), group["pair_prob"].astype(float)))
        values = np.array([float(probs.get(p, np.nan)) for p in PAIR_NAMES], dtype=float)
        values = np.nan_to_num(values, nan=0.0)
        sorted_vals = np.sort(values)[::-1]
        row = {"image_id": image_id}
        for pair_name, val in zip(PAIR_NAMES, values):
            row[f"local_pair_prob_{pair_name}"] = float(val)
        row["local_max_pair_prob"] = float(sorted_vals[0])
        row["local_second_max_pair_prob"] = float(sorted_vals[1]) if len(sorted_vals) > 1 else 0.0
        row["local_pair_prob_gap_top2"] = float(sorted_vals[0] - (sorted_vals[1] if len(sorted_vals) > 1 else 0.0))
        row["local_pred_pair_count"] = int((values >= float(local_threshold)).sum())
        row["local_pair_prob_mean"] = float(values.mean())
        row["local_pair_prob_std"] = float(values.std())
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_table(global_df, local_case_df, local_pair_df, local_threshold):
    g = normalize_global(global_df)
    l = local_case_df[["image_id", "y_true", "y_prob_pair_mil", "top_pair_prob", "top_pair_slip_pred"]].rename(
        columns={
            "y_true": "local_y_true",
            "y_prob_pair_mil": "local_mil_prob",
            "top_pair_prob": "local_top_pair_prob",
            "top_pair_slip_pred": "local_top_pair_slip_pred",
        }
    )
    l["local_mil_logit"] = logit_np(l["local_mil_prob"])
    pair_feat = build_pair_features(local_pair_df, local_threshold)
    out = g.merge(l, on="image_id", how="inner").merge(pair_feat, on="image_id", how="inner")
    if len(out) != len(g) or len(out) != len(l):
        raise RuntimeError(f"Feature merge mismatch: global={len(g)}, local={len(l)}, merged={len(out)}")
    if not (out["y_true"].astype(int) == out["local_y_true"].astype(int)).all():
        raise RuntimeError("Global/local label mismatch in fusion features.")
    out["y_true"] = out["y_true"].astype(int)
    out["simple_mean_prob"] = 0.5 * out["global_prob"] + 0.5 * out["local_mil_prob"]
    out["simple_max_prob"] = np.maximum(out["global_prob"], out["local_mil_prob"])
    out["simple_noisy_or_prob"] = 1.0 - (1.0 - out["global_prob"]) * (1.0 - out["local_mil_prob"])
    return out


def feature_columns(df):
    cols = [
        "global_prob",
        "global_logit",
        "local_mil_prob",
        "local_mil_logit",
        "local_top_pair_prob",
        "local_top_pair_slip_pred",
        "local_max_pair_prob",
        "local_second_max_pair_prob",
        "local_pair_prob_gap_top2",
        "local_pred_pair_count",
        "local_pair_prob_mean",
        "local_pair_prob_std",
        "simple_mean_prob",
        "simple_max_prob",
        "simple_noisy_or_prob",
    ] + [f"local_pair_prob_{p}" for p in PAIR_NAMES]
    return [c for c in cols if c in df.columns]


class FusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(x_train, y_train, x_val, y_val, seed, epochs, patience):
    torch.manual_seed(seed)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)
    model = FusionMLP(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    pos = max(float((y_train == 1).sum()), 1.0)
    neg = max(float((y_train == 0).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32))
    xt = torch.tensor(x_train_s)
    yt = torch.tensor(y_train.astype(np.float32))
    xv = torch.tensor(x_val_s)
    best_state = None
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(xv)).numpy()
        val_metrics = threshold_sweep(y_val, val_prob).iloc[0]
        score = float(val_metrics["selection_score"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, scaler, {"best_epoch": int(best_epoch), "best_selection_score": float(best_score)}


def predict_mlp(model, scaler, x):
    model.eval()
    xs = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(xs))).numpy()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    global_oof = pd.read_csv(args.global_oof_csv)
    local_oof_case = load_glob(args.local_oof_case_glob)
    local_oof_pair = load_glob(args.local_oof_pair_glob)
    global_test = pd.read_csv(args.global_test_case_csv)
    local_test_case = pd.read_csv(args.local_test_case_csv)
    local_test_pair = pd.read_csv(args.local_test_pair_csv)

    train_df = build_feature_table(global_oof, local_oof_case, local_oof_pair, args.local_threshold)
    test_df = build_feature_table(global_test, local_test_case, local_test_pair, args.local_threshold)
    cols = feature_columns(train_df)
    train_df.to_csv(outdir / "fusion_train_oof_features.csv", index=False)
    test_df.to_csv(outdir / "fusion_test_features.csv", index=False)
    (outdir / "feature_columns.json").write_text(json.dumps(cols, indent=2), encoding="utf-8")

    x = train_df[cols].to_numpy(float)
    y = train_df["y_true"].to_numpy(int)
    x_test = test_df[cols].to_numpy(float)
    y_test = test_df["y_true"].to_numpy(int)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=args.seed, stratify=y
    )
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    if args.model_type == "logreg":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        ])
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        sweep = threshold_sweep(y_val, val_prob)
        best_threshold = float(sweep.iloc[0]["threshold"])
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        ])
        final_model.fit(x, y)
        test_prob = final_model.predict_proba(x_test)[:, 1]
        joblib.dump(final_model, outdir / "logreg_model.joblib")
        extra = {"model_type": "logreg"}
    else:
        model, scaler, extra = train_mlp(x_train, y_train, x_val, y_val, args.seed, args.epochs, args.patience)
        val_prob = predict_mlp(model, scaler, x_val)
        sweep = threshold_sweep(y_val, val_prob)
        best_threshold = float(sweep.iloc[0]["threshold"])
        test_prob = predict_mlp(model, scaler, x_test)
        torch.save({"model_state_dict": model.state_dict(), "feature_columns": cols, "extra": extra}, outdir / "best.pt")
        joblib.dump(scaler, outdir / "scaler.joblib")
        extra["model_type"] = "mlp"

    sweep.to_csv(outdir / "threshold_sweep_val.csv", index=False)
    test_metric = metrics(y_test, test_prob, best_threshold)
    test_metric["auroc"] = float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else np.nan
    test_metric["ap"] = float(average_precision_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else np.nan
    test_metric["model_type"] = args.model_type
    pd.DataFrame([test_metric]).to_csv(outdir / "test_metrics.csv", index=False)

    pred = test_df[["image_id", "y_true"]].copy()
    pred["fusion_prob"] = test_prob
    pred["fusion_threshold"] = best_threshold
    pred["fusion_pred"] = (pred["fusion_prob"] >= best_threshold).astype(int)
    pred["confusion_type"] = "TN"
    pred.loc[(pred["y_true"] == 0) & (pred["fusion_pred"] == 1), "confusion_type"] = "FP"
    pred.loc[(pred["y_true"] == 1) & (pred["fusion_pred"] == 0), "confusion_type"] = "FN"
    pred.loc[(pred["y_true"] == 1) & (pred["fusion_pred"] == 1), "confusion_type"] = "TP"
    pred.to_csv(outdir / "test_predictions.csv", index=False)

    summary = {"model_type": args.model_type, "best_threshold": best_threshold, "test_metrics": test_metric, "extra": extra}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        f"Learned Fusion Summary: {args.model_type}",
        "=" * 36,
        f"features: {len(cols)}",
        f"train_oof_cases: {len(train_df)}",
        f"test_cases: {len(test_df)}",
        f"selected_threshold: {best_threshold:.3f}",
        f"accuracy: {test_metric['accuracy']:.4f}",
        f"balanced_acc: {test_metric['balanced_acc']:.4f}",
        f"sensitivity: {test_metric['sensitivity']:.4f}",
        f"specificity: {test_metric['specificity']:.4f}",
        f"precision: {test_metric['precision']:.4f}",
        f"f1: {test_metric['f1']:.4f}",
        f"auroc: {test_metric['auroc']:.4f}",
        f"ap: {test_metric['ap']:.4f}",
        f"tn/fp/fn/tp: {test_metric['tn']}/{test_metric['fp']}/{test_metric['fn']}/{test_metric['tp']}",
    ]
    (outdir / "summary_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
