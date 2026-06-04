# python ./presence/scripts/basic/analyze_pair_crop_cv_ensemble_test.py --config ./presence/configs/resnet50_pair_crop.yaml --cv_dir ./presence/experiments/resnet50_pairCrop_resnet50_bce_caseBalAcc_cv5 --outdir ./presence/experiments/resnet50_pairCrop_resnet50_bce_caseBalAcc_cv5/test_analysis --n_folds 5 --ckpt_name best.pt --ensemble_mode probs
"""
Analyze a CV ensemble trained with pair-crop inputs.

Outputs:
  - pair_crop_cv_ensemble_pair_predictions.csv
      one row per pair crop, including pair/location y_true, pair_prob, pair_pred
  - pair_crop_cv_ensemble_case_predictions.csv
      one row per image_id/case; case probability = max pair probability
  - pair_crop_cv_ensemble_presence_metrics.csv
      main presence metrics at case level
  - pair_crop_cv_ensemble_location_pair_metrics.csv
      pair-level location metrics on test set only
  - pair_crop_cv_ensemble_top1_location_metrics.csv
      positive-case top-1 location hit metrics
  - pair_crop_cv_ensemble_summary_report.txt
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from data.resnet50.dataset_pair_crop import (
    PairCropDataset,
    PairCropDatasetWithMask,
    PAIR_NAMES,
    location_to_pair_set,
)
from models.basic import LumbarCropModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cv_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--ensemble_mode", type=str, default="probs", choices=["probs", "logits"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    return parser.parse_args()


def pair_crop_collate_fn(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    labels = torch.stack([x["label"] for x in batch], dim=0)
    study_labels = torch.stack([x["study_label"] for x in batch], dim=0)
    pair_ids = torch.stack([x["pair_id"] for x in batch], dim=0)
    return {
        "images": images,
        "labels": labels,
        "study_labels": study_labels,
        "pair_ids": pair_ids,
        "pair_names": [x["pair_name"] for x in batch],
        "image_ids": [x["image_id"] for x in batch],
        "sample_ids": [x["sample_id"] for x in batch],
        "location_raw": [x.get("location_raw", "") for x in batch],
    }


def build_dataset(cfg):
    test_image_dir = cfg.get("test_image_dir", cfg.get("image_dir"))
    if test_image_dir is None:
        raise ValueError("Missing test_image_dir/image_dir in config.")

    use_pair_mask = bool(cfg.get("use_pair_mask", cfg.get("use_mask_input", False)))
    dataset_cls = PairCropDatasetWithMask if use_pair_mask else PairCropDataset
    kwargs = dict(
        text_csv_path=cfg["test_csv"],
        image_dir=test_image_dir,
        split="test",
        exclude_s1_positive=bool(cfg.get("exclude_s1_positive", False)),
        aug_cfg=cfg.get("augmentation", {}),
    )
    if use_pair_mask:
        kwargs["mask_dir"] = cfg.get("test_mask_dir", None)
    return dataset_cls(**kwargs)


def build_model(cfg, device):
    model = LumbarCropModel(
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        loss_name=cfg.get("loss_name", "bce"),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.0),
    )
    return model.to(device)


def load_checkpoint_state_dict(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        extra = ckpt.get("extra", {})
    else:
        state = ckpt
        extra = {}
    state = {k.replace("module.", ""): v for k, v in state.items()}
    return state, extra


def apply_temperature_np(logits, temperature):
    temperature = max(float(temperature), 0.05)
    return logits / temperature


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)

    pos_mask = x >= 0
    neg_mask = ~pos_mask

    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))

    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)

    return out


@torch.no_grad()
def run_single_model(model, loader, device, temperature=1.0):
    model.eval()
    rows = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        logits = model(images)["study_logits"].detach().cpu().numpy()
        logits = apply_temperature_np(logits, temperature)
        probs = sigmoid_np(logits)
        labels = batch["labels"].cpu().numpy()
        study_labels = batch["study_labels"].cpu().numpy()
        pair_ids = batch["pair_ids"].cpu().numpy()

        for i in range(len(labels)):
            rows.append({
                "image_id": str(batch["image_ids"][i]),
                "sample_id": str(batch["sample_ids"][i]),
                "pair_name": str(batch["pair_names"][i]),
                "pair_id": int(pair_ids[i]),
                "pair_y_true": int(labels[i]),
                "study_y_true": int(study_labels[i]),
                "pair_logit": float(logits[i]),
                "pair_prob": float(probs[i]),
                "location_raw": batch["location_raw"][i],
            })
    return pd.DataFrame(rows)


def aggregate_fold_predictions(fold_dfs, ensemble_mode):
    base = fold_dfs[0].copy()
    logits = np.stack([df["pair_logit"].values for df in fold_dfs], axis=0)
    probs = np.stack([df["pair_prob"].values for df in fold_dfs], axis=0)

    if ensemble_mode == "logits":
        ens_logits = logits.mean(axis=0)
        ens_probs = sigmoid_np(ens_logits)
    else:
        ens_probs = probs.mean(axis=0)
        ens_probs = np.clip(ens_probs, 1e-6, 1.0 - 1e-6)
        ens_logits = np.log(ens_probs / (1.0 - ens_probs))

    base["pair_logit"] = ens_logits
    base["pair_prob"] = ens_probs
    return base


def safe_binary_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    out = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = np.nan
    try:
        out["ap"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["ap"] = np.nan

    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.update({
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        })
    return out


def build_case_predictions(pair_df, threshold):
    rows = []
    for image_id, g in pair_df.groupby("image_id", sort=False):
        best_idx = g["pair_prob"].idxmax()
        best = g.loc[best_idx]
        row = {
            "image_id": image_id,
            "y_true": int(g["study_y_true"].iloc[0]),
            "y_prob": float(g["pair_prob"].max()),
            "y_logit": float(best["pair_logit"]),
            "y_pred": int(float(g["pair_prob"].max()) >= threshold),
            "top_pair_name": str(best["pair_name"]),
            "top_pair_prob": float(best["pair_prob"]),
            "n_pair_crops": int(len(g)),
            "location_raw": g["location_raw"].iloc[0],
        }
        for pair_name in PAIR_NAMES:
            sub = g[g["pair_name"] == pair_name]
            row[f"pair_prob_{pair_name}"] = float(sub["pair_prob"].iloc[0]) if len(sub) else np.nan
            row[f"pair_pred_{pair_name}"] = int(sub["pair_prob"].iloc[0] >= threshold) if len(sub) else -1
            row[f"pair_true_{pair_name}"] = int(sub["pair_y_true"].iloc[0]) if len(sub) else -1
        rows.append(row)
    case_df = pd.DataFrame(rows)
    case_df["confusion_type"] = "TN"
    case_df.loc[(case_df["y_true"] == 0) & (case_df["y_pred"] == 1), "confusion_type"] = "FP"
    case_df.loc[(case_df["y_true"] == 1) & (case_df["y_pred"] == 0), "confusion_type"] = "FN"
    case_df.loc[(case_df["y_true"] == 1) & (case_df["y_pred"] == 1), "confusion_type"] = "TP"
    return case_df


def compute_top1_location_metrics(case_df):
    rows = []
    for _, row in case_df.iterrows():
        gt_set = location_to_pair_set(row.get("location_raw", ""))
        is_positive = int(row["y_true"]) == 1
        if not is_positive:
            continue
        rows.append({
            "image_id": row["image_id"],
            "gt_pair_set": ";".join(sorted(gt_set)),
            "top_pair_name": row["top_pair_name"],
            "top_pair_prob": row["top_pair_prob"],
            "top1_hit": int(row["top_pair_name"] in gt_set),
            "pred_set_hit": int(any(int(row.get(f"pair_pred_{p}", 0)) == 1 and p in gt_set for p in PAIR_NAMES)),
            "n_gt_pairs": int(len(gt_set)),
        })
    detail = pd.DataFrame(rows)
    if len(detail) == 0:
        return detail, {"n_positive_cases": 0, "top1_acc_positive_cases": np.nan, "pred_set_hit_acc_positive_cases": np.nan}
    metrics = {
        "n_positive_cases": int(len(detail)),
        "top1_acc_positive_cases": float(detail["top1_hit"].mean()),
        "pred_set_hit_acc_positive_cases": float(detail["pred_set_hit"].mean()),
    }
    return detail, metrics


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if str(cfg.get("loss_name", "bce")).lower() != "bce":
        raise ValueError("Pair-crop analysis currently supports BCE checkpoints only.")

    threshold = float(args.threshold if args.threshold is not None else cfg.get("threshold", 0.5))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv_dir = Path(args.cv_dir)

    test_dataset = build_dataset(cfg)
    print(f"[INFO] Test pair samples: {len(test_dataset)}")
    print(f"[INFO] Test cases: {len(set(s['image_id'] for s in test_dataset.samples))}")

    loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        collate_fn=pair_crop_collate_fn,
        pin_memory=True,
    )

    fold_dfs = []
    used_folds = []
    for fold_idx in range(1, args.n_folds + 1):
        ckpt_path = cv_dir / f"fold_{fold_idx}" / args.ckpt_name
        cal_path = cv_dir / f"fold_{fold_idx}" / "calibration_summary.json"
        if not ckpt_path.exists():
            print(f"[WARN] Skip fold {fold_idx}: missing {ckpt_path}")
            continue

        model = build_model(cfg, device)
        state, extra = load_checkpoint_state_dict(str(ckpt_path), device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Fold {fold_idx} missing keys: {missing[:10]}")
        if unexpected:
            print(f"[WARN] Fold {fold_idx} unexpected keys: {unexpected[:10]}")

        cal = {}
        if cal_path.exists():
            with open(cal_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
        temperature = float(args.temperature if args.temperature is not None else cal.get("temperature", extra.get("temperature", 1.0)))

        df = run_single_model(model, loader, device, temperature=temperature)
        df["fold_idx"] = fold_idx
        df["fold_temperature"] = temperature
        fold_dfs.append(df)
        used_folds.append({"fold": fold_idx, "ckpt": str(ckpt_path), "temperature": temperature})
        print(f"[INFO] Loaded fold {fold_idx} | temperature={temperature:.6f}")

    if len(fold_dfs) == 0:
        raise RuntimeError("No fold checkpoints loaded.")

    pair_df = aggregate_fold_predictions(fold_dfs, args.ensemble_mode)
    pair_df["pair_pred"] = (pair_df["pair_prob"] >= threshold).astype(int)
    pair_df["pair_confusion_type"] = "TN"
    pair_df.loc[(pair_df["pair_y_true"] == 0) & (pair_df["pair_pred"] == 1), "pair_confusion_type"] = "FP"
    pair_df.loc[(pair_df["pair_y_true"] == 1) & (pair_df["pair_pred"] == 0), "pair_confusion_type"] = "FN"
    pair_df.loc[(pair_df["pair_y_true"] == 1) & (pair_df["pair_pred"] == 1), "pair_confusion_type"] = "TP"

    case_df = build_case_predictions(pair_df, threshold)

    raw_test_df = pd.read_csv(cfg["test_csv"])
    case_df = case_df.merge(raw_test_df, on="image_id", how="left", suffixes=("", "_csv"))

    presence_metrics = safe_binary_metrics(case_df["y_true"], case_df["y_prob"], threshold)
    location_pair_metrics = safe_binary_metrics(pair_df["pair_y_true"], pair_df["pair_prob"], threshold)
    top1_detail, top1_metrics = compute_top1_location_metrics(case_df)

    prefix = "pair_crop_cv_ensemble"
    pair_df.to_csv(outdir / f"{prefix}_pair_predictions.csv", index=False)
    case_df.to_csv(outdir / f"{prefix}_case_predictions.csv", index=False)
    pd.DataFrame([presence_metrics]).to_csv(outdir / f"{prefix}_presence_metrics.csv", index=False)
    pd.DataFrame([location_pair_metrics]).to_csv(outdir / f"{prefix}_location_pair_metrics.csv", index=False)
    pd.DataFrame([top1_metrics]).to_csv(outdir / f"{prefix}_top1_location_metrics.csv", index=False)
    top1_detail.to_csv(outdir / f"{prefix}_top1_location_detail.csv", index=False)

    pd.crosstab(pair_df["pair_name"], pair_df["pair_confusion_type"], dropna=False).to_csv(
        outdir / f"{prefix}_pair_confusion_by_pair_name.csv"
    )
    wrong_pairs = pair_df[pair_df["pair_y_true"] != pair_df["pair_pred"]].copy()
    wrong_pairs.to_csv(outdir / f"{prefix}_wrong_pair_predictions.csv", index=False)
    wrong_cases = case_df[case_df["y_true"] != case_df["y_pred"]].copy()
    wrong_cases.to_csv(outdir / f"{prefix}_wrong_case_predictions.csv", index=False)

    cv_summary = {}
    cv_summary_path = cv_dir / "cv_summary.json"
    if cv_summary_path.exists():
        with open(cv_summary_path, "r", encoding="utf-8") as f:
            cv_summary = json.load(f)

    meta = {
        "ensemble_mode": args.ensemble_mode,
        "threshold": threshold,
        "used_folds": used_folds,
        "cv_summary": cv_summary,
        "presence_metrics_are_case_level": True,
        "case_probability_rule": "max(pair probabilities for the same image_id)",
        "location_pair_metrics_are_pair_crop_level": True,
    }
    with open(outdir / f"{prefix}_ensemble_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with open(outdir / f"{prefix}_summary_report.txt", "w", encoding="utf-8") as f:
        f.write("Branch: pair_crop_cv_ensemble\n")
        f.write(f"ensemble_mode: {args.ensemble_mode}\n")
        f.write(f"threshold: {threshold}\n")
        f.write("\nPresence metrics, case-level, MAIN metric target\n")
        for k, v in presence_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nLocation metrics, pair-crop level, TEST ANALYSIS ONLY\n")
        for k, v in location_pair_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTop-1 location metrics on positive cases\n")
        for k, v in top1_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nWrong summary\n")
        f.write(f"wrong_cases: {len(wrong_cases)}\n")
        f.write(f"wrong_pairs: {len(wrong_pairs)}\n")

    print("[INFO] Presence metrics:", presence_metrics)
    print("[INFO] Location pair metrics:", location_pair_metrics)
    print("[INFO] Top1 location metrics:", top1_metrics)
    print(f"[INFO] Wrote analysis outputs to {outdir}")


if __name__ == "__main__":
    main()
