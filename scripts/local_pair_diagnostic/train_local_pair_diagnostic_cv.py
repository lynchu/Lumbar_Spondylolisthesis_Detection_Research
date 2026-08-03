#!/usr/bin/env python3
"""Stage 2 local pair diagnostic branch training.

This trains the design-doc local branch independently from the global screening
model: five pair crops + geometry -> pair location, slip%, ordinal grade, and
pair-MIL case presence.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from data.resnet50.dataset_local_pair_diagnostic import (
    PAIR_NAMES,
    DEFAULT_GEOMETRY_FEATURES,
    LocalPairDiagnosticDataset,
    local_pair_collate_fn,
)
from models.local_pair_diagnostic import LocalPairDiagnosticModel
from utils.seed import set_seed


EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


class CaseSubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.samples = [base_dataset.samples[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


class CombinedLocalPairDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.index_map = []
        self.samples = []
        for ds_idx, ds in enumerate(self.datasets):
            for local_idx, sample in enumerate(ds.samples):
                self.index_map.append((ds_idx, local_idx))
                self.samples.append(sample)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, local_idx = self.index_map[idx]
        return self.datasets[ds_idx][local_idx]


def build_dataset(cfg, split_key: str, split_name: str):
    return LocalPairDiagnosticDataset(
        diag_csv=cfg[f"{split_key}_csv"],
        pair_image_dir=cfg[f"{split_key}_pair_image_dir"],
        feature_csv=cfg[f"{split_key}_feature_csv"],
        split=split_name,
        aug_cfg=cfg.get("augmentation", {}),
        geometry_feature_names=cfg.get("geometry_feature_names", DEFAULT_GEOMETRY_FEATURES),
        require_all_pairs=bool(cfg.get("require_all_pairs", True)),
        slip_column=str(cfg.get("slip_column", "estimated_slip_percentage")),
        slip_raw_column=str(cfg.get("slip_raw_column", "estimated_slip_percentage_raw")),
        slip_outlier_column=str(cfg.get("slip_outlier_column", "estimated_slip_percentage_outlier")),
        exclude_slip_outliers=bool(cfg.get("exclude_slip_outliers", True)),
    )


def base_image_id_from_image_id(image_id):
    s = str(image_id).strip()
    for suffix in ("_clahe", "_hflip", "_inverse", "_rotate"):
        if s.endswith(suffix):
            return s[:-len(suffix)]
    return s


def build_case_table(dataset) -> pd.DataFrame:
    rows = []
    for idx, sample in enumerate(dataset.samples):
        rows.append({
            "dataset_idx": idx,
            "image_id": sample["image_id"],
            "base_image_id": base_image_id_from_image_id(sample["image_id"]),
            "study_label": int(sample["study_label"]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No cases available for CV split.")
    return out


def sample_geometry_array(sample, geometry_feature_names):
    rows = []
    feature_lookup = sample["feature_lookup"]
    image_id = sample["image_id"]
    for pair_name in PAIR_NAMES:
        feat_row = feature_lookup[(image_id, pair_name)]
        vals = []
        for col in geometry_feature_names:
            value = pd.to_numeric(getattr(feat_row, col), errors="coerce")
            vals.append(float(value) if not pd.isna(value) else np.nan)
        rows.append(vals)
    return np.asarray(rows, dtype=np.float32)


def fit_geometry_normalizer(dataset, indices, device, geometry_feature_names):
    arrs = [sample_geometry_array(dataset.samples[idx], geometry_feature_names) for idx in indices]
    geom = np.concatenate(arrs, axis=0).astype(np.float32)  # [n_cases*5, d]
    geom = np.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)
    mean = geom.mean(axis=0)
    std = geom.std(axis=0)
    std[~np.isfinite(std) | (std < EPS)] = 1.0
    return (
        torch.tensor(mean, dtype=torch.float32, device=device).view(1, 1, -1),
        torch.tensor(std, dtype=torch.float32, device=device).view(1, 1, -1),
    )


def normalize_geometry(geometry, mean, std):
    geometry = torch.nan_to_num(geometry.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return (geometry - mean) / std


def build_model(cfg, geometry_dim, device):
    model = LocalPairDiagnosticModel(
        geometry_dim=geometry_dim,
        backbone_name=str(cfg.get("backbone_name", "resnet18")),
        backbone_pretrained=bool(cfg.get("imagenet_pretrained", False)),
        token_dim=int(cfg.get("token_dim", 256)),
        geo_hidden_dim=int(cfg.get("geo_hidden_dim", 64)),
        transformer_layers=int(cfg.get("transformer_layers", 2)),
        transformer_heads=int(cfg.get("transformer_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    ckpt = str(cfg.get("local_backbone_ckpt", "") or "")
    if ckpt:
        model.load_backbone_checkpoint(ckpt)
    return model.to(device)


def build_optimizer(model, cfg):
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(cfg.get("backbone_lr", 5e-5))},
            {"params": head_params, "lr": float(cfg.get("head_lr", 3e-4))},
        ],
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(optimizer, cfg):
    name = str(cfg.get("scheduler_name", "cosine")).lower()
    if name == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.get("epochs", 80)),
        eta_min=float(cfg.get("min_lr", 1e-6)),
    )


def compute_pos_weight_from_subset(dataset, indices, device):
    labels = []
    study = []
    for idx in indices:
        sample = dataset.samples[idx]
        pair_set = sample["pair_set"]
        labels.append([1.0 if pair_name in pair_set else 0.0 for pair_name in PAIR_NAMES])
        study.append(float(sample["study_label"]))
    y = np.asarray(labels, dtype=np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = neg / np.maximum(pos, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 50.0)
    study = np.asarray(study)
    study_pos = study.sum()
    study_neg = len(study) - study_pos
    study_pos_weight = study_neg / max(study_pos, 1.0)
    return (
        torch.tensor(pos_weight, dtype=torch.float32, device=device),
        torch.tensor([study_pos_weight], dtype=torch.float32, device=device),
        {PAIR_NAMES[i]: {"pos": int(pos[i]), "neg": int(neg[i]), "pos_weight": float(pos_weight[i])} for i in range(len(PAIR_NAMES))},
    )


def coral_targets(grade_targets):
    thresholds = torch.arange(4, device=grade_targets.device).view(1, 1, 4)
    return (grade_targets.unsqueeze(-1).float() > thresholds.float()).float()


def masked_mean(loss, mask):
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def compute_losses(outputs, batch, criteria, weights):
    pair_labels = batch["pair_labels"]
    study_label = batch["study_label"]
    slip_targets = torch.nan_to_num(batch["slip_targets"], nan=0.0, posinf=100.0, neginf=0.0)
    slip_mask = batch["slip_loss_mask"]
    grade_targets = batch["grade_targets"]

    loss_location = criteria["location"](outputs["location_logits"], pair_labels)
    loss_pair_mil = criteria["pair_mil"](outputs["pair_presence_logit_mil"], study_label)

    slip_loss_raw = criteria["slip"](outputs["slip_percentage"], slip_targets)
    loss_slip = masked_mean(slip_loss_raw, slip_mask)

    ordinal_target = coral_targets(grade_targets)
    ordinal_loss_raw = criteria["ordinal"](outputs["ordinal_logits"], ordinal_target).mean(dim=-1)
    loss_ordinal = masked_mean(ordinal_loss_raw, torch.ones_like(pair_labels))

    total = (
        float(weights.get("location", 1.0)) * loss_location
        + float(weights.get("slip", 0.5)) * loss_slip
        + float(weights.get("ordinal", 0.1)) * loss_ordinal
        + float(weights.get("pair_mil", 0.3)) * loss_pair_mil
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_location": float(loss_location.detach().cpu()),
        "loss_pair_mil": float(loss_pair_mil.detach().cpu()),
        "loss_slip": float(loss_slip.detach().cpu()),
        "loss_ordinal": float(loss_ordinal.detach().cpu()),
    }


def safe_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    out = {
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
            "specificity": float(tn / max(tn + fp, 1)),
            "sensitivity": float(tp / max(tp + fn, 1)),
        })
    return out


@torch.no_grad()
def predict_rows(model, loader, device, geo_mean, geo_std):
    model.eval()
    case_rows = []
    pair_rows = []
    for batch in loader:
        pair_images = batch["pair_images"].to(device, non_blocking=True)
        geometry = normalize_geometry(batch["geometry_features"].to(device, non_blocking=True), geo_mean, geo_std)
        outputs = model(pair_images, geometry)
        loc_probs = outputs["location_probs"].detach().cpu().numpy()
        loc_logits = outputs["location_logits"].detach().cpu().numpy()
        slip_pred = outputs["slip_percentage"].detach().cpu().numpy()
        mil_prob = outputs["pair_presence_prob_mil"].detach().cpu().numpy()
        y_pair = batch["pair_labels"].cpu().numpy()
        y_case = batch["study_label"].cpu().numpy()
        slip_target = batch["slip_targets"].cpu().numpy()
        slip_mask = batch["slip_loss_mask"].cpu().numpy()
        slip_outlier = batch["slip_outlier"].cpu().numpy()

        for b, image_id in enumerate(batch["image_ids"]):
            top_idx = int(np.argmax(loc_probs[b]))
            case_rows.append({
                "image_id": image_id,
                "y_true": int(y_case[b]),
                "y_prob_pair_mil": float(mil_prob[b]),
                "top_pair_name": PAIR_NAMES[top_idx],
                "top_pair_prob": float(loc_probs[b, top_idx]),
                "top_pair_slip_pred": float(slip_pred[b, top_idx]),
                "location_raw": batch["location_raw"][b],
            })
            for p, pair_name in enumerate(PAIR_NAMES):
                pair_rows.append({
                    "image_id": image_id,
                    "pair_name": pair_name,
                    "pair_idx": p,
                    "pair_y_true": int(y_pair[b, p]),
                    "pair_logit": float(loc_logits[b, p]),
                    "pair_prob": float(loc_probs[b, p]),
                    "slip_pred": float(slip_pred[b, p]),
                    "slip_target": float(slip_target[b, p]),
                    "slip_loss_mask": int(slip_mask[b, p] > 0.5),
                    "slip_outlier": int(slip_outlier[b, p] > 0.5),
                })
    return pd.DataFrame(case_rows), pd.DataFrame(pair_rows)


def summarize_predictions(case_df, pair_df, threshold):
    metrics = {f"case_mil_{k}": v for k, v in safe_binary_metrics(case_df["y_true"], case_df["y_prob_pair_mil"], threshold).items()}
    metrics.update({f"pair_{k}": v for k, v in safe_binary_metrics(pair_df["pair_y_true"], pair_df["pair_prob"], threshold).items()})

    for pair_name in PAIR_NAMES:
        g = pair_df[pair_df["pair_name"] == pair_name]
        if len(g) > 0:
            pm = safe_binary_metrics(g["pair_y_true"], g["pair_prob"], threshold)
            metrics[f"{pair_name}_ap"] = pm.get("ap", np.nan)
            metrics[f"{pair_name}_sensitivity"] = pm.get("sensitivity", np.nan)
            metrics[f"{pair_name}_specificity"] = pm.get("specificity", np.nan)
            metrics[f"{pair_name}_f1"] = pm.get("f1", np.nan)

    pos_cases = pair_df[pair_df["pair_y_true"] == 1]["image_id"].unique()
    if len(pos_cases) > 0:
        correct = 0
        total = 0
        for image_id, g in pair_df[pair_df["image_id"].isin(pos_cases)].groupby("image_id"):
            top = g.loc[g["pair_prob"].idxmax()]
            correct += int(top["pair_y_true"] == 1)
            total += 1
        metrics["top1_positive_location_acc"] = float(correct / max(total, 1))
    else:
        metrics["top1_positive_location_acc"] = np.nan

    slip_eval = pair_df[pair_df["slip_loss_mask"] == 1].copy()
    if len(slip_eval) > 0:
        metrics["slip_mae_positive_nonoutlier"] = float(np.mean(np.abs(slip_eval["slip_pred"] - slip_eval["slip_target"])))
        metrics["n_slip_eval_pairs"] = int(len(slip_eval))
    else:
        metrics["slip_mae_positive_nonoutlier"] = np.nan
        metrics["n_slip_eval_pairs"] = 0
    return metrics


def train_one_epoch(model, loader, optimizer, device, geo_mean, geo_std, criteria, weights, grad_clip=None):
    model.train()
    losses = []
    for batch in loader:
        pair_images = batch["pair_images"].to(device, non_blocking=True)
        geometry = normalize_geometry(batch["geometry_features"].to(device, non_blocking=True), geo_mean, geo_std)
        batch_dev = {
            "pair_labels": batch["pair_labels"].to(device, non_blocking=True),
            "study_label": batch["study_label"].to(device, non_blocking=True),
            "slip_targets": batch["slip_targets"].to(device, non_blocking=True),
            "slip_loss_mask": batch["slip_loss_mask"].to(device, non_blocking=True),
            "grade_targets": batch["grade_targets"].to(device, non_blocking=True),
        }
        optimizer.zero_grad(set_to_none=True)
        outputs = model(pair_images, geometry)
        loss, loss_dict = compute_losses(outputs, batch_dev, criteria, weights)
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        losses.append(loss_dict)
    return {k: float(np.mean([d[k] for d in losses])) for k in losses[0]} if losses else {}


@torch.no_grad()
def validate(model, loader, device, geo_mean, geo_std, criteria, weights, threshold):
    model.eval()
    losses = []
    for batch in loader:
        pair_images = batch["pair_images"].to(device, non_blocking=True)
        geometry = normalize_geometry(batch["geometry_features"].to(device, non_blocking=True), geo_mean, geo_std)
        batch_dev = {
            "pair_labels": batch["pair_labels"].to(device, non_blocking=True),
            "study_label": batch["study_label"].to(device, non_blocking=True),
            "slip_targets": batch["slip_targets"].to(device, non_blocking=True),
            "slip_loss_mask": batch["slip_loss_mask"].to(device, non_blocking=True),
            "grade_targets": batch["grade_targets"].to(device, non_blocking=True),
        }
        outputs = model(pair_images, geometry)
        _, loss_dict = compute_losses(outputs, batch_dev, criteria, weights)
        losses.append(loss_dict)
    case_df, pair_df = predict_rows(model, loader, device, geo_mean, geo_std)
    metrics = summarize_predictions(case_df, pair_df, threshold)
    for k in losses[0]:
        metrics[k] = float(np.mean([d[k] for d in losses]))
    return metrics, case_df, pair_df


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, geo_mean, geo_std, feature_names, extra=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "geometry_mean": geo_mean.detach().cpu().numpy().reshape(-1).tolist(),
        "geometry_std": geo_std.detach().cpu().numpy().reshape(-1).tolist(),
        "geometry_feature_names": list(feature_names),
        "extra": extra or {},
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint_state(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    set_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = float(cfg.get("threshold", 0.5))
    num_folds = int(cfg.get("num_folds", 5))
    monitor_metric = str(cfg.get("monitor_metric", "pair_ap"))
    geometry_names = cfg.get("geometry_feature_names", DEFAULT_GEOMETRY_FEATURES)
    loss_weights = cfg.get("loss_weights", {"location": 1.0, "slip": 0.5, "ordinal": 0.1, "pair_mil": 0.3})
    run_folds = cfg.get("run_folds", None)
    if run_folds is not None:
        run_folds = {int(x) for x in run_folds}

    train_dataset = build_dataset(cfg, "train", "train")
    val_dataset = build_dataset(cfg, "val", "valid")
    # Keep original train+valid CV behavior while preserving each split's own pair_image_dir.
    full_dataset = CombinedLocalPairDataset([train_dataset, val_dataset])

    case_table = build_case_table(full_dataset)
    labels = case_table["study_label"].astype(int).values
    groups = case_table["base_image_id"].astype(str).values
    if len(np.unique(labels)) < 2:
        raise RuntimeError(f"Need two classes for StratifiedKFold. Counts={dict(Counter(labels))}")
    print("[INFO] case label counts:", dict(Counter(labels)))
    print(f"[INFO] full CV cases={len(full_dataset)} geometry_dim={len(geometry_names)}")

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    group_cv_by_base_image_id = bool(cfg.get("group_cv_by_base_image_id", True))
    if group_cv_by_base_image_id:
        skf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=int(cfg.get("seed", 42)))
        split_iter = skf.split(case_table["image_id"].values, labels, groups=groups)
        print("[INFO] CV split mode: StratifiedGroupKFold grouped by base_image_id")
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=int(cfg.get("seed", 42)))
        split_iter = skf.split(case_table["image_id"].values, labels)
        print("[INFO] CV split mode: StratifiedKFold by image_id sample, no base_image_id grouping")

    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        fold_dir = save_dir / f"fold_{fold}"
        if run_folds is not None and fold not in run_folds:
            metrics_path = fold_dir / "best_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    fold_metrics.append(json.load(f))
                print(f"[INFO] Skipping fold {fold}; loaded existing metrics from {metrics_path}")
                continue
            raise RuntimeError(f"run_folds excludes fold {fold}, but {metrics_path} does not exist.")

        print("=" * 80)
        print(f"[INFO] Fold {fold}/{num_folds}")
        print("=" * 80)

        train_indices = case_table.iloc[tr_idx]["dataset_idx"].tolist()
        val_indices = case_table.iloc[va_idx]["dataset_idx"].tolist()
        train_groups = set(case_table.iloc[tr_idx]["base_image_id"].astype(str))
        val_groups = set(case_table.iloc[va_idx]["base_image_id"].astype(str))
        overlap_groups = train_groups & val_groups
        if group_cv_by_base_image_id and overlap_groups:
            raise RuntimeError(f"Grouped CV leakage detected in fold {fold}: {sorted(list(overlap_groups))[:10]}")
        print(
            f"[INFO] Fold {fold} split: train_cases={len(train_indices)} val_cases={len(val_indices)} "
            f"train_base_ids={len(train_groups)} val_base_ids={len(val_groups)} "
            f"overlap_base_ids={len(overlap_groups)} group_cv_by_base_image_id={group_cv_by_base_image_id}"
        )
        train_subset = CaseSubset(full_dataset, train_indices)
        val_subset = CaseSubset(full_dataset, val_indices)
        geo_mean, geo_std = fit_geometry_normalizer(full_dataset, train_indices, device, geometry_names)

        train_loader = DataLoader(
            train_subset,
            batch_size=int(cfg.get("batch_size", 16)),
            shuffle=True,
            num_workers=int(cfg.get("num_workers", 4)),
            collate_fn=local_pair_collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=int(cfg.get("batch_size", 16)),
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 4)),
            collate_fn=local_pair_collate_fn,
            pin_memory=True,
        )

        model = build_model(cfg, geometry_dim=len(geometry_names), device=device)
        pair_pos_weight, study_pos_weight, pos_summary = compute_pos_weight_from_subset(full_dataset, train_indices, device)
        if not bool(cfg.get("use_per_level_pos_weight", True)):
            pair_pos_weight = None
        if not bool(cfg.get("use_case_pos_weight", True)):
            study_pos_weight = None
        print("[INFO] per-level train pos_weight:", pos_summary)
        print("[INFO] case pos_weight:", None if study_pos_weight is None else float(study_pos_weight.item()))

        criteria = {
            "location": nn.BCEWithLogitsLoss(pos_weight=pair_pos_weight),
            "pair_mil": nn.BCEWithLogitsLoss(pos_weight=study_pos_weight),
            "slip": nn.SmoothL1Loss(reduction="none", beta=float(cfg.get("slip_smooth_l1_beta", 5.0))),
            "ordinal": nn.BCEWithLogitsLoss(reduction="none"),
        }
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        fold_dir.mkdir(parents=True, exist_ok=True)
        best_metric = -float("inf")
        best_metrics = None
        bad_epochs = 0
        patience = int(cfg.get("early_stop_patience", 15))
        min_delta = float(cfg.get("early_stop_min_delta", 0.0))
        max_epochs = int(cfg.get("epochs", 80))
        start_epoch = 1

        if bool(cfg.get("resume_from_last", False)):
            last_path = fold_dir / "last.pt"
            metrics_path = fold_dir / "best_metrics.json"
            if last_path.exists():
                ckpt = load_checkpoint_state(last_path, model, optimizer, scheduler, device)
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                if metrics_path.exists():
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        best_metrics = json.load(f)
                    best_metric = float(best_metrics.get("monitor_value", best_metrics.get(monitor_metric, -float("inf"))))
                else:
                    best_metric = float(ckpt.get("best_metric", -float("inf")))
                print(
                    f"[INFO] Resuming fold {fold} from {last_path} at epoch {start_epoch}/{max_epochs}; "
                    f"current best {monitor_metric}={best_metric:.6f}"
                )
            else:
                print(f"[INFO] resume_from_last=true, but {last_path} does not exist; starting fold {fold} from epoch 1.")

        for epoch in range(start_epoch, max_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, geo_mean, geo_std, criteria, loss_weights,
                grad_clip=cfg.get("grad_clip_norm", None),
            )
            val_metrics, val_case_df, val_pair_df = validate(
                model, val_loader, device, geo_mean, geo_std, criteria, loss_weights, threshold=threshold,
            )
            if scheduler is not None:
                scheduler.step()

            score = float(val_metrics.get(monitor_metric, np.nan))
            if np.isnan(score):
                raise RuntimeError(f"monitor_metric={monitor_metric} is missing/NaN. Available keys={sorted(val_metrics)}")
            print(
                f"[Fold {fold} | Epoch {epoch}] "
                f"train_loss={train_loss.get('loss', np.nan):.4f} val_loss={val_metrics.get('loss', np.nan):.4f} "
                f"pair_ap={val_metrics.get('pair_ap', np.nan):.4f} pair_f1={val_metrics.get('pair_f1', np.nan):.4f} "
                f"case_mil_bal_acc={val_metrics.get('case_mil_balanced_acc', np.nan):.4f} "
                f"top1={val_metrics.get('top1_positive_location_acc', np.nan):.4f} "
                f"slip_mae={val_metrics.get('slip_mae_positive_nonoutlier', np.nan):.3f}"
            )

            save_checkpoint(
                fold_dir / "last.pt", model, optimizer, scheduler, epoch, best_metric, geo_mean, geo_std,
                geometry_names, extra={"fold": fold, "monitor_metric": monitor_metric, "last_score": score},
            )
            if score > best_metric + min_delta:
                best_metric = score
                bad_epochs = 0
                best_metrics = dict(val_metrics)
                best_metrics.update({"epoch": epoch, "fold": fold, "monitor_metric": monitor_metric, "monitor_value": best_metric})
                save_checkpoint(
                    fold_dir / "best.pt", model, optimizer, scheduler, epoch, best_metric, geo_mean, geo_std,
                    geometry_names, extra={"fold": fold, "monitor_metric": monitor_metric, "best_metrics": best_metrics},
                )
                with open(fold_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(best_metrics, f, indent=2, ensure_ascii=False)
                val_case_df.to_csv(fold_dir / "best_val_case_predictions.csv", index=False)
                val_pair_df.to_csv(fold_dir / "best_val_pair_predictions.csv", index=False)
                print(f"[INFO] Saved fold {fold} best checkpoint: {fold_dir / 'best.pt'}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[INFO] Early stopping fold {fold} at epoch {epoch}")
                    break

        if best_metrics is None:
            raise RuntimeError(f"Fold {fold} did not produce best metrics.")
        fold_metrics.append(best_metrics)
        with open(fold_dir / "fold_complete.json", "w", encoding="utf-8") as f:
            json.dump({
                "fold": fold,
                "monitor_metric": monitor_metric,
                "monitor_value": float(best_metrics.get("monitor_value", best_metrics.get(monitor_metric, np.nan))),
                "best_epoch": int(best_metrics.get("epoch", -1)),
            }, f, indent=2, ensure_ascii=False)

    summary = {}
    for key in sorted({k for m in fold_metrics for k in m.keys()}):
        vals = []
        for m in fold_metrics:
            if key not in m:
                continue
            try:
                value = float(m[key])
            except (TypeError, ValueError):
                continue
            if not pd.isna(value):
                vals.append(value)
        if vals:
            arr = np.asarray(vals, dtype=float)
            summary[key] = {"mean": float(arr.mean()), "std": float(arr.std())}
    with open(save_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"fold_metrics": fold_metrics, "summary": summary}, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved CV summary: {save_dir / 'cv_summary.json'}")


if __name__ == "__main__":
    main()
