# python ./presence/scripts/basic/train_pair_crop_cv.py --config ./presence/configs/resnet50_pair_crop.yaml
"""
Train ResNet50 lumbar spondylolisthesis presence prediction using pair-crop images.

Main differences from lumbar-crop training:
  1. One sample = one adjacent-pair crop image, e.g. image_id_L4-L5.png.
  2. Pair label is generated from the case-level `location` column.
  3. Cross-validation split is done at image_id/case level to avoid leakage.
  4. Validation metrics are aggregated back to case-level presence metrics.
     A case is predicted positive if ANY pair crop probability >= threshold,
     implemented as max(pair_prob) per image_id.
  5. Pair/location metrics are computed for monitoring only and are not the default
     monitor target unless you explicitly set monitor_metric to a pair_* metric.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from collections import Counter
from pathlib import Path

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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from data.resnet50.dataset_pair_crop import (
    PairCropDataset,
    PairCropDatasetWithMask,
    CombinedPairCropDataset,
    PAIR_NAMES,
)
from models.basic import LumbarCropModel
from utils.seed import set_seed


# -------------------------
# Argument / config
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def get_implant_filter_kwargs(cfg):
    """
    Kept for config compatibility with the lumbar-crop script.
    The uploaded PairCropDataset currently does not implement implant filtering,
    so this function only warns if filtering is requested.
    """
    section = cfg.get("implant_filter", {}) or {}
    enabled = bool(section.get("enabled", cfg.get("filter_implant_enabled", False)))
    if enabled:
        print(
            "[WARN] implant_filter.enabled=True, but PairCropDataset does not currently "
            "filter implant categories. Add implant filtering inside dataset_pair_crop.py "
            "if you need the same implant subset behavior as lumbar-crop training."
        )


# -------------------------
# Collate / subset helpers
# -------------------------
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


class CaseSubset(Dataset):
    """Select all pair-crop samples whose image_id is in `case_ids`."""
    def __init__(self, base_dataset, case_ids):
        self.base_dataset = base_dataset
        self.case_ids = set(str(x) for x in case_ids)
        self.indices = [
            i for i, s in enumerate(base_dataset.samples)
            if str(s["image_id"]) in self.case_ids
        ]
        if len(self.indices) == 0:
            raise RuntimeError("CaseSubset is empty. Check image_id matching and pair_crop filenames.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


class OversampledDataset(Dataset):
    """
    Pair-level oversampling after case-level CV split.
    This avoids leakage because the repeated samples are only from the training cases.
    """
    def __init__(self, dataset, repeat_map=None, print_stats=True):
        self.dataset = dataset
        self.repeat_map = repeat_map or {"negative": 1, **{p: 1 for p in PAIR_NAMES}}
        self.indices = []
        original = Counter()
        repeated = Counter()

        for i in range(len(dataset)):
            item = dataset[i]
            key = item["pair_name"] if float(item["label"]) > 0.5 else "negative"
            n = max(1, int(self.repeat_map.get(key, 1)))
            self.indices.extend([i] * n)
            original[key] += 1
            repeated[key] += n

        if print_stats:
            print("[RepeatMap] original train pair distribution:")
            for k, v in sorted(original.items()):
                print(f"  {k}: {v}")
            print("[RepeatMap] repeated train pair distribution:")
            for k, v in sorted(repeated.items()):
                print(f"  {k}: {v}")
            print(f"[RepeatMap] train samples: {len(dataset)} -> {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def build_case_table(dataset) -> pd.DataFrame:
    rows = []
    seen = set()
    for s in dataset.samples:
        image_id = str(s["image_id"])
        if image_id in seen:
            continue
        seen.add(image_id)
        rows.append({
            "image_id": image_id,
            "study_label": int(float(s["study_label"])),
            "location_raw": s.get("location_raw", ""),
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No case rows found in pair-crop dataset.")
    return out


# -------------------------
# Model / loss / checkpoint
# -------------------------
def build_model(cfg, device):
    model = LumbarCropModel(
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        loss_name=cfg.get("loss_name", "bce"),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.0),
    )
    lumbar_ckpt = cfg.get("lumbar_pretrained_ckpt", "")
    if lumbar_ckpt:
        model.load_backbone_checkpoint(lumbar_ckpt)
    return model.to(device)


def build_optimizer(model, cfg):
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(cfg.get("backbone_lr", 1e-4))},
            {"params": head_params, "lr": float(cfg.get("head_lr", 5e-4))},
        ],
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(optimizer, cfg):
    name = str(cfg.get("scheduler_name", "cosine")).lower()
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("plateau_factor", 0.5)),
            patience=int(cfg.get("plateau_patience", 5)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.get("epochs", 100)),
        eta_min=float(cfg.get("min_lr", 1e-6)),
    )


def build_criterion(cfg, device):
    loss_name = str(cfg.get("loss_name", "bce")).lower()
    if loss_name != "bce":
        raise ValueError("Pair-crop script currently supports BCE only. Set loss_name: bce.")
    pos_weight = cfg.get("pos_weight", None)
    if pos_weight is not None:
        pos_weight = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, extra=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "extra": extra or {},
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


# -------------------------
# Metrics
# -------------------------
def _safe_binary_metrics(y_true, y_prob, threshold):
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
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        })
    return out


@torch.no_grad()
def predict_pair_rows(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        logits = model(images)["study_logits"]
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
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
                "pair_logit": float(logits_np[i]),
                "pair_prob": float(probs[i]),
                "location_raw": batch["location_raw"][i],
            })
    return pd.DataFrame(rows)


def compute_case_presence_from_pair_df(pair_df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict]:
    rows = []
    for image_id, g in pair_df.groupby("image_id", sort=False):
        best_idx = g["pair_prob"].astype(float).idxmax()
        best = g.loc[best_idx]
        rows.append({
            "image_id": image_id,
            "y_true": int(g["study_y_true"].iloc[0]),
            "y_prob": float(g["pair_prob"].max()),
            "y_logit": float(g.loc[best_idx, "pair_logit"]),
            "top_pair_name": best["pair_name"],
            "top_pair_prob": float(best["pair_prob"]),
            "n_pair_crops": int(len(g)),
            "location_raw": g["location_raw"].iloc[0],
        })
    case_df = pd.DataFrame(rows)
    case_metrics = _safe_binary_metrics(case_df["y_true"], case_df["y_prob"], threshold)
    return case_df, case_metrics


def compute_pair_location_metrics(pair_df: pd.DataFrame, threshold: float) -> dict:
    metrics = _safe_binary_metrics(pair_df["pair_y_true"], pair_df["pair_prob"], threshold)
    return {f"pair_{k}": v for k, v in metrics.items()}


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm=None):
    model.train()
    losses = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)["study_logits"]
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip_norm is not None and float(grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def validate_pair_crop(model, loader, criterion, device, threshold):
    model.eval()
    losses = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(images)["study_logits"]
        loss = criterion(logits, labels)
        losses.append(float(loss.detach().cpu()))

    pair_df = predict_pair_rows(model, loader, device)
    case_df, case_metrics = compute_case_presence_from_pair_df(pair_df, threshold)
    pair_metrics = compute_pair_location_metrics(pair_df, threshold)

    out = {"loss": float(np.mean(losses)) if losses else np.nan}
    out.update(case_metrics)
    out.update(pair_metrics)
    return out, case_df, pair_df


# -------------------------
# Dataset build
# -------------------------
def build_dataset(cfg, split_key: str, split_name: str):
    csv_key = f"{split_key}_csv"
    img_key = f"{split_key}_image_dir"
    mask_key = f"{split_key}_mask_dir"
    if csv_key not in cfg or img_key not in cfg:
        raise ValueError(f"Missing {csv_key} or {img_key} in config.")

    use_pair_mask = bool(cfg.get("use_pair_mask", cfg.get("use_mask_input", False)))
    dataset_cls = PairCropDatasetWithMask if use_pair_mask else PairCropDataset
    kwargs = dict(
        text_csv_path=cfg[csv_key],
        image_dir=cfg[img_key],
        split=split_name,
        exclude_s1_positive=bool(cfg.get("exclude_s1_positive", False)),
        aug_cfg=cfg.get("augmentation", {}),
    )
    if use_pair_mask:
        kwargs["mask_dir"] = cfg.get(mask_key, None)
    return dataset_cls(**kwargs)


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))
    get_implant_filter_kwargs(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = float(cfg.get("threshold", 0.5))
    num_folds = int(cfg.get("num_folds", 5))
    monitor_metric = str(cfg.get("monitor_metric", "balanced_acc"))

    train_dataset = build_dataset(cfg, "train", "train")
    val_dataset = build_dataset(cfg, "val", "valid")
    full_dataset = CombinedPairCropDataset([train_dataset, val_dataset])

    print(f"[INFO] train pair samples: {len(train_dataset)}")
    print(f"[INFO] valid pair samples: {len(val_dataset)}")
    print(f"[INFO] combined pair samples: {len(full_dataset)}")

    case_table = build_case_table(full_dataset)
    labels = case_table["study_label"].astype(int).values
    if len(np.unique(labels)) < 2:
        raise RuntimeError(f"Need two case-level classes for StratifiedKFold. Counts={dict(Counter(labels))}")

    print("[INFO] case-level label counts:", dict(Counter(labels)))
    print("[INFO] pair-level label counts:", dict(Counter(int(s["label"]) for s in full_dataset.samples)))

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=int(cfg.get("seed", 42)),
    )

    all_fold_metrics = []
    for fold, (train_case_idx, val_case_idx) in enumerate(skf.split(case_table["image_id"].values, labels), start=1):
        print("=" * 80)
        print(f"[INFO] Fold {fold}/{num_folds}")
        print("=" * 80)

        train_case_ids = case_table.iloc[train_case_idx]["image_id"].tolist()
        val_case_ids = case_table.iloc[val_case_idx]["image_id"].tolist()

        train_subset = CaseSubset(full_dataset, train_case_ids)
        val_subset = CaseSubset(full_dataset, val_case_ids)

        if bool(cfg.get("use_repeat_map", False)):
            repeat_map = cfg.get("pair_repeat_map", None)
            train_for_loader = OversampledDataset(
                train_subset,
                repeat_map=repeat_map,
                print_stats=bool(cfg.get("print_repeat_map_stats", True)),
            )
        else:
            train_for_loader = train_subset

        train_loader = DataLoader(
            train_for_loader,
            batch_size=int(cfg.get("batch_size", 8)),
            shuffle=True,
            num_workers=int(cfg.get("num_workers", 4)),
            collate_fn=pair_crop_collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=int(cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 4)),
            collate_fn=pair_crop_collate_fn,
            pin_memory=True,
        )

        model = build_model(cfg, device)
        criterion = build_criterion(cfg, device)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        fold_dir = save_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_path = fold_dir / "best.pt"
        last_path = fold_dir / "last.pt"
        best_metrics_path = fold_dir / "best_metrics.json"

        best_metric = -float("inf")
        best_metrics = None
        bad_epochs = 0
        patience = int(cfg.get("early_stop_patience", 20))
        min_delta = float(cfg.get("early_stop_min_delta", 0.0))

        for epoch in range(1, int(cfg.get("epochs", 100)) + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                grad_clip_norm=cfg.get("grad_clip_norm", None),
            )
            val_metrics, val_case_df, val_pair_df = validate_pair_crop(
                model,
                val_loader,
                criterion,
                device,
                threshold=threshold,
            )

            if scheduler is not None:
                if str(cfg.get("scheduler_name", "cosine")).lower() == "plateau":
                    scheduler.step(float(val_metrics.get(monitor_metric, 0.0)))
                else:
                    scheduler.step()

            msg = (
                f"[Fold {fold} | Epoch {epoch}] "
                f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                f"case_bal_acc={val_metrics.get('balanced_acc', np.nan):.4f} "
                f"case_f1={val_metrics.get('f1', np.nan):.4f} "
                f"case_ap={val_metrics.get('ap', np.nan):.4f} "
                f"pair_f1={val_metrics.get('pair_f1', np.nan):.4f}"
            )
            print(msg)

            score = float(val_metrics.get(monitor_metric, np.nan))
            if np.isnan(score):
                raise RuntimeError(f"monitor_metric={monitor_metric} is NaN or missing in validation metrics.")

            save_checkpoint(
                str(last_path), model, optimizer, scheduler, epoch, best_metric,
                extra={"fold": fold, "monitor_metric": monitor_metric, "last_score": score},
            )

            if score > best_metric + min_delta:
                best_metric = score
                bad_epochs = 0
                best_metrics = dict(val_metrics)
                best_metrics.update({
                    "epoch": epoch,
                    "fold": fold,
                    "monitor_metric": monitor_metric,
                    "monitor_value": best_metric,
                    "n_val_cases": int(val_case_df["image_id"].nunique()),
                    "n_val_pairs": int(len(val_pair_df)),
                })
                save_checkpoint(
                    str(best_path), model, optimizer, scheduler, epoch, best_metric,
                    extra={"fold": fold, "monitor_metric": monitor_metric, "best_metrics": best_metrics},
                )
                with open(best_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(best_metrics, f, indent=2, ensure_ascii=False)
                val_case_df.to_csv(fold_dir / "best_val_case_predictions.csv", index=False)
                val_pair_df.to_csv(fold_dir / "best_val_pair_predictions.csv", index=False)
                print(f"[INFO] Saved fold {fold} best checkpoint to {best_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[INFO] Early stopping fold {fold} at epoch {epoch}.")
                    break

        if best_metrics is None:
            raise RuntimeError(f"Fold {fold} did not produce best metrics.")
        all_fold_metrics.append(best_metrics)

    keys = [
        "loss", "auroc", "ap", "accuracy", "balanced_acc", "precision", "recall", "f1",
        "specificity", "sensitivity", "pair_auroc", "pair_ap", "pair_accuracy",
        "pair_balanced_acc", "pair_precision", "pair_recall", "pair_f1",
    ]
    summary = {}
    print("\n" + "=" * 80)
    print("[CV SUMMARY] case-level metrics are the main training/selection metrics")
    print("=" * 80)
    for key in keys:
        vals = [float(m[key]) for m in all_fold_metrics if key in m and not pd.isna(m[key])]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        summary[key] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"{key}: mean={summary[key]['mean']:.4f}, std={summary[key]['std']:.4f}")

    with open(save_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"fold_metrics": all_fold_metrics, "summary": summary}, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved CV summary to {save_dir / 'cv_summary.json'}")


if __name__ == "__main__":
    main()
