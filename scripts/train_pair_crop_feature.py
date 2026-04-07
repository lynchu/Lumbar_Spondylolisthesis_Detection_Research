#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from data.dataset_pair_crop import (
    PAIR_NAME_TO_ID,
    PairCropLevelDataset,
    get_transform,
)
from engine.metrics import compute_binary_metrics
from models.backbone_resnet import ResNet50FeatureExtractor


EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def infer_feature_csv_from_diag_csv(diag_csv_path: str) -> str:
    return str(Path(diag_csv_path).with_name("pair_crop_features.csv"))


class PairCropImageFeatureDataset(Dataset):
    """
    Pair-level dataset with:
      - image tensor
      - binary pair label
      - numeric features (selected columns from pair_crop_features.csv)

    Keys returned:
      image, label, num_feat, pair_id, pair_name, image_id, sample_id,
      study_label, location_raw
    """
    def __init__(
        self,
        text_csv_path: str,
        image_dir: str,
        feature_csv_path: str,
        feature_names,
        split: str = "test",
        transform=None,
        exclude_s1_positive: bool = True,
        aug_cfg=None,
    ):
        self.base_dataset = PairCropLevelDataset(
            text_csv_path=text_csv_path,
            image_dir=image_dir,
            split=split,
            transform=transform if transform is not None else get_transform(split, aug_cfg=aug_cfg),
            exclude_s1_positive=exclude_s1_positive,
            aug_cfg=aug_cfg,
        )
        self.image_dir = image_dir
        self.transform = self.base_dataset.transform
        self.samples = [dict(x) for x in self.base_dataset.samples]
        self.feature_names = list(feature_names)

        feat_df = pd.read_csv(feature_csv_path)
        required_cols = {"image_id", "pair_name", *self.feature_names}
        missing = [c for c in required_cols if c not in feat_df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {feature_csv_path}: {missing}")

        feat_df = feat_df[["image_id", "pair_name", *self.feature_names]].copy()
        feat_df["image_id"] = feat_df["image_id"].astype(str)
        feat_df["pair_name"] = feat_df["pair_name"].astype(str)

        # keep first if duplicated
        feat_df = feat_df.drop_duplicates(subset=["image_id", "pair_name"], keep="first").copy()

        self.feature_map = {}
        for _, row in feat_df.iterrows():
            key = (str(row["image_id"]), str(row["pair_name"]))
            arr = np.array([pd.to_numeric(row[f], errors="coerce") for f in self.feature_names], dtype=np.float32)
            self.feature_map[key] = arr

        # attach raw numeric features to sample metadata
        kept_samples = []
        n_missing_feature_rows = 0
        for s in self.samples:
            key = (str(s["image_id"]), str(s["pair_name"]))
            if key not in self.feature_map:
                n_missing_feature_rows += 1
                continue
            s["raw_num_feat"] = self.feature_map[key]
            kept_samples.append(s)

        self.samples = kept_samples

        if n_missing_feature_rows > 0:
            print(f"[WARN] Dropped {n_missing_feature_rows} samples due to missing pair features in {feature_csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.image_dir, item["filename"])

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(float(item["label"]), dtype=torch.float32),
            "num_feat": torch.tensor(item["raw_num_feat"], dtype=torch.float32),
            "pair_id": torch.tensor(int(item["pair_id"]), dtype=torch.long),
            "pair_name": item["pair_name"],
            "image_id": item["image_id"],
            "sample_id": item["sample_id"],
            "study_label": torch.tensor(float(item["study_label"]), dtype=torch.float32),
            "location_raw": item["location_raw"],
        }


class CombinedPairCropFeatureDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.samples = []

        for ds in self.datasets:
            for sample in ds.samples:
                merged = dict(sample)
                merged["_image_dir"] = ds.image_dir
                self.samples.append(merged)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Use PairCropFeatureSubset for fold-specific transform/scaler handling.")


class PairCropFeatureSubset(Dataset):
    def __init__(self, full_dataset, indices, transform, feat_mean, feat_std):
        self.full_dataset = full_dataset
        self.indices = list(indices)
        self.transform = transform
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = feat_std.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.full_dataset.samples[self.indices[idx]]
        img_path = os.path.join(item["_image_dir"], item["filename"])

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        raw_feat = np.array(item["raw_num_feat"], dtype=np.float32)
        raw_feat = np.where(np.isnan(raw_feat), self.feat_mean, raw_feat)
        norm_feat = (raw_feat - self.feat_mean) / (self.feat_std + EPS)

        return {
            "image": img,
            "label": torch.tensor(float(item["label"]), dtype=torch.float32),
            "num_feat": torch.tensor(norm_feat, dtype=torch.float32),
            "pair_id": torch.tensor(int(item["pair_id"]), dtype=torch.long),
            "pair_name": item["pair_name"],
            "image_id": item["image_id"],
            "sample_id": item["sample_id"],
            "study_label": torch.tensor(float(item["study_label"]), dtype=torch.float32),
            "location_raw": item["location_raw"],
        }


class PairCropImageFeatureFusionModel(nn.Module):
    """
    Two-branch model:
      1) image branch: ResNet50 -> pooled feature
      2) numeric feature branch: FC -> 64-d embedding -> ReLU
      concat -> FC/ReLU/Dropout -> binary logit
    """
    def __init__(
        self,
        num_features: int,
        backbone_pretrained: bool = False,
        dropout: float = 0.2,
        num_feat_hidden_dim: int = 64,
        fusion_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = ResNet50FeatureExtractor(pretrained=backbone_pretrained)
        image_dim = self.backbone.feature_dim

        self.num_branch = nn.Sequential(
            nn.Linear(num_features, num_feat_hidden_dim),
            nn.ReLU(inplace=True),
        )

        if fusion_hidden_dim is None:
            fusion_hidden_dim = image_dim // 2

        self.fusion_head = nn.Sequential(
            nn.Linear(image_dim + num_feat_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, images, num_feat):
        img_feat = self.backbone(images)
        num_emb = self.num_branch(num_feat)
        fused = torch.cat([img_feat, num_emb], dim=1)
        logits = self.fusion_head(fused).squeeze(-1)

        return {
            "logits": logits,
            "image_features": img_feat,
            "num_embeddings": num_emb,
            "fused_features": fused,
        }


class SimpleEarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.bad_count = 0

    def step(self, score):
        if self.best is None:
            self.best = score
            self.bad_count = 0
            return False

        if self.mode == "max":
            improved = score > self.best + self.min_delta
        else:
            improved = score < self.best - self.min_delta

        if improved:
            self.best = score
            self.bad_count = 0
        else:
            self.bad_count += 1

        return self.bad_count >= self.patience


def build_optimizer(model, backbone_lr, head_lr, weight_decay):
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))


def save_checkpoint(path, model, optimizer, epoch, best_metric, extra=None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def get_labels_from_dataset(dataset):
    return np.array([int(s["label"]) for s in dataset.samples], dtype=int)


def get_groups_from_dataset(dataset):
    return np.array([str(s["image_id"]) for s in dataset.samples])


def compute_train_fold_scaler(full_dataset, train_idx, num_features):
    feats = []
    for i in train_idx:
        x = np.array(full_dataset.samples[int(i)]["raw_num_feat"], dtype=np.float32)
        feats.append(x)
    feats = np.stack(feats, axis=0) if len(feats) > 0 else np.zeros((0, num_features), dtype=np.float32)

    feat_mean = np.nanmean(feats, axis=0)
    feat_std = np.nanstd(feats, axis=0)

    feat_mean = np.where(np.isnan(feat_mean), 0.0, feat_mean).astype(np.float32)
    feat_std = np.where(np.isnan(feat_std) | (feat_std < EPS), 1.0, feat_std).astype(np.float32)

    return feat_mean, feat_std


def compute_pos_weight_from_indices(full_dataset, train_idx):
    labels = np.array([int(full_dataset.samples[int(i)]["label"]) for i in train_idx], dtype=np.float32)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos < 1.0:
        return None
    return n_neg / max(n_pos, 1.0)


def print_dataset_sanity(dataset, name):
    num_samples = len(dataset.samples)
    num_pos = sum(int(s["label"]) for s in dataset.samples)
    num_neg = num_samples - num_pos

    pair_pos_counts = {}
    for pair_name in ["L1-L2", "L2-L3", "L3-L4", "L4-L5"]:
        pair_pos_counts[pair_name] = sum(
            int(s["label"]) for s in dataset.samples if s["pair_name"] == pair_name
        )

    print(f"[{name}] samples={num_samples}, pos={num_pos}, neg={num_neg}")
    print(f"[{name}] positive counts by pair: {pair_pos_counts}")

    preview = []
    for s in dataset.samples:
        if int(s["label"]) == 1:
            preview.append(
                (s["image_id"], s["location_raw"], s["pair_name"], int(s["label"]), s["raw_num_feat"].tolist())
            )
        if len(preview) >= 5:
            break

    if len(preview) > 0:
        print(f"[{name}] positive sample preview:")
        for x in preview:
            print("   ", x)
    else:
        print(f"[{name}] WARNING: no positive samples found.")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        num_feat = batch["num_feat"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images, num_feat)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    running_loss = 0.0
    y_true = []
    y_prob = []

    for batch in loader:
        images = batch["image"].to(device)
        num_feat = batch["num_feat"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images, num_feat)
        loss = criterion(outputs["logits"], labels)
        probs = torch.sigmoid(outputs["logits"])

        running_loss += loss.item() * images.size(0)
        y_true.append(labels.detach().cpu().numpy())
        y_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true) if len(y_true) > 0 else np.array([])
    y_prob = np.concatenate(y_prob) if len(y_prob) > 0 else np.array([])

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


def build_model(cfg, device, num_features):
    model = PairCropImageFeatureFusionModel(
        num_features=num_features,
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        dropout=cfg.get("dropout", 0.2),
        num_feat_hidden_dim=cfg.get("num_feat_hidden_dim", 64),
        fusion_hidden_dim=cfg.get("fusion_hidden_dim", None),
    )
    lumbar_ckpt = cfg.get("lumbar_pretrained_ckpt", "")
    if lumbar_ckpt:
        model.backbone.load_partial_checkpoint(lumbar_ckpt)
    return model.to(device)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    feature_names = cfg.get("feature_names", ["abs_dx_slip_px", "abs_dx_slip_ratio_wavg"])
    if not isinstance(feature_names, (list, tuple)) or len(feature_names) == 0:
        raise ValueError("config field 'feature_names' must be a non-empty list.")

    train_feature_csv = cfg.get("train_feature_csv", infer_feature_csv_from_diag_csv(cfg["train_csv"]))
    val_feature_csv = cfg.get("val_feature_csv", infer_feature_csv_from_diag_csv(cfg["val_csv"]))

    train_dataset = PairCropImageFeatureDataset(
        text_csv_path=cfg["train_csv"],
        image_dir=cfg["train_image_dir"],
        feature_csv_path=train_feature_csv,
        feature_names=feature_names,
        split="train",
        exclude_s1_positive=cfg.get("exclude_s1_positive", True),
        aug_cfg=cfg.get("augmentation", {}),
    )
    val_dataset = PairCropImageFeatureDataset(
        text_csv_path=cfg["val_csv"],
        image_dir=cfg["val_image_dir"],
        feature_csv_path=val_feature_csv,
        feature_names=feature_names,
        split="valid",
        exclude_s1_positive=cfg.get("exclude_s1_positive", True),
        aug_cfg=cfg.get("augmentation", {}),
    )

    print_dataset_sanity(train_dataset, "train_dataset")
    print_dataset_sanity(val_dataset, "val_dataset")
    print(f"[INFO] Using numeric features: {feature_names}")

    full_dataset = CombinedPairCropFeatureDataset([train_dataset, val_dataset])
    labels = get_labels_from_dataset(full_dataset)
    groups = get_groups_from_dataset(full_dataset)

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.get("num_folds", 5),
        shuffle=True,
        random_state=seed,
    )

    base_save_dir = cfg["save_dir"]
    os.makedirs(base_save_dir, exist_ok=True)

    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(np.arange(len(full_dataset)), labels, groups=groups),
        start=1,
    ):
        print("=" * 80)
        print(f"[INFO] Fold {fold}/{cfg.get('num_folds', 5)}")
        print("=" * 80)

        feat_mean, feat_std = compute_train_fold_scaler(
            full_dataset=full_dataset,
            train_idx=train_idx,
            num_features=len(feature_names),
        )

        train_subset = PairCropFeatureSubset(
            full_dataset=full_dataset,
            indices=train_idx.tolist(),
            transform=train_dataset.transform,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        val_subset = PairCropFeatureSubset(
            full_dataset=full_dataset,
            indices=val_idx.tolist(),
            transform=val_dataset.transform,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )

        model = build_model(cfg, device, num_features=len(feature_names))

        pos_weight_cfg = cfg.get("pos_weight", None)
        if pos_weight_cfg == "auto":
            pos_weight_value = compute_pos_weight_from_indices(full_dataset, train_idx)
        else:
            pos_weight_value = pos_weight_cfg

        pos_weight = None
        if pos_weight_value is not None:
            pos_weight = torch.tensor([float(pos_weight_value)], device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = build_optimizer(
            model,
            backbone_lr=cfg.get("backbone_lr", 1e-4),
            head_lr=cfg.get("head_lr", 5e-4),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        scheduler = build_scheduler(optimizer, epochs=cfg["epochs"])

        fold_dir = os.path.join(base_save_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        best_path = os.path.join(fold_dir, "best.pt")

        with open(os.path.join(fold_dir, "feature_scaler.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_names": list(feature_names),
                    "feat_mean": feat_mean.tolist(),
                    "feat_std": feat_std.tolist(),
                },
                f,
                indent=2,
            )

        monitor_metric = cfg.get("monitor_metric", "auroc")
        best_metric = -1.0
        best_metrics = None

        stopper = SimpleEarlyStopping(
            patience=cfg.get("early_stop_patience", 10),
            mode="max",
            min_delta=cfg.get("early_stop_min_delta", 0.0),
        )

        for epoch in range(1, cfg["epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                threshold=cfg.get("threshold", 0.5),
            )
            scheduler.step()

            print(
                f"[Fold {fold} | Epoch {epoch}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"auroc={val_metrics['auroc']:.4f} "
                f"ap={val_metrics['ap']:.4f} "
                f"f1={val_metrics['f1']:.4f} "
                f"bal_acc={val_metrics['balanced_acc']:.4f}"
            )

            score = val_metrics[monitor_metric]
            if score > best_metric:
                best_metric = score
                best_metrics = dict(val_metrics)
                best_metrics["best_epoch"] = epoch
                best_metrics["feature_names"] = list(feature_names)

                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    best_metric,
                    extra={
                        "feature_names": list(feature_names),
                        "feat_mean": feat_mean.tolist(),
                        "feat_std": feat_std.tolist(),
                        "pos_weight": None if pos_weight_value is None else float(pos_weight_value),
                    },
                )
                print(f"[INFO] Saved fold {fold} best checkpoint to {best_path}")

            if stopper.step(score):
                print(
                    f"[INFO] Early stopping fold {fold} at epoch {epoch}. "
                    f"No improvement in {monitor_metric} for {stopper.patience} epochs."
                )
                break

        if best_metrics is None:
            raise RuntimeError(f"Fold {fold} did not produce valid metrics.")

        best_metrics["fold"] = fold
        all_fold_metrics.append(best_metrics)

        with open(os.path.join(fold_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("[CV SUMMARY]")
    print("=" * 80)

    keys = [
        "loss",
        "auroc",
        "ap",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "balanced_acc",
        "specificity",
        "sensitivity",
        "best_epoch",
    ]
    summary = {}
    for key in keys:
        vals = [m[key] for m in all_fold_metrics if key in m]
        vals = np.array(vals, dtype=float)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
        print(f"{key}: mean={summary[key]['mean']:.4f}, std={summary[key]['std']:.4f}")

    with open(os.path.join(base_save_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": list(feature_names),
                "fold_metrics": all_fold_metrics,
                "summary": summary,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
