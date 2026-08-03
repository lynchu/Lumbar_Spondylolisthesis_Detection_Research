#!/usr/bin/env python3
"""Generate OOF validation predictions for the global lumbar-crop branch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.resnet50.collate import lumbar_crop_collate_fn
from data.resnet50.dataset_lumbar_crop import CombinedLumbarCropDataset, LumbarCropDataset
from models.basic import LumbarCropModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cv-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--ckpt-name", default="best.pt")
    return parser.parse_args()


def logit_to_prob(logits, loss_name):
    if str(loss_name).lower() in {"bce", "focal_bce", "focal"}:
        return torch.sigmoid(logits), logits
    probs = torch.softmax(logits, dim=1)[:, 1]
    scalar_logits = logits[:, 1] - logits[:, 0]
    return probs, scalar_logits


def load_checkpoint_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ["model", "model_state_dict", "state_dict"]:
            if key in ckpt:
                return {k.replace("module.", ""): v for k, v in ckpt[key].items()}
    return {k.replace("module.", ""): v for k, v in ckpt.items()}


def build_dataset(cfg):
    train_geometry = {
        "geometry_feature_csv": cfg.get("train_geometry_feature_csv"),
        "geometry_feature_names": cfg.get("geometry_feature_names", "auto"),
        "require_geometry": bool(cfg.get("require_geometry_features", True)),
    } if cfg.get("use_geometry_features", False) else {}
    val_geometry = {
        "geometry_feature_csv": cfg.get("val_geometry_feature_csv"),
        "geometry_feature_names": cfg.get("geometry_feature_names", "auto"),
        "require_geometry": bool(cfg.get("require_geometry_features", True)),
    } if cfg.get("use_geometry_features", False) else {}
    train_ds = LumbarCropDataset(
        cfg["train_csv"],
        cfg["train_image_dir"],
        split="train",
        exclude_s1_positive=cfg.get("exclude_s1_positive", True),
        aug_cfg=cfg.get("augmentation", {}),
        **train_geometry,
    )
    val_ds = LumbarCropDataset(
        cfg["val_csv"],
        cfg["val_image_dir"],
        split="valid",
        exclude_s1_positive=cfg.get("exclude_s1_positive", True),
        aug_cfg=cfg.get("augmentation", {}),
        **val_geometry,
    )
    return CombinedLumbarCropDataset([train_ds, val_ds])


def build_model(cfg, geometry_dim, device):
    model = LumbarCropModel(
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        loss_name=cfg.get("loss_name", "bce"),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.0),
        geometry_dim=geometry_dim,
        geometry_hidden_dim=cfg.get("geometry_hidden_dim", 32),
    )
    return model.to(device)


@torch.no_grad()
def predict_fold(model, loader, device, loss_name):
    rows = []
    model.eval()
    for batch in loader:
        images = batch["images"].to(device)
        geometry = batch.get("geometry_features")
        if geometry is not None:
            geometry = geometry.to(device)
        if getattr(model, "expects_geometry", False):
            out = model(images, geometry_features=geometry)
        else:
            out = model(images)
        probs, scalar_logits = logit_to_prob(out["study_logits"], loss_name)
        for image_id, y_true, prob, logit in zip(
            batch["image_ids"],
            batch["study_labels"].cpu().numpy().astype(int).tolist(),
            probs.detach().cpu().numpy().tolist(),
            scalar_logits.detach().cpu().numpy().tolist(),
        ):
            rows.append({
                "image_id": image_id,
                "y_true": int(y_true),
                "global_prob": float(prob),
                "global_logit": float(logit),
            })
    return rows


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = build_dataset(cfg)
    labels = np.array([int(sample[1]) for sample in dataset.samples])
    geometry_dim = int(getattr(dataset, "geometry_dim", 0)) if cfg.get("use_geometry_features", False) else 0
    skf = StratifiedKFold(n_splits=int(cfg.get("num_folds", 5)), shuffle=True, random_state=cfg.get("seed", 42))
    rows = []
    for fold, (_, val_idx) in enumerate(skf.split(np.arange(len(dataset)), labels), start=1):
        ckpt = Path(args.cv_dir) / f"fold_{fold}" / args.ckpt_name
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        loader = DataLoader(
            Subset(dataset, val_idx.tolist()),
            batch_size=int(cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 2)),
            collate_fn=lumbar_crop_collate_fn,
            pin_memory=True,
        )
        model = build_model(cfg, geometry_dim, device)
        missing, unexpected = model.load_state_dict(load_checkpoint_state_dict(ckpt, device), strict=False)
        if missing:
            print(f"[WARN] fold {fold} missing keys: {missing}")
        if unexpected:
            print(f"[WARN] fold {fold} unexpected keys: {unexpected}")
        fold_rows = predict_fold(model, loader, device, cfg.get("loss_name", "bce"))
        for row in fold_rows:
            row["fold"] = fold
        rows.extend(fold_rows)
        print(f"[OK] fold {fold}: {len(fold_rows)} OOF predictions")
    out = pd.DataFrame(rows).sort_values(["image_id", "fold"]).reset_index(drop=True)
    if out["image_id"].duplicated().any():
        dup = out.loc[out["image_id"].duplicated(), "image_id"].head().tolist()
        raise RuntimeError(f"Duplicate OOF image_id rows detected: {dup}")
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
