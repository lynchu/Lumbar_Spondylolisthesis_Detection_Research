import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import yaml
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from data.dataset_pair_crop import PairCropLevelDataset, CombinedPairCropDataset
from models.pair_crop_binary import PairCropBinaryModel
from losses.classification import StudyBCELoss
from engine.train_one_epoch_pair_crop import train_one_epoch_pair_crop
from engine.validate_pair_crop import validate_pair_crop
from utils.seed import set_seed
from utils.optim import build_optimizer, build_scheduler
from utils.checkpoint import save_checkpoint
from utils.early_stopping import EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


class PairCropTransformSubset(Dataset):
    """
    Transform-aware subset:
    use train transform for training fold and valid transform for validation fold,
    while still splitting from one combined metadata pool.
    """
    def __init__(self, full_dataset, indices, transform):
        self.full_dataset = full_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.full_dataset.samples[self.indices[idx]]
        img_path = os.path.join(item["_image_dir"], item["filename"])

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(float(item["label"]), dtype=torch.float32),
            "pair_id": torch.tensor(int(item["pair_id"]), dtype=torch.long),
            "pair_name": item["pair_name"],
            "image_id": item["image_id"],
            "sample_id": item["sample_id"],
            "study_label": torch.tensor(float(item["study_label"]), dtype=torch.float32),
            "location_raw": item["location_raw"],
        }


def get_labels_from_dataset(dataset):
    labels = []
    for sample in dataset.samples:
        labels.append(int(sample["label"]))
    return np.array(labels)


def get_groups_from_dataset(dataset):
    groups = []
    for sample in dataset.samples:
        groups.append(sample["image_id"])
    return np.array(groups)


def build_model(cfg, device):
    model = PairCropBinaryModel(
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        dropout=cfg.get("dropout", 0.2),
    )
    lumbar_ckpt = cfg.get("lumbar_pretrained_ckpt", "")
    if lumbar_ckpt:
        model.backbone.load_partial_checkpoint(lumbar_ckpt)
    return model.to(device)


def print_dataset_sanity(dataset, name):
    num_samples = len(dataset.samples)
    num_pos = sum(int(s["label"]) for s in dataset.samples)
    num_neg = num_samples - num_pos

    pair_pos_counts = {}
    for pair_name in ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]:
        pair_pos_counts[pair_name] = sum(
            int(s["label"]) for s in dataset.samples if s["pair_name"] == pair_name
        )

    print(f"[{name}] samples={num_samples}, pos={num_pos}, neg={num_neg}")
    print(f"[{name}] positive counts by pair: {pair_pos_counts}")

    # preview = []
    # for s in dataset.samples:
    #     if int(s["label"]) == 1:
    #         preview.append(
    #             (s["image_id"], s["location_raw"], s["pair_name"], int(s["label"]))
    #         )
    #     if len(preview) >= 8:
    #         break

    # if len(preview) > 0:
    #     print(f"[{name}] positive sample preview:")
    #     for x in preview:
    #         print("   ", x)
    # else:
    #     print(f"[{name}] WARNING: no positive samples found.")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = PairCropLevelDataset(
        text_csv_path=cfg["train_csv"],
        image_dir=cfg["train_image_dir"],
        split="train",
        exclude_s1_positive=cfg.get("exclude_s1_positive", False),
        aug_cfg=cfg.get("augmentation", {}),
    )

    val_dataset = PairCropLevelDataset(
        text_csv_path=cfg["val_csv"],
        image_dir=cfg["val_image_dir"],
        split="valid",
        exclude_s1_positive=cfg.get("exclude_s1_positive", False),
        aug_cfg=cfg.get("augmentation", {}),
    )

    print_dataset_sanity(train_dataset, "train_dataset")
    print_dataset_sanity(val_dataset, "val_dataset")

    full_dataset = CombinedPairCropDataset([train_dataset, val_dataset])
    labels = get_labels_from_dataset(full_dataset)
    groups = get_groups_from_dataset(full_dataset)

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.get("num_folds", 5),
        shuffle=True,
        random_state=cfg.get("seed", 42),
    )

    all_fold_metrics = []
    base_save_dir = cfg["save_dir"]
    os.makedirs(base_save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(np.arange(len(full_dataset)), labels, groups=groups),
        start=1,
    ):
        print("=" * 80)
        print(f"[INFO] Fold {fold}/{cfg.get('num_folds', 5)}")
        print("=" * 80)

        train_subset = PairCropTransformSubset(
            full_dataset=full_dataset,
            indices=train_idx.tolist(),
            transform=train_dataset.transform,
        )
        val_subset = PairCropTransformSubset(
            full_dataset=full_dataset,
            indices=val_idx.tolist(),
            transform=val_dataset.transform,
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

        model = build_model(cfg, device)

        pos_weight = None
        if cfg.get("pos_weight", None) is not None:
            pos_weight = torch.tensor([cfg["pos_weight"]], device=device)

        criterion = StudyBCELoss(pos_weight=pos_weight)

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

        monitor_metric = cfg.get("monitor_metric", "auroc")
        best_metric = -1.0
        best_metrics = None

        stopper = EarlyStopping(
            patience=cfg.get("early_stop_patience", 10),
            mode="max",
            min_delta=cfg.get("early_stop_min_delta", 0.0),
        )

        for epoch in range(1, cfg["epochs"] + 1):
            train_loss = train_one_epoch_pair_crop(model, train_loader, optimizer, criterion, device)
            val_metrics = validate_pair_crop(
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
                save_checkpoint(best_path, model, optimizer, epoch, best_metric)
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

        with open(os.path.join(fold_dir, "best_metrics.json"), "w") as f:
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

    with open(os.path.join(base_save_dir, "cv_summary.json"), "w") as f:
        json.dump({"fold_metrics": all_fold_metrics, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()