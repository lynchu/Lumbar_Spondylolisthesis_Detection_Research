# python ./presence/scripts/basic/train_lumbar_crop_cv.py --config ./presence/configs/resnet50_basic_option1.yaml

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from data.dataset_lumbar_crop import LumbarCropDataset, LumbarCropDatasetWithMask, CombinedLumbarCropDataset
from data.collate import lumbar_crop_collate_fn
from models.basic import LumbarCropModel, LumbarCropModelWithMask
from losses.classification import StudyClassificationLoss
from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from utils.seed import set_seed
from utils.optim import build_optimizer_simple, build_scheduler_simple
from utils.checkpoint import save_checkpoint
from utils.early_stopping import EarlyStopping

from collections import Counter
from torch.utils.data import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


PAIR_NAMES = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]
LUMBAR_LEVELS = ["S1", "L5", "L4", "L3", "L2", "L1"]
LEVEL_TO_ORDER = {name: i for i, name in enumerate(LUMBAR_LEVELS)}

DEFAULT_PAIR_REPEAT_MAP = {
    "negative": 1,
    "missing": 1,
    "L5-S1": 2,
    "L4-L5": 1,
    "L3-L4": 2,
    "L2-L3": 4,
    "L1-L2": 4,
}


def parse_location_levels(location):
    if location is None:
        return []
    s = str(location).strip()
    if s == "" or s.lower() == "nan":
        return []

    levels = []
    for x in s.split(";"):
        x = x.strip()
        if x in LEVEL_TO_ORDER and x not in levels:
            levels.append(x)
    return levels


def location_to_adjacent_pair(location):
    levels = parse_location_levels(location)
    if len(levels) < 2:
        return "missing"

    ids = sorted([LEVEL_TO_ORDER[x] for x in levels])
    pair_candidates = []
    for a, b in zip(ids[:-1], ids[1:]):
        if b == a + 1:
            pair_candidates.append((a, b))

    if len(pair_candidates) != 1:
        return "missing"

    a, b = pair_candidates[0]
    return f"{LUMBAR_LEVELS[a]}-{LUMBAR_LEVELS[b]}"


def sample_to_repeat_key(sample):
    _, study_label, location = sample
    if int(study_label) == 0:
        return "negative"
    return location_to_adjacent_pair(location)


def build_repeat_map(cfg):
    user_map = cfg.get("pair_repeat_map", None)
    if user_map is None:
        return dict(DEFAULT_PAIR_REPEAT_MAP)
    out = dict(DEFAULT_PAIR_REPEAT_MAP)
    out.update(dict(user_map))
    return out


class OversampledSubset(Dataset):
    def __init__(self, base_dataset, indices, repeat_map, print_stats=True):
        self.base_dataset = base_dataset
        self.original_indices = list(indices)
        self.repeat_map = dict(repeat_map)

        self.indices = []
        key_counter = Counter()
        repeated_counter = Counter()

        for idx in self.original_indices:
            sample = self.base_dataset.samples[idx]
            key = sample_to_repeat_key(sample)
            repeat_n = int(self.repeat_map.get(key, 1))
            repeat_n = max(repeat_n, 1)

            key_counter[key] += 1
            repeated_counter[key] += repeat_n

            self.indices.extend([idx] * repeat_n)

        if print_stats:
            print("[RepeatMap] original training subset distribution:")
            for k, v in sorted(key_counter.items()):
                print(f"  {k}: {v}")
            print("[RepeatMap] repeated training subset distribution:")
            for k, v in sorted(repeated_counter.items()):
                print(f"  {k}: {v}")
            print(f"[RepeatMap] train subset size: {len(self.original_indices)} -> {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]


def get_labels_from_dataset(dataset):
    return np.array([int(sample[1]) for sample in dataset.samples])


def build_model(cfg, device):
    model = LumbarCropModel(
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        loss_name=cfg.get("loss_name", "bce"),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.0),
    )
    # model = LumbarCropModelWithMask(
    #     backbone_pretrained=cfg.get("imagenet_pretrained", False),
    #     # dropout=cfg.get("dropout", 0.2),
    # )
    lumbar_ckpt = cfg.get("lumbar_pretrained_ckpt", "")
    if lumbar_ckpt:
        model.load_backbone_checkpoint(lumbar_ckpt)
    return model.to(device)


def build_criterion(cfg, device):
    loss_name = cfg.get("loss_name", "bce").lower()

    if loss_name == "bce":
        pos_weight = None
        if cfg.get("pos_weight", None) is not None:
            pos_weight = torch.tensor([cfg["pos_weight"]], device=device, dtype=torch.float32)
        return StudyClassificationLoss(loss_name="bce", pos_weight=pos_weight), loss_name

    if loss_name == "ce":
        class_weight = None
        if cfg.get("class_weight", None) is not None:
            class_weight = torch.tensor(cfg["class_weight"], device=device, dtype=torch.float32)
        return StudyClassificationLoss(
            loss_name="ce",
            class_weight=class_weight,
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        ), loss_name

    raise ValueError(f"Unsupported loss_name: {loss_name}")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = LumbarCropDataset(
        text_csv_path=cfg["train_csv"],
        image_dir=cfg["train_image_dir"],
        split="train",
        exclude_s1_positive=cfg.get("exclude_s1_positive", True),
        aug_cfg=cfg.get("augmentation", {}),
    )
    val_dataset = LumbarCropDataset(
        text_csv_path=cfg["val_csv"],
        image_dir=cfg["val_image_dir"],
        split="valid",
        exclude_s1_positive=cfg.get("exclude_s1_positive", True),
        aug_cfg=cfg.get("augmentation", {}),
    )

    full_dataset = CombinedLumbarCropDataset([train_dataset, val_dataset])
    labels = get_labels_from_dataset(full_dataset)

    skf = StratifiedKFold(
        n_splits=cfg.get("num_folds", 5),
        shuffle=True,
        random_state=cfg.get("seed", 42),
    )

    all_fold_metrics = []
    base_save_dir = cfg["save_dir"]
    os.makedirs(base_save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(full_dataset)), labels), start=1):
        print("=" * 80)
        print(f"[INFO] Fold {fold}/{cfg.get('num_folds', 5)}")
        print("=" * 80)

        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()

        repeat_map = build_repeat_map(cfg)
        if cfg.get("use_repeat_map", False):
            train_subset = OversampledSubset(
                base_dataset=full_dataset,
                indices=train_idx,
                repeat_map=repeat_map,
                print_stats=cfg.get("print_repeat_map_stats", True),
            )
        else:
            train_subset = Subset(full_dataset, train_idx)

        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=cfg["batch_size"], shuffle=True,
                                  num_workers=cfg.get("num_workers", 4), collate_fn=lumbar_crop_collate_fn,
                                  pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=cfg["batch_size"], shuffle=False,
                                num_workers=cfg.get("num_workers", 4), collate_fn=lumbar_crop_collate_fn,
                                pin_memory=True)

        model = build_model(cfg, device)
        criterion, loss_name = build_criterion(cfg, device)
        optimizer = build_optimizer_simple(model, cfg)
        scheduler = build_scheduler_simple(optimizer, cfg)

        fold_dir = os.path.join(base_save_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        best_path = os.path.join(fold_dir, "best.pt")

        monitor_metric = cfg.get("monitor_metric", "ap")
        best_metric = -1.0
        best_metrics = None

        stopper = EarlyStopping(
            patience=cfg.get("early_stop_patience", 10),
            mode="max",
            min_delta=cfg.get("early_stop_min_delta", 0.0),
        )

        for epoch in range(1, cfg["epochs"] + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                loss_name=loss_name,
                grad_clip_norm=cfg.get("grad_clip_norm", None),
                freeze_bn=(
                    cfg.get("freeze_bn_stats", False)
                    and (cfg.get("imagenet_pretrained", False) or bool(cfg.get("lumbar_pretrained_ckpt", "")))
                ),
            )

            val_metrics = validate(
                model, val_loader, criterion, device,
                loss_name=loss_name,
                threshold=cfg.get("threshold", None),
            )

            if scheduler is not None:
                if str(cfg.get("scheduler_name", "none")).lower() == "plateau":
                    scheduler.step(val_metrics[monitor_metric])
                else:
                    scheduler.step()

            msg = (
                f"[Fold {fold} | Epoch {epoch}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"auroc={val_metrics['auroc']:.4f} "
                f"ap={val_metrics['ap']:.4f}"
            )
            if "f1" in val_metrics:
                msg += f" f1={val_metrics['f1']:.4f}"
            if "balanced_acc" in val_metrics:
                msg += f" bal_acc={val_metrics['balanced_acc']:.4f}"
            print(msg)

            score = val_metrics[monitor_metric]
            if score > best_metric:
                best_metric = score
                best_metrics = dict(val_metrics)
                best_metrics["epoch"] = epoch
                best_metrics["monitor_metric"] = monitor_metric
                best_metrics["monitor_value"] = float(score)
                save_checkpoint(best_path, model, optimizer, epoch, best_metric)
                with open(os.path.join(fold_dir, "best_metrics.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"[INFO] Saved fold {fold} best checkpoint to {best_path}")

            if stopper.step(score):
                print(f"[INFO] Early stopping fold {fold} at epoch {epoch}. No improvement in {monitor_metric} for {stopper.patience} epochs.")
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

    keys = ["loss", "auroc", "ap", "accuracy", "f1", "precision", "recall", "balanced_acc", "specificity", "sensitivity"]
    summary = {}

    for key in keys:
        vals = [m[key] for m in all_fold_metrics if key in m]
        if len(vals) == 0:
            continue
        vals = np.array(vals, dtype=float)
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"{key}: mean={summary[key]['mean']:.4f}, std={summary[key]['std']:.4f}")

    with open(os.path.join(base_save_dir, "cv_summary.json"), "w") as f:
        json.dump({"fold_metrics": all_fold_metrics, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
