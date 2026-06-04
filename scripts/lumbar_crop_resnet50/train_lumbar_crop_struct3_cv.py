# python ./presence/scripts/basic/train_lumbar_crop_struct5_cv.py --config ./presence/configs/resnet50_basic_struct5.yaml

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from data.dataset_lumbar_crop_struct3 import (
    LumbarCropDataset,
    LumbarCropDatasetWithMask,
    CombinedLumbarCropDataset,
    implant_category_counts,
)
from data.collate import lumbar_crop_collate_fn
from models.basic import LumbarCropModel, LumbarCropModelWithMask
from losses.classification import StudyClassificationLoss
from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from utils.seed import set_seed
from utils.optim import build_optimizer, build_scheduler
from utils.checkpoint import load_checkpoint, save_checkpoint, save_cv_summary
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
    """
    Parse both old and new location formats.

    Old examples:
      "L4; L5"
      "L5; S1"

    New compact_lumbar_diagnosis examples:
      "L3,L4; L4,L5;"
      "L4,L5;"

    Returns unique lumbar level names. This is only used for repeat-map
    oversampling; it does NOT filter implant groups.
    """
    if location is None:
        return []
    s = str(location).strip()
    if s == "" or s.lower() == "nan":
        return []

    levels = []
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        for token in seg.split(","):
            token = token.strip().upper()
            if token in LEVEL_TO_ORDER and token not in levels:
                levels.append(token)
    return levels


def location_to_adjacent_pair(location):
    levels = parse_location_levels(location)
    if len(levels) < 2:
        return "missing"

    ids = sorted([LEVEL_TO_ORDER[x] for x in levels])  # 0=S1, 1=L5, ...
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


def get_implant_filter_kwargs(cfg):
    """
    Build optional implant filtering kwargs for LumbarCropDataset/LumbarCropDatasetWithMask.

    Default baseline behavior: keep all implant categories.

    Config examples:
      # keep all cases, baseline
      implant_filter:
        enabled: false

      # remove implant category 2
      implant_filter:
        enabled: true
        exclude_values: ["2"]

      # train/evaluate only categories 0 and 3
      implant_filter:
        enabled: true
        include_values: ["0", "3"]

    Backward-compatible flat keys are also supported:
      filter_implant_enabled: true
      implant_include_values: ["0", "3"]
      implant_exclude_values: ["2"]
    """
    section = cfg.get("implant_filter", {}) or {}
    enabled = bool(section.get("enabled", cfg.get("filter_implant_enabled", False)))

    if not enabled:
        return {
            "include_implant_values": None,
            "exclude_implant_values": None,
            "require_implant_column": False,
            "verbose_implant_filter": False,
            "skip_unknown_implant": bool(section.get("skip_unknown", cfg.get("implant_skip_unknown", True))),
        }

    include_values = section.get("include_values", cfg.get("implant_include_values", None))
    exclude_values = section.get("exclude_values", cfg.get("implant_exclude_values", None))
    require_column = bool(section.get("require_column", cfg.get("implant_filter_require_column", True)))
    verbose = bool(section.get("verbose", cfg.get("implant_filter_verbose", True)))
    skip_unknown = bool(section.get("skip_unknown", cfg.get("implant_skip_unknown", True)))

    if include_values is not None and not isinstance(include_values, (list, tuple, set)):
        include_values = [include_values]
    if exclude_values is not None and not isinstance(exclude_values, (list, tuple, set)):
        exclude_values = [exclude_values]

    if include_values is None and exclude_values is None:
        print("[WARN] implant_filter.enabled=True but neither include_values nor exclude_values is set. Keep all cases.")

    print(
        "[INFO] Implant filtering enabled | "
        f"include_values={include_values} | exclude_values={exclude_values} | "
        f"require_column={require_column} | skip_unknown={skip_unknown}"
    )

    return {
        "include_implant_values": include_values,
        "exclude_implant_values": exclude_values,
        "require_implant_column": require_column,
        "verbose_implant_filter": verbose,
        "skip_unknown_implant": skip_unknown,
    }


class OversampledSubset(Dataset):
    """
    Wrap a base dataset + original indices, then duplicate only the selected
    training samples according to repeat_map.

    This avoids leakage across CV folds because oversampling happens AFTER split.
    """
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


def load_existing_fold_metrics(base_save_dir, completed_folds):
    existing = []
    for fold in completed_folds:
        metrics_path = os.path.join(base_save_dir, f"fold_{fold}", "best_metrics.json")
        if not os.path.isfile(metrics_path):
            print(f"[WARN] Completed fold {fold} has no best_metrics.json: {metrics_path}")
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["fold"] = fold
        existing.append(metrics)
    return existing




def inspect_cv_run_simple(base_save_dir, num_folds, fold_prefix="fold_", summary_filename="cv_summary.json",
                          resume_ckpt_name="last.pt"):
    """
    Temporary simple resume logic for the user's current workflow:
    - cv_summary.json exists => whole CV already completed
    - otherwise, look only at whether fold directories exist
    - if fold_3 and fold_4 both exist, resume/start from fold 4
    - more generally, resume/start from the highest existing fold directory
    - if no fold directory exists, start from fold 1

    Note: existence of a fold directory does NOT mean the fold is finished.
    """
    summary_path = os.path.join(base_save_dir, summary_filename)
    if os.path.isfile(summary_path):
        return {
            "status": "already_completed",
            "message": f"CV directory already completed: {summary_path}",
            "completed_folds": [],
            "fold_idx": None,
            "resume_checkpoint": None,
        }

    existing_folds = []
    for fold in range(1, num_folds + 1):
        fold_dir = os.path.join(base_save_dir, f"{fold_prefix}{fold}")
        if os.path.isdir(fold_dir):
            existing_folds.append(fold)

    if not existing_folds:
        return {
            "status": "start_fold",
            "message": "No existing fold directory found. Start from fold 1.",
            "completed_folds": [],
            "fold_idx": 1,
            "resume_checkpoint": None,
        }

    target_fold = max(existing_folds)
    fold_dir = os.path.join(base_save_dir, f"{fold_prefix}{target_fold}")
    resume_ckpt = os.path.join(fold_dir, resume_ckpt_name)

    if os.path.isfile(resume_ckpt):
        return {
            "status": "resume_fold",
            "message": f"Resume from highest existing fold {target_fold}: {resume_ckpt}",
            "completed_folds": [f for f in existing_folds if f < target_fold],
            "fold_idx": target_fold,
            "resume_checkpoint": resume_ckpt,
        }

    return {
        "status": "start_fold",
        "message": f"Start from highest existing fold directory {target_fold} (no last.pt found).",
        "completed_folds": [f for f in existing_folds if f < target_fold],
        "fold_idx": target_fold,
        "resume_checkpoint": None,
    }


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_folds = int(cfg.get("num_folds", 5))

    # Default baseline behavior trains on ALL implant categories.
    # Future ablations can enable implant_filter in the config without editing code.
    use_lumbar_mask = bool(cfg.get("use_lumbar_mask", cfg.get("use_mask_input", False)))
    implant_filter_kwargs = get_implant_filter_kwargs(cfg)

    if use_lumbar_mask:
        # LumbarCropDatasetWithMask uses the mask to crop/filter the RGB image and
        # still returns a normal 3-channel image tensor. Therefore build_model()
        # should keep using LumbarCropModel unless you rewrite the dataset to return
        # a real 4-channel image+mask tensor.
        for required_key in ["train_mask_dir", "val_mask_dir"]:
            if not cfg.get(required_key):
                raise ValueError(f"Config sets use_lumbar_mask=True but misses {required_key}")

        train_dataset = LumbarCropDatasetWithMask(
            text_csv_path=cfg["train_csv"],
            image_dir=cfg["train_image_dir"],
            mask_dir=cfg["train_mask_dir"],
            split="train",
            exclude_s1_positive=cfg.get("exclude_s1_positive", True),
            aug_cfg=cfg.get("augmentation", {}),
            **implant_filter_kwargs,
        )
        val_dataset = LumbarCropDatasetWithMask(
            text_csv_path=cfg["val_csv"],
            image_dir=cfg["val_image_dir"],
            mask_dir=cfg["val_mask_dir"],
            split="valid",
            exclude_s1_positive=cfg.get("exclude_s1_positive", True),
            aug_cfg=cfg.get("augmentation", {}),
            **implant_filter_kwargs,
        )
    else:
        train_dataset = LumbarCropDataset(
            text_csv_path=cfg["train_csv"],
            image_dir=cfg["train_image_dir"],
            split="train",
            exclude_s1_positive=cfg.get("exclude_s1_positive", True),
            aug_cfg=cfg.get("augmentation", {}),
            **implant_filter_kwargs,
        )
        val_dataset = LumbarCropDataset(
            text_csv_path=cfg["val_csv"],
            image_dir=cfg["val_image_dir"],
            split="valid",
            exclude_s1_positive=cfg.get("exclude_s1_positive", True),
            aug_cfg=cfg.get("augmentation", {}),
            **implant_filter_kwargs,
        )

    print(f"[INFO] use_lumbar_mask={use_lumbar_mask}")
    print(f"[INFO] train samples after image/S1 filtering: {len(train_dataset)}")
    print(f"[INFO] valid samples after image/S1 filtering: {len(val_dataset)}")

    # Optional diagnostic only. This does not filter by implant category.
    for split_name, csv_key in [("train", "train_csv"), ("valid", "val_csv")]:
        try:
            _df = pd.read_csv(cfg[csv_key])
            if "implant" in _df.columns:
                counts = implant_category_counts(_df["implant"])
                print(f"[INFO] {split_name} implant multi-label distribution in CSV: {counts}")
        except Exception as e:
            print(f"[WARN] Could not print {split_name} implant distribution: {e}")


    full_dataset = CombinedLumbarCropDataset([train_dataset, val_dataset])
    labels = get_labels_from_dataset(full_dataset)

    if len(full_dataset) == 0:
        raise RuntimeError("No samples remain after image/S1/implant filtering.")
    if len(np.unique(labels)) < 2:
        raise RuntimeError(
            "Need at least two classes after image/S1/implant filtering to run stratified CV. "
            f"Label counts: {dict(Counter(labels.astype(int).tolist()))}"
        )

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=cfg.get("seed", 42),
    )

    base_save_dir = cfg["save_dir"]
    os.makedirs(base_save_dir, exist_ok=True)

    cv_state = inspect_cv_run_simple(
        base_save_dir=base_save_dir,
        num_folds=num_folds,
        fold_prefix="fold_",
        summary_filename="cv_summary.json",
        resume_ckpt_name="last.pt"
    )
    print(f"[INFO] {cv_state['message']}")

    if cv_state["status"] == "already_completed":
        return

    all_fold_metrics = load_existing_fold_metrics(base_save_dir, cv_state.get("completed_folds", []))
    completed_folds = set(cv_state.get("completed_folds", []))
    resume_fold = cv_state.get("fold_idx") if cv_state.get("status") == "resume_fold" else None
    resume_ckpt_path = cv_state.get("resume_checkpoint")

    monitor_metric = cfg.get("monitor_metric", "ap")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(full_dataset)), labels), start=1):
        if cv_state["status"] in {"start_fold", "resume_fold"} and fold < cv_state["fold_idx"]:
            print(f"[INFO] Skip fold {fold}: before target fold {cv_state['fold_idx']}. Treat as previously handled in this temporary resume mode.")
            continue
        if cv_state["status"] in {"start_fold", "resume_fold"} and fold > cv_state["fold_idx"] and fold not in completed_folds:
            # later folds will naturally run after the target fold finishes
            pass

        print("=" * 80)
        print(f"[INFO] Fold {fold}/{num_folds}")
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

        train_loader = DataLoader(
            train_subset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=lumbar_crop_collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=lumbar_crop_collate_fn,
            pin_memory=True,
        )

        model = build_model(cfg, device)
        criterion, loss_name = build_criterion(cfg, device)
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
        last_path = os.path.join(fold_dir, "last.pt")
        best_metrics_path = os.path.join(fold_dir, "best_metrics.json")

        best_metric = -1.0
        best_metrics = None
        start_epoch = 1

        stopper = EarlyStopping(
            patience=cfg.get("early_stop_patience", 10),
            mode="max",
            min_delta=cfg.get("early_stop_min_delta", 0.0),
        )

        if fold == resume_fold and resume_ckpt_path and os.path.isfile(resume_ckpt_path):
            ckpt = load_checkpoint(
                resume_ckpt_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopping=stopper,
                map_location=device,
            )
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = float(ckpt.get("best_metric", best_metric))
            if os.path.isfile(best_metrics_path):
                with open(best_metrics_path, "r", encoding="utf-8") as f:
                    best_metrics = json.load(f)
            print(
                f"[INFO] Resume fold {fold} from checkpoint {resume_ckpt_path} "
                f"at epoch {start_epoch}/{cfg['epochs']} (best_{monitor_metric}={best_metric:.4f})"
            )

            if start_epoch > cfg["epochs"]:
                print(f"[WARN] Fold {fold} checkpoint already reached max epochs. Skipping training loop.")

        for epoch in range(start_epoch, cfg["epochs"] + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                loss_name=loss_name,
                # grad_clip_norm=cfg.get("grad_clip_norm", None),
                # freeze_bn=(
                #     cfg.get("freeze_bn_stats", False)
                #     and (cfg.get("imagenet_pretrained", False) or bool(cfg.get("lumbar_pretrained_ckpt", "")))
                # ),
            )

            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
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

            score = float(val_metrics[monitor_metric])
            save_checkpoint(
                last_path,
                model,
                optimizer,
                epoch,
                best_metric,
                scheduler=scheduler,
                early_stopping=stopper,
                extra_state={
                    "fold": fold,
                    "monitor_metric": monitor_metric,
                    "last_score": score,
                },
            )

            if score > best_metric:
                best_metric = score
                best_metrics = dict(val_metrics)
                best_metrics["epoch"] = epoch
                best_metrics["monitor_metric"] = monitor_metric
                best_metrics["monitor_value"] = best_metric
                best_metrics["fold"] = fold
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    best_metric,
                    scheduler=scheduler,
                    early_stopping=stopper,
                    extra_state={
                        "fold": fold,
                        "monitor_metric": monitor_metric,
                        "best_metrics": best_metrics,
                    },
                )
                with open(best_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(best_metrics, f, indent=2, ensure_ascii=False)
                print(f"[INFO] Saved fold {fold} best checkpoint to {best_path}")

            if stopper.step(score):
                print(
                    f"[INFO] Early stopping fold {fold} at epoch {epoch}. "
                    f"No improvement in {monitor_metric} for {stopper.patience} epochs."
                )
                break

        if best_metrics is None:
            if os.path.isfile(best_metrics_path):
                with open(best_metrics_path, "r", encoding="utf-8") as f:
                    best_metrics = json.load(f)
            else:
                raise RuntimeError(f"Fold {fold} did not produce valid metrics.")

        best_metrics["fold"] = fold
        with open(best_metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2, ensure_ascii=False)

        all_fold_metrics = [m for m in all_fold_metrics if int(m.get("fold", -1)) != fold]
        all_fold_metrics.append(best_metrics)

    all_fold_metrics = sorted(all_fold_metrics, key=lambda x: int(x.get("fold", 0)))

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

    if len(all_fold_metrics) != num_folds:
        print(
            f"[WARN] CV summary not written because only {len(all_fold_metrics)}/{num_folds} folds have best metrics."
        )
        return

    save_cv_summary(
        base_save_dir,
        {"fold_metrics": all_fold_metrics, "summary": summary},
        summary_filename="cv_summary.json",
    )
    print(f"[INFO] Saved CV summary to {os.path.join(base_save_dir, 'cv_summary.json')}")


if __name__ == "__main__":
    main()