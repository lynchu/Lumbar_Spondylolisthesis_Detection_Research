# python ./presence/scripts/basic/analyze_stage1_basic_cv_ensemble_test.py  
# --config ./presence/configs/resnet50_basic.yaml   --cv_dir ./presence/experiments/basic/exp_resnet50_basic_cv5   
# --outdir ./presence/experiments/basic/exp_resnet50_basic_cv5/test_analysis   
# --n_folds 5   --ckpt_name best.pt   --ensemble_mode probs

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# from data.dataset_stage1_aug import Stage1PresenceDataset, LUMBAR_LEVELS
from data.dataset_lumbar_crop import LumbarCropDataset, LumbarCropDatasetWithMask, LUMBAR_LEVELS
from data.collate import lumbar_crop_collate_fn
from engine.metrics import compute_binary_metrics

from models.basic import LumbarCropModel, LumbarCropModelWithMask


PAIR_NAMES = ["L4-L5", "L3-L4", "L2-L3", "L1-L2"]
LEVEL_TO_ORDER = {name: i for i, name in enumerate(LUMBAR_LEVELS)}


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


def parse_location_levels(location):
    if pd.isna(location) or location is None:
        return []
    levels = []
    for x in str(location).split(";"):
        x = x.strip()
        if x in LEVEL_TO_ORDER:
            levels.append(x)
    out = []
    for x in levels:
        if x not in out:
            out.append(x)
    return out


def location_to_gt_pair(location):
    levels = parse_location_levels(location)
    if len(levels) < 2:
        return None

    ids = sorted([LEVEL_TO_ORDER[x] for x in levels])
    pair_candidates = []
    for a, b in zip(ids[:-1], ids[1:]):
        if b == a + 1:
            pair_candidates.append((a, b))

    if len(pair_candidates) != 1:
        return None

    a, b = pair_candidates[0]
    return f"{LUMBAR_LEVELS[a]}-{LUMBAR_LEVELS[b]}"


def location_to_gt_vertebra_set(location):
    return parse_location_levels(location)


def grade_bucket(x):
    if pd.isna(x):
        return "missing"
    try:
        g = int(x)
    except Exception:
        return "missing"

    if g <= 1:
        return "low_0_1"
    elif g == 2:
        return "grade_2"
    else:
        return "grade_3_plus"


def load_checkpoint_state_dict(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
    return cleaned, extra


def apply_temperature_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(float(temperature), 1e-6)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_model(cfg, device):
    model = LumbarCropModel(
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        # dropout=cfg.get("dropout", 0.2),
    )
    # model = LumbarCropModelWithMask(
    #     backbone_pretrained=cfg.get("imagenet_pretrained", False),
    #     # dropout=cfg.get("dropout", 0.2),
    # )

    lumbar_ckpt = cfg.get("lumbar_pretrained_ckpt", "")
    if lumbar_ckpt and hasattr(model, "backbone") and hasattr(model.backbone, "load_partial_checkpoint"):
        model.backbone.load_partial_checkpoint(lumbar_ckpt)

    return model.to(device)


@torch.no_grad()
def run_single_model(model, loader, device, dataset, temperature: float):
    model.eval()
    rows = []
    sample_ptr = 0

    for batch in loader:
        images = batch["images"].to(device)
        study_labels = batch["study_labels"].cpu().numpy()

        batch_size = len(study_labels)
        image_ids = []
        for i in range(batch_size):
            image_id, _, _ = dataset.samples[sample_ptr + i]
            image_ids.append(image_id)
        sample_ptr += batch_size

        outputs = model(images)

        raw_study_logits = outputs["study_logits"].detach().cpu().numpy()
        cal_study_logits = apply_temperature_np(raw_study_logits, temperature)
        study_probs = sigmoid_np(cal_study_logits)

        B = len(image_ids)
        for i in range(B):
            row = {
                "image_id": image_ids[i],
                "y_true": int(study_labels[i]),
                "y_logit": float(cal_study_logits[i]),
                "y_prob": float(study_probs[i]),
            }

            # keep old interface-compatible columns, but whole-lumbar basic has no localization outputs
            row["top_pair_idx"] = -1
            row["top_pair_name"] = None
            row["top_pair_prob"] = np.nan
            row["top_vertebra_idx"] = -1
            row["top_vertebra_name"] = None

            for p in range(4):
                row[f"pair_prob_{PAIR_NAMES[p]}"] = np.nan
                row[f"pair_attn_{PAIR_NAMES[p]}"] = np.nan

            for lv in LUMBAR_LEVELS:
                row[f"vertebra_prob_{lv}"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_fold_predictions(fold_pred_dfs, ensemble_mode: str):
    base = fold_pred_dfs[0].copy()

    study_logits = np.stack([df["y_logit"].values for df in fold_pred_dfs], axis=0)
    study_probs = np.stack([df["y_prob"].values for df in fold_pred_dfs], axis=0)

    if ensemble_mode == "logits":
        ens_logits = study_logits.mean(axis=0)
        ens_probs = sigmoid_np(ens_logits)
    else:
        ens_probs = study_probs.mean(axis=0)
        ens_probs = np.clip(ens_probs, 1e-6, 1.0 - 1e-6)
        ens_logits = np.log(ens_probs / (1.0 - ens_probs))

    base["y_logit"] = ens_logits
    base["y_prob"] = ens_probs

    # localization columns remain NaN / None
    base["top_pair_idx"] = -1
    base["top_pair_name"] = None
    base["top_pair_prob"] = np.nan
    base["top_vertebra_idx"] = -1
    base["top_vertebra_name"] = None

    for p in range(4):
        base[f"pair_prob_{PAIR_NAMES[p]}"] = np.nan
        base[f"pair_attn_{PAIR_NAMES[p]}"] = np.nan

    for lv in LUMBAR_LEVELS:
        base[f"vertebra_prob_{lv}"] = np.nan

    return base


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv_dir = Path(args.cv_dir)

    cv_summary = {}
    cv_summary_path = cv_dir / "cv_summary.json"
    if cv_summary_path.exists():
        with open(cv_summary_path, "r", encoding="utf-8") as f:
            cv_summary = json.load(f)

    threshold = args.threshold if args.threshold is not None else float(cfg.get("threshold", 0.5))

    test_image_dir = cfg.get("test_image_dir", cfg.get("image_dir"))
    if test_image_dir is None:
        raise ValueError("Missing test_image_dir in config.")

    # test_dataset = LumbarCropDataset(
    #     text_csv_path=cfg["test_csv"],
    #     image_dir=test_image_dir,
    #     split="test",
    #     exclude_s1_positive=cfg.get("exclude_s1_positive", False),
    #     aug_cfg=cfg.get("augmentation", {}),
    # )
    test_dataset = LumbarCropDatasetWithMask(
        text_csv_path=cfg["test_csv"],
        image_dir=test_image_dir,
        mask_dir=cfg["test_mask_dir"],
        split="test",
        exclude_s1_positive=cfg.get("exclude_s1_positive", False),
        aug_cfg=cfg.get("augmentation", {}),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=lumbar_crop_collate_fn,
        pin_memory=True,
    )
    raw_test_df = pd.read_csv(cfg["test_csv"])

    fold_pred_dfs = []
    used_folds = []

    for fold_idx in range(1, args.n_folds + 1):
        fold_dir = cv_dir / f"fold_{fold_idx}"
        ckpt_path = fold_dir / args.ckpt_name
        cal_path = fold_dir / "calibration_summary.json"

        if not ckpt_path.exists():
            print(f"[WARN] Skip fold {fold_idx}: checkpoint not found: {ckpt_path}")
            continue

        cal = {}
        if cal_path.exists():
            with open(cal_path, "r", encoding="utf-8") as f:
                cal = json.load(f)

        model = build_model(cfg, device)
        state_dict, extra = load_checkpoint_state_dict(str(ckpt_path), device)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Fold {fold_idx} missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Fold {fold_idx} unexpected keys: {unexpected}")

        if args.temperature is not None:
            temperature = float(args.temperature)
        else:
            temperature = float(cal.get("temperature", extra.get("temperature", 1.0)))

        pred_df = run_single_model(model, test_loader, device, test_dataset, temperature=temperature)
        pred_df["fold_idx"] = fold_idx
        pred_df["fold_temperature"] = temperature

        fold_pred_dfs.append(pred_df)
        used_folds.append({
            "fold": fold_idx,
            "ckpt": str(ckpt_path),
            "temperature": temperature,
        })

        print(f"[INFO] Loaded fold {fold_idx} | temperature={temperature:.6f}")

    if len(fold_pred_dfs) == 0:
        raise RuntimeError("No fold checkpoints were loaded successfully.")

    pred_df = aggregate_fold_predictions(fold_pred_dfs, ensemble_mode=args.ensemble_mode)
    pred_df["y_pred"] = (pred_df["y_prob"] >= float(threshold)).astype(int)

    pred_df["confusion_type"] = "TN"
    pred_df.loc[(pred_df["y_true"] == 0) & (pred_df["y_pred"] == 1), "confusion_type"] = "FP"
    pred_df.loc[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 0), "confusion_type"] = "FN"
    pred_df.loc[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 1), "confusion_type"] = "TP"

    merged = pred_df.merge(raw_test_df, on="image_id", how="left")
    merged["gt_pair"] = merged["location"].apply(location_to_gt_pair)
    merged["gt_vertebra_set"] = merged["location"].apply(location_to_gt_vertebra_set)
    merged["has_vertebra_gt"] = merged["gt_vertebra_set"].apply(lambda x: len(x) > 0)
    merged["grade_bucket"] = merged["grade"].apply(grade_bucket)
    merged["has_location"] = ~merged["location"].isna() & (merged["location"].astype(str).str.strip() != "")
    merged["has_grade"] = ~merged["grade"].isna()
    merged["vertebra_top1_hit"] = np.nan

    prefix = "basic_cv_ensemble"
    merged.to_csv(os.path.join(args.outdir, f"{prefix}_case_predictions.csv"), index=False)

    overall = compute_binary_metrics(
        y_true=merged["y_true"].values,
        y_prob=merged["y_prob"].values,
        threshold=float(threshold),
    )
    pd.DataFrame([overall]).to_csv(
        os.path.join(args.outdir, f"{prefix}_overall_metrics.csv"),
        index=False,
    )

    pd.crosstab(merged["grade_bucket"], merged["confusion_type"], dropna=False).to_csv(
        os.path.join(args.outdir, f"{prefix}_study_confusion_by_grade.csv")
    )

    pair_eval_df = merged[merged["gt_pair"].notna()].copy()
    pd.crosstab(pair_eval_df["gt_pair"], pair_eval_df["confusion_type"], dropna=False).to_csv(
        os.path.join(args.outdir, f"{prefix}_study_confusion_by_location_pair.csv")
    )

    usable_pair = merged[
        (merged["y_true"] == 1) &
        (merged["gt_pair"].notna()) &
        (merged["top_pair_name"].notna())
    ].copy()
    pd.crosstab(usable_pair["gt_pair"], usable_pair["top_pair_name"], dropna=False).to_csv(
        os.path.join(args.outdir, f"{prefix}_adjacent_pair_error_table.csv")
    )

    wrong = merged[merged["y_true"] != merged["y_pred"]].copy()
    wrong.to_csv(os.path.join(args.outdir, f"{prefix}_wrong_cases.csv"), index=False)

    pos_with_grade = merged[(merged["y_true"] == 1) & (merged["grade"].notna())].copy()
    grade_perf_rows = []
    for bucket, gdf in pos_with_grade.groupby("grade_bucket"):
        tp = int(((gdf["y_true"] == 1) & (gdf["y_pred"] == 1)).sum())
        fn = int(((gdf["y_true"] == 1) & (gdf["y_pred"] == 0)).sum())
        recall = tp / max(tp + fn, 1)
        grade_perf_rows.append({
            "grade_bucket": bucket,
            "n": len(gdf),
            "tp": tp,
            "fn": fn,
            "recall": recall,
            "avg_prob": float(gdf["y_prob"].mean()),
        })
    pd.DataFrame(grade_perf_rows).to_csv(
        os.path.join(args.outdir, f"{prefix}_positive_recall_by_grade.csv"),
        index=False,
    )

    pd.crosstab(
        merged["has_location"].map({True: "has_location", False: "missing_location"}),
        merged["confusion_type"]
    ).to_csv(os.path.join(args.outdir, f"{prefix}_confusion_by_location_availability.csv"))

    pd.crosstab(
        merged["has_grade"].map({True: "has_grade", False: "missing_grade"}),
        merged["confusion_type"]
    ).to_csv(os.path.join(args.outdir, f"{prefix}_confusion_by_grade_availability.csv"))

    # keep file for compatibility, but metric is not meaningful for whole-lumbar basic
    pd.DataFrame([{
        "n_cases": 0,
        "vertebra_top1_accuracy": np.nan,
    }]).to_csv(
        os.path.join(args.outdir, f"{prefix}_vertebra_top1_metrics.csv"),
        index=False,
    )

    pd.crosstab(
        merged["top_vertebra_name"],
        merged["confusion_type"],
        dropna=False
    ).to_csv(os.path.join(args.outdir, f"{prefix}_study_confusion_by_top_vertebra.csv"))

    with open(os.path.join(args.outdir, f"{prefix}_ensemble_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "ensemble_mode": args.ensemble_mode,
                "threshold": float(threshold),
                "used_folds": used_folds,
                "cv_summary": cv_summary,
            },
            f,
            indent=2,
        )

    with open(os.path.join(args.outdir, f"{prefix}_summary_report.txt"), "w", encoding="utf-8") as f:
        f.write("Branch: basic_cv_ensemble\n")
        f.write(f"ensemble_mode: {args.ensemble_mode}\n")
        f.write(f"threshold: {threshold}\n")
        f.write("Overall metrics\n")
        for k, v in overall.items():
            f.write(f"{k}: {v}\n")
        f.write("\nWrong-case summary\n")
        f.write(f"Total wrong: {len(wrong)}\n")
        f.write(f"FP: {int((merged['confusion_type'] == 'FP').sum())}\n")
        f.write(f"FN: {int((merged['confusion_type'] == 'FN').sum())}\n")
        f.write("\nPositive cases with usable GT pair labels\n")
        f.write(f"{len(usable_pair)}\n")
        f.write("\nGrade buckets among positives with grade\n")
        for row in grade_perf_rows:
            f.write(str(row) + "\n")


if __name__ == "__main__":
    main()