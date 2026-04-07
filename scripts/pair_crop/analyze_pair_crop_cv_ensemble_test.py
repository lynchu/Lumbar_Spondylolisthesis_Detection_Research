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

from data.dataset_pair_crop import PairCropLevelDataset, PAIR_NAMES, location_to_pair_set
from engine.metrics import compute_binary_metrics
from models.pair_crop_binary import PairCropBinaryModel


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
    model = PairCropBinaryModel(
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        dropout=cfg.get("dropout", 0.2),
    )

    lumbar_ckpt = cfg.get("lumbar_pretrained_ckpt", "")
    if lumbar_ckpt and hasattr(model, "backbone") and hasattr(model.backbone, "load_partial_checkpoint"):
        model.backbone.load_partial_checkpoint(lumbar_ckpt)

    return model.to(device)


@torch.no_grad()
def run_single_model(model, loader, device, temperature: float):
    model.eval()
    rows = []

    for batch in loader:
        images = batch["image"].to(device)

        outputs = model(images)
        raw_logits = outputs["logits"].detach().cpu().numpy()
        cal_logits = apply_temperature_np(raw_logits, temperature)
        probs = sigmoid_np(cal_logits)

        labels = batch["label"].cpu().numpy()
        pair_ids = batch["pair_id"].cpu().numpy()
        pair_names = batch["pair_name"]
        image_ids = batch["image_id"]
        sample_ids = batch["sample_id"]
        study_labels = batch["study_label"].cpu().numpy()
        location_raws = batch["location_raw"]

        for i in range(len(labels)):
            row = {
                "image_id": image_ids[i],
                "sample_id": sample_ids[i],
                "pair_id": int(pair_ids[i]),
                "pair_name": pair_names[i],
                "y_true": int(labels[i]),
                "y_logit": float(cal_logits[i]),
                "y_prob": float(probs[i]),
                "study_label": int(study_labels[i]),
                "location_raw": location_raws[i],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_fold_predictions(fold_pred_dfs, ensemble_mode: str):
    base = fold_pred_dfs[0].copy()

    base_keys = base["sample_id"].tolist()
    for i, df in enumerate(fold_pred_dfs[1:], start=2):
        cur_keys = df["sample_id"].tolist()
        if cur_keys != base_keys:
            raise RuntimeError(
                f"Sample order mismatch across folds when aggregating predictions. "
                f"Fold 1 and fold {i} are not aligned."
            )

    logits = np.stack([df["y_logit"].values for df in fold_pred_dfs], axis=0)
    probs = np.stack([df["y_prob"].values for df in fold_pred_dfs], axis=0)

    if ensemble_mode == "logits":
        ens_logits = logits.mean(axis=0)
        ens_probs = sigmoid_np(ens_logits)
    else:
        ens_probs = probs.mean(axis=0)
        ens_probs = np.clip(ens_probs, 1e-6, 1.0 - 1e-6)
        ens_logits = np.log(ens_probs / (1.0 - ens_probs))

    base["y_logit"] = ens_logits
    base["y_prob"] = ens_probs
    return base


def build_pair_level_analysis_df(pred_df, raw_test_df, threshold: float):
    merged = pred_df.merge(raw_test_df, on="image_id", how="left", suffixes=("", "_csv"))

    merged["gt_pair_set"] = merged["location"].apply(location_to_pair_set)
    merged["gt_pair_set_str"] = merged["gt_pair_set"].apply(
        lambda s: ";".join(sorted(list(s))) if isinstance(s, set) and len(s) > 0 else ""
    )
    merged["gt_positive_for_this_pair"] = merged.apply(
        lambda row: int(row["pair_name"] in row["gt_pair_set"]) if isinstance(row["gt_pair_set"], set) else 0,
        axis=1,
    )

    # sanity check: dataset-generated y_true should match location-derived pair label
    merged["label_match_check"] = (
        merged["y_true"].astype(int) == merged["gt_positive_for_this_pair"].astype(int)
    )

    merged["y_pred"] = (merged["y_prob"] >= float(threshold)).astype(int)

    merged["confusion_type"] = "TN"
    merged.loc[(merged["y_true"] == 0) & (merged["y_pred"] == 1), "confusion_type"] = "FP"
    merged.loc[(merged["y_true"] == 1) & (merged["y_pred"] == 0), "confusion_type"] = "FN"
    merged.loc[(merged["y_true"] == 1) & (merged["y_pred"] == 1), "confusion_type"] = "TP"

    merged["grade_bucket"] = merged["grade"].apply(grade_bucket)
    merged["has_location"] = ~merged["location"].isna() & (merged["location"].astype(str).str.strip() != "")
    merged["has_grade"] = ~merged["grade"].isna()

    return merged


def build_study_level_df(pair_df, threshold: float):
    rows = []

    for image_id, gdf in pair_df.groupby("image_id", sort=False):
        gdf = gdf.sort_values("pair_id").reset_index(drop=True)

        top_idx = int(np.argmax(gdf["y_prob"].values))
        top_row = gdf.iloc[top_idx]

        y_true_study = int(gdf["study_label"].iloc[0])
        y_prob_study = float(gdf["y_prob"].max())
        y_logit_study = float(gdf["y_logit"].max())
        y_pred_study = int(y_prob_study >= float(threshold))

        gt_pair_set = top_row["gt_pair_set"]
        if not isinstance(gt_pair_set, set):
            gt_pair_set = set()

        rows.append({
            "image_id": image_id,
            "study_label": y_true_study,
            "y_true": y_true_study,
            "y_logit": y_logit_study,
            "y_prob": y_prob_study,
            "y_pred": y_pred_study,
            "top_pair_name": str(top_row["pair_name"]),
            "top_pair_prob": float(top_row["y_prob"]),
            "top_pair_is_gt": int(top_row["pair_name"] in gt_pair_set),
            "gt_pair_set": gt_pair_set,
            "gt_pair_set_str": ";".join(sorted(list(gt_pair_set))) if len(gt_pair_set) > 0 else "",
            "location": top_row.get("location", np.nan),
            "grade": top_row.get("grade", np.nan),
            "grade_bucket": top_row.get("grade_bucket", "missing"),
            "has_location": bool(top_row.get("has_location", False)),
            "has_grade": bool(top_row.get("has_grade", False)),
        })

    study_df = pd.DataFrame(rows)

    study_df["confusion_type"] = "TN"
    study_df.loc[(study_df["y_true"] == 0) & (study_df["y_pred"] == 1), "confusion_type"] = "FP"
    study_df.loc[(study_df["y_true"] == 1) & (study_df["y_pred"] == 0), "confusion_type"] = "FN"
    study_df.loc[(study_df["y_true"] == 1) & (study_df["y_pred"] == 1), "confusion_type"] = "TP"

    return study_df


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

    test_dataset = PairCropLevelDataset(
        text_csv_path=cfg["test_csv"],
        image_dir=cfg["test_image_dir"],
        split="test",
        exclude_s1_positive=cfg.get("exclude_s1_positive", False),
        aug_cfg=cfg.get("augmentation", {}),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
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

        pred_df = run_single_model(model, test_loader, device, temperature=temperature)
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
    pair_df = build_pair_level_analysis_df(pred_df, raw_test_df, threshold=threshold)
    study_df = build_study_level_df(pair_df, threshold=threshold)

    prefix = "pair_crop_cv_ensemble"

    # save raw case predictions
    pair_df.to_csv(os.path.join(args.outdir, f"{prefix}_pair_case_predictions.csv"), index=False)
    study_df.to_csv(os.path.join(args.outdir, f"{prefix}_study_case_predictions.csv"), index=False)

    # pair-level overall metrics
    pair_overall = compute_binary_metrics(
        y_true=pair_df["y_true"].values,
        y_prob=pair_df["y_prob"].values,
        threshold=float(threshold),
    )
    pd.DataFrame([pair_overall]).to_csv(
        os.path.join(args.outdir, f"{prefix}_pair_overall_metrics.csv"),
        index=False,
    )

    # study-level overall metrics
    study_overall = compute_binary_metrics(
        y_true=study_df["y_true"].values,
        y_prob=study_df["y_prob"].values,
        threshold=float(threshold),
    )
    pd.DataFrame([study_overall]).to_csv(
        os.path.join(args.outdir, f"{prefix}_study_overall_metrics.csv"),
        index=False,
    )

    # label sanity check
    label_check_df = pd.DataFrame([{
        "n_rows": int(len(pair_df)),
        "n_label_match": int(pair_df["label_match_check"].sum()),
        "n_label_mismatch": int((~pair_df["label_match_check"]).sum()),
        "match_ratio": float(pair_df["label_match_check"].mean()),
    }])
    label_check_df.to_csv(
        os.path.join(args.outdir, f"{prefix}_label_match_check.csv"),
        index=False,
    )

    # pair-level confusion by pair type
    pd.crosstab(pair_df["pair_name"], pair_df["confusion_type"], dropna=False).to_csv(
        os.path.join(args.outdir, f"{prefix}_pair_confusion_by_pair_name.csv")
    )

    # pair-level confusion by grade
    pd.crosstab(pair_df["grade_bucket"], pair_df["confusion_type"], dropna=False).to_csv(
        os.path.join(args.outdir, f"{prefix}_pair_confusion_by_grade.csv")
    )

    # pair-level confusion by location availability
    pd.crosstab(
        pair_df["has_location"].map({True: "has_location", False: "missing_location"}),
        pair_df["confusion_type"],
        dropna=False,
    ).to_csv(os.path.join(args.outdir, f"{prefix}_pair_confusion_by_location_availability.csv"))

    # pair-level confusion by grade availability
    pd.crosstab(
        pair_df["has_grade"].map({True: "has_grade", False: "missing_grade"}),
        pair_df["confusion_type"],
        dropna=False,
    ).to_csv(os.path.join(args.outdir, f"{prefix}_pair_confusion_by_grade_availability.csv"))

    # pair-level positive recall by pair type
    pair_pos_rows = []
    for pair_name, gdf in pair_df[pair_df["y_true"] == 1].groupby("pair_name"):
        tp = int(((gdf["y_true"] == 1) & (gdf["y_pred"] == 1)).sum())
        fn = int(((gdf["y_true"] == 1) & (gdf["y_pred"] == 0)).sum())
        recall = tp / max(tp + fn, 1)
        pair_pos_rows.append({
            "pair_name": pair_name,
            "n": int(len(gdf)),
            "tp": tp,
            "fn": fn,
            "recall": float(recall),
            "avg_prob": float(gdf["y_prob"].mean()),
        })
    pd.DataFrame(pair_pos_rows).to_csv(
        os.path.join(args.outdir, f"{prefix}_pair_positive_recall_by_pair.csv"),
        index=False,
    )

    # pair-level positive recall by grade
    grade_perf_rows = []
    pos_with_grade = pair_df[(pair_df["y_true"] == 1) & (pair_df["grade"].notna())].copy()
    for bucket, gdf in pos_with_grade.groupby("grade_bucket"):
        tp = int(((gdf["y_true"] == 1) & (gdf["y_pred"] == 1)).sum())
        fn = int(((gdf["y_true"] == 1) & (gdf["y_pred"] == 0)).sum())
        recall = tp / max(tp + fn, 1)
        grade_perf_rows.append({
            "grade_bucket": bucket,
            "n": int(len(gdf)),
            "tp": tp,
            "fn": fn,
            "recall": float(recall),
            "avg_prob": float(gdf["y_prob"].mean()),
        })
    pd.DataFrame(grade_perf_rows).to_csv(
        os.path.join(args.outdir, f"{prefix}_pair_positive_recall_by_grade.csv"),
        index=False,
    )

    # pair-level wrong cases
    pair_wrong = pair_df[pair_df["y_true"] != pair_df["y_pred"]].copy()
    pair_wrong.to_csv(os.path.join(args.outdir, f"{prefix}_pair_wrong_cases.csv"), index=False)

    # study-level confusion by grade
    pd.crosstab(study_df["grade_bucket"], study_df["confusion_type"], dropna=False).to_csv(
        os.path.join(args.outdir, f"{prefix}_study_confusion_by_grade.csv")
    )

    # study-level confusion by top predicted pair
    pd.crosstab(study_df["top_pair_name"], study_df["confusion_type"], dropna=False).to_csv(
        os.path.join(args.outdir, f"{prefix}_study_confusion_by_top_pair.csv")
    )

    # study-level top pair hit
    positive_study_df = study_df[(study_df["y_true"] == 1) & (study_df["gt_pair_set_str"] != "")].copy()

    top_pair_hit_df = pd.DataFrame([{
        "n_cases": int(len(positive_study_df)),
        "top_pair_hit_accuracy": float(positive_study_df["top_pair_is_gt"].mean()) if len(positive_study_df) > 0 else np.nan,
    }])
    top_pair_hit_df.to_csv(
        os.path.join(args.outdir, f"{prefix}_study_top_pair_hit_metrics.csv"),
        index=False,
    )

    # study-level GT pair set vs top predicted pair
    pd.crosstab(
        positive_study_df["gt_pair_set_str"],
        positive_study_df["top_pair_name"],
        dropna=False,
    ).to_csv(os.path.join(args.outdir, f"{prefix}_study_top_pair_error_table.csv"))

    # study-level wrong cases
    study_wrong = study_df[study_df["y_true"] != study_df["y_pred"]].copy()
    study_wrong.to_csv(os.path.join(args.outdir, f"{prefix}_study_wrong_cases.csv"), index=False)

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
        f.write("Branch: pair_crop_cv_ensemble\n")
        f.write(f"ensemble_mode: {args.ensemble_mode}\n")
        f.write(f"threshold: {threshold}\n")

        f.write("\n[Pair-level overall metrics]\n")
        for k, v in pair_overall.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[Study-level overall metrics]\n")
        for k, v in study_overall.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[Label match check]\n")
        f.write(f"n_rows: {int(len(pair_df))}\n")
        f.write(f"n_label_match: {int(pair_df['label_match_check'].sum())}\n")
        f.write(f"n_label_mismatch: {int((~pair_df['label_match_check']).sum())}\n")
        f.write(f"match_ratio: {float(pair_df['label_match_check'].mean())}\n")

        f.write("\n[Pair-level wrong-case summary]\n")
        f.write(f"Total wrong: {len(pair_wrong)}\n")
        f.write(f"FP: {int((pair_df['confusion_type'] == 'FP').sum())}\n")
        f.write(f"FN: {int((pair_df['confusion_type'] == 'FN').sum())}\n")

        f.write("\n[Study-level wrong-case summary]\n")
        f.write(f"Total wrong: {len(study_wrong)}\n")
        f.write(f"FP: {int((study_df['confusion_type'] == 'FP').sum())}\n")
        f.write(f"FN: {int((study_df['confusion_type'] == 'FN').sum())}\n")

        f.write("\n[Study-level top-pair hit among positive studies with usable GT pair labels]\n")
        if len(positive_study_df) > 0:
            f.write(f"n_cases: {len(positive_study_df)}\n")
            f.write(f"top_pair_hit_accuracy: {float(positive_study_df['top_pair_is_gt'].mean())}\n")
        else:
            f.write("n_cases: 0\n")
            f.write("top_pair_hit_accuracy: nan\n")

        f.write("\n[Pair-level positive recall by grade]\n")
        for row in grade_perf_rows:
            f.write(str(row) + "\n")

    print("=" * 80)
    print("[DONE] Pair-crop CV ensemble analysis finished.")
    print(f"Outputs saved to: {args.outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()