# python ./presence/scripts/basic/analyze_lumbar_crop_twostream_cv_ensemble_test.py --config ./presence/configs/resnet50_basic.yaml   --cv_dir ./presence/expe
# riments/resnet50_basic0_lumbarCrop_noGradClip_smallLR_bce_balacc_cv5/ --outdir ./presence/experiments/resnet50_basic0_lumbar
# Crop_noGradClip_smallLR_bce_balacc_cv5/test_analysis --n_folds 5   --ckpt_name best.pt   --ensemble_mode probs

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path

import cv2

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset_lumbar_crop_twostream import (
    LumbarCropTwoStreamDataset,
    LUMBAR_LEVELS,
    parse_implant_categories,
    has_invalid_implant_value,
    implant_category_counts,
    normalize_implant_value,
)
# Pair/ROI analysis is intentionally not used in Exp 7.

from engine.metrics import compute_binary_metrics

from models.basic_twostream import LumbarCropTwoStreamLateFusionModel


PAIR_NAMES = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]
LEVEL_TO_ORDER = {name: i for i, name in enumerate(LUMBAR_LEVELS)}


def twostream_collate_fn(batch):
    images = torch.stack([x["image_tensor"] for x in batch], dim=0)
    edge_images = torch.stack([x["edge_tensor"] for x in batch], dim=0)
    study_labels = torch.stack([x["study_label"] for x in batch], dim=0)
    image_ids = [x["image_id"] for x in batch]
    return {
        "images": images,
        "edge_images": edge_images,
        "study_labels": study_labels,
        "image_ids": image_ids,
    }


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


def get_implant_filter_kwargs(cfg):
    """
    Optional implant filtering for analysis/test subset selection.

    Default keeps all implant categories. To evaluate a subset:
      implant_filter:
        enabled: true
        include_values: ["2"]

    Or remove a subset:
      implant_filter:
        enabled: true
        exclude_values: ["2"]
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

    print(
        "[INFO] Analysis implant filtering enabled | "
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


def parse_location_levels(location):
    """Parse old location format ('L4; L5') and new format ('L4,L5;')."""
    if pd.isna(location) or location is None:
        return []

    out = []
    for seg in str(location).split(";"):
        seg = seg.strip()
        if not seg:
            continue
        for token in seg.split(","):
            token = token.strip().upper()
            if token in LEVEL_TO_ORDER and token not in out:
                out.append(token)
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


def implant_description(implant):
    return {
        "0": "no implant",
        "1": "TADs / Temporary Anchorage Devices",
        "2": "implant or blurry region between adjacent vertebrae",
        "3": "other implant not directly intersecting lumbar vertebra region",
        "missing": "missing implant value",
        "invalid_or_unknown": "invalid/unknown implant value, skipped",
    }.get(str(implant), "unknown implant category")


def implant_description_multi(implant_value):
    cats = parse_implant_categories(implant_value)
    if len(cats) == 0 or has_invalid_implant_value(implant_value):
        return implant_description("invalid_or_unknown")
    return "; ".join(implant_description(c) for c in cats)


def build_implant_wrong_case_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """
    One detailed CSV with both requested views using multi-label implant membership:
      1. distribution of implant categories among wrong cases
      2. wrong-case ratio within all cases of each implant category

    A case with implant="1, 2" contributes once to category 1 and once to category 2.
    Rows with invalid/unknown values such as "?" are skipped in this category summary.
    """
    if "implant" not in merged.columns:
        merged = merged.copy()
        merged["implant"] = "missing"

    df = merged.copy()
    df["implant_raw"] = df["implant"].fillna("missing").astype(str).str.strip()
    df["implant_categories"] = df["implant_raw"].apply(parse_implant_categories)
    df["implant_invalid_or_unknown"] = df["implant_raw"].apply(has_invalid_implant_value)
    df["is_wrong"] = (df["y_true"] != df["y_pred"]).astype(int)

    valid_df = df[~df["implant_invalid_or_unknown"]].copy()
    skipped_df = df[df["implant_invalid_or_unknown"]].copy()

    total_cases = int(len(valid_df))
    total_wrong_cases = int(valid_df["is_wrong"].sum())
    total_case_memberships = int(valid_df["implant_categories"].apply(len).sum())
    total_wrong_memberships = int(valid_df.loc[valid_df["is_wrong"] == 1, "implant_categories"].apply(len).sum())

    rows = []
    for implant in ["0", "1", "2", "3"]:
        g = valid_df[valid_df["implant_categories"].apply(lambda cats: implant in cats)].copy()
        wrong_g = g[g["is_wrong"] == 1]
        rows.append({
            "implant": implant,
            "implant_description": implant_description(implant),
            "total_cases_in_implant_category": int(len(g)),
            "wrong_cases_in_implant_category": int(len(wrong_g)),
            "wrong_case_distribution_count": int(len(wrong_g)),
            "wrong_case_distribution_ratio_among_all_wrong_cases": float(len(wrong_g) / total_wrong_cases) if total_wrong_cases > 0 else 0.0,
            "wrong_case_distribution_ratio_among_all_wrong_memberships": float(len(wrong_g) / total_wrong_memberships) if total_wrong_memberships > 0 else 0.0,
            "wrong_case_ratio_within_implant_category": float(len(wrong_g) / len(g)) if len(g) > 0 else 0.0,
            "case_distribution_ratio_among_all_cases": float(len(g) / total_cases) if total_cases > 0 else 0.0,
            "case_distribution_ratio_among_all_memberships": float(len(g) / total_case_memberships) if total_case_memberships > 0 else 0.0,
            "fp": int((g["confusion_type"] == "FP").sum()),
            "fn": int((g["confusion_type"] == "FN").sum()),
            "tp": int((g["confusion_type"] == "TP").sum()),
            "tn": int((g["confusion_type"] == "TN").sum()),
            "avg_y_prob_all_cases": float(g["y_prob"].mean()) if len(g) > 0 else np.nan,
            "avg_y_prob_wrong_cases": float(wrong_g["y_prob"].mean()) if len(wrong_g) > 0 else np.nan,
        })

    if len(skipped_df) > 0:
        rows.append({
            "implant": "invalid_or_unknown_skipped",
            "implant_description": implant_description("invalid_or_unknown"),
            "total_cases_in_implant_category": int(len(skipped_df)),
            "wrong_cases_in_implant_category": int(skipped_df["is_wrong"].sum()),
            "wrong_case_distribution_count": int(skipped_df["is_wrong"].sum()),
            "wrong_case_distribution_ratio_among_all_wrong_cases": np.nan,
            "wrong_case_distribution_ratio_among_all_wrong_memberships": np.nan,
            "wrong_case_ratio_within_implant_category": float(skipped_df["is_wrong"].mean()) if len(skipped_df) > 0 else np.nan,
            "case_distribution_ratio_among_all_cases": np.nan,
            "case_distribution_ratio_among_all_memberships": np.nan,
            "fp": int((skipped_df["confusion_type"] == "FP").sum()),
            "fn": int((skipped_df["confusion_type"] == "FN").sum()),
            "tp": int((skipped_df["confusion_type"] == "TP").sum()),
            "tn": int((skipped_df["confusion_type"] == "TN").sum()),
            "avg_y_prob_all_cases": float(skipped_df["y_prob"].mean()) if len(skipped_df) > 0 else np.nan,
            "avg_y_prob_wrong_cases": float(skipped_df.loc[skipped_df["is_wrong"] == 1, "y_prob"].mean()) if int(skipped_df["is_wrong"].sum()) > 0 else np.nan,
        })

    return pd.DataFrame(rows)



# =========================
# Brightness / contrast analysis helpers
# =========================
def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _find_case_image_path(image_dir: Path, image_id: str) -> Path | None:
    """Support both lumbar-crop names and raw image names."""
    candidates = [
        image_dir / f"{image_id}_lumbar.png",
        image_dir / f"{image_id}.png",
    ]
    for p in candidates:
        if p.is_file():
            return p
    matches = sorted(image_dir.glob(f"{image_id}*.png"))
    return matches[0] if matches else None


def _find_case_mask_path(mask_dir: Path | None, image_path: Path, image_id: str) -> Path | None:
    if mask_dir is None:
        return None
    candidates = [
        mask_dir / image_path.name,
        mask_dir / f"{image_id}_lumbar.png",
        mask_dir / f"{image_id}.png",
    ]
    for p in candidates:
        if p.is_file():
            return p
    matches = sorted(mask_dir.glob(f"{image_id}*.png"))
    return matches[0] if matches else None


def _basic_intensity_stats(gray: np.ndarray, prefix: str) -> dict:
    arr = gray.astype(np.float32)
    if arr.size == 0:
        return {f"{prefix}_{k}": np.nan for k in [
            "mean", "std", "min", "max", "p1", "p5", "p25", "p50", "p75", "p95", "p99",
            "contrast_p99_p1", "contrast_p95_p5", "cv",
        ]}

    p1, p5, p25, p50, p75, p95, p99 = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    mean = float(arr.mean())
    std = float(arr.std())
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_p1": float(p1),
        f"{prefix}_p5": float(p5),
        f"{prefix}_p25": float(p25),
        f"{prefix}_p50": float(p50),
        f"{prefix}_p75": float(p75),
        f"{prefix}_p95": float(p95),
        f"{prefix}_p99": float(p99),
        f"{prefix}_contrast_p99_p1": float(p99 - p1),
        f"{prefix}_contrast_p95_p5": float(p95 - p5),
        f"{prefix}_cv": float(std / (mean + 1e-6)),
    }



def compute_case_intensity_stats(image_id: str, image_dir: Path, mask_dir: Path | None = None) -> dict:
    """
    Compute brightness/contrast statistics on the image used by lumbar-crop analysis.

    The main `raw_*` columns are computed on the full lumbar crop. If a matching mask exists,
    additional `mask_fg_*` and `mask_bg_*` columns are added, so you can compare intensity
    inside the lumbar mask vs background/outside-mask region.
    """
    out = {"intensity_image_found": False, "intensity_mask_found": False}
    image_path = _find_case_image_path(image_dir, str(image_id))
    if image_path is None:
        return out

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return out

    out["intensity_image_found"] = True
    out["intensity_image_path"] = str(image_path)
    out["intensity_image_width"] = int(img.shape[1])
    out["intensity_image_height"] = int(img.shape[0])
    out.update(_basic_intensity_stats(img, "raw"))

    mask_path = _find_case_mask_path(mask_dir, image_path, str(image_id))
    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 0
            out["intensity_mask_found"] = True
            out["intensity_mask_path"] = str(mask_path)
            out["mask_fg_area_ratio"] = float(mask_bool.mean())
            if mask_bool.any():
                out.update(_basic_intensity_stats(img[mask_bool], "mask_fg"))
            if (~mask_bool).any():
                out.update(_basic_intensity_stats(img[~mask_bool], "mask_bg"))
            if mask_bool.any() and (~mask_bool).any():
                out["mask_fg_bg_mean_diff"] = float(out.get("mask_fg_mean", np.nan) - out.get("mask_bg_mean", np.nan))
                out["mask_fg_bg_abs_mean_diff"] = float(abs(out["mask_fg_bg_mean_diff"]))
    return out


def add_intensity_stats(merged: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    image_dir_value = cfg.get("test_image_dir", cfg.get("image_dir"))
    if image_dir_value is None:
        print("[WARN] Missing test_image_dir/image_dir. Skip intensity statistics.")
        return merged

    image_dir = Path(image_dir_value).resolve()
    use_lumbar_mask = bool(cfg.get("use_lumbar_mask", cfg.get("use_mask_input", False)))
    mask_dir_value = cfg.get("test_mask_dir", None)
    mask_dir = Path(mask_dir_value).resolve() if (use_lumbar_mask and mask_dir_value is not None) else None

    print(f"[INFO] Computing brightness/contrast statistics from image_dir={image_dir}")
    if mask_dir is not None:
        print(f"[INFO] Also computing mask foreground/background intensity statistics from mask_dir={mask_dir}")

    rows = []
    for image_id in merged["image_id"].astype(str).tolist():
        rows.append(compute_case_intensity_stats(image_id, image_dir=image_dir, mask_dir=mask_dir))
    stats_df = pd.DataFrame(rows)
    return pd.concat([merged.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)


def add_quantile_bucket_columns(df: pd.DataFrame, columns: list[str], suffix: str = "bucket") -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        bucket_col = f"{col}_{suffix}"
        if values.notna().sum() < 3 or values.nunique(dropna=True) < 3:
            out[bucket_col] = "missing_or_constant"
            continue
        try:
            out[bucket_col] = pd.qcut(values, q=3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
            out.loc[values.isna(), bucket_col] = "missing"
        except Exception:
            out[bucket_col] = "missing_or_constant"
    return out


def _summary_for_group(g: pd.DataFrame) -> dict:
    n = int(len(g))
    wrong = int((g["y_true"] != g["y_pred"]).sum()) if n > 0 else 0
    return {
        "n_cases": n,
        "wrong": wrong,
        "wrong_rate": float(wrong / n) if n > 0 else np.nan,
        "fp": int((g["confusion_type"] == "FP").sum()) if "confusion_type" in g.columns else 0,
        "fn": int((g["confusion_type"] == "FN").sum()) if "confusion_type" in g.columns else 0,
        "tp": int((g["confusion_type"] == "TP").sum()) if "confusion_type" in g.columns else 0,
        "tn": int((g["confusion_type"] == "TN").sum()) if "confusion_type" in g.columns else 0,
        "avg_y_prob": float(g["y_prob"].mean()) if "y_prob" in g.columns and n > 0 else np.nan,
        "avg_raw_mean": float(pd.to_numeric(g.get("raw_mean", pd.Series(dtype=float)), errors="coerce").mean()) if n > 0 else np.nan,
        "avg_raw_std": float(pd.to_numeric(g.get("raw_std", pd.Series(dtype=float)), errors="coerce").mean()) if n > 0 else np.nan,
        "avg_raw_contrast_p99_p1": float(pd.to_numeric(g.get("raw_contrast_p99_p1", pd.Series(dtype=float)), errors="coerce").mean()) if n > 0 else np.nan,
    }


def write_intensity_and_gradcam_group_summaries(merged: pd.DataFrame, outdir: str, prefix: str) -> pd.DataFrame:
    """
    Add bucket columns and write CSVs useful for Grad-CAM review grouping.

    Outputs:
      - *_intensity_error_summary.csv: all-case performance by intensity bucket
      - *_gradcam_group_summary.csv: wrong-case counts by confusion/implant/intensity bucket
      - *_wrong_cases_gradcam_groups.csv: wrong cases with group labels and suggested gradcam group names
      - *_intensity_descriptive_stats_by_confusion.csv
    """
    bucket_source_cols = [
        "raw_mean",
        "raw_std",
        "raw_contrast_p99_p1",
        "raw_contrast_p95_p5",
        "mask_fg_mean",
        "mask_fg_bg_abs_mean_diff",
    ]
    merged = add_quantile_bucket_columns(merged, bucket_source_cols)
    bucket_cols = [f"{c}_bucket" for c in bucket_source_cols if f"{c}_bucket" in merged.columns]

    # All-case error summary by each bucket column.
    summary_rows = []
    for bucket_col in bucket_cols:
        for bucket_value, g in merged.groupby(bucket_col, dropna=False):
            row = {
                "bucket_column": bucket_col,
                "bucket_value": bucket_value,
            }
            row.update(_summary_for_group(g))
            summary_rows.append(row)
    intensity_summary = pd.DataFrame(summary_rows)
    intensity_summary.to_csv(os.path.join(outdir, f"{prefix}_intensity_error_summary.csv"), index=False)

    # Descriptive stats by confusion type.
    stat_cols = [c for c in bucket_source_cols if c in merged.columns]
    if stat_cols:
        desc = merged.groupby("confusion_type")[stat_cols].agg(["count", "mean", "std", "median", "min", "max"])
        desc.columns = ["_".join(col).strip() for col in desc.columns.values]
        desc.reset_index().to_csv(os.path.join(outdir, f"{prefix}_intensity_descriptive_stats_by_confusion.csv"), index=False)

    wrong = merged[merged["y_true"] != merged["y_pred"]].copy()
    if len(wrong) == 0:
        pd.DataFrame().to_csv(os.path.join(outdir, f"{prefix}_wrong_cases_gradcam_groups.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(outdir, f"{prefix}_gradcam_group_summary.csv"), index=False)
        return merged

    # Convenient grouping keys for manually reviewing Grad-CAM results.
    for required_col in ["raw_mean_bucket", "raw_contrast_p99_p1_bucket", "raw_std_bucket"]:
        if required_col not in wrong.columns:
            wrong[required_col] = "missing"

    wrong["gradcam_group_basic"] = (
        wrong["confusion_type"].astype(str)
        + "__brightness_" + wrong["raw_mean_bucket"].astype(str)
        + "__contrast_" + wrong["raw_contrast_p99_p1_bucket"].astype(str)
    )

    # One wrong case can have multiple implant categories. Explode to build membership-aware group summary.
    wrong["implant_category_list"] = wrong.get("implant_raw", wrong.get("implant", "missing")).apply(parse_implant_categories)
    wrong_exploded = wrong.explode("implant_category_list")
    wrong_exploded["implant_category_list"] = wrong_exploded["implant_category_list"].fillna("missing")
    wrong_exploded["implant_group_description"] = wrong_exploded["implant_category_list"].apply(implant_description)
    wrong_exploded["gradcam_group_implant_intensity"] = (
        wrong_exploded["confusion_type"].astype(str)
        + "__implant_" + wrong_exploded["implant_category_list"].astype(str)
        + "__brightness_" + wrong_exploded["raw_mean_bucket"].astype(str)
        + "__contrast_" + wrong_exploded["raw_contrast_p99_p1_bucket"].astype(str)
    )

    # Save case-level wrong rows with grouping keys. This is the file to use when browsing Grad-CAMs.
    wrong_exploded.to_csv(os.path.join(outdir, f"{prefix}_wrong_cases_gradcam_groups.csv"), index=False)

    group_rows = []
    group_cols = [
        "gradcam_group_implant_intensity",
        "confusion_type",
        "implant_category_list",
        "implant_group_description",
        "raw_mean_bucket",
        "raw_contrast_p99_p1_bucket",
        "raw_std_bucket",
    ]
    for keys, g in wrong_exploded.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_wrong_memberships"] = int(len(g))
        row["n_unique_wrong_cases"] = int(g["image_id"].nunique()) if "image_id" in g.columns else int(len(g))
        row["avg_y_prob"] = float(g["y_prob"].mean()) if "y_prob" in g.columns and len(g) > 0 else np.nan
        row["avg_raw_mean"] = float(pd.to_numeric(g.get("raw_mean", pd.Series(dtype=float)), errors="coerce").mean())
        row["avg_raw_contrast_p99_p1"] = float(pd.to_numeric(g.get("raw_contrast_p99_p1", pd.Series(dtype=float)), errors="coerce").mean())
        row["avg_raw_std"] = float(pd.to_numeric(g.get("raw_std", pd.Series(dtype=float)), errors="coerce").mean())
        row["example_image_ids"] = ";".join(g["image_id"].astype(str).head(10).tolist()) if "image_id" in g.columns else ""
        group_rows.append(row)

    group_summary = pd.DataFrame(group_rows).sort_values(
        ["n_unique_wrong_cases", "n_wrong_memberships"], ascending=False
    ) if group_rows else pd.DataFrame()
    group_summary.to_csv(os.path.join(outdir, f"{prefix}_gradcam_group_summary.csv"), index=False)

    return merged

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


def softmax_positive_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return probs[:, 1]


def build_model(cfg, device):
    model = LumbarCropTwoStreamLateFusionModel(
        image_backbone_pretrained=cfg.get("image_backbone_imagenet_pretrained", cfg.get("imagenet_pretrained", False)),
        edge_backbone_pretrained=cfg.get("edge_backbone_imagenet_pretrained", False),
        loss_name=cfg.get("loss_name", "bce"),
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.2),
        freeze_edge_backbone=cfg.get("freeze_edge_backbone", False),
    )
    return model.to(device)


@torch.no_grad()
def run_single_model(model, loader, device, dataset, temperature: float, loss_name: str):
    model.eval()
    rows = []
    sample_ptr = 0
    loss_name = loss_name.lower()

    for batch in loader:
        images = batch["images"].to(device)
        study_labels = batch["study_labels"].cpu().numpy()

        batch_size = len(study_labels)
        image_ids = []
        for j in range(batch_size):
            image_id, _, _ = dataset.samples[sample_ptr + j]
            image_ids.append(image_id)
        sample_ptr += batch_size

        edge_images = batch["edge_images"].to(device)
        outputs = model(images, edge_images)
        raw_study_logits = outputs["study_logits"].detach().cpu().numpy()

        if loss_name == "bce":
            cal_study_logits = apply_temperature_np(raw_study_logits, temperature)   # shape (B,)
            study_probs = sigmoid_np(cal_study_logits)                               # shape (B,)
            scalar_logits = cal_study_logits                                         # shape (B,)
        elif loss_name == "ce":
            cal_study_logits = apply_temperature_np(raw_study_logits, temperature)   # shape (B, 2)
            study_probs = softmax_positive_np(cal_study_logits)                      # shape (B,)
            scalar_logits = cal_study_logits[:, 1] - cal_study_logits[:, 0]          # shape (B,)
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

        B = len(image_ids)
        for i in range(B):
            row = {
                "image_id": image_ids[i],
                "y_true": int(study_labels[i]),
                "y_logit": float(scalar_logits[i]),
                "y_prob": float(study_probs[i]),
            }

            row["top_pair_idx"] = -1
            row["top_pair_name"] = None
            row["top_pair_prob"] = np.nan
            row["top_vertebra_idx"] = -1
            row["top_vertebra_name"] = None

            for p in range(len(PAIR_NAMES)):
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

    base["top_pair_idx"] = -1
    base["top_pair_name"] = None
    base["top_pair_prob"] = np.nan
    base["top_vertebra_idx"] = -1
    base["top_vertebra_name"] = None

    for p in range(len(PAIR_NAMES)):
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
    loss_name = cfg.get("loss_name", "bce").lower()
    implant_filter_kwargs = get_implant_filter_kwargs(cfg)

    cv_summary = {}
    cv_summary_path = cv_dir / "cv_summary.json"
    if cv_summary_path.exists():
        with open(cv_summary_path, "r", encoding="utf-8") as f:
            cv_summary = json.load(f)

    threshold = args.threshold if args.threshold is not None else float(cfg.get("threshold", 0.5))

    test_image_dir = cfg.get("test_image_dir", cfg.get("image_dir"))
    test_edge_map_dir = cfg.get("test_edge_map_dir", None)
    if test_image_dir is None:
        raise ValueError("Missing test_image_dir in config.")
    if test_edge_map_dir is None:
        raise ValueError("Missing test_edge_map_dir in config for Exp 7 two-stream analysis.")

    test_dataset = LumbarCropTwoStreamDataset(
        text_csv_path=cfg["test_csv"],
        image_dir=test_image_dir,
        edge_map_dir=test_edge_map_dir,
        split="test",
        exclude_s1_positive=cfg.get("exclude_s1_positive", False),
        aug_cfg=cfg.get("augmentation", {}),
        **implant_filter_kwargs,
    )

    if len(test_dataset) == 0:
        raise RuntimeError("No test samples remain after image/S1/implant filtering.")
    print(f"[INFO] Test samples after image/S1/implant filtering: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=twostream_collate_fn,
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

        pred_df = run_single_model(
            model,
            test_loader,
            device,
            test_dataset,
            temperature=temperature,
            loss_name=loss_name,
        )
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

    if "implant" not in merged.columns:
        print("[WARN] test_csv has no implant column. Implant analysis will use 'missing'.")
        merged["implant"] = "missing"
    merged["implant_raw"] = merged["implant"].fillna("missing").astype(str).str.strip()
    merged["implant"] = merged["implant_raw"].apply(normalize_implant_value)
    merged["implant_categories"] = merged["implant_raw"].apply(lambda x: ",".join(parse_implant_categories(x)))
    merged["implant_invalid_or_unknown"] = merged["implant_raw"].apply(has_invalid_implant_value)
    merged["implant_description"] = merged["implant_raw"].apply(implant_description_multi)

    merged["gt_pair"] = merged["location"].apply(location_to_gt_pair) if "location" in merged.columns else None
    merged["gt_vertebra_set"] = merged["location"].apply(location_to_gt_vertebra_set) if "location" in merged.columns else [[] for _ in range(len(merged))]
    merged["has_vertebra_gt"] = merged["gt_vertebra_set"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
    merged["has_location"] = (
        ~merged["location"].isna() & (merged["location"].astype(str).str.strip() != "")
        if "location" in merged.columns else False
    )

    prefix = "twostream_cv_ensemble"

    # Add brightness/contrast statistics before saving case-level outputs and wrong-case outputs.
    merged = add_intensity_stats(merged, cfg)
    merged = write_intensity_and_gradcam_group_summaries(merged, args.outdir, prefix)

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

    pair_eval_df = merged[merged["gt_pair"].notna()].copy()
    pd.crosstab(pair_eval_df["gt_pair"], pair_eval_df["confusion_type"], dropna=False).to_csv(
        os.path.join(args.outdir, f"{prefix}_study_confusion_by_location_pair.csv")
    )

    wrong = merged[merged["y_true"] != merged["y_pred"]].copy()
    wrong.to_csv(os.path.join(args.outdir, f"{prefix}_wrong_cases.csv"), index=False)

    implant_wrong_summary = build_implant_wrong_case_summary(merged)
    implant_wrong_summary.to_csv(
        os.path.join(args.outdir, f"{prefix}_implant_wrong_case_summary.csv"),
        index=False,
    )

    pd.crosstab(
        merged["has_location"].map({True: "has_location", False: "missing_location"}),
        merged["confusion_type"]
    ).to_csv(os.path.join(args.outdir, f"{prefix}_confusion_by_location_availability.csv"))


    with open(os.path.join(args.outdir, f"{prefix}_ensemble_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "ensemble_mode": args.ensemble_mode,
                "threshold": float(threshold),
                "used_folds": used_folds,
                "cv_summary": cv_summary,
                "loss_name": loss_name,
            },
            f,
            indent=2,
        )

    with open(os.path.join(args.outdir, f"{prefix}_summary_report.txt"), "w", encoding="utf-8") as f:
        f.write("Branch: twostream_late_fusion_cv_ensemble\n")
        f.write(f"ensemble_mode: {args.ensemble_mode}\n")
        f.write(f"loss_name: {loss_name}\n")
        f.write(f"threshold: {threshold}\n")
        f.write("Overall metrics\n")
        for k, v in overall.items():
            f.write(f"{k}: {v}\n")
        f.write("\nWrong-case summary\n")
        f.write(f"Total wrong: {len(wrong)}\n")
        f.write(f"FP: {int((merged['confusion_type'] == 'FP').sum())}\n")
        f.write(f"FN: {int((merged['confusion_type'] == 'FN').sum())}\n")
        f.write("\nImplant wrong-case summary\n")
        if 'implant_wrong_summary' in locals():
            f.write(implant_wrong_summary.to_string(index=False))
            f.write("\n")


if __name__ == "__main__":
    main()