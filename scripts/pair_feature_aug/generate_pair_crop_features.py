#!/usr/bin/env python3
"""
Generate pair-crop numerical features for every original and augmented lumbar image.

This script is intentionally strict about output IDs:
- It uses `crop_image_name` from lumbar_crop_position.csv / lumbar_position.csv as the source of truth.
- `0001-Lat_lumbar.png`          -> output image_id `0001-Lat`
- `0001-Lat_clahe_lumbar.png`    -> output image_id `0001-Lat_clahe`
- `0001-Lat_hflip_lumbar.png`    -> output image_id `0001-Lat_hflip`
- `0001-Lat_inverse_lumbar.png`  -> output image_id `0001-Lat_inverse`
- `0001-Lat_rotate_lumbar.png`   -> output image_id `0001-Lat_rotate`

The raw position CSV `image_id` is kept as `position_image_id` only.
Each augmented version has its own geometry, so features are computed separately
for each `(position_image_id, crop_image_name)` group.
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

EPS = 1e-6
AUG_SUFFIXES = ("clahe", "hflip", "inverse", "rotate")

PAIR_SPECS_WITH_S1 = [
    ("L5-S1", "L5", "S1"),
    ("L4-L5", "L4", "L5"),
    ("L3-L4", "L3", "L4"),
    ("L2-L3", "L2", "L3"),
    ("L1-L2", "L1", "L2"),
]
PAIR_SPECS_WITHOUT_S1 = [
    ("L4-L5", "L4", "L5"),
    ("L3-L4", "L3", "L4"),
    ("L2-L3", "L2", "L3"),
    ("L1-L2", "L1", "L2"),
]

FEATURE_13 = [
    "abs_dx_slip_px",
    "abs_dx_slip_ratio_wavg",
    "abs_left_edge_dx_px",
    "abs_left_edge_dx_ratio_wavg",
    "abs_right_edge_dx_px",
    "abs_right_edge_dx_ratio_wavg",
    "dx_slip_px",
    "left_edge_dx_px",
    "right_edge_dx_px",
    "dy_slip_px",
    "abs_dy_slip_px",
    "pair_interface_width",
    "pair_reference_width",
]

CORNER_ANGLE_FEATURES_4 = [
    "min_top_left_corner_angle_deg",
    "min_top_right_corner_angle_deg",
    "min_bottom_right_corner_angle_deg",
    "min_bottom_left_corner_angle_deg",
]


def angle_between_vectors_deg(a: np.ndarray, b: np.ndarray) -> float:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= EPS or nb <= EPS:
        return np.nan
    cosine = float(np.dot(a, b) / (na * nb + EPS))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def vertebra_corner_angles(row: pd.Series) -> dict[str, float]:
    tl = point_xy(row, "top_left_x", "top_left_y")
    tr = point_xy(row, "top_right_x", "top_right_y")
    br = point_xy(row, "bottom_right_x", "bottom_right_y")
    bl = point_xy(row, "bottom_left_x", "bottom_left_y")
    return {
        "top_left": angle_between_vectors_deg(tr - tl, bl - tl),
        "top_right": angle_between_vectors_deg(tl - tr, br - tr),
        "bottom_right": angle_between_vectors_deg(tr - br, bl - br),
        "bottom_left": angle_between_vectors_deg(tl - bl, br - bl),
    }


def finite_min(values, default=np.nan):
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return default
    return float(np.min(vals))


def pair_min_corner_angle_features(upper: pd.Series, lower: pd.Series) -> dict[str, float]:
    upper_angles = vertebra_corner_angles(upper)
    lower_angles = vertebra_corner_angles(lower)
    return {
        "min_top_left_corner_angle_deg": finite_min([upper_angles["top_left"], lower_angles["top_left"]]),
        "min_top_right_corner_angle_deg": finite_min([upper_angles["top_right"], lower_angles["top_right"]]),
        "min_bottom_right_corner_angle_deg": finite_min([upper_angles["bottom_right"], lower_angles["bottom_right"]]),
        "min_bottom_left_corner_angle_deg": finite_min([upper_angles["bottom_left"], lower_angles["bottom_left"]]),
    }


def point_xy(row: pd.Series, x_col: str, y_col: str) -> np.ndarray:
    return np.array([safe_float(row[x_col]), safe_float(row[y_col])], dtype=np.float64)


def norm_or_nan(vec: np.ndarray) -> float:
    if not np.isfinite(vec).all():
        return np.nan
    return float(np.linalg.norm(vec))


def estimate_slip_percentage_from_corners(upper: pd.Series, lower: pd.Series) -> float:
    """Estimate Meyerding-style slip percentage from segmentation corners.

    The displacement is measured by projecting the upper inferior midpoint to
    lower superior midpoint offset onto the lower superior endplate axis. The
    denominator is the lower superior endplate width, with robust fallbacks.
    The returned value is an absolute percentage because image orientation and
    horizontal-flip augmentation can invert the signed direction.
    """
    lower_top_left = point_xy(lower, "top_left_x", "top_left_y")
    lower_top_right = point_xy(lower, "top_right_x", "top_right_y")
    upper_bottom_mid = point_xy(upper, "bottom_mid_x", "bottom_mid_y")
    lower_top_mid = point_xy(lower, "top_mid_x", "top_mid_y")

    lower_axis = lower_top_right - lower_top_left
    lower_width = norm_or_nan(lower_axis)
    fallback_widths = [
        safe_float(lower.get("top_length")),
        safe_float(lower.get("average_width")),
        safe_float(lower.get("average_top_length")),
    ]
    if not np.isfinite(lower_width) or lower_width <= EPS:
        lower_width = finite_mean(fallback_widths)
    if not np.isfinite(lower_width) or lower_width <= EPS:
        return np.nan

    axis_norm = norm_or_nan(lower_axis)
    if not np.isfinite(axis_norm) or axis_norm <= EPS:
        axis_unit = np.array([1.0, 0.0], dtype=np.float64)
    else:
        axis_unit = lower_axis / axis_norm

    offset_vec = upper_bottom_mid - lower_top_mid
    if not np.isfinite(offset_vec).all():
        return np.nan

    projected_translation_px = float(np.dot(offset_vec, axis_unit))
    return float(abs(projected_translation_px) / (lower_width + EPS) * 100.0)

def clip_slip_percentage(value: float, max_percent: float = 100.0) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(np.clip(value, 0.0, max_percent))


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def finite_mean(values, default=np.nan):
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return default
    return float(np.mean(vals))


def strip_lumbar_suffix_from_stem(stem: str) -> str:
    stem = str(stem).strip()
    # Important: only remove the final crop suffix, not augmentation suffix.
    for suffix in ("_lumbar", "_global", "_crop"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def diag_image_id_from_crop_image_name(crop_image_name) -> Optional[str]:
    if crop_image_name is None or pd.isna(crop_image_name):
        return None
    s = str(crop_image_name).strip()
    if not s:
        return None
    return strip_lumbar_suffix_from_stem(Path(s).stem)


def split_aug(diag_image_id: str):
    s = str(diag_image_id).strip()
    for aug in AUG_SUFFIXES:
        suffix = f"_{aug}"
        if s.endswith(suffix):
            return s[: -len(suffix)], aug, 1
    return s, "original", 0


def read_csv_keep_ids(path: Path) -> pd.DataFrame:
    # dtype=str for image/name-like columns prevents pandas from turning 0001 into 1.
    return pd.read_csv(
        path,
        dtype={
            "image_id": "string",
            "crop_image_name": "string",
            "crop_type": "string",
            "pair_name": "string",
        },
    )


def required_columns():
    return [
        "image_id", "lumbar_id", "crop_image_name",
        "top_left_x", "top_left_y", "top_right_x", "top_right_y",
        "bottom_left_x", "bottom_left_y", "bottom_right_x", "bottom_right_y",
        "top_mid_x", "top_mid_y", "bottom_mid_x", "bottom_mid_y",
        "average_width", "average_height", "top_length", "bottom_length",
        "average_top_length", "average_bottom_length",
    ]


def validate_columns(df: pd.DataFrame, path: Path):
    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def level_to_name(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in {"L1", "L2", "L3", "L4", "L5", "S1"}:
        return s
    try:
        iv = int(float(s))
    except Exception:
        return None
    # Your position files use 0=S1, 1=L5, ..., 5=L1.
    zero_based_reverse = {0: "S1", 1: "L5", 2: "L4", 3: "L3", 4: "L2", 5: "L1"}
    one_based_forward = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "S1"}
    return zero_based_reverse.get(iv) or one_based_forward.get(iv)


def resolve_pair_specs(cfg: dict):
    include_s1 = bool(cfg.get("include_s1_pair_features", cfg.get("include_l5s1", True)))
    return PAIR_SPECS_WITH_S1 if include_s1 else PAIR_SPECS_WITHOUT_S1


def compute_pair_features(upper: pd.Series, lower: pd.Series, *, diag_image_id: str, position_image_id: str, crop_image_name: str, pair_name: str):
    upper_bottom_mid_x = safe_float(upper["bottom_mid_x"])
    upper_bottom_mid_y = safe_float(upper["bottom_mid_y"])
    lower_top_mid_x = safe_float(lower["top_mid_x"])
    lower_top_mid_y = safe_float(lower["top_mid_y"])

    upper_bottom_left_x = safe_float(upper["bottom_left_x"])
    upper_bottom_right_x = safe_float(upper["bottom_right_x"])
    lower_top_left_x = safe_float(lower["top_left_x"])
    lower_top_right_x = safe_float(lower["top_right_x"])

    upper_bottom_length = safe_float(upper.get("bottom_length"))
    lower_top_length = safe_float(lower.get("top_length"))
    upper_avg_width = safe_float(upper.get("average_width"))
    lower_avg_width = safe_float(lower.get("average_width"))

    dx_slip_px = upper_bottom_mid_x - lower_top_mid_x
    dy_slip_px = upper_bottom_mid_y - lower_top_mid_y
    left_edge_dx_px = upper_bottom_left_x - lower_top_left_x
    right_edge_dx_px = upper_bottom_right_x - lower_top_right_x

    pair_interface_width = finite_mean([upper_bottom_length, lower_top_length])
    pair_reference_width = finite_mean([upper_avg_width, lower_avg_width])
    wavg = pair_interface_width if np.isfinite(pair_interface_width) and pair_interface_width > EPS else pair_reference_width
    if not np.isfinite(wavg) or wavg <= EPS:
        wavg = EPS

    estimated_slip_percentage_raw = estimate_slip_percentage_from_corners(upper, lower)
    corner_angle_features = pair_min_corner_angle_features(upper, lower)
    base_image_id, augmentation, is_augmented = split_aug(diag_image_id)

    return {
        "image_id": diag_image_id,                  # diagnosis-style ID: 0001-Lat_clahe
        "base_image_id": base_image_id,             # leakage-safe case ID: 0001-Lat
        "augmentation": augmentation,               # original / clahe / hflip / inverse / rotate
        "is_augmented": int(is_augmented),
        "position_image_id": position_image_id,      # raw position CSV ID: 0001_clahe
        "crop_image_name": crop_image_name,
        "pair_name": pair_name,
        "upper_level": upper["level_name"],
        "lower_level": lower["level_name"],

        # 6 core absolute horizontal features
        "abs_dx_slip_px": abs(dx_slip_px),
        "abs_dx_slip_ratio_wavg": abs(dx_slip_px) / (wavg + EPS),
        "estimated_slip_percentage_raw": estimated_slip_percentage_raw,
        "estimated_slip_percentage": clip_slip_percentage(estimated_slip_percentage_raw),
        "estimated_slip_percentage_outlier": int(np.isfinite(estimated_slip_percentage_raw) and estimated_slip_percentage_raw > 100.0),
        "abs_left_edge_dx_px": abs(left_edge_dx_px),
        "abs_left_edge_dx_ratio_wavg": abs(left_edge_dx_px) / (wavg + EPS),
        "abs_right_edge_dx_px": abs(right_edge_dx_px),
        "abs_right_edge_dx_ratio_wavg": abs(right_edge_dx_px) / (wavg + EPS),

        # 3 signed horizontal features
        "dx_slip_px": dx_slip_px,
        "left_edge_dx_px": left_edge_dx_px,
        "right_edge_dx_px": right_edge_dx_px,

        # 2 vertical context features
        "dy_slip_px": dy_slip_px,
        "abs_dy_slip_px": abs(dy_slip_px),

        # 2 local scale/context features
        "pair_interface_width": pair_interface_width,
        "pair_reference_width": pair_reference_width,

        # 4 pair-level vertebral corner angle features.
        # Each is the smaller same-corner angle across the upper/lower vertebra.
        **corner_angle_features,

        # extra raw scale columns kept for debugging
        "upper_bottom_length": upper_bottom_length,
        "lower_top_length": lower_top_length,
        "upper_average_width": upper_avg_width,
        "lower_average_width": lower_avg_width,
    }


def build_pair_feature_df(position_csv: Path, diag_csv: Optional[Path], pair_specs, skip_missing_diag_ids: bool = False):
    df = read_csv_keep_ids(position_csv)
    validate_columns(df, position_csv)

    # Derive diagnosis-style image_id from crop_image_name for every row.
    df["diag_image_id"] = df["crop_image_name"].apply(diag_image_id_from_crop_image_name)
    bad = df["diag_image_id"].isna() | (df["diag_image_id"].astype(str).str.strip() == "")
    if bad.any():
        examples = df.loc[bad, ["image_id", "crop_image_name"]].head(10).to_dict("records")
        raise ValueError(f"Could not derive diagnosis-style image_id from crop_image_name. Examples: {examples}")

    df["position_image_id"] = df["image_id"].astype(str).str.strip()
    df["level_name"] = df["lumbar_id"].apply(level_to_name)
    df = df[df["level_name"].notna()].copy()

    # Validate exact match with diagnosis CSV. No zero-padding correction, no remapping.
    if diag_csv is not None and Path(diag_csv).is_file():
        diag = read_csv_keep_ids(Path(diag_csv))
        if "image_id" not in diag.columns:
            raise ValueError(f"Diagnosis CSV has no image_id column: {diag_csv}")
        diag_ids = set(diag["image_id"].dropna().astype(str).str.strip())
        generated_ids = set(df["diag_image_id"].dropna().astype(str).str.strip())
        missing = sorted(generated_ids - diag_ids)
        if missing:
            examples_df = df[df["diag_image_id"].isin(missing)][["position_image_id", "diag_image_id", "crop_image_name"]].drop_duplicates().head(20)
            if skip_missing_diag_ids:
                print(
                    f"[WARN] Skipping {len(missing)} generated image_id values absent from diagnosis CSV {diag_csv}. "
                    f"Examples: {missing[:10]}"
                )
                df = df[df["diag_image_id"].isin(diag_ids)].copy()
            else:
                raise ValueError(
                    f"Generated image_id values from crop_image_name were not found in diagnosis CSV {diag_csv}.\n"
                    f"Missing examples: {missing[:20]}\n"
                    f"Rows: {examples_df.to_dict('records')}"
                )

    rows = []
    group_cols = ["position_image_id", "diag_image_id"]
    # Include crop_image_name in group validation but not as grouping key because all six lumbar levels share it.
    for (position_image_id, diag_image_id), gdf in df.groupby(group_cols, sort=False):
        crop_names = gdf["crop_image_name"].dropna().astype(str).str.strip().unique().tolist()
        crop_image_name = crop_names[0] if crop_names else ""
        if len(crop_names) > 1:
            raise ValueError(
                f"One position_image_id/diag_image_id group has multiple crop_image_name values: "
                f"position_image_id={position_image_id}, diag_image_id={diag_image_id}, crop_names={crop_names[:10]}"
            )

        # Each group should have one row per lumbar level. Duplicates are resolved conservatively.
        gdf = gdf.drop_duplicates(subset=["level_name"], keep="first").copy()
        lookup = {row["level_name"]: row for _, row in gdf.iterrows()}
        for pair_name, upper_level, lower_level in pair_specs:
            if upper_level not in lookup or lower_level not in lookup:
                continue
            rows.append(
                compute_pair_features(
                    lookup[upper_level],
                    lookup[lower_level],
                    diag_image_id=str(diag_image_id),
                    position_image_id=str(position_image_id),
                    crop_image_name=crop_image_name,
                    pair_name=pair_name,
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No pair rows generated from {position_csv}. Check lumbar_id encoding and crop_image_name.")

    # Strict output guard: image_id must be diagnosis-style with -Lat.
    raw_like = out[~out["image_id"].astype(str).str.contains("-Lat", regex=False)]
    if len(raw_like) > 0:
        raise RuntimeError(
            "Internal error: output image_id still contains raw position IDs. "
            f"Examples: {raw_like[['image_id', 'position_image_id', 'crop_image_name']].head(20).to_dict('records')}"
        )

    # Useful sanity summary.
    n_versions = out[["base_image_id", "image_id"]].drop_duplicates().groupby("base_image_id")["image_id"].nunique()
    print("[CHECK] augmented versions per base_image_id:")
    print(n_versions.value_counts().sort_index().to_string())
    print("[CHECK] rows per image_id distribution:")
    print(out.groupby("image_id")["pair_name"].nunique().value_counts().sort_index().to_string())

    return out


def infer_position_csv_from_diag_csv(diag_csv: Path) -> Optional[Path]:
    diag_csv = Path(diag_csv)
    candidates = [
        diag_csv.with_name("lumbar_crop_position.csv"),
        diag_csv.with_name("lumbar_position.csv"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def process_one_split(position_csv: Path, output_csv: Path, pair_specs, diag_csv: Optional[Path], skip_missing_diag_ids: bool = False):
    out_df = build_pair_feature_df(position_csv=Path(position_csv), diag_csv=diag_csv, pair_specs=pair_specs, skip_missing_diag_ids=skip_missing_diag_ids)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[OK] {position_csv} -> {output_csv} | rows={len(out_df)} | unique_image_ids={out_df['image_id'].nunique()}")


def str_to_bool(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {x}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--train_pos_csv", type=str, default=None)
    p.add_argument("--valid_pos_csv", type=str, default=None)
    p.add_argument("--test_pos_csv", type=str, default=None)
    p.add_argument("--train_diag_csv", type=str, default=None)
    p.add_argument("--valid_diag_csv", type=str, default=None)
    p.add_argument("--test_diag_csv", type=str, default=None)
    p.add_argument("--train_out_csv", type=str, default=None)
    p.add_argument("--valid_out_csv", type=str, default=None)
    p.add_argument("--test_out_csv", type=str, default=None)
    p.add_argument("--include_s1_pair_features", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    override = str_to_bool(args.include_s1_pair_features)
    if override is not None:
        cfg["include_s1_pair_features"] = override
    pair_specs = resolve_pair_specs(cfg)
    skip_missing_diag_ids = bool(cfg.get("skip_missing_diag_ids", False))
    print(f"[INFO] pair_specs={[p[0] for p in pair_specs]}")
    print(f"[INFO] skip_missing_diag_ids={skip_missing_diag_ids}")

    # Resolve diagnosis CSVs from explicit args first, then config.
    train_diag = Path(args.train_diag_csv or cfg.get("train_diag_csv", cfg.get("train_csv", ""))) if (args.train_diag_csv or cfg.get("train_diag_csv") or cfg.get("train_csv")) else None
    valid_diag = Path(args.valid_diag_csv or cfg.get("valid_diag_csv", cfg.get("val_diag_csv", cfg.get("val_csv", "")))) if (args.valid_diag_csv or cfg.get("valid_diag_csv") or cfg.get("val_diag_csv") or cfg.get("val_csv")) else None
    test_diag = Path(args.test_diag_csv or cfg.get("test_diag_csv", cfg.get("test_csv", ""))) if (args.test_diag_csv or cfg.get("test_diag_csv") or cfg.get("test_csv")) else None

    jobs = []
    if train_diag:
        pos = Path(args.train_pos_csv or cfg.get("train_position_csv", "")) if (args.train_pos_csv or cfg.get("train_position_csv")) else infer_position_csv_from_diag_csv(train_diag)
        out = Path(args.train_out_csv or cfg.get("train_feature_csv", "")) if (args.train_out_csv or cfg.get("train_feature_csv")) else train_diag.with_name("pair_crop_features.csv")
        jobs.append((pos, out, train_diag))
    if valid_diag:
        pos = Path(args.valid_pos_csv or cfg.get("val_position_csv", cfg.get("valid_position_csv", ""))) if (args.valid_pos_csv or cfg.get("val_position_csv") or cfg.get("valid_position_csv")) else infer_position_csv_from_diag_csv(valid_diag)
        out = Path(args.valid_out_csv or cfg.get("val_feature_csv", cfg.get("valid_feature_csv", ""))) if (args.valid_out_csv or cfg.get("val_feature_csv") or cfg.get("valid_feature_csv")) else valid_diag.with_name("pair_crop_features.csv")
        jobs.append((pos, out, valid_diag))
    if test_diag:
        pos = Path(args.test_pos_csv or cfg.get("test_position_csv", "")) if (args.test_pos_csv or cfg.get("test_position_csv")) else infer_position_csv_from_diag_csv(test_diag)
        out = Path(args.test_out_csv or cfg.get("test_feature_csv", "")) if (args.test_out_csv or cfg.get("test_feature_csv")) else test_diag.with_name("pair_crop_features.csv")
        jobs.append((pos, out, test_diag))

    # If no diagnosis/config jobs, allow direct position-only generation.
    if not jobs:
        for pos_arg, out_arg in [
            (args.train_pos_csv, args.train_out_csv),
            (args.valid_pos_csv, args.valid_out_csv),
            (args.test_pos_csv, args.test_out_csv),
        ]:
            if pos_arg:
                pos = Path(pos_arg)
                out = Path(out_arg) if out_arg else pos.with_name("pair_crop_features.csv")
                jobs.append((pos, out, None))

    if not jobs:
        raise ValueError("No inputs found. Use --config or explicit --train_pos_csv/--train_diag_csv paths.")

    seen = set()
    unique_jobs = []
    for pos, out, diag in jobs:
        key = (str(pos), str(out), str(diag) if diag else "")
        if key not in seen:
            unique_jobs.append((pos, out, diag))
            seen.add(key)

    for pos, out, diag in unique_jobs:
        if not Path(pos).is_file():
            raise FileNotFoundError(f"Position CSV not found: {pos}")
        if diag is not None and not Path(diag).is_file():
            raise FileNotFoundError(f"Diagnosis CSV not found: {diag}")
        process_one_split(position_csv=Path(pos), output_csv=Path(out), pair_specs=pair_specs, diag_csv=Path(diag) if diag else None, skip_missing_diag_ids=skip_missing_diag_ids)


if __name__ == "__main__":
    main()
