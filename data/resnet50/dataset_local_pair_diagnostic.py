import os
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.resnet50.dataset_pair_crop import get_transform


PAIR_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
PAIR_TO_ID = {name: i for i, name in enumerate(PAIR_NAMES)}
DEFAULT_GEOMETRY_FEATURES = [
    "estimated_slip_percentage",
    "estimated_slip_percentage_outlier",
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
    "min_top_left_corner_angle_deg",
    "min_top_right_corner_angle_deg",
    "min_bottom_right_corner_angle_deg",
    "min_bottom_left_corner_angle_deg",
]


def normalize_pair_name(a: str, b: str) -> Optional[str]:
    a = str(a).strip().upper().replace(" ", "")
    b = str(b).strip().upper().replace(" ", "")
    order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "S1": 6}
    if a not in order or b not in order:
        return None
    if abs(order[a] - order[b]) != 1:
        return None
    upper, lower = (a, b) if order[a] < order[b] else (b, a)
    pair = f"{upper}-{lower}"
    return pair if pair in PAIR_TO_ID else None


def location_to_pair_set(location) -> set[str]:
    """Parse explicit pair labels such as 'L3,L4; L4,L5;' or 'L5-S1'."""
    if location is None or pd.isna(location):
        return set()
    s = str(location).strip().upper().replace(" ", "")
    if not s or s == "NAN":
        return set()

    pair_set = set()
    for chunk in re.split(r";+", s):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.fullmatch(r"(L[1-5]|S1)[,-](L[1-5]|S1)", chunk)
        if m:
            pair = normalize_pair_name(m.group(1), m.group(2))
            if pair is not None:
                pair_set.add(pair)
    if pair_set:
        return pair_set

    # Legacy format fallback: adjacent levels listed separately.
    levels = []
    seen = set()
    order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "S1": 6}
    for token in re.split(r"[;,]+", s):
        token = token.strip()
        if token in order and token not in seen:
            levels.append(token)
            seen.add(token)
    levels.sort(key=lambda x: order[x])
    for a, b in zip(levels[:-1], levels[1:]):
        pair = normalize_pair_name(a, b)
        if pair is not None:
            pair_set.add(pair)
    return pair_set


def _safe_grade(value) -> int:
    try:
        if pd.isna(value):
            return 0
        grade = int(float(value))
    except Exception:
        return 0
    return int(np.clip(grade, 0, 4))


class LocalPairDiagnosticDataset(Dataset):
    """One sample is one case with all five adjacent-pair crops and geometry rows."""

    def __init__(
        self,
        diag_csv: str,
        pair_image_dir: str,
        feature_csv: str,
        split: str = "test",
        aug_cfg: Optional[dict] = None,
        geometry_feature_names: Optional[Sequence[str]] = None,
        require_all_pairs: bool = True,
        slip_column: str = "estimated_slip_percentage",
        slip_raw_column: str = "estimated_slip_percentage_raw",
        slip_outlier_column: str = "estimated_slip_percentage_outlier",
        exclude_slip_outliers: bool = True,
    ):
        self.diag_csv = str(diag_csv)
        self.pair_image_dir = str(pair_image_dir)
        self.feature_csv = str(feature_csv)
        self.transform = get_transform(split=split, aug_cfg=aug_cfg or {})
        self.geometry_feature_names = list(geometry_feature_names or DEFAULT_GEOMETRY_FEATURES)
        self.require_all_pairs = bool(require_all_pairs)
        self.slip_column = slip_column
        self.slip_raw_column = slip_raw_column
        self.slip_outlier_column = slip_outlier_column
        self.exclude_slip_outliers = bool(exclude_slip_outliers)

        diag = pd.read_csv(diag_csv, dtype={"image_id": str})
        feat = pd.read_csv(feature_csv, dtype={"image_id": str, "pair_name": str})
        diag["image_id"] = diag["image_id"].astype(str).str.strip()
        feat["image_id"] = feat["image_id"].astype(str).str.strip()
        feat["pair_name"] = feat["pair_name"].astype(str).str.strip()

        if "spondylolisthesis_label" not in diag.columns:
            raise ValueError(f"{diag_csv} missing spondylolisthesis_label")
        if "location" not in diag.columns:
            diag["location"] = ""
        if "grade" not in diag.columns:
            diag["grade"] = 0

        missing_features = [c for c in self.geometry_feature_names if c not in feat.columns]
        if missing_features:
            raise ValueError(f"{feature_csv} missing geometry columns: {missing_features}")
        if self.slip_column not in feat.columns:
            raise ValueError(f"{feature_csv} missing slip column: {self.slip_column}")
        if self.slip_raw_column not in feat.columns:
            feat[self.slip_raw_column] = feat[self.slip_column]
        if self.slip_outlier_column not in feat.columns:
            feat[self.slip_outlier_column] = 0

        pair_files = self._index_pair_files(pair_image_dir)
        feat_lookup = {
            (str(row.image_id), str(row.pair_name)): row
            for row in feat.itertuples(index=False)
            if str(row.pair_name) in PAIR_TO_ID
        }

        samples = []
        skipped_missing_pairs = 0
        skipped_missing_features = 0
        for row in diag.itertuples(index=False):
            image_id = str(getattr(row, "image_id")).strip()
            if image_id not in pair_files:
                skipped_missing_pairs += 1
                continue
            has_pairs = all(pair in pair_files[image_id] for pair in PAIR_NAMES)
            has_features = all((image_id, pair) in feat_lookup for pair in PAIR_NAMES)
            if self.require_all_pairs and not has_pairs:
                skipped_missing_pairs += 1
                continue
            if self.require_all_pairs and not has_features:
                skipped_missing_features += 1
                continue

            pair_set = location_to_pair_set(getattr(row, "location", ""))
            samples.append({
                "image_id": image_id,
                "study_label": int(float(getattr(row, "spondylolisthesis_label", 0))),
                "location_raw": getattr(row, "location", ""),
                "grade": _safe_grade(getattr(row, "grade", 0)),
                "pair_files": pair_files[image_id],
                "pair_set": pair_set,
                "feature_lookup": feat_lookup,
            })

        if not samples:
            raise RuntimeError(
                f"No local-pair samples built from diag_csv={diag_csv}, pair_image_dir={pair_image_dir}, feature_csv={feature_csv}."
            )
        self.samples = samples
        print(
            f"[INFO] LocalPairDiagnosticDataset split={split} cases={len(self.samples)} "
            f"skipped_missing_pairs={skipped_missing_pairs} skipped_missing_features={skipped_missing_features}"
        )

    @staticmethod
    def _index_pair_files(pair_image_dir: str) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for fname in os.listdir(pair_image_dir):
            if not fname.lower().endswith(".png"):
                continue
            stem = os.path.splitext(fname)[0]
            if "_" not in stem:
                continue
            image_id, pair_name = stem.rsplit("_", 1)
            if pair_name not in PAIR_TO_ID:
                continue
            out.setdefault(image_id, {})[pair_name] = fname
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pair_images = []
        geometry = []
        pair_labels = []
        slip_targets = []
        slip_raw = []
        slip_outlier = []
        grade_targets = []

        for pair_name in PAIR_NAMES:
            img_path = os.path.join(self.pair_image_dir, item["pair_files"][pair_name])
            image = Image.open(img_path).convert("RGB")
            pair_images.append(self.transform(image))

            feat_row = item["feature_lookup"][(item["image_id"], pair_name)]
            geometry.append([
                float(pd.to_numeric(getattr(feat_row, c), errors="coerce"))
                if not pd.isna(pd.to_numeric(getattr(feat_row, c), errors="coerce")) else np.nan
                for c in self.geometry_feature_names
            ])

            is_pos = int(pair_name in item["pair_set"])
            pair_labels.append(float(is_pos))
            slip_value = float(pd.to_numeric(getattr(feat_row, self.slip_column), errors="coerce"))
            raw_value = float(pd.to_numeric(getattr(feat_row, self.slip_raw_column), errors="coerce"))
            outlier_value = int(float(pd.to_numeric(getattr(feat_row, self.slip_outlier_column), errors="coerce")))
            slip_targets.append(slip_value if np.isfinite(slip_value) else np.nan)
            slip_raw.append(raw_value if np.isfinite(raw_value) else np.nan)
            slip_outlier.append(outlier_value)
            grade_targets.append(item["grade"] if is_pos else 0)

        pair_labels_t = torch.tensor(pair_labels, dtype=torch.float32)
        slip_outlier_t = torch.tensor(slip_outlier, dtype=torch.float32)
        slip_mask = pair_labels_t.clone()
        if self.exclude_slip_outliers:
            slip_mask = slip_mask * (1.0 - slip_outlier_t)

        return {
            "pair_images": torch.stack(pair_images, dim=0),
            "geometry_features": torch.tensor(np.asarray(geometry, dtype=np.float32), dtype=torch.float32),
            "pair_labels": pair_labels_t,
            "study_label": torch.tensor(float(item["study_label"]), dtype=torch.float32),
            "slip_targets": torch.tensor(slip_targets, dtype=torch.float32),
            "slip_targets_raw": torch.tensor(slip_raw, dtype=torch.float32),
            "slip_loss_mask": slip_mask.float(),
            "slip_outlier": slip_outlier_t,
            "grade_targets": torch.tensor(grade_targets, dtype=torch.long),
            "image_id": item["image_id"],
            "location_raw": item["location_raw"],
        }


def local_pair_collate_fn(batch):
    return {
        "pair_images": torch.stack([x["pair_images"] for x in batch], dim=0),
        "geometry_features": torch.stack([x["geometry_features"] for x in batch], dim=0),
        "pair_labels": torch.stack([x["pair_labels"] for x in batch], dim=0),
        "study_label": torch.stack([x["study_label"] for x in batch], dim=0),
        "slip_targets": torch.stack([x["slip_targets"] for x in batch], dim=0),
        "slip_targets_raw": torch.stack([x["slip_targets_raw"] for x in batch], dim=0),
        "slip_loss_mask": torch.stack([x["slip_loss_mask"] for x in batch], dim=0),
        "slip_outlier": torch.stack([x["slip_outlier"] for x in batch], dim=0),
        "grade_targets": torch.stack([x["grade_targets"] for x in batch], dim=0),
        "image_ids": [x["image_id"] for x in batch],
        "location_raw": [x["location_raw"] for x in batch],
    }
