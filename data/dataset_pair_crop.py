import os
import re
from typing import Optional
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import numpy as np


PAIR_NAMES = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]
PAIR_NAME_TO_ID = {name: idx for idx, name in enumerate(PAIR_NAMES)}


class ApplyCLAHE:
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"ApplyCLAHE expects PIL.Image, got {type(img)}")

        gray = np.array(img.convert("L"))
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        gray_clahe = clahe.apply(gray)

        # keep 3 channels for pretrained RGB backbones
        rgb = np.stack([gray_clahe, gray_clahe, gray_clahe], axis=-1)
        return Image.fromarray(rgb)
    

def get_transform(split: str = "test", aug_cfg: Optional[dict] = None):
    aug_cfg = aug_cfg or {}

    raw_image_size = aug_cfg.get("image_size", 224)
    if isinstance(raw_image_size, (list, tuple)):
        if len(raw_image_size) != 2:
            raise ValueError(
                f"augmentation.image_size must be an int or a 2-element list/tuple, got: {raw_image_size}"
            )
        image_size = (int(raw_image_size[0]), int(raw_image_size[1]))  # (H, W)
    else:
        image_size = int(raw_image_size)

    rotate_deg = float(aug_cfg.get("rotation_deg", 0.0))
    brightness = float(aug_cfg.get("brightness", 0.08))
    contrast = float(aug_cfg.get("contrast", 0.10))
    use_hflip = bool(aug_cfg.get("use_horizontal_flip", False))
    hflip_p = float(aug_cfg.get("horizontal_flip_prob", 0.5))

    clahe_clip_limit = float(aug_cfg.get("clahe_clip_limit", 4.0))
    clahe_tile_grid_size = tuple(aug_cfg.get("clahe_tile_grid_size", [8, 8]))

    train_tfms = [
        ApplyCLAHE(
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        ),
        transforms.Resize(image_size),
        # transforms.RandomRotation(
        #     degrees=rotate_deg,
        #     interpolation=transforms.InterpolationMode.BILINEAR,
        #     fill=0,
        # ),
        # transforms.ColorJitter(
        #     brightness=brightness,
        #     contrast=contrast,
        # ),
    ]
    print(image_size)

    if use_hflip:
        train_tfms.append(transforms.RandomHorizontalFlip(p=hflip_p))

    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.479, 0.479, 0.479], std=[0.167, 0.167, 0.167]),
    ])

    test_tfms = transforms.Compose([
        ApplyCLAHE(
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        ),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.479, 0.479, 0.479], std=[0.167, 0.167, 0.167]),
    ])

    if split == "train":
        return transforms.Compose(train_tfms)
    return test_tfms


def has_s1_in_location(location: str) -> bool:
    if not location:
        return False
    return "S1" in str(location).upper()


def parse_location_levels(location):
    """
    Examples:
      'L4; L5'    -> ['L5', 'L4']? no
      final order should follow anatomical adjacency from caudal to cranial:
      'L5; S1'    -> ['S1', 'L5']
      'L4; L5'    -> ['L5', 'L4']
      'L3;L2'     -> ['L3', 'L2']? no

    To build adjacent pairs robustly, we sort by:
      S1(0), L5(1), L4(2), L3(3), L2(4), L1(5)

    So:
      'L5;S1' -> ['S1', 'L5']
      'L4;L5' -> ['L5', 'L4']
      'L3;L2' -> ['L3', 'L2'] after sort? Actually order becomes ['L3','L2'] only if mapping reversed,
      but we want adjacent traversal low->high in spine:
      S1, L5, L4, L3, L2, L1
    """
    if location is None:
        return []

    s = str(location).strip()
    if s == "" or s.lower() == "nan":
        return []

    level_order = {
        "S1": 0,
        "L5": 1,
        "L4": 2,
        "L3": 3,
        "L2": 4,
        "L1": 5,
    }

    levels = []
    seen = set()

    for x in re.split(r"[;,]", s):
        x = x.strip().upper().replace(" ", "")
        if x in level_order and x not in seen:
            levels.append(x)
            seen.add(x)

    levels.sort(key=lambda x: level_order[x])
    return levels


def id_to_level_name(idx: int) -> str:
    mapping = {
        0: "S1",
        1: "L5",
        2: "L4",
        3: "L3",
        4: "L2",
        5: "L1",
    }
    return mapping[idx]


def location_to_pair_set(location):
    """
    Map report location to a set of adjacent pairs.

    Examples:
      'L5;S1'      -> {'L5-S1'}
      'L4;L5'      -> {'L4-L5'}
      'L4;L5;S1'   -> {'L5-S1', 'L4-L5'}
      'L2;L4'      -> set()
    """
    levels = parse_location_levels(location)
    if len(levels) < 2:
        return set()

    level_order = {
        "S1": 0,
        "L5": 1,
        "L4": 2,
        "L3": 3,
        "L2": 4,
        "L1": 5,
    }
    ids = [level_order[x] for x in levels]

    pair_set = set()
    for a, b in zip(ids[:-1], ids[1:]):
        if b == a + 1:
            low = id_to_level_name(a)
            high = id_to_level_name(b)

            # pair name convention:
            # S1-L5 should be written as L5-S1
            # L5-L4 should be written as L4-L5
            pair_name = f"{high}-{low}"
            if pair_name in PAIR_NAME_TO_ID:
                pair_set.add(pair_name)

    return pair_set


class PairCropLevelDataset(Dataset):
    """
    Independent pair-level dataset:
      one pair_crop image = one sample
    """
    def __init__(
        self,
        text_csv_path: str,
        image_dir: str,
        split: str = "test",
        transform=None,
        exclude_s1_positive: bool = False,
        aug_cfg: Optional[dict] = None,
    ):
        self.text_df = pd.read_csv(text_csv_path)
        self.image_dir = image_dir
        self.transform = transform if transform is not None else get_transform(split, aug_cfg=aug_cfg)

        pair_dict = {}
        for fname in os.listdir(image_dir):
            if not fname.lower().endswith(".png"):
                continue

            stem = os.path.splitext(fname)[0]
            if "_" not in stem:
                continue

            image_id, pair_name = stem.rsplit("_", 1)
            if pair_name not in PAIR_NAME_TO_ID:
                continue

            if image_id not in pair_dict:
                pair_dict[image_id] = {}
            pair_dict[image_id][pair_name] = fname

        self.samples = []

        for _, row in self.text_df.iterrows():
            image_id = str(row["image_id"])
            if image_id not in pair_dict:
                continue

            study_label = int(row.get("spondylolisthesis_label", 0))
            loc_raw = row.get("location", "")
            location = "" if pd.isna(loc_raw) else str(loc_raw)

            gt_pair_set = location_to_pair_set(location)

            # only exclude if user explicitly wants to exclude S1-positive/no-usable-pair cases
            if exclude_s1_positive and study_label == 1 and len(gt_pair_set) == 0:
                continue

            for pair_name, fname in pair_dict[image_id].items():
                label = 1.0 if pair_name in gt_pair_set else 0.0

                self.samples.append({
                    "image_id": image_id,
                    "pair_name": pair_name,
                    "pair_id": PAIR_NAME_TO_ID[pair_name],
                    "filename": fname,
                    "label": label,
                    "study_label": study_label,
                    "location_raw": location,
                    "sample_id": f"{image_id}_{pair_name}",
                })

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
            "pair_id": torch.tensor(int(item["pair_id"]), dtype=torch.long),
            "pair_name": item["pair_name"],
            "image_id": item["image_id"],
            "sample_id": item["sample_id"],
            "study_label": torch.tensor(float(item["study_label"]), dtype=torch.float32),
            "location_raw": item["location_raw"],
        }


class CombinedPairCropDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.index_map = []
        self.samples = []

        for ds_idx, ds in enumerate(self.datasets):
            for local_idx, sample in enumerate(ds.samples):
                self.index_map.append((ds_idx, local_idx))
                merged = dict(sample)
                merged["_image_dir"] = ds.image_dir
                self.samples.append(merged)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, local_idx = self.index_map[idx]
        return self.datasets[ds_idx][local_idx]
    

def summarize_empty_pair_locations(text_csv_path: str):
    df = pd.read_csv(text_csv_path)

    total_rows = 0
    candidate_rows = 0
    empty_pair_rows = 0
    counter = Counter()

    for _, row in df.iterrows():
        location = row.get("location", "")
        levels = parse_location_levels(location)
        pair_set = location_to_pair_set(location)

        total_rows += 1

        if len(levels) >= 2:
            candidate_rows += 1
            if len(pair_set) == 0:
                empty_pair_rows += 1
                counter[str(location)] += 1

    print(f"Total rows: {total_rows}")
    print(f"Rows with >=2 parsed levels: {candidate_rows}")
    print(f"Rows with empty pair_set: {empty_pair_rows}")
    print("\\nBreakdown:")
    for loc, cnt in counter.most_common():
        print(f"{loc}: {cnt}")


if __name__ == "__main__":
    summarize_empty_pair_locations("./dataset/train/lumbar_diagnosis.csv")

