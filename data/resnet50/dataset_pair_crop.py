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
    ]

    if use_hflip:
        train_tfms.append(transforms.RandomHorizontalFlip(p=hflip_p))

    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.479, 0.479, 0.479], std=[0.167, 0.167, 0.167]),
    ])

    test_tfms = transforms.Compose([
        # ApplyCLAHE(
        #     clip_limit=clahe_clip_limit,
        #     tile_grid_size=clahe_tile_grid_size,
        # ),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.479, 0.479, 0.479], std=[0.167, 0.167, 0.167]),
    ])

    if split == "train":
        return transforms.Compose(train_tfms)
    return test_tfms


def parse_location_levels(location):
    """
    New location format example:
        "L3,L4; L4,L5;"

    Meaning:
        - ';' separates positive adjacent pairs
        - ',' separates the two vertebrae within one positive pair

    Return:
        unique levels sorted from low to high spine order:
        ["L4", "L3", "L5"] would finally be sorted by level_order
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

    seen = set()
    levels = []

    pair_chunks = [chunk.strip() for chunk in s.split(";") if chunk.strip() != ""]

    for chunk in pair_chunks:
        parts = [x.strip().upper().replace(" ", "") for x in chunk.split(",")]
        for x in parts:
            if x in level_order and x not in seen:
                seen.add(x)
                levels.append(x)

    levels.sort(key=lambda x: level_order[x])
    return levels


def location_to_pair_set(location):
    """
    New location format example:
        "L3,L4; L4,L5;"

    Returns:
        {"L3-L4", "L4-L5"}
    """
    if location is None:
        return set()

    s = str(location).strip()
    if s == "" or s.lower() == "nan":
        return set()

    level_order = {
        "S1": 0,
        "L5": 1,
        "L4": 2,
        "L3": 3,
        "L2": 4,
        "L1": 5,
    }

    pair_set = set()
    pair_chunks = [chunk.strip() for chunk in s.split(";") if chunk.strip() != ""]

    for chunk in pair_chunks:
        parts = [x.strip().upper().replace(" ", "") for x in chunk.split(",")]
        if len(parts) != 2:
            continue

        a, b = parts
        if a not in level_order or b not in level_order:
            continue

        ida, idb = level_order[a], level_order[b]

        # require adjacent levels only
        if abs(ida - idb) != 1:
            continue

        low_id = min(ida, idb)
        high_id = max(ida, idb)

        low = id_to_level_name(low_id)
        high = id_to_level_name(high_id)
        pair_name = f"{high}-{low}"

        if pair_name in PAIR_NAME_TO_ID:
            pair_set.add(pair_name)

    return pair_set


def has_s1_in_location(location: str) -> bool:
    """
    True if any positive pair contains S1, e.g.
        "L5,S1;"
    """
    if location is None:
        return False

    s = str(location).strip()
    if s == "" or s.lower() == "nan":
        return False

    pair_chunks = [chunk.strip() for chunk in s.split(";") if chunk.strip() != ""]
    for chunk in pair_chunks:
        parts = [x.strip().upper().replace(" ", "") for x in chunk.split(",")]
        if "S1" in parts:
            return True
    return False


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


def get_boundary(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min, y_min, x_max, y_max = float("inf"), float("inf"), 0, 0

    if contours:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    return x_min, y_min, x_max, y_max


def keep_largest_component(mask):
    mask = mask.astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if stats.shape[0] <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    filtered_mask = np.zeros_like(mask)
    filtered_mask[labels == largest_label] = 1
    return filtered_mask


class PairCropDataset(Dataset):
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


class PairCropDatasetWithMask(Dataset):
    """
    Independent pair-level dataset with per-pair mask:
      one pair_crop image + one pair_crop_mask image = one sample

    Expected mask naming:
      same filename under mask_dir as pair_crop image, e.g.
      pair_crop/xxx_L4-L5.png
      pair_crop_mask/xxx_L4-L5.png

    Mask convention:
      255 = foreground, 0 = background
    """
    def __init__(
        self,
        text_csv_path: str,
        image_dir: str,
        split: str = "test",
        transform=None,
        exclude_s1_positive: bool = False,
        aug_cfg: Optional[dict] = None,
        mask_dir: str | None = None,
    ):
        self.text_df = pd.read_csv(text_csv_path)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
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

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} does not exist.")

        org_img = cv2.imread(img_path)
        if org_img is None:
            raise ValueError(f"Failed to read image {img_path}. Check if the file is corrupted.")

        if self.mask_dir is not None:
            mask_path_png = os.path.join(self.mask_dir, item["filename"])
            mask_path_txt = os.path.join(self.mask_dir, item["filename"].replace(".png", ".txt"))

            if os.path.exists(mask_path_png):
                mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE)
            elif os.path.exists(mask_path_txt):
                mask = np.loadtxt(mask_path_txt, dtype=np.uint8)
            else:
                raise FileNotFoundError(
                    f"Neither mask PNG nor TXT found for {item['filename']} in mask_dir={self.mask_dir}"
                )

            if mask is None:
                raise ValueError(f"Failed to read mask for {item['filename']}")

            mask = mask.astype(np.uint8)
            mask[mask != 255] = 0
            mask[mask == 255] = 1

            _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if stats.shape[0] > 1:
                average_size = np.mean(stats[1:, cv2.CC_STAT_AREA])
            else:
                average_size = 0

            mask = cv2.resize(mask, (org_img.shape[1], org_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            kernel_size = max(int(average_size * 0.025), 100)
            kernel = np.ones((kernel_size * 2, kernel_size), np.uint8)
            expanded_mask = cv2.dilate(mask, kernel, iterations=1)
            filtered_mask = keep_largest_component(expanded_mask)

            org_img = cv2.bitwise_and(org_img, org_img, mask=filtered_mask)

            x_min, y_min, x_max, y_max = get_boundary(filtered_mask)
            if x_max > x_min and y_max > y_min:
                org_img = org_img[y_min:y_max, x_min:x_max]

        img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        transformed_img = self.transform(img_pil)

        if transformed_img.shape[0] == 1:
            transformed_img = torch.cat((transformed_img, transformed_img, transformed_img), 0)

        return {
            "image": transformed_img,
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

