import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import numpy as np


LUMBAR_LEVELS = ["S1", "L5", "L4", "L3", "L2", "L1"]
LUMBAR_LEVEL_TO_ID = {name: idx for idx, name in enumerate(LUMBAR_LEVELS)}
NUM_LUMBAR_LEVELS = len(LUMBAR_LEVELS)


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

        rgb = np.stack([gray_clahe, gray_clahe, gray_clahe], axis=-1)
        return Image.fromarray(rgb)


def get_transform(split: str = "test", aug_cfg: dict | None = None):
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

    train_tfms = [
        transforms.Resize(image_size),
    ]

    if use_hflip:
        train_tfms.append(transforms.RandomHorizontalFlip(p=hflip_p))

    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.479, 0.479, 0.479], std=[0.167, 0.167, 0.167]),
    ])

    test_tfms = transforms.Compose([
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



class LumbarCropDataset(Dataset):
    def __init__(
        self,
        text_csv_path: str,
        image_dir: str,
        split: str = "test",
        transform=None,
        exclude_s1_positive: bool = True,
        aug_cfg: dict | None = None,
    ):
        self.text_df = pd.read_csv(text_csv_path)
        self.image_dir = image_dir
        self.split = split
        self.transform = transform if transform is not None else get_transform(split, aug_cfg=aug_cfg)

        png_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
        if len(png_files) == 0:
            raise RuntimeError(f"No .png files found in image_dir={image_dir}")

        self.image_dict = {}
        for fname in png_files:
            stem = os.path.splitext(fname)[0]
            base_id = stem[:-7] if stem.endswith("_lumbar") else stem
            self.image_dict[base_id] = fname

        self.samples = []
        for _, row in self.text_df.iterrows():
            image_id = str(row["image_id"])

            if image_id not in self.image_dict:
                continue

            study_label = int(row.get("spondylolisthesis_label", 0))
            loc_raw = row.get("location", "")
            location = "" if pd.isna(loc_raw) else str(loc_raw)

            if exclude_s1_positive and study_label == 1 and has_s1_in_location(location):
                continue

            self.samples.append((image_id, study_label, location))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, study_label, location = self.samples[idx]

        fname = self.image_dict[image_id]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        img = self.transform(img)

        return {
            "image_tensor": img,
            "study_label": torch.tensor(float(study_label), dtype=torch.float32),
            "image_id": image_id,
        }
    

class LumbarCropDatasetWithMask(Dataset):
    def __init__(
        self,
        text_csv_path: str,
        image_dir: str,
        split: str = "test",
        transform=None,
        exclude_s1_positive: bool = True,
        aug_cfg: dict | None = None,
        mask_dir: str | None = None,
    ):
        self.text_df = pd.read_csv(text_csv_path)
        self.image_dir = image_dir
        self.split = split
        self.mask_dir = mask_dir
        self.transform = transform if transform is not None else get_transform(split, aug_cfg=aug_cfg)

        png_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
        if len(png_files) == 0:
            raise RuntimeError(f"No .png files found in image_dir={image_dir}")

        self.image_dict = {}
        for fname in png_files:
            stem = os.path.splitext(fname)[0]
            base_id = stem[:-7] if stem.endswith("_lumbar") else stem
            self.image_dict[base_id] = fname

        self.samples = []
        for _, row in self.text_df.iterrows():
            image_id = str(row["image_id"])

            if image_id not in self.image_dict:
                continue

            study_label = int(row.get("spondylolisthesis_label", 0))
            loc_raw = row.get("location", "")
            location = "" if pd.isna(loc_raw) else str(loc_raw)

            if exclude_s1_positive and study_label == 1 and has_s1_in_location(location):
                continue

            self.samples.append((image_id, study_label, location))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, study_label, location = self.samples[idx]
        fname = self.image_dict[image_id]
        img_path = os.path.join(self.image_dir, fname)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} does not exist.")

        org_img = cv2.imread(img_path)
        if org_img is None:
            raise ValueError(f"Failed to read image {img_path}. Check if the file is corrupted.")

        if self.mask_dir is not None:
            # assume mask file uses the same filename under mask_dir
            # and stores 255 for vertebra+sacrum, 0 for background
            mask_path_png = os.path.join(self.mask_dir, fname)
            mask_path_txt = os.path.join(self.mask_dir, fname.replace(".png", ".txt"))

            if os.path.exists(mask_path_png):
                mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE)
            elif os.path.exists(mask_path_txt):
                mask = np.loadtxt(mask_path_txt, dtype=np.uint8)
            else:
                raise FileNotFoundError(
                    f"Neither mask PNG nor TXT found for {fname} in mask_dir={self.mask_dir}"
                )

            if mask is None:
                raise ValueError(f"Failed to read mask for {fname}")

            # your case: 255 = vertebra + sacrum, 0 = background
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
            "image_tensor": transformed_img,
            "study_label": torch.tensor(float(study_label), dtype=torch.float32),
            "image_id": image_id,
        }


class CombinedLumbarCropDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.index_map = []
        self.samples = []

        for ds_idx, ds in enumerate(self.datasets):
            for local_idx, sample in enumerate(ds.samples):
                self.index_map.append((ds_idx, local_idx))
                self.samples.append(sample)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, local_idx = self.index_map[idx]
        return self.datasets[ds_idx][local_idx]