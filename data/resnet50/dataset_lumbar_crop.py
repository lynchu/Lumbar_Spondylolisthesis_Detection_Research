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
PAIR_TO_IDS = {
    "L5-S1": (0, 1),
    "L4-L5": (1, 2),
    "L3-L4": (2, 3),
    "L2-L3": (3, 4),
    "L1-L2": (4, 5),
}



VALID_IMPLANT_CATEGORIES = {"0", "1", "2", "3"}
IMPLANT_CATEGORY_DESCRIPTIONS = {
    "0": "no implant",
    "1": "TADs / Temporary Anchorage Devices",
    "2": "implant or blurry region between adjacent vertebrae",
    "3": "other implant not directly intersecting lumbar vertebra region",
}


def normalize_implant_token(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def parse_implant_categories(value, keep_invalid: bool = False) -> list[str]:
    """
    Parse implant as a multi-label field.

    "0" -> ["0"], "1," -> ["1"], "1, 2" -> ["1", "2"],
    "1, 2, 3" -> ["1", "2", "3"], "?" -> [] by default.
    """
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return []

    cats = []
    invalid = []
    for tok in s.split(","):
        tok = normalize_implant_token(tok)
        if tok == "":
            continue
        if tok in VALID_IMPLANT_CATEGORIES:
            if tok not in cats:
                cats.append(tok)
        else:
            if tok not in invalid:
                invalid.append(tok)

    cats = sorted(cats, key=lambda x: int(x))
    return cats + invalid if keep_invalid else cats


# Backward-compatible alias used by older scripts.
def normalize_implant_value(value) -> str:
    cats = parse_implant_categories(value, keep_invalid=True)
    if len(cats) == 0:
        return "missing"
    return ", ".join(cats)


def has_invalid_implant_value(value) -> bool:
    parsed_all = parse_implant_categories(value, keep_invalid=True)
    parsed_valid = parse_implant_categories(value, keep_invalid=False)
    if len(parsed_valid) == 0:
        return True
    return any(x not in VALID_IMPLANT_CATEGORIES for x in parsed_all)


def implant_category_counts(values) -> dict:
    """
    Count implant categories as multi-label membership.
    A row with "1, 2" contributes to both category 1 and category 2.
    Rows like "?" are counted under invalid_or_unknown_skipped.
    """
    counts = {k: 0 for k in sorted(VALID_IMPLANT_CATEGORIES, key=int)}
    invalid = 0
    for v in values:
        cats = parse_implant_categories(v)
        if len(cats) == 0 or has_invalid_implant_value(v):
            invalid += 1
            continue
        for cat in cats:
            counts[cat] += 1
    if invalid > 0:
        counts["invalid_or_unknown_skipped"] = invalid
    return counts


def filter_dataframe_by_implant(
    df: pd.DataFrame,
    include_implant_values=None,
    exclude_implant_values=None,
    require_implant_column: bool = False,
    verbose: bool = True,
    dataset_name: str = "dataset",
    skip_unknown_implant: bool = True,
) -> pd.DataFrame:
    """
    Optional multi-label implant filtering for future ablation experiments.

    Default behavior keeps all valid implant categories 0/1/2/3 but skips unknown
    rows such as "?". Multi-label rows are treated as category membership sets.

    For implant="1, 2": include ["1"] keeps it, include ["2"] keeps it,
    and exclude ["2"] removes it.
    """
    if "implant" not in df.columns:
        msg = f"implant filtering was requested for {dataset_name}, but the CSV does not contain an 'implant' column."
        if require_implant_column:
            raise ValueError(msg)
        if verbose and (include_implant_values is not None or exclude_implant_values is not None):
            print(f"[WARN] {msg} Keep all rows instead.")
        return df

    out = df.copy()
    before = len(out)

    if skip_unknown_implant:
        valid_mask = out["implant"].apply(lambda x: (len(parse_implant_categories(x)) > 0) and (not has_invalid_implant_value(x)))
        skipped = int((~valid_mask).sum())
        if skipped > 0 and verbose:
            bad_counts = out.loc[~valid_mask, "implant"].fillna("missing").astype(str).str.strip().value_counts(dropna=False).sort_index().to_dict()
            print(f"[ImplantFilter] {dataset_name}: skip {skipped} rows with unknown/invalid implant values: {bad_counts}")
        out = out[valid_mask].copy()

    if include_implant_values is not None:
        include_set = set()
        for v in include_implant_values:
            include_set.update(parse_implant_categories(v))
        out = out[out["implant"].apply(lambda x: bool(set(parse_implant_categories(x)) & include_set))].copy()
    else:
        include_set = None

    if exclude_implant_values is not None:
        exclude_set = set()
        for v in exclude_implant_values:
            exclude_set.update(parse_implant_categories(v))
        out = out[~out["implant"].apply(lambda x: bool(set(parse_implant_categories(x)) & exclude_set))].copy()
    else:
        exclude_set = None

    if verbose:
        after = len(out)
        print(
            f"[ImplantFilter] {dataset_name}: {before} -> {after} rows | "
            f"include={sorted(include_set, key=int) if include_set is not None else None} | "
            f"exclude={sorted(exclude_set, key=int) if exclude_set is not None else None} | "
            f"skip_unknown_implant={skip_unknown_implant}"
        )
        print(f"[ImplantFilter] {dataset_name} before multi-label counts: {implant_category_counts(df['implant'])}")
        print(f"[ImplantFilter] {dataset_name} after multi-label counts : {implant_category_counts(out['implant'])}")

    return out

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


class PercentileNormalize:
    """
    Deterministic per-image percentile normalization.

    Maps [p_low, p_high] intensity range to [0, 255].
    This reduces global exposure/brightness variation without strongly
    enhancing local contrast like CLAHE.
    """
    def __init__(self, p_low=1.0, p_high=99.0):
        self.p_low = float(p_low)
        self.p_high = float(p_high)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"PercentileNormalize expects PIL.Image, got {type(img)}")

        gray = np.array(img.convert("L")).astype(np.float32)

        lo = np.percentile(gray, self.p_low)
        hi = np.percentile(gray, self.p_high)

        if hi <= lo + 1e-6:
            norm = np.clip(gray, 0, 255)
        else:
            norm = (gray - lo) / (hi - lo + 1e-6)
            norm = np.clip(norm, 0.0, 1.0) * 255.0

        norm = norm.astype(np.uint8)
        rgb = np.stack([norm, norm, norm], axis=-1)
        return Image.fromarray(rgb)


class RandomGamma:
    """
    Random gamma augmentation for training only.
    gamma_range=(0.8, 1.2) means:
      gamma < 1 brightens mid-tones
      gamma > 1 darkens mid-tones
    """
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        self.gamma_range = gamma_range
        self.p = float(p)

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def get_transform(split: str = "test", aug_cfg: dict | None = None):
    aug_cfg = aug_cfg or {}

    raw_image_size = aug_cfg.get("image_size", 224)
    if isinstance(raw_image_size, (list, tuple)):
        if len(raw_image_size) != 2:
            raise ValueError(
                f"augmentation.image_size must be an int or a 2-element list/tuple, got: {raw_image_size}"
            )
        image_size = (int(raw_image_size[0]), int(raw_image_size[1]))
    else:
        image_size = int(raw_image_size)

    use_imagenet_norm = bool(aug_cfg.get("use_imagenet_norm", False))
    if use_imagenet_norm:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = aug_cfg.get("mean", [0.4564, 0.4564, 0.4564])
        std = aug_cfg.get("std", [0.1944, 0.1944, 0.1944])

    # -------------------------
    # Deterministic intensity preprocessing
    # Applied to train / valid / test equally.
    # Options:
    #   none
    #   percentile
    #   clahe
    #   percentile_clahe
    # -------------------------
    intensity_mode = str(aug_cfg.get("intensity_mode", "none")).lower()

    percentile_low = float(aug_cfg.get("percentile_low", 1.0))
    percentile_high = float(aug_cfg.get("percentile_high", 99.0))

    clahe_clip_limit = float(aug_cfg.get("clahe_clip_limit", 2.0))
    clahe_tile_grid_size = tuple(aug_cfg.get("clahe_tile_grid_size", [8, 8]))

    deterministic_tfms = []

    if intensity_mode in ["percentile", "percentile_clahe"]:
        deterministic_tfms.append(
            PercentileNormalize(
                p_low=percentile_low,
                p_high=percentile_high,
            )
        )

    if intensity_mode in ["clahe", "percentile_clahe"]:
        deterministic_tfms.append(
            ApplyCLAHE(
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        )

    deterministic_tfms.append(transforms.Resize(image_size))

    # -------------------------
    # Training-only augmentation
    # Controlled by augmentation.enable_train_augmentation.
    # No vertical flip is used here.
    # -------------------------
    enable_train_augmentation = bool(aug_cfg.get("enable_train_augmentation", False))
    train_tfms = list(deterministic_tfms)

    if enable_train_augmentation:
        use_rotation = bool(aug_cfg.get("use_random_rotation", False))
        if use_rotation:
            rotation_degrees = aug_cfg.get("rotation_degrees", 5.0)
            rotation_prob = float(aug_cfg.get("rotation_prob", 1.0))
            rotation_fill = int(aug_cfg.get("rotation_fill", 0))
            rotation_transform = transforms.RandomRotation(
                degrees=rotation_degrees,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=rotation_fill,
            )
            if rotation_prob >= 1.0:
                train_tfms.append(rotation_transform)
            elif rotation_prob > 0.0:
                train_tfms.append(transforms.RandomApply([rotation_transform], p=rotation_prob))

        use_hflip = bool(aug_cfg.get("use_horizontal_flip", False))
        if use_hflip:
            hflip_p = float(aug_cfg.get("horizontal_flip_prob", 0.5))
            train_tfms.append(transforms.RandomHorizontalFlip(p=hflip_p))

        # Explicitly do not use vertical flip for lateral lumbar X-ray.
        # If a config accidentally sets it, warn and ignore it.
        if bool(aug_cfg.get("use_vertical_flip", False)):
            print("[WARN] augmentation.use_vertical_flip=True is ignored. Vertical flip is intentionally disabled.")

        use_random_brightness_contrast = bool(
            aug_cfg.get("use_random_brightness_contrast", False)
        )
        if use_random_brightness_contrast:
            train_tfms.append(
                transforms.ColorJitter(
                    brightness=float(aug_cfg.get("brightness_limit", 0.10)),
                    contrast=float(aug_cfg.get("contrast_limit", 0.10)),
                )
            )

        use_random_gamma = bool(aug_cfg.get("use_random_gamma", False))
        if use_random_gamma:
            train_tfms.append(
                RandomGamma(
                    gamma_range=tuple(aug_cfg.get("gamma_range", [0.9, 1.1])),
                    p=float(aug_cfg.get("gamma_prob", 0.5)),
                )
            )

    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # valid / test: deterministic preprocessing only
    test_tfms = list(deterministic_tfms)
    test_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if split == "train":
        return transforms.Compose(train_tfms)
    return transforms.Compose(test_tfms)


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


def parse_location_levels(location):
    if location is None:
        return []
    s = str(location).strip()
    if s == "" or s.lower() == "nan":
        return []

    parts = []
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        for token in seg.split(","):
            token = token.strip().upper()
            if token in LUMBAR_LEVEL_TO_ID and token not in parts:
                parts.append(token)
    return parts


def location_to_adjacent_pair(location):
    levels = parse_location_levels(location)
    if len(levels) < 2:
        return "missing"

    ids = sorted(LUMBAR_LEVEL_TO_ID[x] for x in levels)
    pair_candidates = []
    for a, b in zip(ids[:-1], ids[1:]):
        if b == a + 1:
            pair_candidates.append((a, b))

    if len(pair_candidates) != 1:
        return "missing"

    a, b = pair_candidates[0]
    return f"{LUMBAR_LEVELS[b]}-{LUMBAR_LEVELS[a]}"


def _resolve_geometry_feature_names(df: pd.DataFrame, feature_names=None) -> list[str]:
    if feature_names is None or feature_names == "auto":
        excluded = {"image_id", "base_image_id", "augmentation", "is_augmented"}
        return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    return [str(c) for c in feature_names]


def load_case_geometry_features(feature_csv: str | None, feature_names=None):
    if not feature_csv:
        return None, [], 0

    df = pd.read_csv(feature_csv)
    if "image_id" not in df.columns:
        raise ValueError(f"Geometry feature CSV must contain image_id: {feature_csv}")

    names = _resolve_geometry_feature_names(df, feature_names)
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"Geometry feature CSV {feature_csv} is missing columns: {missing}")

    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    for c in names:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    features = {
        row["image_id"]: torch.tensor(row[names].to_numpy(dtype=np.float32), dtype=torch.float32)
        for _, row in df.iterrows()
    }
    return features, names, len(names)


class LumbarCropDataset(Dataset):
    def __init__(
        self,
        text_csv_path: str,
        image_dir: str,
        split: str = "test",
        transform=None,
        exclude_s1_positive: bool = True,
        aug_cfg: dict | None = None,
        include_implant_values=None,
        exclude_implant_values=None,
        require_implant_column: bool = False,
        verbose_implant_filter: bool = True,
        skip_unknown_implant: bool = True,
        geometry_feature_csv: str | None = None,
        geometry_feature_names=None,
        require_geometry: bool = False,
    ):
        self.geometry_features, self.geometry_feature_names, self.geometry_dim = load_case_geometry_features(
            geometry_feature_csv, geometry_feature_names
        )
        self.require_geometry = bool(require_geometry)
        self.text_df = pd.read_csv(text_csv_path)
        self.text_df = filter_dataframe_by_implant(
            self.text_df,
            include_implant_values=include_implant_values,
            exclude_implant_values=exclude_implant_values,
            require_implant_column=require_implant_column,
            verbose=verbose_implant_filter,
            dataset_name=f"{split}:{os.path.basename(str(text_csv_path))}",
            skip_unknown_implant=skip_unknown_implant,
        )
        self.image_dir = image_dir
        self.split = split
        self.transform = get_transform(split, aug_cfg=aug_cfg)

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
        image_id, study_label, _ = self.samples[idx]

        fname = self.image_dict[image_id]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        img = self.transform(img)

        item = {
            "image_tensor": img,
            "study_label": torch.tensor(float(study_label), dtype=torch.float32),
            "image_id": image_id,
        }
        if self.geometry_features is not None:
            if image_id not in self.geometry_features:
                if self.require_geometry:
                    raise KeyError(f"Missing geometry features for image_id={image_id}")
                item["geometry_features"] = torch.zeros(self.geometry_dim, dtype=torch.float32)
            else:
                item["geometry_features"] = self.geometry_features[image_id]
        return item


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
        include_implant_values=None,
        exclude_implant_values=None,
        require_implant_column: bool = False,
        verbose_implant_filter: bool = True,
        skip_unknown_implant: bool = True,
        geometry_feature_csv: str | None = None,
        geometry_feature_names=None,
        require_geometry: bool = False,
    ):
        self.geometry_features, self.geometry_feature_names, self.geometry_dim = load_case_geometry_features(
            geometry_feature_csv, geometry_feature_names
        )
        self.require_geometry = bool(require_geometry)
        self.text_df = pd.read_csv(text_csv_path)
        self.text_df = filter_dataframe_by_implant(
            self.text_df,
            include_implant_values=include_implant_values,
            exclude_implant_values=exclude_implant_values,
            require_implant_column=require_implant_column,
            verbose=verbose_implant_filter,
            dataset_name=f"{split}:{os.path.basename(str(text_csv_path))}",
            skip_unknown_implant=skip_unknown_implant,
        )
        self.image_dir = image_dir
        self.split = split
        self.mask_dir = mask_dir
        self.transform = get_transform(split, aug_cfg=aug_cfg)

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
        image_id, study_label, _ = self.samples[idx]
        fname = self.image_dict[image_id]
        img_path = os.path.join(self.image_dir, fname)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} does not exist.")

        org_img = cv2.imread(img_path)
        if org_img is None:
            raise ValueError(f"Failed to read image {img_path}. Check if the file is corrupted.")

        if self.mask_dir is not None:
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

        item = {
            "image_tensor": transformed_img,
            "study_label": torch.tensor(float(study_label), dtype=torch.float32),
            "image_id": image_id,
        }
        if self.geometry_features is not None:
            if image_id not in self.geometry_features:
                if self.require_geometry:
                    raise KeyError(f"Missing geometry features for image_id={image_id}")
                item["geometry_features"] = torch.zeros(self.geometry_dim, dtype=torch.float32)
            else:
                item["geometry_features"] = self.geometry_features[image_id]
        return item


class CombinedLumbarCropDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.index_map = []
        self.samples = []

        self.geometry_dim = 0
        self.geometry_feature_names = []
        for ds in self.datasets:
            if getattr(ds, "geometry_dim", 0):
                if self.geometry_dim and self.geometry_dim != ds.geometry_dim:
                    raise ValueError("All combined datasets must use the same geometry feature dimension.")
                self.geometry_dim = ds.geometry_dim
                self.geometry_feature_names = list(getattr(ds, "geometry_feature_names", []))

        for ds_idx, ds in enumerate(self.datasets):
            for local_idx, sample in enumerate(ds.samples):
                self.index_map.append((ds_idx, local_idx))
                self.samples.append(sample)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, local_idx = self.index_map[idx]
        return self.datasets[ds_idx][local_idx]