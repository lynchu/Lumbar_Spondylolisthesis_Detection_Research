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
PAIR_NAMES = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]
PAIR_TO_INDEX = {name: i for i, name in enumerate(PAIR_NAMES)}



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
        mean = [0.4564, 0.4564, 0.4564]  # Sacrum1: [0.3928, 0.3928, 0.3928]
        std = [0.1944, 0.1944, 0.1944]  # Sacrum1: [0.2332, 0.2332, 0.2332]

        use_hflip = bool(aug_cfg.get("use_horizontal_flip", False))
    hflip_p = float(aug_cfg.get("horizontal_flip_prob", 0.5))

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
    # -------------------------
    train_tfms = list(deterministic_tfms)

    if use_hflip:
        train_tfms.append(transforms.RandomHorizontalFlip(p=hflip_p))

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


def _canonical_pair_from_two_levels(a: str, b: str):
    """Return canonical adjacent pair name such as L4-L5, or None."""
    a = str(a).strip().upper()
    b = str(b).strip().upper()
    if a not in LUMBAR_LEVEL_TO_ID or b not in LUMBAR_LEVEL_TO_ID:
        return None
    ia, ib = LUMBAR_LEVEL_TO_ID[a], LUMBAR_LEVEL_TO_ID[b]
    lo, hi = sorted([ia, ib])
    if hi != lo + 1:
        return None
    # LUMBAR_LEVELS is ordered inferior -> superior: S1, L5, L4, ...
    # PAIR_NAMES uses superior-inferior format: L5-S1, L4-L5, ...
    pair = f"{LUMBAR_LEVELS[hi]}-{LUMBAR_LEVELS[lo]}"
    return pair if pair in PAIR_TO_INDEX else None


def location_to_pair_names(location):
    """
    Convert location text to zero or more adjacent-pair labels.

    Examples:
      "L4,L5;" -> ["L4-L5"]
      "L4,L5; L5,S1;" -> ["L5-S1", "L4-L5"] in PAIR_NAMES order
      "L4; L5" -> ["L4-L5"]
      negative/missing location -> []
    """
    if location is None or pd.isna(location):
        return []

    s = str(location).strip()
    if s == "" or s.lower() == "nan":
        return []

    found = []

    # First handle explicit pair segments such as "L4,L5; L5,S1;".
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        toks = [t.strip().upper() for t in seg.split(",") if t.strip()]
        if len(toks) >= 2:
            pair = _canonical_pair_from_two_levels(toks[0], toks[1])
            if pair is not None and pair not in found:
                found.append(pair)

    # Fallback for older level-list style such as "L4; L5".
    if not found:
        levels = parse_location_levels(location)
        ids = sorted(LUMBAR_LEVEL_TO_ID[x] for x in levels)
        for lo, hi in zip(ids[:-1], ids[1:]):
            if hi == lo + 1:
                pair = f"{LUMBAR_LEVELS[hi]}-{LUMBAR_LEVELS[lo]}"
                if pair in PAIR_TO_INDEX and pair not in found:
                    found.append(pair)

    return [p for p in PAIR_NAMES if p in found]


def location_to_pair_label_vector(location, study_label=1):
    """Return a 5-dim multi-hot label in PAIR_NAMES order."""
    vec = torch.zeros(len(PAIR_NAMES), dtype=torch.float32)
    try:
        y = int(float(study_label))
    except Exception:
        y = 0
    if y <= 0:
        return vec
    for pair in location_to_pair_names(location):
        vec[PAIR_TO_INDEX[pair]] = 1.0
    return vec


def image_id_to_position_key(image_id: str) -> str:
    """Map augmented diagnosis image_id to the base key used by lumbar_crop_position.csv.

    Examples:
      0028-Lat          -> 0028
      0028-Lat_clahe    -> 0028
      0028-Lat_hflip    -> 0028
    """
    s = str(image_id)
    if "-Lat" in s:
        return s.split("-Lat", 1)[0]
    return s.split("_", 1)[0]


def _normalize_image_size_for_roi(image_size):
    if image_size is None:
        return (512, 256)
    if isinstance(image_size, int):
        return (int(image_size), int(image_size))
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError(f"augmentation.image_size must be int or [height, width], got {image_size}")


def _get_effective_bottom_crop_ratio(aug_cfg: dict) -> float:
    """Mirror SacrumVerticalTrim crop behavior when ROI masks are built.

    The ROI coordinates in lumbar_crop_position.csv are defined on the original lumbar crop.
    If deterministic sacrum trimming crops the bottom of the image before Resize, the y-coordinate
    scaling must use the remaining height instead of the original full height.
    """
    aug_cfg = aug_cfg or {}
    st = aug_cfg.get("sacrum_trim", {}) or {}
    if not bool(st.get("enabled", False)):
        return 0.0
    mode = str(st.get("mode", "crop")).lower()
    if mode != "crop":
        return 0.0
    if st.get("crop_bottom_ratio", None) is not None:
        ratio = float(st.get("crop_bottom_ratio"))
    else:
        region = float(st.get("sacrum_region_height_ratio", 0.20))
        keep = float(st.get("keep_sacrum_ratio", 0.50))
        ratio = region * max(0.0, 1.0 - keep)
    min_remaining = float(st.get("min_remaining_height_ratio", 0.70))
    ratio = max(0.0, min(float(ratio), 1.0 - min_remaining))
    return ratio


class LumbarPositionROIMaker:
    """Build pair-level ViT patch masks from lumbar_crop_position.csv.

    The CSV is expected to contain one row per vertebra and columns:
      image_id, lumbar_id, crop_bbox_x1, crop_bbox_y1, crop_bbox_x2, crop_bbox_y2, crop_width, crop_height

    lumbar_id follows LUMBAR_LEVELS order: 0=S1, 1=L5, 2=L4, 3=L3, 4=L2, 5=L1.
    For each pair, the ROI is the union of the two vertebra boxes plus a configurable margin,
    converted to the fixed ViT patch grid.
    """
    def __init__(self, position_csv: str, aug_cfg: dict | None = None, verbose: bool = True):
        self.position_csv = position_csv
        self.aug_cfg = aug_cfg or {}
        roi_cfg = (self.aug_cfg.get("pair_roi", {}) or {})
        self.enabled = bool(roi_cfg.get("enabled", False)) and bool(position_csv)
        self.verbose = bool(verbose)
        self.image_size = _normalize_image_size_for_roi(self.aug_cfg.get("image_size", [512, 256]))
        self.patch_size = roi_cfg.get("patch_size", 16)
        if isinstance(self.patch_size, (list, tuple)):
            self.patch_h, self.patch_w = int(self.patch_size[0]), int(self.patch_size[1])
        else:
            self.patch_h = self.patch_w = int(self.patch_size)
        self.grid_h = int(self.image_size[0]) // self.patch_h
        self.grid_w = int(self.image_size[1]) // self.patch_w
        self.margin_w = float(roi_cfg.get("margin_width_ratio", 0.15))
        self.margin_h = float(roi_cfg.get("margin_height_ratio", 0.15))
        self.min_patches_per_pair = int(roi_cfg.get("min_patches_per_pair", 1))
        self.bottom_crop_ratio = _get_effective_bottom_crop_ratio(self.aug_cfg)
        self.position_by_image = {}

        if not self.enabled:
            return
        if not os.path.isfile(position_csv):
            raise FileNotFoundError(f"pair_roi.position_csv not found: {position_csv}")
        df = pd.read_csv(position_csv)
        required = ["image_id", "lumbar_id", "crop_bbox_x1", "crop_bbox_y1", "crop_bbox_x2", "crop_bbox_y2", "crop_width", "crop_height"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"pair_roi.position_csv missing required columns: {missing}")
        for image_key, g in df.groupby("image_id"):
            self.position_by_image[str(image_key)] = g.copy()
        if self.verbose:
            print(
                f"[PairROI] enabled | csv={position_csv} | images={len(self.position_by_image)} | "
                f"grid={self.grid_h}x{self.grid_w} | patch={self.patch_h}x{self.patch_w} | "
                f"bottom_crop_ratio={self.bottom_crop_ratio:.4f}"
            )

    def empty_masks(self):
        return (
            torch.zeros((len(PAIR_NAMES), self.grid_h * self.grid_w), dtype=torch.float32),
            torch.zeros((len(PAIR_NAMES),), dtype=torch.float32),
        )

    def __call__(self, image_id: str):
        masks, valid = self.empty_masks()
        if not self.enabled:
            return masks, valid
        key = image_id_to_position_key(image_id)
        g = self.position_by_image.get(str(key), None)
        if g is None or len(g) == 0:
            return masks, valid

        by_lid = {int(r["lumbar_id"]): r for _, r in g.iterrows() if not pd.isna(r.get("lumbar_id", np.nan))}
        for pair_idx, pair_name in enumerate(PAIR_NAMES):
            lids = PAIR_TO_IDS[pair_name]
            if lids[0] not in by_lid or lids[1] not in by_lid:
                continue
            rows = [by_lid[lids[0]], by_lid[lids[1]]]
            x1 = min(float(r["crop_bbox_x1"]) for r in rows)
            y1 = min(float(r["crop_bbox_y1"]) for r in rows)
            x2 = max(float(r["crop_bbox_x2"]) for r in rows)
            y2 = max(float(r["crop_bbox_y2"]) for r in rows)
            crop_w = max(float(rows[0].get("crop_width", 1.0)), 1.0)
            crop_h = max(float(rows[0].get("crop_height", 1.0)), 1.0)

            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            x1 -= w * self.margin_w
            x2 += w * self.margin_w
            y1 -= h * self.margin_h
            y2 += h * self.margin_h

            kept_h = crop_h * (1.0 - self.bottom_crop_ratio)
            kept_h = max(kept_h, 1.0)
            x1 = max(0.0, min(x1, crop_w))
            x2 = max(0.0, min(x2, crop_w))
            y1 = max(0.0, min(y1, kept_h))
            y2 = max(0.0, min(y2, kept_h))
            if x2 <= x1 or y2 <= y1:
                continue

            # Map original/trimmed crop coordinates to resized model coordinates, then patches.
            rx1 = x1 / crop_w * float(self.image_size[1])
            rx2 = x2 / crop_w * float(self.image_size[1])
            ry1 = y1 / kept_h * float(self.image_size[0])
            ry2 = y2 / kept_h * float(self.image_size[0])

            px1 = int(np.floor(rx1 / self.patch_w))
            px2 = int(np.ceil(rx2 / self.patch_w))
            py1 = int(np.floor(ry1 / self.patch_h))
            py2 = int(np.ceil(ry2 / self.patch_h))
            px1 = max(0, min(px1, self.grid_w - 1))
            py1 = max(0, min(py1, self.grid_h - 1))
            px2 = max(px1 + 1, min(px2, self.grid_w))
            py2 = max(py1 + 1, min(py2, self.grid_h))

            m2d = torch.zeros((self.grid_h, self.grid_w), dtype=torch.float32)
            m2d[py1:py2, px1:px2] = 1.0
            if int(m2d.sum().item()) >= self.min_patches_per_pair:
                masks[pair_idx] = m2d.flatten()
                valid[pair_idx] = 1.0
        return masks, valid



def normalize_grade_label(grade):
    if grade is None or pd.isna(grade):
        return "unknown"
    s = str(grade).strip()
    if s == "" or s.lower() == "nan":
        return "unknown"
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def resolve_pair_roi_position_csv(aug_cfg: dict | None, split: str) -> str | None:
    aug_cfg = aug_cfg or {}
    pair_roi_cfg = aug_cfg.get("pair_roi", {}) or {}

    if not bool(pair_roi_cfg.get("enabled", False)):
        return None

    split_key = str(split).lower()
    if split_key == "train":
        position_csv = pair_roi_cfg.get("train_position_csv")
    elif split_key in {"valid", "val", "validation"}:
        position_csv = pair_roi_cfg.get("valid_position_csv")
    elif split_key == "test":
        position_csv = pair_roi_cfg.get("test_position_csv")
    else:
        position_csv = None

    if position_csv is None:
        position_csv = pair_roi_cfg.get("position_csv")

    if position_csv is None:
        raise ValueError(
            f"pair_roi.enabled=True but no position CSV was provided for split={split}. "
            "Set augmentation.pair_roi.train_position_csv, "
            "augmentation.pair_roi.valid_position_csv, and "
            "augmentation.pair_roi.test_position_csv."
        )

    return position_csv


def format_location_label(location):
    """
    Convert location CSV field into a readable level phrase.

    Example:
      "L4,L5; L5,S1;" -> "the L4, L5 and L5, S1 levels"
      "L4,L5;" -> "the L4, L5 level"
    """
    if location is None or pd.isna(location):
        return "unknown level"

    pairs = []
    for seg in str(location).split(";"):
        seg = seg.strip()
        if not seg:
            continue
        toks = [t.strip().upper() for t in seg.split(",") if t.strip()]
        if len(toks) >= 2:
            pair = f"{toks[0]}, {toks[1]}"
        elif len(toks) == 1:
            pair = toks[0]
        else:
            continue
        if pair not in pairs:
            pairs.append(pair)

    if len(pairs) == 0:
        return "unknown level"
    if len(pairs) == 1:
        return f"the {pairs[0]} level"
    return "the " + " and ".join(pairs) + " levels"


def generate_spondylolisthesis_report(spondylolisthesis_label, location=None, grade=None):
    """
    Generate a label-derived report for the current binary presence task.

    Positive example:
      Grade 1 lumbar spondylolisthesis at the L4, L5 and L5, S1 levels.

    Negative example:
      No lumbar spondylolisthesis.
    """
    try:
        y = int(float(spondylolisthesis_label))
    except Exception:
        y = 0

    if y == 0:
        return "No lumbar spondylolisthesis."

    grade_label = normalize_grade_label(grade)
    location_label = format_location_label(location)
    return f"Grade {grade_label} lumbar spondylolisthesis at {location_label}."


class LumbarCropDataset(Dataset):
    def __init__(
        self,
        text_csv_path: str,
        image_dir: str,
        split: str = "test",
        transform=None,
        exclude_s1_positive: bool = True,
        aug_cfg: dict | None = None,
        pair_roi_position_csv: str | None = None,
        include_implant_values=None,
        exclude_implant_values=None,
        require_implant_column: bool = False,
        verbose_implant_filter: bool = True,
        skip_unknown_implant: bool = True,
    ):
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
        self.aug_cfg = aug_cfg or {}
        self.transform = get_transform(split, aug_cfg=self.aug_cfg)
        self.pair_roi_position_csv = resolve_pair_roi_position_csv(aug_cfg, split)
        self.roi_maker = LumbarPositionROIMaker(resolve_pair_roi_position_csv(aug_cfg, split), aug_cfg=self.aug_cfg, verbose=(split == 'train'))

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
            grade = row.get("grade", None)
            text_report = generate_spondylolisthesis_report(study_label, location, grade)

            if exclude_s1_positive and study_label == 1 and has_s1_in_location(location):
                continue

            self.samples.append((image_id, study_label, location))
            if not hasattr(self, "text_reports"):
                self.text_reports = {}
            self.text_reports[image_id] = text_report

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, study_label, location = self.samples[idx]

        fname = self.image_dict[image_id]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        img = self.transform(img)

        pair_labels = location_to_pair_label_vector(location, study_label=study_label)
        pair_roi_masks, pair_roi_valid = self.roi_maker(image_id)

        out = {
            "image_tensor": img,
            "study_label": torch.tensor(float(study_label), dtype=torch.float32),
            "pair_labels": pair_labels,
            "pair_roi_masks": pair_roi_masks,
            "pair_roi_valid": pair_roi_valid,
            "image_id": image_id,
        }
        if hasattr(self, "text_reports"):
            out["text_report"] = self.text_reports.get(image_id, "")
        return out


class LumbarCropDatasetWithMask(Dataset):
    def __init__(
        self,
        text_csv_path: str,
        image_dir: str,
        split: str = "test",
        transform=None,
        exclude_s1_positive: bool = True,
        aug_cfg: dict | None = None,
        pair_roi_position_csv: str | None = None,
        mask_dir: str | None = None,
        include_implant_values=None,
        exclude_implant_values=None,
        require_implant_column: bool = False,
        verbose_implant_filter: bool = True,
        skip_unknown_implant: bool = True,
    ):
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
        self.aug_cfg = aug_cfg or {}
        self.transform = get_transform(split, aug_cfg=self.aug_cfg)
        self.roi_maker = LumbarPositionROIMaker(pair_roi_position_csv or (self.aug_cfg.get('pair_roi', {}) or {}).get('position_csv', ''), aug_cfg=self.aug_cfg, verbose=(split == 'train'))

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
            grade = row.get("grade", None)
            text_report = generate_spondylolisthesis_report(study_label, location, grade)

            if exclude_s1_positive and study_label == 1 and has_s1_in_location(location):
                continue

            self.samples.append((image_id, study_label, location))
            if not hasattr(self, "text_reports"):
                self.text_reports = {}
            self.text_reports[image_id] = text_report

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

        pair_labels = location_to_pair_label_vector(location, study_label=study_label)
        pair_roi_masks, pair_roi_valid = self.roi_maker(image_id)

        out = {
            "image_tensor": transformed_img,
            "study_label": torch.tensor(float(study_label), dtype=torch.float32),
            "pair_labels": pair_labels,
            "pair_roi_masks": pair_roi_masks,
            "pair_roi_valid": pair_roi_valid,
            "image_id": image_id,
        }
        if hasattr(self, "text_reports"):
            out["text_report"] = self.text_reports.get(image_id, "")
        return out


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