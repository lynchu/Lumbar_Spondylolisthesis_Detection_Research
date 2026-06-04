# python ./presence/scripts/pair_crop_resnet50/visualize_pair_crop_wrong_cases_gradcam.py \
#   --wrong_cases_csv ./presence/experiments/resnet50_pairCrop_aug_b16_bce_balacc_cv5/test_analysis/pair_crop_cv_ensemble_wrong_case_predictions.csv \
#   --pair_predictions_csv ./presence/experiments/resnet50_pairCrop_aug_b16_bce_balacc_cv5/test_analysis/pair_crop_cv_ensemble_pair_predictions.csv \
#   --config ./presence/configs/resnet50/resnet50_pair_crop.yaml \
#   --cv_dir ./presence/experiments/resnet50_pairCrop_aug_b16_bce_balacc_cv5 \
#   --mode case \
#   --overwrite



#!/usr/bin/env python3
"""
Visualize Grad-CAM for pair-crop lumbar spondylolisthesis presence/location errors.

Supported inputs:
1) Case-level wrong cases from pair-crop analysis:
   pair_crop_cv_ensemble_wrong_case_predictions.csv
   -> saves one panel per case, containing all available adjacent pair crops.

2) Pair-level wrong predictions from pair-crop analysis:
   pair_crop_cv_ensemble_wrong_pair_predictions.csv
   -> saves one panel per wrong pair crop.

Default behavior:
- Loads all fold checkpoints and averages Grad-CAM maps across folds.
- Uses the same pair-crop filename convention as PairCropDataset:
    {image_id}_{pair_name}.png
- Targets ResNet50 last conv layer by default:
    backbone.layer4.2.conv3

Example, case-level wrong cases:
python ./presence/scripts/pair_crop_resnet50/visualize_pair_crop_wrong_cases_gradcam.py \
  --wrong_cases_csv ./presence/experiments/resnet50_pairCrop_aug_b16_bce_balacc_cv5/test_analysis/pair_crop_cv_ensemble_wrong_case_predictions.csv \
  --pair_predictions_csv ./presence/experiments/resnet50_pairCrop_aug_b16_bce_balacc_cv5/test_analysis/pair_crop_cv_ensemble_pair_predictions.csv \
  --config ./presence/configs/resnet50/resnet50_pair_crop.yaml \
  --cv_dir ./presence/experiments/resnet50_pairCrop_aug_b16_bce_balacc_cv5 \
  --mode case \
  --overwrite

Example, pair-level wrong pairs:
python ./presence/scripts/pair_crop_resnet50/visualize_pair_crop_wrong_cases_gradcam.py \
  --wrong_cases_csv ./presence/experiments/resnet50_pairCrop_aug_b16_bce_balacc_cv5/test_analysis/pair_crop_cv_ensemble_wrong_pair_predictions.csv \
  --config ./presence/configs/resnet50/resnet50_pair_crop.yaml \
  --cv_dir ./presence/experiments/resnet50_pairCrop_aug_b16_bce_balacc_cv5 \
  --mode pair \
  --overwrite
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from models.basic import LumbarCropModel

PAIR_NAMES = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]


# -------------------------
# CLI / config
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrong_cases_csv",
        type=str,
        required=True,
        help=(
            "Either pair_crop_cv_ensemble_wrong_case_predictions.csv "
            "or pair_crop_cv_ensemble_wrong_pair_predictions.csv"
        ),
    )
    parser.add_argument(
        "--pair_predictions_csv",
        type=str,
        default=None,
        help=(
            "Optional full pair prediction CSV. Recommended for --mode case because it provides "
            "pair_prob / pair_pred / pair_y_true for every pair crop."
        ),
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--cv_dir",
        type=str,
        default=None,
        help="CV experiment root containing fold_1/, fold_2/, ...; if omitted, infer from wrong_cases_csv.",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, default="case", choices=["case", "pair"])
    parser.add_argument(
        "--cam_target",
        type=str,
        default="positive",
        choices=["positive", "pred"],
        help=(
            "positive: always backprop the positive-class logit. "
            "pred: backprop positive logit for predicted-positive pairs, and -logit for predicted-negative pairs."
        ),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--target_layer", type=str, default=None, help="Optional dotted module path, e.g. backbone.layer4.2.conv3")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_grayscale_cam", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.40)
    parser.add_argument(
        "--only_pairs",
        type=str,
        default=None,
        help="Optional comma-separated pair names to visualize, e.g. L4-L5,L5-S1.",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_cv_dir_from_csv(csv_path: Path) -> Path:
    # .../<exp>/test_analysis/file.csv -> .../<exp>
    if csv_path.parent.name == "test_analysis":
        return csv_path.parent.parent
    return csv_path.parent.parent


def infer_gradcam_dir(csv_path: Path, mode: str) -> Path:
    if csv_path.parent.name == "test_analysis":
        return csv_path.parent.parent / f"gradcam_pair_crop_{mode}"
    return csv_path.parent / f"gradcam_pair_crop_{mode}"


# -------------------------
# Model / checkpoint
# -------------------------
def build_model(cfg: dict, device: str):
    model = LumbarCropModel(
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        loss_name=cfg.get("loss_name", "bce"),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.0),
    )
    lumbar_ckpt = cfg.get("lumbar_pretrained_ckpt", "")
    if lumbar_ckpt and hasattr(model, "load_backbone_checkpoint"):
        try:
            model.load_backbone_checkpoint(lumbar_ckpt)
        except Exception as e:
            print(f"[WARN] Failed to load lumbar_pretrained_ckpt='{lumbar_ckpt}': {e}")
    return model.to(device)


def load_checkpoint_state_dict(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        extra = ckpt.get("extra", {})
    else:
        state = ckpt
        extra = {}
    state = {k.replace("module.", ""): v for k, v in state.items()}
    return state, extra


# -------------------------
# Preprocessing: match PairCropDataset test path
# -------------------------
class ApplyCLAHE:
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"ApplyCLAHE expects PIL.Image, got {type(img)}")
        gray = np.array(img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        gray_clahe = clahe.apply(gray)
        rgb = np.stack([gray_clahe, gray_clahe, gray_clahe], axis=-1)
        return Image.fromarray(rgb)


class PercentileNormalize:
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


def get_transform(aug_cfg: Optional[dict], split: str = "test"):
    aug_cfg = aug_cfg or {}
    raw_image_size = aug_cfg.get("image_size", [512, 256])
    if isinstance(raw_image_size, (list, tuple)):
        if len(raw_image_size) != 2:
            raise ValueError(f"augmentation.image_size must be int or length-2, got {raw_image_size}")
        image_size = (int(raw_image_size[0]), int(raw_image_size[1]))
    else:
        image_size = int(raw_image_size)

    use_imagenet_norm = bool(aug_cfg.get("use_imagenet_norm", False))
    if use_imagenet_norm:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = aug_cfg.get("mean", [0.479, 0.479, 0.479])
        std = aug_cfg.get("std", [0.167, 0.167, 0.167])

    intensity_mode = str(aug_cfg.get("intensity_mode", "none")).lower()
    deterministic_tfms = []
    if intensity_mode in ["percentile", "percentile_clahe"]:
        deterministic_tfms.append(
            PercentileNormalize(
                p_low=float(aug_cfg.get("percentile_low", 1.0)),
                p_high=float(aug_cfg.get("percentile_high", 99.0)),
            )
        )
    if intensity_mode in ["clahe", "percentile_clahe"]:
        deterministic_tfms.append(
            ApplyCLAHE(
                clip_limit=float(aug_cfg.get("clahe_clip_limit", 2.0)),
                tile_grid_size=tuple(aug_cfg.get("clahe_tile_grid_size", [8, 8])),
            )
        )
    deterministic_tfms.append(transforms.Resize(image_size))

    tfm = list(deterministic_tfms)
    tfm.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(tfm), deterministic_tfms


def find_pair_image_filename(image_dir: Path, image_id: str, pair_name: str) -> Optional[str]:
    candidates = [
        f"{image_id}_{pair_name}.png",
        f"{image_id}_{pair_name.replace('-', '_')}.png",
    ]
    for fname in candidates:
        if (image_dir / fname).is_file():
            return fname
    # Conservative fallback: pair name should still be in the stem.
    matches = sorted(image_dir.glob(f"{image_id}*{pair_name}*.png"))
    if matches:
        return matches[0].name
    return None


def preprocess_pair_crop(
    image_id: str,
    pair_name: str,
    image_dir: Path,
    transform,
    deterministic_tfms: List,
) -> Tuple[torch.Tensor, np.ndarray, str]:
    fname = find_pair_image_filename(image_dir, image_id, pair_name)
    if fname is None:
        raise FileNotFoundError(f"Cannot find pair crop for image_id={image_id}, pair_name={pair_name} under {image_dir}")

    img_path = image_dir / fname
    img = Image.open(img_path).convert("RGB")

    # Visualization image: deterministic preprocessing + resize, before ToTensor/Normalize.
    vis_pil = img
    for t in deterministic_tfms:
        vis_pil = t(vis_pil)
    vis_rgb = np.array(vis_pil.convert("RGB")).astype(np.uint8)

    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, vis_rgb, fname


# -------------------------
# Grad-CAM
# -------------------------
def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    cur = model
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def resolve_target_layer(model: nn.Module, target_layer_path: Optional[str] = None) -> Tuple[nn.Module, str]:
    if target_layer_path:
        return get_module_by_path(model, target_layer_path), target_layer_path

    for p in ["backbone.layer4.2.conv3", "backbone.layer4.2", "layer4.2.conv3", "layer4.2"]:
        try:
            return get_module_by_path(model, p), p
        except Exception:
            pass

    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        layer = model.backbone.layer4[-1]
        if hasattr(layer, "conv3"):
            return layer.conv3, "backbone.layer4[-1].conv3"
        return layer, "backbone.layer4[-1]"

    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last_name, last_mod = name, mod
    if last_mod is None:
        raise RuntimeError("Could not resolve a Conv2d target Grad-CAM layer.")
    return last_mod, last_name


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, target_score: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        target_score.backward(retain_graph=False)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations
        gradients = self.gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()

        if np.allclose(cam.max(), cam.min()):
            cam = np.zeros_like(cam, dtype=np.float32)
        else:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)


# -------------------------
# Visualization helpers
# -------------------------
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def overlay_cam_on_image(vis_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.40) -> Tuple[np.ndarray, np.ndarray]:
    h, w = vis_rgb.shape[:2]
    cam_uint8 = np.uint8(np.clip(cam, 0.0, 1.0) * 255.0)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    if heatmap_rgb.shape[:2] != (h, w):
        heatmap_rgb = cv2.resize(heatmap_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = np.clip(vis_rgb.astype(np.float32) * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha, 0, 255)
    return heatmap_rgb.astype(np.uint8), overlay.astype(np.uint8)


def draw_text_block(lines: List[str], width: int, pad: int = 8) -> Image.Image:
    font = ImageFont.load_default()
    line_h = 14
    height = pad * 2 + line_h * len(lines)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = pad
    for line in lines:
        draw.text((pad, y), line[:240], fill=(0, 0, 0), font=font)
        y += line_h
    return img


def add_label_below(img: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    label_h = 18
    out = Image.new("RGB", (img.width, img.height + label_h), (255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((6, img.height + 2), label, fill=(0, 0, 0), font=font)
    return out


def compose_pair_row(vis_rgb: np.ndarray, heatmap_rgb: np.ndarray, overlay_rgb: np.ndarray, pair_info: Dict) -> Image.Image:
    orig = add_label_below(Image.fromarray(vis_rgb), "Input pair crop")
    heat = add_label_below(Image.fromarray(heatmap_rgb), "Grad-CAM")
    over = add_label_below(Image.fromarray(overlay_rgb), "Overlay")

    spacer = 10
    row_w = orig.width + heat.width + over.width + spacer * 2
    row_h = max(orig.height, heat.height, over.height)
    image_row = Image.new("RGB", (row_w, row_h), (255, 255, 255))
    image_row.paste(orig, (0, 0))
    image_row.paste(heat, (orig.width + spacer, 0))
    image_row.paste(over, (orig.width + spacer + heat.width + spacer, 0))

    info_lines = [
        f"pair={pair_info.get('pair_name','')} | pair_true={pair_info.get('pair_y_true','')} | pair_pred={pair_info.get('pair_pred','')} | pair_prob={pair_info.get('pair_prob', np.nan):.4f} | pair_conf={pair_info.get('pair_confusion_type','')}",
        f"image_file={pair_info.get('image_fname','')} | target={pair_info.get('cam_target','')} | layer={pair_info.get('target_layer','')} | folds={pair_info.get('used_folds','')}",
    ]
    info = draw_text_block(info_lines, row_w)
    out = Image.new("RGB", (row_w, info.height + image_row.height), (255, 255, 255))
    out.paste(info, (0, 0))
    out.paste(image_row, (0, info.height))
    return out


def compose_case_panel(pair_rows: List[Image.Image], case_info: Dict) -> Image.Image:
    if not pair_rows:
        raise ValueError("No pair rows to compose.")
    width = max(r.width for r in pair_rows)
    header_lines = [
        f"image_id={case_info.get('image_id','')} | case_conf={case_info.get('confusion_type','')} | y_true={case_info.get('y_true','')} | y_pred={case_info.get('y_pred','')} | y_prob={case_info.get('y_prob', np.nan):.4f}",
        f"top_pair={case_info.get('top_pair_name','')} | top_pair_prob={case_info.get('top_pair_prob', np.nan):.4f} | location={case_info.get('location','')} | grade={case_info.get('grade','')} | implant={case_info.get('implant','')}",
    ]
    header = draw_text_block(header_lines, width)
    total_h = header.height + sum(r.height for r in pair_rows) + 8 * (len(pair_rows) - 1)
    canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
    y = 0
    canvas.paste(header, (0, y))
    y += header.height
    for row in pair_rows:
        canvas.paste(row, (0, y))
        y += row.height + 8
    return canvas


def choose_target_score(outputs: Dict, cam_target: str, pair_pred: int) -> torch.Tensor:
    logits = outputs["study_logits"]
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    elif logits.ndim != 1:
        logits = logits.view(-1)
    pos_score = logits[0]
    if cam_target == "positive":
        return pos_score
    return pos_score if int(pair_pred) == 1 else -pos_score


def normalize_cam_mean(cams: List[np.ndarray]) -> np.ndarray:
    cam = np.mean(np.stack(cams, axis=0), axis=0).astype(np.float32)
    if np.allclose(cam.max(), cam.min()):
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).astype(np.float32)


# -------------------------
# Metadata helpers
# -------------------------
def get_pair_info_from_case_row(case_row: pd.Series, pair_name: str, pair_df: Optional[pd.DataFrame]) -> Dict:
    image_id = str(case_row["image_id"])
    info = {
        "image_id": image_id,
        "pair_name": pair_name,
        "pair_prob": np.nan,
        "pair_pred": -1,
        "pair_y_true": -1,
        "pair_confusion_type": "",
    }

    # Prefer full pair prediction CSV because it has exact pair-level rows.
    if pair_df is not None:
        sub = pair_df[(pair_df["image_id"].astype(str) == image_id) & (pair_df["pair_name"].astype(str) == pair_name)]
        if len(sub) > 0:
            r = sub.iloc[0]
            info.update({
                "pair_prob": float(r.get("pair_prob", np.nan)),
                "pair_pred": int(r.get("pair_pred", -1)),
                "pair_y_true": int(r.get("pair_y_true", -1)),
                "pair_confusion_type": safe_str(r.get("pair_confusion_type", "")),
            })
            return info

    # Fallback: use columns from case prediction CSV.
    prob_col = f"pair_prob_{pair_name}"
    pred_col = f"pair_pred_{pair_name}"
    true_col = f"pair_true_{pair_name}"
    if prob_col in case_row:
        info["pair_prob"] = float(case_row.get(prob_col, np.nan))
    if pred_col in case_row:
        info["pair_pred"] = int(case_row.get(pred_col, -1))
    if true_col in case_row:
        info["pair_y_true"] = int(case_row.get(true_col, -1))
    if info["pair_y_true"] in (0, 1) and info["pair_pred"] in (0, 1):
        if info["pair_y_true"] == 0 and info["pair_pred"] == 0:
            info["pair_confusion_type"] = "TN"
        elif info["pair_y_true"] == 0 and info["pair_pred"] == 1:
            info["pair_confusion_type"] = "FP"
        elif info["pair_y_true"] == 1 and info["pair_pred"] == 0:
            info["pair_confusion_type"] = "FN"
        else:
            info["pair_confusion_type"] = "TP"
    return info


def enrich_case_metadata(wrong_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = wrong_df.copy()
    test_csv = Path(cfg["test_csv"])
    if test_csv.is_file():
        raw = pd.read_csv(test_csv)
        if "image_id" in raw.columns:
            raw["image_id"] = raw["image_id"].astype(str)
            out["image_id"] = out["image_id"].astype(str)
            add_cols = [c for c in ["image_id", "location", "grade", "implant", "text"] if c in raw.columns]
            for c in add_cols:
                if c != "image_id" and c in out.columns:
                    add_cols.remove(c)
            if len(add_cols) > 1:
                out = out.merge(raw[add_cols], on="image_id", how="left")
    return out


# -------------------------
# Core visualization
# -------------------------
def load_models(cfg: dict, cv_dir: Path, n_folds: int, ckpt_name: str, device: str, target_layer_arg: Optional[str], temperature_arg: Optional[float]):
    loaded = []
    for fold_idx in range(1, n_folds + 1):
        ckpt_path = cv_dir / f"fold_{fold_idx}" / ckpt_name
        cal_path = cv_dir / f"fold_{fold_idx}" / "calibration_summary.json"
        if not ckpt_path.is_file():
            print(f"[WARN] Skip fold {fold_idx}: checkpoint not found: {ckpt_path}")
            continue

        model = build_model(cfg, device)
        state, extra = load_checkpoint_state_dict(str(ckpt_path), device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Fold {fold_idx} missing keys: {missing[:10]}")
        if unexpected:
            print(f"[WARN] Fold {fold_idx} unexpected keys: {unexpected[:10]}")
        model.eval()

        cal = {}
        if cal_path.is_file():
            with open(cal_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
        temperature = float(temperature_arg if temperature_arg is not None else cal.get("temperature", extra.get("temperature", 1.0)))
        temperature = max(temperature, 0.05)

        target_layer, target_layer_name = resolve_target_layer(model, target_layer_arg)
        cam_helper = GradCAM(model, target_layer)
        loaded.append({
            "fold_idx": fold_idx,
            "model": model,
            "cam": cam_helper,
            "temperature": temperature,
            "target_layer_name": target_layer_name,
        })
        print(f"[INFO] Loaded fold {fold_idx}: {ckpt_path} | temperature={temperature:.6f} | target_layer={target_layer_name}")

    if not loaded:
        raise RuntimeError("No checkpoints were loaded successfully.")
    return loaded


def generate_pair_cam(
    image_id: str,
    pair_name: str,
    pair_info: Dict,
    image_dir: Path,
    transform,
    deterministic_tfms: List,
    loaded: List[Dict],
    device: str,
    cam_target: str,
    alpha: float,
) -> Tuple[Image.Image, Dict, Optional[np.ndarray]]:
    input_tensor, vis_rgb, image_fname = preprocess_pair_crop(
        image_id=image_id,
        pair_name=pair_name,
        image_dir=image_dir,
        transform=transform,
        deterministic_tfms=deterministic_tfms,
    )
    input_tensor = input_tensor.to(device)

    cams = []
    fold_probs = []
    used_folds = []
    with torch.enable_grad():
        for item in loaded:
            model = item["model"]
            cam_helper = item["cam"]
            temp = item["temperature"]
            outputs = model(input_tensor)
            logits = outputs["study_logits"]
            scalar_logit = logits.view(-1)[0]
            prob = float(sigmoid_np(np.array([scalar_logit.detach().cpu().item() / temp]))[0])
            fold_probs.append(prob)

            score = choose_target_score(outputs, cam_target=cam_target, pair_pred=int(pair_info.get("pair_pred", 0)))
            cam = cam_helper.generate(input_tensor, score)
            cams.append(cam)
            used_folds.append(item["fold_idx"])

    cam_mean = normalize_cam_mean(cams)
    heatmap_rgb, overlay_rgb = overlay_cam_on_image(vis_rgb, cam_mean, alpha=alpha)
    pair_info = dict(pair_info)
    pair_info.update({
        "image_fname": image_fname,
        "used_folds": ",".join(map(str, used_folds)),
        "cam_target": cam_target,
        "target_layer": loaded[0]["target_layer_name"],
        "fold_prob_mean_for_this_pair": float(np.mean(fold_probs)) if fold_probs else np.nan,
    })
    row_img = compose_pair_row(vis_rgb, heatmap_rgb, overlay_rgb, pair_info)
    return row_img, pair_info, cam_mean


def main():
    args = parse_args()
    wrong_cases_csv = Path(args.wrong_cases_csv).resolve()
    cfg = load_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cv_dir = Path(args.cv_dir).resolve() if args.cv_dir else infer_cv_dir_from_csv(wrong_cases_csv)
    outdir = infer_gradcam_dir(wrong_cases_csv, args.mode)
    outdir.mkdir(parents=True, exist_ok=True)

    threshold = float(args.threshold if args.threshold is not None else cfg.get("threshold", 0.5))
    image_dir_value = cfg.get("test_image_dir", cfg.get("image_dir"))
    if image_dir_value is None:
        raise ValueError("Missing test_image_dir/image_dir in config.")
    image_dir = Path(image_dir_value).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Pair crop test image dir not found: {image_dir}")

    wrong_df = pd.read_csv(wrong_cases_csv)
    wrong_df["image_id"] = wrong_df["image_id"].astype(str)
    wrong_df = enrich_case_metadata(wrong_df, cfg)
    if args.start_idx > 0:
        wrong_df = wrong_df.iloc[args.start_idx:].reset_index(drop=True)
    if args.max_cases is not None:
        wrong_df = wrong_df.iloc[:args.max_cases].reset_index(drop=True)
    if len(wrong_df) == 0:
        raise RuntimeError("No rows to visualize after start_idx/max_cases.")

    pair_df = None
    if args.pair_predictions_csv:
        pair_df = pd.read_csv(args.pair_predictions_csv)
        pair_df["image_id"] = pair_df["image_id"].astype(str)

    if args.only_pairs:
        requested_pairs = [x.strip() for x in args.only_pairs.split(",") if x.strip()]
        bad = [p for p in requested_pairs if p not in PAIR_NAMES]
        if bad:
            raise ValueError(f"Invalid pair names in --only_pairs: {bad}. Valid={PAIR_NAMES}")
        pair_names_to_use = requested_pairs
    else:
        pair_names_to_use = list(PAIR_NAMES)

    transform, deterministic_tfms = get_transform(cfg.get("augmentation", {}), split="test")
    loaded = load_models(
        cfg=cfg,
        cv_dir=cv_dir,
        n_folds=args.n_folds,
        ckpt_name=args.ckpt_name,
        device=device,
        target_layer_arg=args.target_layer,
        temperature_arg=args.temperature,
    )

    manifest_rows = []

    for idx, row in wrong_df.iterrows():
        image_id = str(row["image_id"])

        if args.mode == "pair":
            pair_name = str(row.get("pair_name", ""))
            if pair_name not in PAIR_NAMES:
                print(f"[WARN] Skip row {idx}: invalid/missing pair_name={pair_name}")
                continue
            pair_names_for_this_row = [pair_name]
            case_conf = safe_str(row.get("pair_confusion_type", row.get("confusion_type", "wrong_pair")))
        else:
            pair_names_for_this_row = pair_names_to_use
            case_conf = safe_str(row.get("confusion_type", "wrong_case"))

        safe_case_conf = case_conf.replace("/", "_").replace(" ", "_")
        out_png = outdir / f"{idx:04d}_{image_id}_{safe_case_conf}_pair_gradcam.png"
        if out_png.exists() and not args.overwrite:
            print(f"[INFO] Skip existing: {out_png}")
            manifest_rows.append({"image_id": image_id, "output_png": str(out_png), "status": "skipped_exists"})
            continue

        try:
            pair_row_imgs = []
            pair_infos = []
            gray_outputs = []

            for pair_name in pair_names_for_this_row:
                if args.mode == "pair":
                    pair_info = {
                        "image_id": image_id,
                        "pair_name": pair_name,
                        "pair_prob": float(row.get("pair_prob", np.nan)),
                        "pair_pred": int(row.get("pair_pred", int(float(row.get("pair_prob", 0.0)) >= threshold))),
                        "pair_y_true": int(row.get("pair_y_true", -1)),
                        "pair_confusion_type": safe_str(row.get("pair_confusion_type", "")),
                    }
                else:
                    pair_info = get_pair_info_from_case_row(row, pair_name, pair_df)
                    # If no pair_pred exists, derive it from pair_prob.
                    if int(pair_info.get("pair_pred", -1)) < 0 and not pd.isna(pair_info.get("pair_prob", np.nan)):
                        pair_info["pair_pred"] = int(float(pair_info["pair_prob"]) >= threshold)

                try:
                    row_img, info, cam_mean = generate_pair_cam(
                        image_id=image_id,
                        pair_name=pair_name,
                        pair_info=pair_info,
                        image_dir=image_dir,
                        transform=transform,
                        deterministic_tfms=deterministic_tfms,
                        loaded=loaded,
                        device=device,
                        cam_target=args.cam_target,
                        alpha=float(args.alpha),
                    )
                    pair_row_imgs.append(row_img)
                    pair_infos.append(info)

                    if args.save_grayscale_cam and cam_mean is not None:
                        out_gray = outdir / f"{idx:04d}_{image_id}_{pair_name}_{safe_case_conf}_cam_gray.png"
                        Image.fromarray(np.uint8(np.clip(cam_mean, 0.0, 1.0) * 255)).save(out_gray)
                        gray_outputs.append(str(out_gray))

                except FileNotFoundError as e:
                    print(f"[WARN] Missing pair crop: {e}")
                    manifest_rows.append({
                        "image_id": image_id,
                        "pair_name": pair_name,
                        "status": f"missing_pair_crop: {e}",
                    })

            if not pair_row_imgs:
                raise RuntimeError("No pair crop rows were generated for this case.")

            if args.mode == "pair":
                panel = pair_row_imgs[0]
            else:
                case_info = {
                    "image_id": image_id,
                    "confusion_type": safe_str(row.get("confusion_type", "")),
                    "y_true": int(row.get("y_true", -1)),
                    "y_pred": int(row.get("y_pred", -1)),
                    "y_prob": float(row.get("y_prob", np.nan)),
                    "top_pair_name": safe_str(row.get("top_pair_name", "")),
                    "top_pair_prob": float(row.get("top_pair_prob", np.nan)),
                    "location": safe_str(row.get("location", row.get("location_raw", ""))),
                    "grade": safe_str(row.get("grade", "")),
                    "implant": safe_str(row.get("implant", "")),
                }
                panel = compose_case_panel(pair_row_imgs, case_info)

            panel.save(out_png)

            manifest_rows.append({
                "image_id": image_id,
                "mode": args.mode,
                "case_confusion_type": safe_str(row.get("confusion_type", row.get("pair_confusion_type", ""))),
                "y_true": row.get("y_true", row.get("study_y_true", np.nan)),
                "y_pred": row.get("y_pred", np.nan),
                "y_prob": row.get("y_prob", np.nan),
                "top_pair_name": row.get("top_pair_name", ""),
                "visualized_pairs": ";".join([p.get("pair_name", "") for p in pair_infos]),
                "cam_target": args.cam_target,
                "target_layer": loaded[0]["target_layer_name"],
                "output_png": str(out_png),
                "output_gray_pngs": ";".join(gray_outputs),
                "status": "ok",
            })
            print(f"[OK] Saved Grad-CAM panel: {out_png}")

        except Exception as e:
            manifest_rows.append({
                "image_id": image_id,
                "mode": args.mode,
                "output_png": str(out_png),
                "status": f"error: {e}",
            })
            print(f"[ERROR] Failed image_id={image_id}: {e}")

    manifest_csv = outdir / "pair_crop_gradcam_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(f"[DONE] Saved manifest: {manifest_csv}")
    print(f"[DONE] Grad-CAM output dir: {outdir}")

    for item in loaded:
        try:
            item["cam"].remove()
        except Exception:
            pass


if __name__ == "__main__":
    main()
