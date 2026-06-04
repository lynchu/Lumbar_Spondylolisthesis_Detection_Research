#!/usr/bin/env python3
"""
Visualize Grad-CAM for wrong test cases from lumbar-crop classification.

Default behavior:
- Input: wrong-case CSV from test_analysis, e.g.
  presence/experiments/.../test_analysis/basic_cv_ensemble_wrong_cases.csv
- Output dir: sibling directory named `gradcam` at the same level as `test_analysis`
  e.g. presence/experiments/.../gradcam/
- Loads fold checkpoints from the sibling CV directory and averages Grad-CAM maps
  across folds, so the visualization aligns better with ensemble-based wrong cases.
- Targets the last convolutional layer of ResNet50 (auto-detected; usually
  model.backbone.layer4[-1].conv3).

Notes:
- If your model uses lumbar masks, this script applies the same mask-based crop/filter
  preprocessing as the dataset class before feeding the image into the model.
- This script saves one PNG per wrong case and also writes gradcam_manifest.csv.
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
try:
    from models.basic_roi import LumbarCropModelWithMask
except Exception:
    LumbarCropModelWithMask = None


# -------------------------
# Utility: config / model
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrong_cases_csv", type=str, required=True,
                        help="Path to basic_cv_ensemble_wrong_cases.csv")
    parser.add_argument("--config", type=str, required=True,
                        help="Training/analyze YAML config used for the experiment")
    parser.add_argument("--cv_dir", type=str, default=None,
                        help="CV experiment root containing fold_1/, fold_2/, ...; if omitted, infer from wrong_cases_csv")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu; default auto")
    parser.add_argument("--cam_target", type=str, default="pred", choices=["pred", "positive"],
                        help="Grad-CAM target: predicted class score or positive-class score")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="If set, only visualize the first N wrong cases")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index inside wrong_cases_csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target_layer", type=str, default=None,
                        help="Optional dotted module path, e.g. backbone.layer4.2.conv3")
    parser.add_argument("--save_grayscale_cam", action="store_true",
                        help="Also save raw grayscale Grad-CAM map as a PNG")
    parser.add_argument("--alpha", type=float, default=0.40,
                        help="Overlay alpha for heatmap")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: str):
    # In the current lumbar-crop pipeline, using a mask often means mask-guided
    # preprocessing on the RGB image, not necessarily a 4-channel model.
    use_four_channel = bool(
        cfg.get("use_pair_roi", False)
        or cfg.get("use_four_channel_input", False)
        or cfg.get("model_uses_mask_channel", False)
    )

    if use_four_channel:
        if LumbarCropModelWithMask is None:
            raise ImportError("LumbarCropModelWithMask is unavailable, but config requests a 4-channel model.")
        model = LumbarCropModelWithMask(
            backbone_pretrained=cfg.get("imagenet_pretrained", False),
            loss_name=cfg.get("loss_name", "bce"),
            hidden_dim=cfg.get("hidden_dim", 64),
            dropout=cfg.get("dropout", 0.0),
        )
    else:
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
            print(f"[WARN] Failed to load lumbar_pretrained_ckpt '{lumbar_ckpt}': {e}")

    return model.to(device)


def load_checkpoint_state_dict(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return cleaned


# -------------------------
# Utility: preprocessing
# -------------------------
def get_transform(aug_cfg: Optional[dict], split: str = "test"):
    aug_cfg = aug_cfg or {}
    raw_image_size = aug_cfg.get("image_size", 224)
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
        mean = [0.3928, 0.3928, 0.3928]
        std = [0.2332, 0.2332, 0.2332]

    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm


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


def find_image_filename(image_dir: Path, image_id: str) -> Optional[str]:
    candidates = [
        f"{image_id}_lumbar.png",
        f"{image_id}.png",
    ]
    for fname in candidates:
        if (image_dir / fname).is_file():
            return fname
    # fallback: any png starting with image_id
    for p in image_dir.glob(f"{image_id}*.png"):
        return p.name
    return None


def preprocess_case(
    image_id: str,
    image_dir: Path,
    transform,
    use_lumbar_mask: bool = False,
    mask_dir: Optional[Path] = None,
) -> Tuple[torch.Tensor, np.ndarray, str]:
    """
    Returns:
      input_tensor: normalized tensor [1, 3, H, W]
      vis_rgb: uint8 RGB image after mask-processing / crop, before normalization, resized to network input size
      image_fname: actual filename used
    """
    fname = find_image_filename(image_dir, image_id)
    if fname is None:
        raise FileNotFoundError(f"Cannot find image for image_id={image_id} under {image_dir}")

    img_path = image_dir / fname
    org_img = cv2.imread(str(img_path))
    if org_img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    if use_lumbar_mask:
        if mask_dir is None:
            raise ValueError("use_lumbar_mask=True but mask_dir is None")

        mask_path_png = mask_dir / fname
        mask_path_txt = mask_dir / fname.replace(".png", ".txt")

        if mask_path_png.is_file():
            mask = cv2.imread(str(mask_path_png), cv2.IMREAD_GRAYSCALE)
        elif mask_path_txt.is_file():
            mask = np.loadtxt(str(mask_path_txt), dtype=np.uint8)
        else:
            raise FileNotFoundError(
                f"Neither mask PNG nor TXT found for {fname} in mask_dir={mask_dir}"
            )

        if mask is None:
            raise ValueError(f"Failed to read mask for {fname}")

        mask = mask.astype(np.uint8)
        mask[mask != 255] = 0
        mask[mask == 255] = 1

        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        average_size = np.mean(stats[1:, cv2.CC_STAT_AREA]) if stats.shape[0] > 1 else 0

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

    # Visualization copy after resize but before normalization.
    resize_only = None
    for t in transform.transforms:
        if isinstance(t, transforms.Resize):
            resize_only = t
            break
    if resize_only is not None:
        vis_pil = resize_only(img_pil)
    else:
        vis_pil = img_pil
    vis_rgb = np.array(vis_pil).astype(np.uint8)

    input_tensor = transform(img_pil)
    if input_tensor.ndim == 3 and input_tensor.shape[0] == 1:
        input_tensor = torch.cat((input_tensor, input_tensor, input_tensor), dim=0)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor, vis_rgb, fname


# -------------------------
# Utility: Grad-CAM
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

    candidate_paths = [
        "backbone.layer4.2.conv3",
        "backbone.layer4.2",
        "backbone.layer4[-1].conv3",  # handled below if the direct dotted access fails
        "backbone.layer4[-1]",
        "layer4.2.conv3",
        "layer4.2",
    ]

    # direct known good paths first
    direct_candidates = [
        "backbone.layer4.2.conv3",
        "backbone.layer4.2",
        "layer4.2.conv3",
        "layer4.2",
    ]
    for p in direct_candidates:
        try:
            mod = get_module_by_path(model, p)
            return mod, p
        except Exception:
            pass

    # layer4[-1] style fallback
    try:
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            layer = model.backbone.layer4[-1]
            if hasattr(layer, "conv3"):
                return layer.conv3, "backbone.layer4[-1].conv3"
            return layer, "backbone.layer4[-1]"
    except Exception:
        pass

    try:
        if hasattr(model, "layer4"):
            layer = model.layer4[-1]
            if hasattr(layer, "conv3"):
                return layer.conv3, "layer4[-1].conv3"
            return layer, "layer4[-1]"
    except Exception:
        pass

    # final fallback: last Conv2d in the whole model
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last_name, last_mod = name, mod
    if last_mod is None:
        raise RuntimeError("Could not resolve a target Grad-CAM layer.")
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
# Utility: visualization
# -------------------------
def overlay_cam_on_image(vis_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.40) -> Tuple[np.ndarray, np.ndarray]:
    h, w = vis_rgb.shape[:2]
    cam_uint8 = np.uint8(np.clip(cam, 0.0, 1.0) * 255.0)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    if heatmap_rgb.shape[:2] != (h, w):
        heatmap_rgb = cv2.resize(heatmap_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    overlay = np.clip(vis_rgb.astype(np.float32) * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha, 0, 255)
    return heatmap_rgb.astype(np.uint8), overlay.astype(np.uint8)


def draw_header(text_lines: List[str], width: int, pad: int = 8) -> Image.Image:
    font = ImageFont.load_default()
    line_h = 14
    height = pad * 2 + line_h * len(text_lines)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = pad
    for line in text_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    return img


def add_label_below(img: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    pad = 6
    label_h = 18
    canvas = Image.new("RGB", (img.width, img.height + label_h), (255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, img.height + 2), label, fill=(0, 0, 0), font=font)
    return canvas


def compose_panel(vis_rgb: np.ndarray, heatmap_rgb: np.ndarray, overlay_rgb: np.ndarray, info: Dict) -> Image.Image:
    orig = Image.fromarray(vis_rgb)
    heat = Image.fromarray(heatmap_rgb)
    over = Image.fromarray(overlay_rgb)

    orig = add_label_below(orig, "Input to model")
    heat = add_label_below(heat, "Grad-CAM heatmap")
    over = add_label_below(over, "Overlay")

    spacer = 10
    row_w = orig.width + heat.width + over.width + spacer * 2
    row_h = max(orig.height, heat.height, over.height)
    row = Image.new("RGB", (row_w, row_h), (255, 255, 255))
    row.paste(orig, (0, 0))
    row.paste(heat, (orig.width + spacer, 0))
    row.paste(over, (orig.width + spacer + heat.width + spacer, 0))

    text_lines = [
        f"image_id={info.get('image_id', '')} | confusion={info.get('confusion_type', '')} | y_true={info.get('y_true', '')} | y_pred={info.get('y_pred', '')} | y_prob={info.get('y_prob', np.nan):.4f}",
        f"location={info.get('location', '')} | grade={info.get('grade', '')} | implant={info.get('implant', '')}",
        f"target_layer={info.get('target_layer_name', '')} | cam_target={info.get('cam_target', '')} | used_folds={info.get('used_folds', '')}",
    ]
    header = draw_header(text_lines, row_w)

    canvas = Image.new("RGB", (row_w, header.height + row.height), (255, 255, 255))
    canvas.paste(header, (0, 0))
    canvas.paste(row, (0, header.height))
    return canvas


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


# -------------------------
# Core logic
# -------------------------
def choose_target_score(outputs: Dict, loss_name: str, cam_target: str, pred_class: int) -> torch.Tensor:
    study_logits = outputs["study_logits"]
    loss_name = loss_name.lower()

    if loss_name == "bce":
        if study_logits.ndim == 2 and study_logits.shape[1] == 1:
            study_logits = study_logits[:, 0]
        elif study_logits.ndim != 1:
            study_logits = study_logits.view(-1)

        pos_score = study_logits[0]
        if cam_target == "positive":
            return pos_score
        # predicted class target
        return pos_score if int(pred_class) == 1 else -pos_score

    if loss_name == "ce":
        if study_logits.ndim != 2 or study_logits.shape[1] < 2:
            raise ValueError(f"Expected CE logits of shape [B, 2], got {tuple(study_logits.shape)}")
        if cam_target == "positive":
            return study_logits[0, 1]
        return study_logits[0, int(pred_class)]

    raise ValueError(f"Unsupported loss_name: {loss_name}")


def infer_cv_dir_from_wrong_cases_csv(wrong_cases_csv: Path) -> Path:
    # .../<exp>/test_analysis/basic_cv_ensemble_wrong_cases.csv -> .../<exp>
    parent = wrong_cases_csv.parent
    if parent.name == "test_analysis":
        return parent.parent
    return parent.parent


def build_case_info(row: pd.Series, target_layer_name: str, cam_target: str, used_folds: List[int]) -> Dict:
    return {
        "image_id": safe_str(row.get("image_id", "")),
        "confusion_type": safe_str(row.get("confusion_type", "")),
        "y_true": int(row.get("y_true", -1)),
        "y_pred": int(row.get("y_pred", -1)),
        "y_prob": float(row.get("y_prob", np.nan)),
        "location": safe_str(row.get("location", "")),
        "grade": safe_str(row.get("grade", "")),
        "implant": safe_str(row.get("implant", "")),
        "target_layer_name": target_layer_name,
        "cam_target": cam_target,
        "used_folds": ",".join(map(str, used_folds)),
    }


def main():
    args = parse_args()
    wrong_cases_csv = Path(args.wrong_cases_csv).resolve()
    cfg = load_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cv_dir = Path(args.cv_dir).resolve() if args.cv_dir else infer_cv_dir_from_wrong_cases_csv(wrong_cases_csv)

    # Output dir: sibling of test_analysis named gradcam
    if wrong_cases_csv.parent.name == "test_analysis":
        gradcam_dir = wrong_cases_csv.parent.parent / "gradcam"
    else:
        gradcam_dir = wrong_cases_csv.parent / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    wrong_df = pd.read_csv(wrong_cases_csv)
    if args.start_idx > 0:
        wrong_df = wrong_df.iloc[args.start_idx:].reset_index(drop=True)
    if args.max_cases is not None:
        wrong_df = wrong_df.iloc[:args.max_cases].reset_index(drop=True)
    if len(wrong_df) == 0:
        raise RuntimeError("No wrong cases to visualize after applying start_idx/max_cases.")

    # Merge raw test CSV if some metadata columns are missing.
    test_csv = Path(cfg["test_csv"]).resolve()
    if test_csv.is_file():
        raw_test_df = pd.read_csv(test_csv)
        merge_cols = [c for c in ["image_id", "location", "grade", "implant", "text"] if c in raw_test_df.columns]
        if "image_id" in merge_cols:
            for col in ["location", "grade", "implant", "text"]:
                if col not in wrong_df.columns and col in raw_test_df.columns:
                    wrong_df = wrong_df.merge(raw_test_df[["image_id", col]], on="image_id", how="left")

    loss_name = str(cfg.get("loss_name", "bce")).lower()
    test_image_dir = Path(cfg.get("test_image_dir", cfg.get("image_dir"))).resolve()
    if not test_image_dir.is_dir():
        raise FileNotFoundError(f"test_image_dir not found: {test_image_dir}")

    use_lumbar_mask = bool(cfg.get("use_lumbar_mask", cfg.get("use_mask_input", False)))
    test_mask_dir = cfg.get("test_mask_dir", None)
    test_mask_dir = Path(test_mask_dir).resolve() if (use_lumbar_mask and test_mask_dir is not None) else None
    if use_lumbar_mask and (test_mask_dir is None or not test_mask_dir.is_dir()):
        raise FileNotFoundError(f"use_lumbar_mask=True but test_mask_dir is invalid: {test_mask_dir}")

    transform = get_transform(cfg.get("augmentation", {}), split="test")

    # Load fold models once.
    loaded = []
    target_layer_name_ref = None
    for fold_idx in range(1, args.n_folds + 1):
        ckpt_path = cv_dir / f"fold_{fold_idx}" / args.ckpt_name
        if not ckpt_path.is_file():
            print(f"[WARN] Skip fold {fold_idx}: checkpoint not found: {ckpt_path}")
            continue

        model = build_model(cfg, device)
        state_dict = load_checkpoint_state_dict(str(ckpt_path), device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Fold {fold_idx} missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Fold {fold_idx} unexpected keys: {unexpected}")
        model.eval()

        target_layer, target_layer_name = resolve_target_layer(model, args.target_layer)
        if target_layer_name_ref is None:
            target_layer_name_ref = target_layer_name
        cam_helper = GradCAM(model, target_layer)

        loaded.append({
            "fold_idx": fold_idx,
            "model": model,
            "cam": cam_helper,
            "target_layer_name": target_layer_name,
        })
        print(f"[INFO] Loaded fold {fold_idx}: {ckpt_path} | target_layer={target_layer_name}")

    if len(loaded) == 0:
        raise RuntimeError("No checkpoints were loaded successfully.")

    manifest_rows = []

    for i, row in wrong_df.iterrows():
        image_id = str(row["image_id"])
        y_pred = int(row.get("y_pred", 0))

        out_png = gradcam_dir / f"{i:04d}_{image_id}_{safe_str(row.get('confusion_type', 'wrong'))}_gradcam.png"
        out_gray = gradcam_dir / f"{i:04d}_{image_id}_{safe_str(row.get('confusion_type', 'wrong'))}_cam_gray.png"
        if out_png.exists() and (not args.overwrite):
            print(f"[INFO] Skip existing: {out_png}")
            manifest_rows.append({
                "image_id": image_id,
                "confusion_type": safe_str(row.get("confusion_type", "")),
                "output_png": str(out_png),
                "output_gray_png": str(out_gray) if args.save_grayscale_cam else "",
                "status": "skipped_exists",
            })
            continue

        try:
            input_tensor, vis_rgb, image_fname = preprocess_case(
                image_id=image_id,
                image_dir=test_image_dir,
                transform=transform,
                use_lumbar_mask=use_lumbar_mask,
                mask_dir=test_mask_dir,
            )
            input_tensor = input_tensor.to(device)

            cams = []
            used_folds = []
            with torch.enable_grad():
                for item in loaded:
                    model = item["model"]
                    cam_helper = item["cam"]

                    outputs = model(input_tensor)
                    score = choose_target_score(outputs, loss_name=loss_name, cam_target=args.cam_target, pred_class=y_pred)
                    cam = cam_helper.generate(input_tensor, score)
                    cams.append(cam)
                    used_folds.append(item["fold_idx"])

            cam_mean = np.mean(np.stack(cams, axis=0), axis=0).astype(np.float32)
            if np.allclose(cam_mean.max(), cam_mean.min()):
                cam_mean = np.zeros_like(cam_mean, dtype=np.float32)
            else:
                cam_mean = (cam_mean - cam_mean.min()) / (cam_mean.max() - cam_mean.min() + 1e-8)

            heatmap_rgb, overlay_rgb = overlay_cam_on_image(vis_rgb, cam_mean, alpha=float(args.alpha))
            info = build_case_info(row, target_layer_name_ref, args.cam_target, used_folds)
            panel = compose_panel(vis_rgb, heatmap_rgb, overlay_rgb, info)
            panel.save(out_png)

            if args.save_grayscale_cam:
                gray_img = Image.fromarray(np.uint8(np.clip(cam_mean, 0.0, 1.0) * 255.0))
                gray_img.save(out_gray)

            manifest_rows.append({
                "image_id": image_id,
                "image_fname": image_fname,
                "confusion_type": safe_str(row.get("confusion_type", "")),
                "y_true": int(row.get("y_true", -1)),
                "y_pred": int(row.get("y_pred", -1)),
                "y_prob": float(row.get("y_prob", np.nan)),
                "location": safe_str(row.get("location", "")),
                "grade": safe_str(row.get("grade", "")),
                "implant": safe_str(row.get("implant", "")),
                "target_layer": target_layer_name_ref,
                "cam_target": args.cam_target,
                "used_folds": ",".join(map(str, used_folds)),
                "output_png": str(out_png),
                "output_gray_png": str(out_gray) if args.save_grayscale_cam else "",
                "status": "ok",
            })
            print(f"[OK] Saved Grad-CAM: {out_png}")

        except Exception as e:
            manifest_rows.append({
                "image_id": image_id,
                "confusion_type": safe_str(row.get("confusion_type", "")),
                "output_png": str(out_png),
                "output_gray_png": str(out_gray) if args.save_grayscale_cam else "",
                "status": f"error: {e}",
            })
            print(f"[ERROR] Failed on image_id={image_id}: {e}")

    manifest_csv = gradcam_dir / "gradcam_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(f"[DONE] Saved manifest: {manifest_csv}")
    print(f"[DONE] Grad-CAM output dir: {gradcam_dir}")

    # Cleanup hooks.
    for item in loaded:
        try:
            item["cam"].remove()
        except Exception:
            pass


if __name__ == "__main__":
    main()
