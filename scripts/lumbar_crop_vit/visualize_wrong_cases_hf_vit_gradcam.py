#!/usr/bin/env python3
"""
Visualize Grad-CAM-style patch-token maps for wrong test cases from ViT lumbar-crop classifiers.

This is the ViT counterpart of the ResNet wrong-case Grad-CAM script.
It supports:
  1) Google Hugging Face ViT, e.g. google/vit-base-patch16-224
     pretrained_source: transformers_hf
  2) OpenAI CLIP ViT vision encoder, e.g. openai/clip-vit-base-patch16
     pretrained_source: transformers_clip_vision
  3) timm ViT / timm HF Hub ViT, as a fallback

Default behavior:
- Input: wrong-case CSV from test_analysis, e.g.
  presence/experiments/.../test_analysis/basic_cv_ensemble_wrong_cases.csv
- Output dir: sibling directory named `vit_gradcam` at the same level as `test_analysis`
  e.g. presence/experiments/.../vit_gradcam/
- Loads fold checkpoints from the sibling CV directory and averages Grad-CAM maps across folds.
- For transformer backbones, hooks a late LayerNorm module and computes a CAM over patch tokens.

Notes:
- This is not CNN Grad-CAM over convolution maps. For ViT, activations are token sequences
  shaped (B, N_tokens, hidden_dim). The script removes CLS when present, reshapes the
  patch tokens into the image patch grid, and upsamples to the model input size.
- For ViT-B/16 with image_size=[512, 256], patch grid = 32 x 16 = 512 patches.
- The script keeps the same preprocessing as your dataset as closely as possible:
  percentile normalization, CLAHE, resize, output channels, and normalization values.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# from models.basic_vit_hf import LumbarCropModel
from models.basic_vit_openai_clip import LumbarCropModel


# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrong_cases_csv", type=str, required=True,
                        help="Path to basic_cv_ensemble_wrong_cases.csv")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config used for the Google/OpenAI ViT experiment")
    parser.add_argument("--cv_dir", type=str, default=None,
                        help="CV experiment root containing fold_1/, fold_2/, ...; if omitted, infer from wrong_cases_csv")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Optional output directory. Default: sibling of test_analysis named vit_gradcam")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu; default auto")
    parser.add_argument("--cam_target", type=str, default="pred", choices=["pred", "positive"],
                        help="Grad-CAM target: predicted class score or positive-class score")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target_layer", type=str, default=None,
                        help=("Optional dotted module path. Examples: "
                              "backbone.vit.encoder.layer.11.layernorm_before, "
                              "backbone.vision.vision_model.encoder.layers.11.layer_norm1"))
    parser.add_argument("--save_grayscale_cam", action="store_true",
                        help="Also save the raw grayscale CAM map as PNG")
    parser.add_argument("--alpha", type=float, default=0.40)
    return parser.parse_args()


# -------------------------
# YAML / model helpers
# -------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_image_size(image_size):
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return image_size
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError(f"image_size must be int or [height, width], got: {image_size}")


def build_model(cfg: dict, device: str):
    aug_cfg = cfg.get("augmentation", {}) or {}
    model = LumbarCropModel(
        backbone_pretrained=cfg.get("imagenet_pretrained", False),
        loss_name=cfg.get("loss_name", "bce"),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.0),
        backbone_name=cfg.get("backbone_name", "resnet50"),
        image_size=aug_cfg.get("image_size", None),
        input_channels=int(aug_cfg.get("input_channels", 3)),
        pretrained_source=cfg.get("pretrained_source", "auto"),
        hf_hub_model_id=cfg.get("hf_hub_model_id", None),
        transformers_model_id=cfg.get("transformers_model_id", None),
    )
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
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


# -------------------------
# Preprocessing helpers matching dataset_lumbar_crop.py
# -------------------------
class ApplyCLAHE:
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)

    def __call__(self, img):
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


class ForceOutputChannels:
    def __init__(self, input_channels: int = 3):
        self.input_channels = int(input_channels)
        if self.input_channels not in (1, 3):
            raise ValueError(f"input_channels must be 1 or 3, got {self.input_channels}")

    def __call__(self, img):
        return img.convert("L") if self.input_channels == 1 else img.convert("RGB")


def get_eval_transform_and_vis_transform(aug_cfg: Optional[dict]):
    aug_cfg = aug_cfg or {}
    raw_image_size = aug_cfg.get("image_size", 224)
    if isinstance(raw_image_size, (list, tuple)):
        image_size = (int(raw_image_size[0]), int(raw_image_size[1]))
    else:
        image_size = int(raw_image_size)

    input_channels = int(aug_cfg.get("input_channels", 3))
    if input_channels not in (1, 3):
        raise ValueError(f"augmentation.input_channels must be 1 or 3, got {input_channels}")

    custom_mean = aug_cfg.get("norm_mean", None)
    custom_std = aug_cfg.get("norm_std", None)
    use_imagenet_norm = bool(aug_cfg.get("use_imagenet_norm", False))
    if custom_mean is not None or custom_std is not None:
        if custom_mean is None or custom_std is None:
            raise ValueError("augmentation.norm_mean and augmentation.norm_std must be set together.")
        mean = [float(x) for x in custom_mean]
        std = [float(x) for x in custom_std]
    elif input_channels == 1:
        if use_imagenet_norm:
            mean = [float(aug_cfg.get("grayscale_imagenet_mean", (0.485 + 0.456 + 0.406) / 3.0))]
            std = [float(aug_cfg.get("grayscale_imagenet_std", (0.229 + 0.224 + 0.225) / 3.0))]
        else:
            mean = [float(aug_cfg.get("grayscale_mean", 0.4564))]
            std = [float(aug_cfg.get("grayscale_std", 0.1944))]
    else:
        if use_imagenet_norm:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.4564, 0.4564, 0.4564]
            std = [0.1944, 0.1944, 0.1944]

    if len(mean) != input_channels or len(std) != input_channels:
        raise ValueError(
            f"norm_mean/norm_std length must match input_channels={input_channels}, "
            f"got len(mean)={len(mean)}, len(std)={len(std)}"
        )

    deterministic_tfms = []
    intensity_mode = str(aug_cfg.get("intensity_mode", "none")).lower()
    if intensity_mode in ["percentile", "percentile_clahe"]:
        deterministic_tfms.append(PercentileNormalize(
            p_low=float(aug_cfg.get("percentile_low", 1.0)),
            p_high=float(aug_cfg.get("percentile_high", 99.0)),
        ))
    if intensity_mode in ["clahe", "percentile_clahe"]:
        deterministic_tfms.append(ApplyCLAHE(
            clip_limit=float(aug_cfg.get("clahe_clip_limit", 2.0)),
            tile_grid_size=tuple(aug_cfg.get("clahe_tile_grid_size", [8, 8])),
        ))

    # The model input transform exactly follows deterministic validation/test preprocessing.
    model_tfms = list(deterministic_tfms)
    model_tfms.extend([
        transforms.Resize(image_size),
        ForceOutputChannels(input_channels=input_channels),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Visualization image should include deterministic intensity preprocessing + resize,
    # but not channel forcing/tensor normalization.
    vis_tfms = list(deterministic_tfms)
    vis_tfms.append(transforms.Resize(image_size))

    return transforms.Compose(model_tfms), transforms.Compose(vis_tfms), input_channels


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
        return int(x_min), int(y_min), int(x_max), int(y_max)
    return 0, 0, 0, 0


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
    for fname in [f"{image_id}_lumbar.png", f"{image_id}.png", f"{image_id}-Lat_lumbar.png"]:
        if (image_dir / fname).is_file():
            return fname
    for p in sorted(image_dir.glob(f"{image_id}*.png")):
        return p.name
    return None


def preprocess_case(
    image_id: str,
    image_dir: Path,
    model_transform,
    vis_transform,
    use_lumbar_mask: bool = False,
    mask_dir: Optional[Path] = None,
) -> Tuple[torch.Tensor, np.ndarray, str]:
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
            raise FileNotFoundError(f"No mask PNG/TXT for {fname} under {mask_dir}")
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
        filtered_mask = keep_largest_component(cv2.dilate(mask, kernel, iterations=1))
        org_img = cv2.bitwise_and(org_img, org_img, mask=filtered_mask)
        x_min, y_min, x_max, y_max = get_boundary(filtered_mask)
        if x_max > x_min and y_max > y_min:
            org_img = org_img[y_min:y_max, x_min:x_max]

    img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    input_tensor = model_transform(img_pil).unsqueeze(0)
    vis_pil = vis_transform(img_pil).convert("RGB")
    vis_rgb = np.array(vis_pil).astype(np.uint8)
    return input_tensor, vis_rgb, fname


# -------------------------
# Module path / target layer resolution
# -------------------------
def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    cur: Any = model
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def try_get_module(model: nn.Module, path: str):
    try:
        return get_module_by_path(model, path), path
    except Exception:
        return None, None


def resolve_target_layer(model: nn.Module, cfg: dict, target_layer_path: Optional[str] = None) -> Tuple[nn.Module, str]:
    if target_layer_path:
        return get_module_by_path(model, target_layer_path), target_layer_path

    source = str(cfg.get("pretrained_source", "")).lower()
    backbone_name = str(cfg.get("backbone_name", "")).lower()

    # Late pre-attention LayerNorm is a common practical target for ViT CAM.
    candidates = []
    if "clip" in source or "clip" in backbone_name:
        candidates.extend([
            "backbone.vision.vision_model.encoder.layers.11.layer_norm1",
            "backbone.vision.vision_model.encoder.layers.11.layer_norm2",
            "backbone.vision.vision_model.post_layernorm",
        ])
    if "transformers" in source or "google" in backbone_name or "vit-base" in backbone_name:
        candidates.extend([
            "backbone.vit.encoder.layer.11.layernorm_before",
            "backbone.vit.encoder.layer.11.layernorm_after",
            "backbone.vit.layernorm",
        ])
    # timm ViT fallback.
    candidates.extend([
        "backbone.blocks.11.norm1",
        "backbone.blocks.11.norm2",
        "backbone.norm",
    ])

    for path in candidates:
        mod, resolved = try_get_module(model, path)
        if mod is not None:
            return mod, resolved

    # Last LayerNorm fallback.
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm):
            last_name, last_mod = name, mod
    if last_mod is not None:
        return last_mod, last_name

    raise RuntimeError(
        "Could not resolve a ViT target layer. Pass --target_layer explicitly. "
        "Examples: google='backbone.vit.encoder.layer.11.layernorm_before', "
        "openai='backbone.vision.vision_model.encoder.layers.11.layer_norm1'."
    )


def first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            t = first_tensor(item)
            if t is not None:
                return t
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    return None


def infer_patch_size(model: nn.Module, cfg: dict) -> Tuple[int, int]:
    aug_cfg = cfg.get("augmentation", {}) or {}
    if "patch_size" in aug_cfg:
        ps = aug_cfg["patch_size"]
        if isinstance(ps, (list, tuple)):
            return int(ps[0]), int(ps[1])
        return int(ps), int(ps)

    # HF Google ViT wrapper.
    vit = getattr(getattr(model, "backbone", None), "vit", None)
    if vit is not None and hasattr(vit, "config") and hasattr(vit.config, "patch_size"):
        ps = vit.config.patch_size
        return (int(ps[0]), int(ps[1])) if isinstance(ps, (list, tuple)) else (int(ps), int(ps))

    # HF CLIP vision wrapper.
    vision = getattr(getattr(model, "backbone", None), "vision", None)
    if vision is not None and hasattr(vision, "config") and hasattr(vision.config, "patch_size"):
        ps = vision.config.patch_size
        return (int(ps[0]), int(ps[1])) if isinstance(ps, (list, tuple)) else (int(ps), int(ps))

    # timm ViT.
    patch_embed = getattr(getattr(model, "backbone", None), "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "patch_size"):
        ps = patch_embed.patch_size
        return (int(ps[0]), int(ps[1])) if isinstance(ps, (list, tuple)) else (int(ps), int(ps))

    return 16, 16


# -------------------------
# ViT Grad-CAM helper
# -------------------------
class ViTGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, patch_size: Tuple[int, int]):
        self.model = model
        self.target_layer = target_layer
        self.patch_size = patch_size
        self.activations = None
        self.gradients = None
        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        t = first_tensor(out)
        if t is None:
            raise RuntimeError(f"Forward hook on {module} did not receive a Tensor output.")
        # A new forward pass should invalidate any stale gradients from the previous case.
        # Do NOT clear activations inside generate(), because forward already happened
        # before we call backward.
        self.activations = t
        self.gradients = None

    def _backward_hook(self, module, grad_input, grad_output):
        t = first_tensor(grad_output)
        if t is None:
            raise RuntimeError(f"Backward hook on {module} did not receive Tensor gradients.")
        self.gradients = t

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def _token_cam(self, input_tensor: torch.Tensor) -> np.ndarray:
        activations = self.activations
        gradients = self.gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Conv fallback, mostly for safety.
        if activations.ndim == 4:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            return normalize_cam(cam)

        if activations.ndim != 3:
            raise RuntimeError(
                f"Expected ViT activations shape (B, N_tokens, D), got {tuple(activations.shape)}. "
                "Try a different --target_layer."
            )

        B, N, D = activations.shape
        ph, pw = self.patch_size
        H, W = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
        grid_h, grid_w = H // ph, W // pw
        n_patches = grid_h * grid_w

        acts = activations
        grads = gradients
        if N == n_patches + 1:
            acts = acts[:, 1:, :]
            grads = grads[:, 1:, :]
        elif N == n_patches:
            pass
        else:
            # Fallback for target layers that emit non-interpolated or unexpected token count.
            # Try to remove CLS and infer a rectangular grid close to input aspect ratio.
            if N > 1:
                acts_try = acts[:, 1:, :]
                grads_try = grads[:, 1:, :]
                n_try = acts_try.shape[1]
            else:
                acts_try, grads_try, n_try = acts, grads, N
            if n_try == n_patches:
                acts, grads = acts_try, grads_try
            else:
                raise RuntimeError(
                    f"Token count mismatch: target layer has N={N}, but input {H}x{W} with patch "
                    f"{ph}x{pw} expects {n_patches} patches (+ optional CLS). "
                    "For Google/OpenAI ViT, use the recommended late LayerNorm target layer."
                )

        # Grad-CAM over token sequence: average gradients over patch tokens, then channel-weight activations.
        weights = grads.mean(dim=1, keepdim=True)       # (B, 1, D)
        cam_tokens = (weights * acts).sum(dim=-1)       # (B, n_patches)
        cam_tokens = F.relu(cam_tokens)
        cam_grid = cam_tokens.reshape(B, 1, grid_h, grid_w)
        cam_grid = F.interpolate(cam_grid, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam_grid[0, 0].detach().cpu().numpy()
        return normalize_cam(cam)

    def generate(self, input_tensor: torch.Tensor, target_score: torch.Tensor) -> np.ndarray:
        # The caller already ran outputs = model(input_tensor), so the forward hook
        # has already captured activations. If we clear self.activations here, Grad-CAM
        # will fail with "hooks did not capture activations/gradients".
        self.model.zero_grad(set_to_none=True)
        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM forward activations are missing before backward. "
                "Make sure outputs = model(input_tensor) is called before cam.generate(...), "
                "and that --target_layer is inside the executed forward graph."
            )
        self.gradients = None
        target_score.backward(retain_graph=False)
        return self._token_cam(input_tensor)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    if np.allclose(cam.max(), cam.min()):
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).astype(np.float32)


# -------------------------
# Visualization helpers
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


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


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
    orig = add_label_below(Image.fromarray(vis_rgb), "Input to model")
    heat = add_label_below(Image.fromarray(heatmap_rgb), "ViT Grad-CAM heatmap")
    over = add_label_below(Image.fromarray(overlay_rgb), "Overlay")

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
        f"backbone={info.get('backbone_name', '')} | source={info.get('pretrained_source', '')}",
        f"target_layer={info.get('target_layer_name', '')} | cam_target={info.get('cam_target', '')} | used_folds={info.get('used_folds', '')}",
    ]
    header = draw_header(text_lines, row_w)
    canvas = Image.new("RGB", (row_w, header.height + row.height), (255, 255, 255))
    canvas.paste(header, (0, 0))
    canvas.paste(row, (0, header.height))
    return canvas


# -------------------------
# Scoring helpers
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
        return pos_score if int(pred_class) == 1 else -pos_score

    if loss_name == "ce":
        if study_logits.ndim != 2 or study_logits.shape[1] < 2:
            raise ValueError(f"Expected CE logits of shape [B, 2], got {tuple(study_logits.shape)}")
        if cam_target == "positive":
            return study_logits[0, 1]
        return study_logits[0, int(pred_class)]
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def infer_cv_dir_from_wrong_cases_csv(wrong_cases_csv: Path) -> Path:
    parent = wrong_cases_csv.parent
    if parent.name == "test_analysis":
        return parent.parent
    return parent.parent


def infer_outdir(wrong_cases_csv: Path) -> Path:
    if wrong_cases_csv.parent.name == "test_analysis":
        return wrong_cases_csv.parent.parent / "vit_gradcam"
    return wrong_cases_csv.parent / "vit_gradcam"


def build_case_info(row: pd.Series, cfg: dict, target_layer_name: str, cam_target: str, used_folds: List[int]) -> Dict:
    return {
        "image_id": safe_str(row.get("image_id", "")),
        "confusion_type": safe_str(row.get("confusion_type", "")),
        "y_true": int(row.get("y_true", -1)),
        "y_pred": int(row.get("y_pred", -1)),
        "y_prob": float(row.get("y_prob", np.nan)),
        "location": safe_str(row.get("location", "")),
        "grade": safe_str(row.get("grade", "")),
        "implant": safe_str(row.get("implant", "")),
        "backbone_name": safe_str(cfg.get("backbone_name", "")),
        "pretrained_source": safe_str(cfg.get("pretrained_source", "")),
        "target_layer_name": target_layer_name,
        "cam_target": cam_target,
        "used_folds": ",".join(map(str, used_folds)),
    }


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    wrong_cases_csv = Path(args.wrong_cases_csv).resolve()
    cfg = load_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cv_dir = Path(args.cv_dir).resolve() if args.cv_dir else infer_cv_dir_from_wrong_cases_csv(wrong_cases_csv)
    outdir = Path(args.outdir).resolve() if args.outdir else infer_outdir(wrong_cases_csv)
    outdir.mkdir(parents=True, exist_ok=True)

    wrong_df = pd.read_csv(wrong_cases_csv)
    if args.start_idx > 0:
        wrong_df = wrong_df.iloc[args.start_idx:].reset_index(drop=True)
    if args.max_cases is not None:
        wrong_df = wrong_df.iloc[:args.max_cases].reset_index(drop=True)
    if len(wrong_df) == 0:
        raise RuntimeError("No wrong cases to visualize after applying start_idx/max_cases.")

    # Merge raw test metadata if missing.
    test_csv = Path(cfg["test_csv"]).resolve()
    if test_csv.is_file():
        raw_test_df = pd.read_csv(test_csv)
        if "image_id" in raw_test_df.columns:
            merge_cols = [c for c in ["image_id", "location", "grade", "implant", "text", "generated_text_report"] if c in raw_test_df.columns]
            for col in merge_cols:
                if col != "image_id" and col not in wrong_df.columns:
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

    model_transform, vis_transform, input_channels = get_eval_transform_and_vis_transform(cfg.get("augmentation", {}))
    print(f"[INFO] device={device} | input_channels={input_channels} | cv_dir={cv_dir}")
    print(f"[INFO] output_dir={outdir}")

    # Load fold models and attach Grad-CAM hooks once.
    loaded = []
    target_layer_name_ref = None
    patch_size_ref = None
    for fold_idx in range(1, args.n_folds + 1):
        ckpt_path = cv_dir / f"fold_{fold_idx}" / args.ckpt_name
        if not ckpt_path.is_file():
            print(f"[WARN] Skip fold {fold_idx}: checkpoint not found: {ckpt_path}")
            continue

        model = build_model(cfg, device)
        state_dict = load_checkpoint_state_dict(str(ckpt_path), device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Fold {fold_idx} missing keys: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        if unexpected:
            print(f"[WARN] Fold {fold_idx} unexpected keys: {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
        model.eval()

        target_layer, target_layer_name = resolve_target_layer(model, cfg, args.target_layer)
        patch_size = infer_patch_size(model, cfg)
        if target_layer_name_ref is None:
            target_layer_name_ref = target_layer_name
            patch_size_ref = patch_size
        cam_helper = ViTGradCAM(model, target_layer, patch_size=patch_size)
        loaded.append({
            "fold_idx": fold_idx,
            "model": model,
            "cam": cam_helper,
            "target_layer_name": target_layer_name,
            "patch_size": patch_size,
        })
        print(f"[INFO] Loaded fold {fold_idx}: {ckpt_path} | target_layer={target_layer_name} | patch_size={patch_size}")

    if len(loaded) == 0:
        raise RuntimeError("No checkpoints were loaded successfully.")

    manifest_rows = []
    for i, row in wrong_df.iterrows():
        image_id = str(row["image_id"])
        confusion = safe_str(row.get("confusion_type", "wrong")) or "wrong"
        y_pred = int(row.get("y_pred", 0))
        out_png = outdir / f"{i:04d}_{image_id}_{confusion}_vit_gradcam.png"
        out_gray = outdir / f"{i:04d}_{image_id}_{confusion}_vit_cam_gray.png"
        if out_png.exists() and not args.overwrite:
            print(f"[INFO] Skip existing: {out_png}")
            manifest_rows.append({
                "image_id": image_id,
                "confusion_type": confusion,
                "output_png": str(out_png),
                "output_gray_png": str(out_gray) if args.save_grayscale_cam else "",
                "status": "skipped_exists",
            })
            continue

        try:
            input_tensor, vis_rgb, image_fname = preprocess_case(
                image_id=image_id,
                image_dir=test_image_dir,
                model_transform=model_transform,
                vis_transform=vis_transform,
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

            cam_mean = normalize_cam(np.mean(np.stack(cams, axis=0), axis=0))
            heatmap_rgb, overlay_rgb = overlay_cam_on_image(vis_rgb, cam_mean, alpha=float(args.alpha))
            info = build_case_info(row, cfg, target_layer_name_ref, args.cam_target, used_folds)
            panel = compose_panel(vis_rgb, heatmap_rgb, overlay_rgb, info)
            panel.save(out_png)

            if args.save_grayscale_cam:
                Image.fromarray(np.uint8(np.clip(cam_mean, 0.0, 1.0) * 255.0)).save(out_gray)

            manifest_rows.append({
                "image_id": image_id,
                "image_fname": image_fname,
                "confusion_type": confusion,
                "y_true": int(row.get("y_true", -1)),
                "y_pred": int(row.get("y_pred", -1)),
                "y_prob": float(row.get("y_prob", np.nan)),
                "location": safe_str(row.get("location", "")),
                "grade": safe_str(row.get("grade", "")),
                "implant": safe_str(row.get("implant", "")),
                "backbone_name": safe_str(cfg.get("backbone_name", "")),
                "pretrained_source": safe_str(cfg.get("pretrained_source", "")),
                "target_layer": target_layer_name_ref,
                "patch_size": str(patch_size_ref),
                "cam_target": args.cam_target,
                "used_folds": ",".join(map(str, used_folds)),
                "output_png": str(out_png),
                "output_gray_png": str(out_gray) if args.save_grayscale_cam else "",
                "status": "ok",
            })
            print(f"[OK] Saved ViT Grad-CAM: {out_png}")
        except Exception as e:
            manifest_rows.append({
                "image_id": image_id,
                "confusion_type": confusion,
                "output_png": str(out_png),
                "output_gray_png": str(out_gray) if args.save_grayscale_cam else "",
                "status": f"error: {e}",
            })
            print(f"[ERROR] Failed on image_id={image_id}: {e}")

    manifest_csv = outdir / "vit_gradcam_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(f"[DONE] Saved manifest: {manifest_csv}")
    print(f"[DONE] ViT Grad-CAM output dir: {outdir}")

    for item in loaded:
        try:
            item["cam"].remove()
        except Exception:
            pass


if __name__ == "__main__":
    main()