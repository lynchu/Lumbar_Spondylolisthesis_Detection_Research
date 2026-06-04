import torch
import torch.nn as nn
import torchvision

try:
    import timm
except ImportError:  # keep ResNet path usable even if timm is not installed yet
    timm = None

try:
    from transformers import ViTModel, ViTConfig, CLIPVisionModel, CLIPVisionConfig
except ImportError:  # optional; only needed for pretrained_source='transformers_hf' or 'transformers_clip_vision'
    ViTModel = None
    ViTConfig = None
    CLIPVisionModel = None
    CLIPVisionConfig = None


class TransformersViTBackbone(nn.Module):
    """HF Transformers ViT wrapper returning one feature vector per image.

    This supports public checkpoints such as google/vit-base-patch16-224 and
    google/vit-base-patch16-224-in21k. For non-224 inputs, forward() uses
    interpolate_pos_encoding=True so 512x256 lumbar crops are accepted.
    """
    def __init__(self, model_id: str, pretrained: bool = True, input_channels: int = 3):
        super().__init__()
        if ViTModel is None or ViTConfig is None:
            raise ImportError(
                "transformers is required for pretrained_source='transformers_hf'. "
                "Install with: pip install transformers"
            )
        self.model_id = str(model_id)
        self.input_channels = int(input_channels)

        if self.input_channels != 3:
            raise ValueError(
                "transformers_hf ViT checkpoints are configured for 3-channel RGB by default. "
                "Use pretrained_source='timm_hf_hub' for timm checkpoints when input_channels=1, "
                "or use 3-channel grayscale replication."
            )

        if pretrained:
            self.vit = ViTModel.from_pretrained(self.model_id)
        else:
            cfg = ViTConfig.from_pretrained(self.model_id)
            self.vit = ViTModel(cfg)

        self.num_features = int(self.vit.config.hidden_size)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected pixel_values (B, C, H, W), got {tuple(x.shape)}")
        out = self.vit(pixel_values=x, interpolate_pos_encoding=True, return_dict=True)
        # pooler_output exists for standard ViTModel; CLS token fallback is safer.
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]




class TransformersCLIPVisionBackbone(nn.Module):
    """HF Transformers OpenAI CLIP vision wrapper returning one feature vector per image.

    This supports checkpoints such as openai/clip-vit-base-patch16. It uses only
    the vision encoder, not the CLIP text encoder, then your LumbarCropModel adds
    a BCE classification head for binary spondylolisthesis presence prediction.

    Practical notes:
      - OpenAI CLIP ViT-B/16 is pretrained for 3-channel images, so input_channels
        must be 3. For grayscale X-rays, keep the dataset in 3-channel mode.
      - For non-224 inputs such as 512x256, recent Transformers versions support
        interpolate_pos_encoding=True. If your local Transformers is too old, update
        with: pip install -U transformers
    """
    def __init__(self, model_id: str, pretrained: bool = True, input_channels: int = 3):
        super().__init__()
        if CLIPVisionModel is None or CLIPVisionConfig is None:
            raise ImportError(
                "transformers is required for pretrained_source='transformers_clip_vision'. "
                "Install with: pip install -U transformers"
            )
        self.model_id = str(model_id)
        self.input_channels = int(input_channels)
        if self.input_channels != 3:
            raise ValueError(
                "openai/clip-vit-base-patch16 expects 3-channel images. "
                "Use augmentation.input_channels=3 and let grayscale X-rays be replicated to RGB."
            )

        if pretrained:
            self.vision = CLIPVisionModel.from_pretrained(self.model_id)
        else:
            cfg = CLIPVisionConfig.from_pretrained(self.model_id)
            self.vision = CLIPVisionModel(cfg)

        self.num_features = int(self.vision.config.hidden_size)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected pixel_values (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"CLIP vision backbone was built with input_channels={self.input_channels}, "
                f"but got images with C={x.shape[1]}."
            )

        try:
            out = self.vision(
                pixel_values=x,
                interpolate_pos_encoding=True,
                return_dict=True,
            )
        except TypeError as e:
            # Older Transformers versions may not expose interpolate_pos_encoding.
            # This fallback only works for native CLIP input sizes, usually 224x224.
            try:
                out = self.vision(pixel_values=x, return_dict=True)
            except Exception as inner:
                raise RuntimeError(
                    "CLIPVisionModel failed on this input size. For 512x256 crops, "
                    "install a recent Transformers version that supports "
                    "interpolate_pos_encoding=True: pip install -U transformers"
                ) from inner

        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]

class LumbarCropModel(nn.Module):
    """
    Single lumbar-crop image classifier for binary spondylolisthesis presence.

    Supported backbones:
      - resnet50: torchvision ResNet-50, same as the previous implementation
      - any timm model name, e.g. vit_base_patch16_224
      - HF Transformers ViT, e.g. google/vit-base-patch16-224
      - HF OpenAI CLIP vision encoder, e.g. openai/clip-vit-base-patch16

    Input channel count is config-controlled: (B, input_channels, H, W).
    For ViT-B/16 at H=512, W=256, the patch size is 16x16, so the image becomes
    a 32x16 patch grid = 512 image tokens (+ 1 CLS token internally).
    """
    def __init__(
        self,
        backbone_pretrained: bool = False,
        loss_name: str = "bce",
        hidden_dim: int = 64,
        dropout: float = 0.0,
        backbone_name: str = "resnet50",
        image_size=None,
        input_channels: int = 3,
        pretrained_source: str = "auto",
        hf_hub_model_id: str | None = None,
        transformers_model_id: str | None = None,
    ):
        super().__init__()
        self.loss_name = loss_name.lower()
        self.backbone_name = str(backbone_name or "resnet50")
        self.pretrained_source = str(pretrained_source or "auto").lower()
        self.hf_hub_model_id = hf_hub_model_id
        self.transformers_model_id = transformers_model_id
        self.input_channels = int(input_channels)
        if self.input_channels not in (1, 3):
            raise ValueError(f"input_channels must be 1 or 3, got {self.input_channels}")

        self.backbone, fc_dim = self._build_backbone(
            backbone_name=self.backbone_name,
            pretrained=backbone_pretrained,
            image_size=image_size,
            input_channels=self.input_channels,
            pretrained_source=self.pretrained_source,
            hf_hub_model_id=self.hf_hub_model_id,
            transformers_model_id=self.transformers_model_id,
        )

        self.fc_combined1 = nn.Linear(fc_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        out_dim = 1 if self.loss_name == "bce" else 2
        self.fc_combined2 = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _is_resnet50(backbone_name: str) -> bool:
        return str(backbone_name).lower() in {"resnet50", "torchvision_resnet50"}

    @staticmethod
    def _normalize_image_size(image_size):
        if image_size is None:
            return None
        if isinstance(image_size, int):
            return image_size
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            return (int(image_size[0]), int(image_size[1]))
        raise ValueError(f"image_size must be int or [height, width], got: {image_size}")

    def _build_backbone(
        self,
        backbone_name: str,
        pretrained: bool,
        image_size=None,
        input_channels: int = 3,
        pretrained_source: str = "auto",
        hf_hub_model_id: str | None = None,
        transformers_model_id: str | None = None,
    ):
        if self._is_resnet50(backbone_name):
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = torchvision.models.resnet50(weights=weights)
            if input_channels == 1:
                old_conv = backbone.conv1
                new_conv = nn.Conv2d(
                    in_channels=1,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                with torch.no_grad():
                    # Average pretrained RGB filters into one grayscale filter.
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
                backbone.conv1 = new_conv
            fc_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return backbone, fc_dim

        pretrained_source = str(pretrained_source or "auto").lower()

        # Option A0: OpenAI CLIP ViT vision encoder from Hugging Face Transformers,
        # e.g. openai/clip-vit-base-patch16. This uses only the vision encoder
        # and attaches our normal BCE classification head.
        if pretrained_source in {"transformers_clip_vision", "clip_vision_hf", "openai_clip", "clip"}:
            model_id = transformers_model_id or hf_hub_model_id or backbone_name
            backbone = TransformersCLIPVisionBackbone(
                model_id=model_id,
                pretrained=pretrained,
                input_channels=int(input_channels),
            )
            return backbone, int(backbone.num_features)

        # Option A: Hugging Face Transformers checkpoints, e.g. google/vit-base-patch16-224.
        # This path supports arbitrary HxW at forward time using interpolate_pos_encoding=True.
        if pretrained_source in {"transformers_hf", "hf_transformers", "transformers"}:
            model_id = transformers_model_id or hf_hub_model_id or backbone_name
            backbone = TransformersViTBackbone(
                model_id=model_id,
                pretrained=pretrained,
                input_channels=int(input_channels),
            )
            return backbone, int(backbone.num_features)

        if timm is None:
            raise ImportError(
                "timm is required for timm backbones. Install with: pip install timm"
            )

        # Option B: timm checkpoint hosted on Hugging Face Hub.
        # Example: hf_hub_model_id='timm/vit_base_patch16_224.augreg_in1k'.
        if pretrained_source in {"timm_hf_hub", "hf_hub", "huggingface_hub"}:
            if not hf_hub_model_id:
                raise ValueError(
                    "pretrained_source='timm_hf_hub' requires hf_hub_model_id, e.g. "
                    "'timm/vit_base_patch16_224.augreg_in1k'."
                )
            timm_model_name = f"hf_hub:{hf_hub_model_id}"
        else:
            # Option C: normal timm model name with built-in/default pretrained weights.
            timm_model_name = backbone_name

        # For ViT models, passing img_size=(H, W) avoids positional embedding shape issues
        # and allows non-square inputs such as 512x256.
        timm_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "in_chans": int(input_channels),
        }
        norm_img_size = self._normalize_image_size(image_size)
        if norm_img_size is not None:
            timm_kwargs["img_size"] = norm_img_size

        try:
            backbone = timm.create_model(timm_model_name, **timm_kwargs)
        except TypeError:
            # Some timm backbones do not accept img_size; fall back for CNN-like models.
            timm_kwargs.pop("img_size", None)
            backbone = timm.create_model(timm_model_name, **timm_kwargs)

        fc_dim = getattr(backbone, "num_features", None)
        if fc_dim is None:
            raise ValueError(f"Cannot infer feature dimension for timm backbone: {timm_model_name}")
        return backbone, int(fc_dim)

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (B, C, H, W), got {tuple(images.shape)}")
        if images.shape[1] != self.input_channels:
            raise ValueError(
                f"Model was built with input_channels={self.input_channels}, "
                f"but got images with C={images.shape[1]}. Check augmentation.input_channels."
            )
        img_features = self.backbone(images)
        if isinstance(img_features, (list, tuple)):
            img_features = img_features[-1]
        if img_features.ndim > 2:
            img_features = torch.flatten(img_features, start_dim=1)

        x = self.fc_combined1(img_features)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc_combined2(x)

        if self.loss_name == "bce":
            logits = logits.squeeze(-1)

        return {
            "study_logits": logits,
            "pooled_features": img_features,
        }

    def load_backbone_checkpoint(self, ckpt_path: str):
        """Load a partial backbone checkpoint. Intended mainly for same-family backbones."""
        state = torch.load(ckpt_path, map_location="cpu")

        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]

        cleaned = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            if nk.startswith("backbone."):
                nk = nk[len("backbone."):]
            cleaned[nk] = v

        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)

        print(f"[INFO] Loaded checkpoint from {ckpt_path}")
        print(f"[INFO] Missing keys: {len(missing)}")
        print(f"[INFO] Unexpected keys: {len(unexpected)}")
        if missing:
            print("[INFO] Example missing keys:", missing[:10])
        if unexpected:
            print("[INFO] Example unexpected keys:", unexpected[:10])


class LumbarCropModelWithMask(nn.Module):
    def __init__(self, backbone_pretrained: bool = False):
        super().__init__()
        # This legacy model is for explicit 4-channel image+mask tensors only.
        self.input_channels = 4

        self.backbone = self.load_pretrained_model(backbone_pretrained)
        fc_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_combined1 = nn.Linear(fc_dim, 64)
        self.activation = nn.LeakyReLU()
        self.fc_combined2 = nn.Linear(64, 1)

    def load_pretrained_model(self, pretrained: bool):
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        model = self.modify_first_layer_for_4_channels(model)
        return model

    def modify_first_layer_for_4_channels(self, model):
        first_layer_name = list(model._modules.keys())[0]
        first_layer = model._modules[first_layer_name]

        if isinstance(first_layer, nn.Conv2d) and first_layer.in_channels == 3:
            new_first_layer = nn.Conv2d(
                in_channels=4,
                out_channels=first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=first_layer.bias is not None,
            )
            with torch.no_grad():
                new_first_layer.weight[:, :3, :, :] = first_layer.weight.clone()
                nn.init.xavier_normal_(new_first_layer.weight[:, 3:, :, :])
                if first_layer.bias is not None:
                    new_first_layer.bias.copy_(first_layer.bias)

            model._modules[first_layer_name] = new_first_layer

        return model

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (B, C, H, W), got {tuple(images.shape)}")
        if images.shape[1] != self.input_channels:
            raise ValueError(
                f"Model was built with input_channels={self.input_channels}, "
                f"but got images with C={images.shape[1]}. Check augmentation.input_channels."
            )
        img_features = self.backbone(images)
        combined = self.activation(self.fc_combined1(img_features))
        output = self.fc_combined2(combined)

        return {
            "study_logits": output.squeeze(-1),
            "pooled_features": img_features,
        }
