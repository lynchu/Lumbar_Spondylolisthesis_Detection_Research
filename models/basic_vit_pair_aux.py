import torch
import torch.nn as nn
import torchvision
import timm



class LumbarCropModel(nn.Module):
    """
    Single lumbar-crop image classifier for binary spondylolisthesis presence.

    Supported backbones:
      - resnet50: torchvision ResNet-50, same as the previous implementation
      - any timm model name, e.g. vit_base_patch16_224

    Input is always a 3-channel tensor: (B, 3, H, W).
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
        use_pair_aux_head: bool = False,
        num_pairs: int = 5,
        pair_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.loss_name = loss_name.lower()
        self.backbone_name = str(backbone_name or "resnet50")
        self.use_pair_aux_head = bool(use_pair_aux_head)
        self.num_pairs = int(num_pairs)

        self.backbone, fc_dim = self._build_backbone(
            backbone_name=self.backbone_name,
            pretrained=backbone_pretrained,
            image_size=image_size,
        )

        self.fc_combined1 = nn.Linear(fc_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        out_dim = 1 if self.loss_name == "bce" else 2
        self.fc_combined2 = nn.Linear(hidden_dim, out_dim)

        # Optional auxiliary head for anatomical pair supervision.
        # It uses the same pooled global ViT feature as a first-stage baseline.
        # Later ROI-token pooling can replace this head input without changing labels/loss.
        if self.use_pair_aux_head:
            ph = int(pair_hidden_dim) if pair_hidden_dim is not None else int(hidden_dim)
            self.pair_head = nn.Sequential(
                nn.Linear(fc_dim, ph),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(ph, self.num_pairs),
            )
        else:
            self.pair_head = None

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

    def _build_backbone(self, backbone_name: str, pretrained: bool, image_size=None):
        if self._is_resnet50(backbone_name):
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = torchvision.models.resnet50(weights=weights)
            fc_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return backbone, fc_dim

        if timm is None:
            raise ImportError(
                "timm is required for non-ResNet backbones. Install with: pip install timm"
            )

        # For ViT models, passing img_size=(H, W) avoids positional embedding shape issues
        # and allows non-square inputs such as 512x256.
        timm_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
        }
        norm_img_size = self._normalize_image_size(image_size)
        if norm_img_size is not None:
            timm_kwargs["img_size"] = norm_img_size

        try:
            backbone = timm.create_model(backbone_name, **timm_kwargs)
        except TypeError:
            # Some timm backbones do not accept img_size; fall back for CNN-like models.
            timm_kwargs.pop("img_size", None)
            backbone = timm.create_model(backbone_name, **timm_kwargs)

        fc_dim = getattr(backbone, "num_features", None)
        if fc_dim is None:
            raise ValueError(f"Cannot infer feature dimension for timm backbone: {backbone_name}")
        return backbone, int(fc_dim)

    def forward(self, images):
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

        out = {
            "study_logits": logits,
            "pooled_features": img_features,
        }
        if self.pair_head is not None:
            out["pair_logits"] = self.pair_head(img_features)
        return out

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
        img_features = self.backbone(images)
        combined = self.activation(self.fc_combined1(img_features))
        output = self.fc_combined2(combined)

        return {
            "study_logits": output.squeeze(-1),
            "pooled_features": img_features,
        }
