import torch
import torch.nn as nn
import torchvision

try:
    import timm
except ImportError:
    timm = None


class LumbarPairCropBinaryModel(nn.Module):
    """Binary classifier for one adjacent-vertebra pair crop.

    Input:  (B, 3, H, W) pair crop image
    Output: pair_logits, shape (B,)
    """
    def __init__(
        self,
        backbone_pretrained: bool = False,
        backbone_name: str = "resnet50",
        image_size=None,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        loss_name: str = "bce",
    ):
        super().__init__()
        self.backbone_name = str(backbone_name or "resnet50")
        self.loss_name = str(loss_name).lower()
        if self.loss_name != "bce":
            raise ValueError("Pair-crop model currently uses binary BCE loss only; set loss_name: bce")

        self.backbone, fc_dim = self._build_backbone(self.backbone_name, backbone_pretrained, image_size)
        self.fc_dim = int(fc_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fc_dim, int(hidden_dim)),
            nn.LeakyReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    @staticmethod
    def _normalize_image_size(image_size):
        if image_size is None:
            return None
        if isinstance(image_size, int):
            return int(image_size)
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            return (int(image_size[0]), int(image_size[1]))
        raise ValueError(f"image_size must be int or [height, width], got: {image_size}")

    def _build_backbone(self, backbone_name: str, pretrained: bool, image_size=None):
        name = str(backbone_name).lower()
        if name in {"resnet50", "torchvision_resnet50"}:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = torchvision.models.resnet50(weights=weights)
            fc_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, fc_dim

        if timm is None:
            raise ImportError("timm is required for non-ResNet backbones. Install with: pip install timm")

        kwargs = {"pretrained": pretrained, "num_classes": 0}
        norm_size = self._normalize_image_size(image_size)
        if norm_size is not None:
            kwargs["img_size"] = norm_size
        try:
            model = timm.create_model(backbone_name, **kwargs)
        except TypeError:
            kwargs.pop("img_size", None)
            model = timm.create_model(backbone_name, **kwargs)
        fc_dim = getattr(model, "num_features", None)
        if fc_dim is None:
            raise ValueError(f"Cannot infer feature dimension for timm backbone: {backbone_name}")
        return model, int(fc_dim)

    def forward(self, images):
        feats = self.backbone(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)
        pair_logits = self.classifier(feats).squeeze(-1)
        return {
            "pair_logits": pair_logits,
            # Alias kept so old helper code can still treat the output as binary logits if needed.
            "study_logits": pair_logits,
            "pooled_features": feats,
        }

    def load_backbone_checkpoint(self, ckpt_path: str):
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
        print(f"[INFO] Loaded backbone checkpoint from {ckpt_path}")
        print(f"[INFO] Missing keys: {len(missing)}")
        print(f"[INFO] Unexpected keys: {len(unexpected)}")
        if missing:
            print("[INFO] Example missing keys:", missing[:10])
        if unexpected:
            print("[INFO] Example unexpected keys:", unexpected[:10])
