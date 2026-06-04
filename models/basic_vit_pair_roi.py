import torch
import torch.nn as nn
import torchvision

try:
    import timm
except ImportError:  # keep ResNet path usable even if timm is not installed yet
    timm = None


class LumbarCropModel(nn.Module):
    """
    Lumbar-crop classifier with optional pair auxiliary supervision.

    Supported modes:
      1) Baseline global image classification:
           image -> backbone pooled feature -> presence head
      2) Global pair auxiliary head:
           pooled feature -> 5 pair logits
      3) ROI-token pair auxiliary head:
           image -> timm ViT patch tokens -> pair ROI token pooling -> 5 pair logits

    For ViT-B/16 at image_size=[512, 256], the patch grid is 32 x 16 = 512 patch tokens.
    ROI pooling uses a binary mask over these 512 tokens for each anatomical pair.
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
        use_pair_roi_pooling: bool = False,
        pair_roi_fallback_to_global: bool = True,
    ):
        super().__init__()
        self.loss_name = loss_name.lower()
        self.backbone_name = str(backbone_name or "resnet50")
        self.use_pair_aux_head = bool(use_pair_aux_head)
        self.num_pairs = int(num_pairs)
        self.use_pair_roi_pooling = bool(use_pair_roi_pooling)
        self.pair_roi_fallback_to_global = bool(pair_roi_fallback_to_global)

        self.backbone, fc_dim = self._build_backbone(
            backbone_name=self.backbone_name,
            pretrained=backbone_pretrained,
            image_size=image_size,
        )
        self.fc_dim = int(fc_dim)

        self.fc_combined1 = nn.Linear(fc_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        out_dim = 1 if self.loss_name == "bce" else 2
        self.fc_combined2 = nn.Linear(hidden_dim, out_dim)

        if self.use_pair_aux_head:
            ph = int(pair_hidden_dim) if pair_hidden_dim is not None else int(hidden_dim)
            if self.use_pair_roi_pooling:
                # Applied independently to each pair ROI feature: (B, 5, D) -> (B, 5)
                self.pair_head = nn.Sequential(
                    nn.Linear(fc_dim, ph),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ph, 1),
                )
            else:
                # First-stage baseline: use global pooled feature for all 5 pair logits.
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
            raise ImportError("timm is required for non-ResNet backbones. Install with: pip install timm")

        timm_kwargs = {"pretrained": pretrained, "num_classes": 0}
        norm_img_size = self._normalize_image_size(image_size)
        if norm_img_size is not None:
            timm_kwargs["img_size"] = norm_img_size

        try:
            backbone = timm.create_model(backbone_name, **timm_kwargs)
        except TypeError:
            timm_kwargs.pop("img_size", None)
            backbone = timm.create_model(backbone_name, **timm_kwargs)

        fc_dim = getattr(backbone, "num_features", None)
        if fc_dim is None:
            raise ValueError(f"Cannot infer feature dimension for timm backbone: {backbone_name}")
        return backbone, int(fc_dim)

    def _is_timm_vit_like(self) -> bool:
        return hasattr(self.backbone, "patch_embed") and hasattr(self.backbone, "blocks") and hasattr(self.backbone, "forward_features")

    def _num_patches(self):
        patch_embed = getattr(self.backbone, "patch_embed", None)
        if patch_embed is None:
            return None
        n = getattr(patch_embed, "num_patches", None)
        if n is not None:
            return int(n)
        grid_size = getattr(patch_embed, "grid_size", None)
        if grid_size is not None:
            return int(grid_size[0]) * int(grid_size[1])
        return None

    def _backbone_features_and_tokens(self, images):
        """Return pooled feature and patch-token sequence when available.

        For timm ViT, `forward_features` returns token sequence `(B, N+1, D)` in most versions.
        We remove the CLS token and keep `(B, N, D)` for ROI token pooling.
        """
        if self.use_pair_roi_pooling:
            if not self._is_timm_vit_like():
                raise RuntimeError("pair_aux.use_roi_pooling=True currently requires a timm ViT-like backbone with patch tokens.")
            feats = self.backbone.forward_features(images)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            if feats.ndim != 3:
                raise RuntimeError(
                    f"Expected timm ViT forward_features to return (B, N_tokens, D), got {tuple(feats.shape)}."
                )

            # Get the same global feature timm would normally return.
            if hasattr(self.backbone, "forward_head"):
                try:
                    pooled = self.backbone.forward_head(feats, pre_logits=True)
                except TypeError:
                    pooled = self.backbone.forward_head(feats)
            else:
                pooled = feats[:, 0]
            if isinstance(pooled, (list, tuple)):
                pooled = pooled[-1]
            if pooled.ndim > 2:
                pooled = torch.flatten(pooled, start_dim=1)

            n_patches = self._num_patches()
            if n_patches is not None and feats.shape[1] == n_patches + 1:
                patch_tokens = feats[:, 1:, :]
            elif n_patches is not None and feats.shape[1] == n_patches:
                patch_tokens = feats
            else:
                # Conservative fallback: assume first token is CLS if token count is not expected.
                patch_tokens = feats[:, 1:, :] if feats.shape[1] > 1 else feats
            return pooled, patch_tokens

        img_features = self.backbone(images)
        if isinstance(img_features, (list, tuple)):
            img_features = img_features[-1]
        if img_features.ndim > 2:
            img_features = torch.flatten(img_features, start_dim=1)
        return img_features, None

    def _pool_pair_roi_tokens(self, patch_tokens, pair_roi_masks, pooled_features=None):
        if pair_roi_masks is None:
            raise RuntimeError("pair_aux.use_roi_pooling=True but batch/model input has no pair_roi_masks.")
        if patch_tokens is None:
            raise RuntimeError("pair_aux.use_roi_pooling=True but patch_tokens are None.")

        masks = pair_roi_masks.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        if masks.ndim == 4:  # (B, 5, Gh, Gw)
            masks = masks.flatten(2)
        if masks.ndim != 3:
            raise RuntimeError(f"pair_roi_masks must be (B, 5, N) or (B, 5, Gh, Gw), got {tuple(masks.shape)}")
        if masks.shape[0] != patch_tokens.shape[0] or masks.shape[2] != patch_tokens.shape[1]:
            raise RuntimeError(
                f"pair_roi_masks shape {tuple(masks.shape)} does not match patch_tokens shape {tuple(patch_tokens.shape)}."
            )

        denom = masks.sum(dim=-1, keepdim=True)  # (B, 5, 1)
        safe_denom = denom.clamp_min(1.0)
        pair_features = torch.einsum("bpn,bnd->bpd", masks, patch_tokens) / safe_denom
        pair_roi_valid = (denom.squeeze(-1) > 0).float()

        if self.pair_roi_fallback_to_global and pooled_features is not None:
            invalid = pair_roi_valid <= 0
            if bool(invalid.any()):
                fallback = pooled_features.unsqueeze(1).expand(-1, masks.shape[1], -1)
                pair_features = torch.where(invalid.unsqueeze(-1), fallback, pair_features)

        return pair_features, pair_roi_valid

    def forward(self, images, pair_roi_masks=None):
        img_features, patch_tokens = self._backbone_features_and_tokens(images)

        x = self.fc_combined1(img_features)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc_combined2(x)
        if self.loss_name == "bce":
            logits = logits.squeeze(-1)

        out = {"study_logits": logits, "pooled_features": img_features}

        if self.pair_head is not None:
            if self.use_pair_roi_pooling:
                pair_features, pair_roi_valid = self._pool_pair_roi_tokens(
                    patch_tokens=patch_tokens,
                    pair_roi_masks=pair_roi_masks,
                    pooled_features=img_features,
                )
                pair_logits = self.pair_head(pair_features).squeeze(-1)
                out["pair_features"] = pair_features
                out["pair_roi_valid"] = pair_roi_valid
                out["pair_logits"] = pair_logits
            else:
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
