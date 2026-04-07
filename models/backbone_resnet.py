import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()

        self.backbone = tvm.resnet50(
            weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        """
        x: (B, 3, H, W)
        return: (B, D)
        """
        return self.backbone(x)

    def load_partial_checkpoint(self, ckpt_path: str):
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

            # If checkpoint was saved from a larger model and backbone keys were prefixed
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