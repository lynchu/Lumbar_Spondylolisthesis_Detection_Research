import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.feature_dim = base.fc.in_features
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)  # (B, D)
        return x

    def load_partial_checkpoint(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

        cleaned = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            cleaned[nk] = v

        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        print(f"[INFO] Loaded checkpoint from {ckpt_path}")
        print(f"[INFO] Missing keys: {len(missing)}")
        print(f"[INFO] Unexpected keys: {len(unexpected)}")