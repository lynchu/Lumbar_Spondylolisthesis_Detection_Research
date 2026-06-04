import torch
import torch.nn as nn
import torchvision


class LumbarCropModel(nn.Module):
    def __init__(
        self,
        backbone_pretrained: bool = False,
        loss_name: str = "bce",
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.loss_name = loss_name.lower()

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if backbone_pretrained else None
        self.backbone = torchvision.models.resnet50(weights=weights)
        # self.backbone = torchvision.models.resnet50(pretrained=backbone_pretrained)
        fc_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_combined1 = nn.Linear(fc_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        out_dim = 1 if self.loss_name == "bce" else 2
        self.fc_combined2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, images):
        img_features = self.backbone(images)
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


class LumbarCropModelWithMask(nn.Module):
    def __init__(self, backbone_pretrained: bool = False):
        super().__init__()

        self.backbone = self.load_pretrained_model(backbone_pretrained)
        fc_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_combined1 = nn.Linear(fc_dim, 64) # d//2
        self.activation = nn.LeakyReLU()
        self.fc_combined2 = nn.Linear(64, 1)

    def load_pretrained_model(self, pretrained: bool):
        model = torchvision.models.resnet50(pretrained=pretrained)
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
        """
        images: (B, 4, H, W)
        """
        img_features = self.backbone(images)
        combined = self.activation(self.fc_combined1(img_features))
        output = self.fc_combined2(combined)

        return {
            "study_logits": output.squeeze(-1),
            "pooled_features": img_features,
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

        print(f"[INFO] Loaded checkpoint from {ckpt_path}")
        print(f"[INFO] Missing keys: {len(missing)}")
        print(f"[INFO] Unexpected keys: {len(unexpected)}")
        if missing:
            print("[INFO] Example missing keys:", missing[:10])
        if unexpected:
            print("[INFO] Example unexpected keys:", unexpected[:10])