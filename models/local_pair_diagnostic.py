import torch
import torch.nn as nn
import torchvision


PAIR_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


class LocalPairDiagnosticModel(nn.Module):
    """Segmentation-guided local pair diagnostic branch.

    Inputs:
      pair_images: [B, 5, 3, H, W]
      geometry_features: [B, 5, d_geo]

    Outputs:
      location_logits: [B, 5]
      pair_presence_prob_mil: [B]
      slip_percentage: [B, 5]
      ordinal_logits: [B, 5, 4]
      pair_tokens: [B, 5, token_dim]
    """

    def __init__(
        self,
        geometry_dim: int,
        backbone_name: str = "resnet18",
        backbone_pretrained: bool = False,
        token_dim: int = 256,
        geo_hidden_dim: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.geometry_dim = int(geometry_dim)
        self.backbone_name = str(backbone_name).lower()
        self.token_dim = int(token_dim)

        self.backbone, fc_dim = self._build_backbone(self.backbone_name, bool(backbone_pretrained))
        self.visual_proj = nn.Sequential(
            nn.Linear(fc_dim, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.geo_mlp = nn.Sequential(
            nn.Linear(self.geometry_dim, int(geo_hidden_dim)),
            nn.LayerNorm(int(geo_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(geo_hidden_dim), int(geo_hidden_dim)),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.token_dim + int(geo_hidden_dim), self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.level_pos_embed = nn.Parameter(torch.zeros(1, len(PAIR_NAMES), self.token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(transformer_heads),
            dim_feedforward=self.token_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pair_transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))

        self.location_head = nn.Linear(self.token_dim, 1)
        self.slip_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.token_dim // 2, 1),
        )
        self.ordinal_head = nn.Linear(self.token_dim, 4)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.level_pos_embed, std=0.02)

    @staticmethod
    def _build_backbone(backbone_name: str, pretrained: bool):
        name = str(backbone_name).lower()
        if name in {"resnet18", "torchvision_resnet18"}:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnet18(weights=weights)
        elif name in {"resnet34", "torchvision_resnet34"}:
            weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnet34(weights=weights)
        elif name in {"resnet50", "torchvision_resnet50"}:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = torchvision.models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported local pair backbone: {backbone_name}")
        fc_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, int(fc_dim)

    def forward(self, pair_images, geometry_features):
        bsz, n_pair, channels, height, width = pair_images.shape
        if n_pair != len(PAIR_NAMES):
            raise ValueError(f"Expected {len(PAIR_NAMES)} pair crops, got {n_pair}")

        x = pair_images.reshape(bsz * n_pair, channels, height, width)
        visual = self.backbone(x)
        if visual.ndim > 2:
            visual = torch.flatten(visual, start_dim=1)
        visual = self.visual_proj(visual).reshape(bsz, n_pair, self.token_dim)

        geom = torch.nan_to_num(geometry_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        geom = self.geo_mlp(geom)
        fused = self.fusion_mlp(torch.cat([visual, geom], dim=-1))
        pair_tokens = self.pair_transformer(fused + self.level_pos_embed)

        location_logits = self.location_head(pair_tokens).squeeze(-1)
        location_probs = torch.sigmoid(location_logits)
        pair_presence_prob_mil = 1.0 - torch.prod(1.0 - location_probs.clamp(1e-6, 1.0 - 1e-6), dim=1)
        pair_presence_logit_mil = torch.logit(pair_presence_prob_mil.clamp(1e-6, 1.0 - 1e-6))

        slip_raw = self.slip_head(pair_tokens).squeeze(-1)
        slip_percentage = torch.sigmoid(slip_raw) * 100.0
        ordinal_logits = self.ordinal_head(pair_tokens)

        return {
            "pair_tokens": pair_tokens,
            "location_logits": location_logits,
            "location_probs": location_probs,
            "pair_presence_prob_mil": pair_presence_prob_mil,
            "pair_presence_logit_mil": pair_presence_logit_mil,
            "slip_raw": slip_raw,
            "slip_percentage": slip_percentage,
            "ordinal_logits": ordinal_logits,
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
        print(f"[INFO] Loaded local pair backbone checkpoint from {ckpt_path}")
        print(f"[INFO] Missing keys: {len(missing)}")
        print(f"[INFO] Unexpected keys: {len(unexpected)}")
        if missing:
            print("[INFO] Example missing keys:", missing[:10])
        if unexpected:
            print("[INFO] Example unexpected keys:", unexpected[:10])
