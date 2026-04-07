import torch
import torch.nn as nn

from models.backbone_resnet_attention import ResNet50FeatureExtractor


class Stage1BasicPresenceModel(nn.Module):
    """
    Basic stage-1 model for whole lumbar crop:
    - shared ResNet50 backbone
    - spatial attention pooling over feature map
    - optional mean over image dimension N (still keeps old interface compatibility)
    - study-level binary classifier
    """
    def __init__(self, backbone_pretrained=False, dropout=0.2, attn_dim=128):
        super().__init__()
        self.backbone = ResNet50FeatureExtractor(pretrained=backbone_pretrained)
        d = self.backbone.feature_dim

        # spatial attention over feature-map locations
        self.spatial_attn = nn.Sequential(
            nn.Linear(d, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )

    @staticmethod
    def masked_mean(x, mask, dim=1, eps=1e-6):
        mask_f = mask.unsqueeze(-1).float()
        x_sum = (x * mask_f).sum(dim=dim)
        denom = mask_f.sum(dim=dim).clamp_min(eps)
        return x_sum / denom

    def spatial_attention_pool(self, fmap):
        """
        fmap: (B, C, H, W)
        returns:
          pooled: (B, C)
          attn:   (B, H*W)
        """
        B, C, H, W = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)   # (B, HW, C)

        attn_logits = self.spatial_attn(tokens).squeeze(-1)  # (B, HW)
        attn = torch.softmax(attn_logits, dim=1)             # (B, HW)

        pooled = torch.sum(tokens * attn.unsqueeze(-1), dim=1)  # (B, C)
        return pooled, attn

    def forward(self, images, mask):
        """
        images: (B, N, 3, H, W)
        mask:   (B, N) bool

        For whole lumbar crop, N=1.
        """
        b, n, c, h, w = images.shape
        x = images.view(b * n, c, h, w)

        fmap = self.backbone.forward_features(x)            # (B*N, C, h', w')
        pooled_img, spatial_attn = self.spatial_attention_pool(fmap)  # (B*N, C), (B*N, HW)

        d = pooled_img.size(-1)
        pooled_img = pooled_img.view(b, n, d)               # (B, N, D)
        spatial_attn = spatial_attn.view(b, n, -1)          # (B, N, HW)

        pooled_study = self.masked_mean(pooled_img, mask)   # (B, D)
        logits = self.classifier(pooled_study).squeeze(-1)  # (B,)

        return {
            "study_logits": logits,
            "pooled_features": pooled_study,
            "vertebra_features": pooled_img,   # keep old key for compatibility
            "spatial_attn": spatial_attn,      # (B, N, HW)
            "feature_map_hw": torch.tensor([fmap.shape[-2], fmap.shape[-1]], device=fmap.device),
        }