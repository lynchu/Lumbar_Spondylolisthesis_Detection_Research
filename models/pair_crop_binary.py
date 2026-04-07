import torch
import torch.nn as nn

from models.backbone_resnet import ResNet50FeatureExtractor


class PairCropBinaryModel(nn.Module):
    """
    Binary classifier for one pair_crop image.
    Fallback version that does not require backbone.forward_features().
    """
    def __init__(self, backbone_pretrained=False, dropout=0.2, attn_dim=128):
        super().__init__()
        self.backbone = ResNet50FeatureExtractor(pretrained=backbone_pretrained)
        d = self.backbone.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )

    def forward(self, images):
        """
        images: (B, 3, H, W)
        """
        pooled = self.backbone(images)   # (B, C)
        logits = self.classifier(pooled).squeeze(-1)

        return {
            "logits": logits,
            "pooled_features": pooled,
        }