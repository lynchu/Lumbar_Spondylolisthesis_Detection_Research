import torch
import torch.nn as nn

from models.backbone_resnet import ResNet50FeatureExtractor


class Stage1BasicPresenceModel(nn.Module):
    def __init__(self, backbone_pretrained=False, dropout=0.2):
        super().__init__()
        self.backbone = ResNet50FeatureExtractor(pretrained=backbone_pretrained)
        d = self.backbone.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(d, 64),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(64, 1),
        # )

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        """
        feats = self.backbone(images)                # (B, D)
        logits = self.classifier(feats).squeeze(-1) # (B,)

        return {
            "study_logits": logits,
            "pooled_features": feats,
        }