import torch
import torch.nn as nn


class StudyClassificationLoss(nn.Module):
    def __init__(
        self,
        loss_name: str = "bce",
        pos_weight=None,
        class_weight=None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.loss_name = loss_name.lower()

        if self.loss_name == "bce":
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.loss_name == "ce":
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weight,
                label_smoothing=label_smoothing,
            )
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

    def forward(self, logits, targets):
        return self.criterion(logits, targets)