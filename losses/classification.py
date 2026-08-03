import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.65, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = None if alpha is None else float(alpha)
        self.gamma = float(gamma)
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", pos_weight.detach().clone().float())

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = ((1.0 - pt).clamp_min(1e-8) ** self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()

class StudyClassificationLoss(nn.Module):
    def __init__(
        self,
        loss_name: str = "bce",
        pos_weight=None,
        class_weight=None,
        label_smoothing: float = 0.0,
        focal_alpha: float = 0.65,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.loss_name = loss_name.lower()

        if self.loss_name == "bce":
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.loss_name in {"focal_bce", "focal"}:
            self.criterion = BinaryFocalWithLogitsLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                pos_weight=pos_weight,
            )
        elif self.loss_name == "ce":
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weight,
                label_smoothing=label_smoothing,
            )
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

    def forward(self, logits, targets):
        return self.criterion(logits, targets)