import torch.nn as nn


class StudyBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.criterion(logits, targets)
    
