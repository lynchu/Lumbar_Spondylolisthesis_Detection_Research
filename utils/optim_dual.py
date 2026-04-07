import torch


def build_optimizer(model, backbone_lr=1e-4, head_lr=5e-4, weight_decay=1e-4):
    backbone_params = []

    # Old single-backbone models
    if hasattr(model, "backbone"):
        backbone_params.extend(list(model.backbone.parameters()))

    # Dual-backbone models
    if hasattr(model, "global_backbone"):
        backbone_params.extend(list(model.global_backbone.parameters()))
    if hasattr(model, "pair_backbone"):
        backbone_params.extend(list(model.pair_backbone.parameters()))

    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)