import torch



# ===========================================
# Version0: 1 Optimizer + 2 LR(Backbone/Head)
# ===========================================

def build_optimizer(model, backbone_lr=1e-4, head_lr=5e-4, weight_decay=1e-4):

    backbone_lr = float(backbone_lr)
    head_lr = float(head_lr)
    weight_decay = float(weight_decay)

    backbone_params = list(model.backbone.parameters())
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


# ========================================
# Version1: Simple 1 Optimizer/LR Training
# ========================================

def build_optimizer_simple(model, cfg):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler_simple(optimizer, cfg):
    scheduler_name = str(cfg.get("scheduler_name", "none")).lower()

    if scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("plateau_factor", 0.5)),
            patience=int(cfg.get("plateau_patience", 5)),
            min_lr=float(cfg.get("min_lr", 1e-7)),
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["epochs"]),
            eta_min=float(cfg.get("min_lr", 1e-7)),
        )
    raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")



# ===============================
# Version2: Three-staged Training
# ===============================

def get_head_parameters(model):
    head_params = []
    for name in ["fc_combined1", "fc_combined2"]:
        if hasattr(model, name):
            head_params += list(getattr(model, name).parameters())
    return head_params


def build_optimizer_staged(model, cfg, stage_name: str):
    weight_decay = cfg.get("weight_decay", 1e-4)

    if stage_name == "head_only":
        params = [
            {
                "params": [p for p in get_head_parameters(model) if p.requires_grad],
                "lr": cfg.get("head_lr", 5e-4),
            }
        ]

    elif stage_name == "layer4_only":
        params = [
            {
                "params": [p for p in model.backbone.layer4.parameters() if p.requires_grad],
                "lr": cfg.get("layer4_lr", cfg.get("backbone_lr", 1e-4)),
            },
            {
                "params": [p for p in get_head_parameters(model) if p.requires_grad],
                "lr": cfg.get("head_lr", 5e-4),
            },
        ]

    elif stage_name == "full_finetune":
        params = [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": cfg.get("full_backbone_lr", 2e-5),
            },
            {
                "params": [p for p in get_head_parameters(model) if p.requires_grad],
                "lr": cfg.get("full_head_lr", cfg.get("head_lr", 5e-4) * 0.5),
            },
        ]

    else:
        raise ValueError(f"Unsupported stage_name: {stage_name}")

    return torch.optim.AdamW(params, weight_decay=weight_decay)


def build_scheduler_staged(optimizer, cfg):
    scheduler_name = cfg.get("scheduler_name", "none").lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.get("plateau_factor", 0.5),
            patience=cfg.get("plateau_patience", 5),
            min_lr=cfg.get("min_lr", 1e-7),
        )

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["epochs"],
            eta_min=cfg.get("min_lr", 1e-7),
        )

    raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")