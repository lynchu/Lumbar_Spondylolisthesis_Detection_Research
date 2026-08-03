import torch
import torch.nn as nn
from tqdm import tqdm


def freeze_bn_stats(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()


def train_one_epoch(model, loader, optimizer, criterion, device, loss_name="bce", grad_clip_norm=None, freeze_bn=False):
    model.train()
    if freeze_bn and hasattr(model, "backbone"):
        model.backbone.apply(freeze_bn_stats)
    running_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["images"].to(device)
        study_labels = batch["study_labels"].to(device)

        masks = None
        if "mask_tensors" in batch:
            masks = batch["mask_tensors"].to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

        if str(loss_name).lower() in {"bce", "focal_bce", "focal"}:
            targets = study_labels.float()
        elif str(loss_name).lower() == "ce":
            targets = study_labels.long()
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

        optimizer.zero_grad(set_to_none=True)

        geometry_features = batch.get("geometry_features")
        if geometry_features is not None:
            geometry_features = geometry_features.to(device)

        if getattr(model, "expects_roi_separately", False):
            outputs = model(images, roi=masks)
            batch_n = images.size(0)
        else:
            if masks is not None:
                images = torch.cat([images, masks], dim=1)
            if getattr(model, "expects_geometry", False):
                outputs = model(images, geometry_features=geometry_features)
            else:
                outputs = model(images)
            batch_n = images.size(0)

        logits = outputs["study_logits"]
        loss = criterion(logits, targets)
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        running_loss += loss.item() * batch_n

    return running_loss / len(loader.dataset)
