import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from engine.metrics import compute_binary_metrics


@torch.no_grad()
def validate(model, loader, criterion, device, loss_name="bce", threshold=0.5):
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_targets = []

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["images"].to(device)
        study_labels = batch["study_labels"].to(device)

        masks = None
        if "mask_tensors" in batch:
            masks = batch["mask_tensors"].to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

        if loss_name == "bce":
            targets = study_labels.float()
        elif loss_name == "ce":
            targets = study_labels.long()
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

        if getattr(model, "expects_roi_separately", False):
            outputs = model(images, roi=masks)
            batch_n = images.size(0)
        else:
            if masks is not None:
                images = torch.cat([images, masks], dim=1)
            outputs = model(images)
            batch_n = images.size(0)

        logits = outputs["study_logits"]
        loss = criterion(logits, targets)

        if loss_name == "bce":
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)[:, 1]

        running_loss += loss.item() * batch_n
        all_probs.extend(probs.cpu().tolist())
        all_targets.extend(study_labels.long().cpu().tolist())

    y_true = np.asarray(all_targets).astype(int)
    y_prob = np.asarray(all_probs).astype(float)

    metrics = {
        "loss": running_loss / len(loader.dataset),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "ap": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }

    if threshold is not None:
        hard_metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        for k, v in hard_metrics.items():
            if k != "threshold":
                metrics[k] = v

    return metrics
