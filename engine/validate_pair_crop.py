import torch
import numpy as np

from engine.metrics import compute_binary_metrics


@torch.no_grad()
def validate_pair_crop(model, loader, criterion, device, threshold=0.5):
    model.eval()

    running_loss = 0.0
    y_true = []
    y_prob = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs["logits"], labels)

        probs = torch.sigmoid(outputs["logits"])

        running_loss += loss.item() * images.size(0)
        y_true.append(labels.detach().cpu().numpy())
        y_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true) if len(y_true) > 0 else np.array([])
    y_prob = np.concatenate(y_prob) if len(y_prob) > 0 else np.array([])

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics