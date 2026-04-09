import torch
from tqdm import tqdm

from engine.metrics import compute_binary_metrics


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_targets = []

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["images"].to(device)
        study_labels = batch["study_labels"].to(device)

        outputs = model(images)
        loss = criterion(outputs["study_logits"], study_labels)

        probs = torch.sigmoid(outputs["study_logits"])

        running_loss += loss.item() * images.size(0)
        all_probs.extend(probs.cpu().tolist())
        all_targets.extend(study_labels.cpu().tolist())

    metrics = compute_binary_metrics(all_targets, all_probs, threshold=threshold)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics