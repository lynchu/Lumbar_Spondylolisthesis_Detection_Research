import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["images"].to(device)
        study_labels = batch["study_labels"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs["study_logits"], study_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)