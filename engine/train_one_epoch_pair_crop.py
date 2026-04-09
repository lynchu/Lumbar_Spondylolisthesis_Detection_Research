import torch
from tqdm import tqdm


def train_one_epoch_pair_crop(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs["logits"], labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{running_loss / max(len(loader.dataset), 1):.4f}"})

    return running_loss / len(loader.dataset)