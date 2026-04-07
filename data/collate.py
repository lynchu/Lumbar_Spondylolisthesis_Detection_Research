import torch


def single_crop_collate_fn(batch):
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    c, h, w = batch[0]["image_tensor"].shape[1:]

    images = torch.zeros(batch_size, max_len, c, h, w, dtype=batch[0]["image_tensor"].dtype)
    vertebra_labels = torch.zeros(batch_size, max_len, dtype=torch.float32)
    lumbar_ids = torch.full((batch_size, max_len), fill_value=-1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    study_labels = torch.zeros(batch_size, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    image_ids = []

    for i, item in enumerate(batch):
        n = item["length"]
        images[i, :n] = item["image_tensor"]
        vertebra_labels[i, :n] = item["vertebra_labels"]
        lumbar_ids[i, :n] = item["lumbar_ids"]
        mask[i, :n] = True
        study_labels[i] = item["study_label"]
        lengths[i] = n
        image_ids.append(item["image_id"])

    return {
        "images": images,                  # (B, N, 3, 224, 224)
        "vertebra_labels": vertebra_labels,# (B, N)
        "lumbar_ids": lumbar_ids,          # (B, N)
        "mask": mask,                      # (B, N)
        "study_labels": study_labels,      # (B,)
        "lengths": lengths,                # (B,)
        "image_ids": image_ids,
    }


def lumbar_crop_collate_fn(batch):
    images = torch.stack([item["image_tensor"] for item in batch], dim=0)  # (B, 3, 224, 224)
    study_labels = torch.stack([item["study_label"] for item in batch], dim=0)  # (B,)
    image_ids = [item["image_id"] for item in batch]

    return {
        "images": images,
        "study_labels": study_labels,
        "image_ids": image_ids,
    }