# utils/collate.py
import torch


def mffn_yolo_collate_fn(batch):
    multi_view_batch = {}
    for key in batch[0]["multi_view"]:
        multi_view_batch[key] = torch.stack([item["multi_view"][key] for item in batch])
    targets = [item["targets"] for item in batch]
    return multi_view_batch, targets