import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import classification_metrics


@torch.no_grad()
def evaluate_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    logits_batches = []
    label_batches = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits_batches.append(model(inputs).detach().cpu())
        label_batches.append(labels.detach().cpu())
    logits = torch.cat(logits_batches, dim=0).numpy()
    labels = torch.cat(label_batches, dim=0).numpy()
    metrics = classification_metrics(logits, labels)
    return metrics, logits, labels

