"""postprocessing.py — logits → sigmoid → threshold → remove small blobs."""
from typing import List
import numpy as np
import torch
from skimage.measure import label


def _remove_small_objects(binary: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return binary
    labeled = label(binary)
    cleaned = np.zeros_like(binary)
    for rid in range(1, labeled.max() + 1):
        if (labeled == rid).sum() >= min_area:
            cleaned[labeled == rid] = 1
    return cleaned


def postprocess(logits: torch.Tensor, thresholds: List[float],
                min_soma_area: int = 500, min_process_area: int = 200) -> np.ndarray:
    if logits.dim() == 4:
        logits = logits.squeeze(0)
    probs = torch.sigmoid(logits).cpu().numpy()
    th = np.array(thresholds, dtype=np.float32).reshape(4, 1, 1)
    pred = (probs >= th).astype(np.uint8)
    pred[1] = _remove_small_objects(pred[1], min_soma_area)
    pred[2] = _remove_small_objects(pred[2], min_process_area)
    return pred


def get_probability_maps(logits: torch.Tensor) -> np.ndarray:
    if logits.dim() == 4:
        logits = logits.squeeze(0)
    return torch.sigmoid(logits).cpu().numpy()
