"""preprocessing.py — TIFF reading, normalisation, resizing."""
from pathlib import Path
from typing import List
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F


def list_tiff_frames(path: Path) -> List[int]:
    with tiff.TiffFile(str(path)) as tf:
        n = len(tf.pages)
    return list(range(n))


def list_tiff_files(folder: Path) -> List[Path]:
    files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def read_frame(path: Path, frame_index: int = 0) -> np.ndarray:
    with tiff.TiffFile(str(path)) as tf:
        arr = tf.pages[frame_index].asarray()
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D frame at index {frame_index}, got shape {arr.shape}.")
    return arr.astype(np.float32)


def normalize(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(img, p_low), np.percentile(img, p_high)
    img = np.clip(img, lo, hi)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img.astype(np.float32)


def preprocess(img: np.ndarray, img_size: int = 512) -> torch.Tensor:
    img = normalize(img)
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return t


def get_frame_thumbnail(path: Path, frame_index: int = 0, max_size: int = 256) -> np.ndarray:
    img = read_frame(path, frame_index)
    img = normalize(img)
    h, w = img.shape
    scale = max_size / max(h, w)
    if scale < 1.0:
        nh, nw = int(h*scale), int(w*scale)
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        img = t.squeeze().numpy()
    return img
