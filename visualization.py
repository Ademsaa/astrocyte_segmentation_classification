"""visualization.py — Result figures and frame-selection grid."""
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import config


def show_selection_grid(frames, frame_indices, tiff_name="", cols=5):
    n = len(frames)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis("off")
    for i, (img, idx) in enumerate(zip(frames, frame_indices)):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Frame {idx}", fontsize=9)
        axes[i].axis("off")
    title = f"Select a frame — {tiff_name}" if tiff_name else "Select a frame"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()


def build_overlay(pred, gt, background=None):
    p, g = pred.astype(bool), gt.astype(bool)
    overlay = np.zeros((*p.shape, 3), dtype=np.float32)
    overlay[..., 1] = p & g    # TP green
    overlay[..., 0] = p & ~g   # FP red
    overlay[..., 2] = ~p & g   # FN blue
    return overlay


def plot_results(raw_image, prob_maps, pred_masks,
                 class_names=config.CLASS_NAMES, title="", gt_masks=None):
    n_classes = len(class_names)
    n_cols = 4 if gt_masks is not None else 3
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(n_cols*4, n_classes*3.5))
    if n_classes == 1: axes = axes[np.newaxis, :]
    col_titles = ["Input image", "Probability map", "Prediction"]
    if gt_masks is not None: col_titles.append("TP / FP / FN")
    for c, name in enumerate(class_names):
        axes[c,0].imshow(raw_image, cmap="gray")
        axes[c,0].set_ylabel(name, fontsize=11, fontweight="bold")
        if c == 0: axes[c,0].set_title(col_titles[0], fontsize=10)
        axes[c,0].axis("off")
        im = axes[c,1].imshow(prob_maps[c], cmap=config.PROB_CMAP, vmin=0, vmax=1)
        if c == 0: axes[c,1].set_title(col_titles[1], fontsize=10)
        plt.colorbar(im, ax=axes[c,1], fraction=0.046, pad=0.04)
        axes[c,1].axis("off")
        axes[c,2].imshow(pred_masks[c], cmap="gray", vmin=0, vmax=1)
        if c == 0: axes[c,2].set_title(col_titles[2], fontsize=10)
        axes[c,2].axis("off")
        if gt_masks is not None:
            overlay = build_overlay(pred_masks[c], gt_masks[c])
            axes[c,3].imshow(raw_image, cmap="gray", alpha=0.4)
            axes[c,3].imshow(overlay, alpha=0.8)
            if c == 0: axes[c,3].set_title(col_titles[3], fontsize=10)
            axes[c,3].axis("off")
    if gt_masks is not None:
        fig.legend(handles=[
            mpatches.Patch(color="green", label="True Positive"),
            mpatches.Patch(color="red",   label="False Positive"),
            mpatches.Patch(color="blue",  label="False Negative"),
        ], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    if title: fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def save_results(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"[visualization] Saved → {output_path}")
