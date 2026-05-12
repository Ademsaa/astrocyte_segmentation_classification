#!/usr/bin/env python3
"""
run_inference.py — Main entry-point for astrocyte segmentation.

Usage examples
--------------
  python run_inference.py                                           # fully auto (finds .pth + creates output folder)
  python run_inference.py --tiff path/to/image.tif                 # pick frame interactively
  python run_inference.py --tiff path/to/image.tif --frame 3       # fully non-interactive
  python run_inference.py --folder path/to/folder                  # pick file + frame
  python run_inference.py --tiff img.tif --all-frames --save out/  # batch all frames to a specific folder
  python run_inference.py --tiff img.tif --checkpoint my.pth       # use a specific checkpoint
  python run_inference.py --no-show --tiff img.tif                 # save only, no GUI window
"""
import argparse, sys
from pathlib import Path
import matplotlib.pyplot as plt
import config
from path_utils import find_checkpoint, resolve_output_dir
from inference import load_model, predict
from postprocessing import postprocess, get_probability_maps
from preprocessing import (list_tiff_files, list_tiff_frames,
                            preprocess, read_frame, normalize, get_frame_thumbnail)
from visualization import plot_results, save_results, show_selection_grid


def _pick_tiff_from_folder(folder):
    files = list_tiff_files(folder)
    if not files:
        print(f"[ERROR] No .tif/.tiff files found in '{folder}'."); return None
    print("\nAvailable TIFF files:")
    for i, f in enumerate(files): print(f"  [{i}] {f.name}")
    while True:
        raw = input("\nEnter file number (or 'q' to quit): ").strip()
        if raw.lower() == "q": return None
        try:
            idx = int(raw)
            if 0 <= idx < len(files): return files[idx]
        except ValueError: pass
        print(f"  Please enter a number between 0 and {len(files)-1}.")


def _pick_frame(tiff_path, show_thumbnails=True):
    frame_indices = list_tiff_frames(tiff_path)
    n = len(frame_indices)
    if n == 1:
        print("[info] Single-frame TIFF — using frame 0."); return 0
    print(f"\n'{tiff_path.name}' has {n} frame(s): {frame_indices[0]} … {frame_indices[-1]}")
    if show_thumbnails:
        print("[info] Loading thumbnails …")
        thumbs = [get_frame_thumbnail(tiff_path, i) for i in frame_indices]
        show_selection_grid(thumbs, frame_indices, tiff_name=tiff_path.name)
    while True:
        raw = input(f"\nEnter frame index [0–{n-1}] (or 'q' to quit): ").strip()
        if raw.lower() == "q": return None
        try:
            idx = int(raw)
            if idx in frame_indices: return idx
        except ValueError: pass
        print(f"  Please enter a number from {frame_indices}.")


def segment_frame(model, tiff_path, frame_index, output_dir, show=True):
    print(f"\n[inference] Processing '{tiff_path.name}' frame {frame_index} …")
    raw      = read_frame(tiff_path, frame_index)
    img_norm = normalize(raw)
    tensor   = preprocess(raw, img_size=config.IMG_SIZE)
    logits   = predict(model, tensor)

    prob_maps  = get_probability_maps(logits)
    pred_masks = postprocess(logits,
                             thresholds=config.THRESHOLDS,
                             min_soma_area=config.MIN_SOMA_AREA,
                             min_process_area=config.MIN_PROCESS_AREA)

    print("\n  Predicted positive pixel fractions:")
    for name, mask in zip(config.CLASS_NAMES, pred_masks):
        print(f"    {name:12s}: {mask.mean()*100:.2f}%")

    fig = plot_results(img_norm, prob_maps, pred_masks,
                       title=f"{tiff_path.stem}  |  frame {frame_index}")

    # ── Auto-save result next to checkpoint ──────────────────────────────────
    out_file = output_dir / f"{tiff_path.stem}_frame{frame_index:04d}_result.png"
    save_results(fig, out_file)

    if show: plt.show()
    else:    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Astrocyte segmentation — U-Net-DC.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--tiff",   type=Path, metavar="PATH",
                     help="Path to a single TIFF file.")
    src.add_argument("--folder", type=Path, metavar="DIR",
                     help="Folder of TIFF files (interactive pick).")
    p.add_argument("--frame",        type=int,  default=None,
                   help="Frame index to process (skips interactive pick).")
    p.add_argument("--all-frames",   action="store_true",
                   help="Process every frame in the TIFF.")
    p.add_argument("--checkpoint",   type=Path, default=None, metavar="PATH",
                   help="Path to a .pth checkpoint (overrides config + auto-detect).")
    p.add_argument("--save",         type=Path, default=None, metavar="DIR",
                   help="Output directory for PNG results (overrides auto location).")
    p.add_argument("--no-show",      action="store_true",
                   help="Do not open a GUI window — only save files.")
    p.add_argument("--no-thumbnails",action="store_true",
                   help="Skip the frame thumbnail grid.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Resolve checkpoint (auto-detect if not specified) ─────────────────
    if args.checkpoint:
        # CLI flag overrides everything
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_path}"); sys.exit(1)
    else:
        ckpt_path = find_checkpoint()   # auto-search cwd recursively

    # ── 2. Resolve output directory (auto-create next to checkpoint) ─────────
    output_dir = resolve_output_dir(ckpt_path, cli_save=args.save)

    # ── 3. Resolve TIFF source ───────────────────────────────────────────────
    tiff_path = None
    if args.tiff:
        tiff_path = args.tiff
        if not tiff_path.exists():
            print(f"[ERROR] File not found: {tiff_path}"); sys.exit(1)
    elif args.folder:
        folder = args.folder
        if not folder.is_dir():
            print(f"[ERROR] Not a directory: {folder}"); sys.exit(1)
        tiff_path = _pick_tiff_from_folder(folder)
    else:
        raw = input("Enter path to a TIFF file or folder: ").strip()
        p   = Path(raw)
        tiff_path = _pick_tiff_from_folder(p) if p.is_dir() else (p if p.is_file() else None)
        if tiff_path is None:
            print(f"[ERROR] Path not found: {p}"); sys.exit(1)
    if tiff_path is None:
        print("Exiting."); sys.exit(0)

    # ── 4. Load model ────────────────────────────────────────────────────────
    model = load_model(checkpoint_path=ckpt_path)
    show  = not args.no_show

    # ── 5. Run inference ─────────────────────────────────────────────────────
    if args.all_frames:
        frames = list_tiff_frames(tiff_path)
        print(f"[info] Processing all {len(frames)} frame(s) …")
        for fi in frames:
            segment_frame(model, tiff_path, fi, output_dir=output_dir, show=show)
    else:
        fi = args.frame if args.frame is not None else \
             _pick_frame(tiff_path, show_thumbnails=not args.no_thumbnails)
        if fi is None:
            print("Exiting."); sys.exit(0)
        segment_frame(model, tiff_path, fi, output_dir=output_dir, show=show)

    print(f"\n[done] Results saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
