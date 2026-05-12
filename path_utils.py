"""
path_utils.py — Auto-detect checkpoint and auto-create output directory.
"""
from pathlib import Path
from typing import Optional
import config


def find_checkpoint(search_root: Optional[Path] = None) -> Path:
    """
    Returns a checkpoint Path using this priority order:
      1. config.CHECKPOINT_PATH  (if set and the file exists)
      2. --checkpoint CLI flag   (passed in as `override`)
      3. Auto-search: recursively scan `search_root` (default = cwd)
         for *.pth files; prefer files whose name contains
         'best', 'unetdc', or '512'; pick the most recently modified.
    Raises FileNotFoundError if nothing is found.
    """
    # 1. Pinned in config
    if config.CHECKPOINT_PATH is not None:
        p = Path(config.CHECKPOINT_PATH)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"[config] CHECKPOINT_PATH points to a missing file:\n  {p}\n"
            "Update CHECKPOINT_PATH in config.py."
        )

    # 3. Auto-search
    root = Path(search_root) if search_root else Path.cwd()
    candidates = list(root.rglob("*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"[auto-detect] No .pth files found under '{root}'.\n"
            "Either:\n"
            "  • Copy your checkpoint here, or\n"
            "  • Set CHECKPOINT_PATH in config.py, or\n"
            "  • Pass --checkpoint /path/to/model.pth on the command line."
        )

    def _score(p: Path) -> int:
        name = p.name.lower()
        return sum([
            "best"   in name,
            "unetdc" in name,
            "512"    in name,
        ])

    # Sort: highest score first, then most recently modified
    candidates.sort(key=lambda p: (_score(p), p.stat().st_mtime), reverse=True)
    chosen = candidates[0]
    print(f"[auto-detect] Found checkpoint: {chosen}")
    if len(candidates) > 1:
        print(f"[auto-detect] Other candidates ({len(candidates)-1}):")
        for c in candidates[1:4]:
            print(f"              {c}")
    return chosen


def resolve_output_dir(checkpoint_path: Path, cli_save: Optional[Path] = None) -> Path:
    """
    Returns the output directory using this priority order:
      1. CLI --save flag         (cli_save argument)
      2. config.OUTPUT_DIR       (if set)
      3. Auto: <checkpoint_parent>/inference_results/
    The directory is created if it does not yet exist.
    """
    if cli_save is not None:
        out = Path(cli_save)
    elif config.OUTPUT_DIR is not None:
        out = Path(config.OUTPUT_DIR)
    else:
        # Place results next to the checkpoint file
        out = checkpoint_path.parent / "inference_results"

    out.mkdir(parents=True, exist_ok=True)
    print(f"[output] Results will be saved to: {out.resolve()}")
    return out
