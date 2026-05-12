# Astrocyte Segmentation — Inference Pipeline

## Quick start
1. `pip install -r requirements.txt`
2. Edit `config.py` → set `CHECKPOINT_PATH` to your `.pth` file
3. `python run_inference.py --tiff your_image.tif`

## All run options
| Command | What it does |
|---------|-------------|
| `python run_inference.py` | Fully interactive |
| `python run_inference.py --tiff img.tif` | Pick frame interactively |
| `python run_inference.py --tiff img.tif --frame 0` | Fully non-interactive |
| `python run_inference.py --folder my_folder/` | Pick file + frame |
| `python run_inference.py --tiff img.tif --all-frames --save out/ --no-show` | Batch all frames |
