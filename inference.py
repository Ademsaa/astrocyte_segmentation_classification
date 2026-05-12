"""inference.py — Model loading and forward pass."""
from pathlib import Path
from typing import Union
import torch
from model import UNetDC512
import config


def load_model(checkpoint_path: Union[str, Path] = config.CHECKPOINT_PATH,
               device: Union[str, torch.device, None] = None) -> UNetDC512:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}\nUpdate CHECKPOINT_PATH in config.py.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model = UNetDC512(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES,
                      base=config.BASE_CHANNELS).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[inference] Loaded: {checkpoint_path}")
    print(f"[inference] Device: {device}")
    return model


@torch.inference_mode()
def predict(model: UNetDC512, img_tensor: torch.Tensor,
            device: Union[str, torch.device, None] = None) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    img_tensor = img_tensor.to(device)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = model(img_tensor)
    return logits
