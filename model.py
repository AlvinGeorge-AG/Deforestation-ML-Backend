"""
model.py — Loads the U-Net deforestation detection model once at startup.

Architecture (V3):
  - Encoder: ResNet-34 (no pretrained weights — we load our own)
  - Input:   8 channels (B2, B3, B4, B8 × 2 years)
  - Output:  1 channel  (binary deforestation mask, logits)
  - Trained with BCEWithLogitsLoss on Dynamic World labels
"""

import torch
import segmentation_models_pytorch as smp


def load_model(path: str = "best_model.pth") -> torch.nn.Module:
    """
    Load and return the trained U-Net model in eval mode.

    The V3 model expects 8 input channels:
      [B2_before, B3_before, B4_before, B8_before,
       B2_after,  B3_after,  B4_after,  B8_after]

    Args:
        path: Path to the saved state dict (.pth file).

    Returns:
        torch.nn.Module ready for inference.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # we load our own trained weights
        in_channels=8,          # V3: 4 bands × 2 years
        classes=1,
    )
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model
