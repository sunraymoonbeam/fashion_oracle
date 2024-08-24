import os
import logging
from collections import OrderedDict
from typing import List

import gdown
import torch
from torch.nn import Module

from .u2net_architecture import U2NET

logger = logging.getLogger(__name__)


def download_segmentation_model(
    download_url: str = "https://drive.google.com/uc?id=11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY",
    save_checkpoint_path: str = "models/segmentation/u2net.pth",
) -> None:
    """
    Download the segmentation model checkpoint if it does not exist.

    Args:
        download_url (str): URL to download the model checkpoint.
        save_checkpoint_path (str): Path to save the downloaded checkpoint.
    """
    if not os.path.exists(save_checkpoint_path):
        os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)
        gdown.download(download_url, save_checkpoint_path, quiet=False)
        logger.info(f"Model downloaded successfully from {download_url}")
    else:
        logger.info("Model already exists.")


def load_model_checkpoint(
    model: Module, checkpoint_path: str = "models/segmentation/u2net.pth"
) -> Module:
    """
    Load model weights from a checkpoint.

    Args:
        model (Module): The model to load weights into.
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        Module: The model with loaded weights.
    """
    if not os.path.exists(checkpoint_path):
        logger.error(
            f"No model checkpoints at given path {checkpoint_path}, please download the model first."
        )
        return model

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = {
        k[7:]: v for k, v in checkpoint.items()
    }  # remove `module.` prefix in keys
    model.load_state_dict(new_state_dict)
    return model


def load_segmentation_model(
    checkpoint_path: str = "models/segmentation/u2net.pth", device: str = "cpu"
) -> Module:
    """
    Load and prepare the segmentation model.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (str): Device to load the model on (e.g., 'cpu' or 'cuda').

    Returns:
        Module: The loaded and prepared segmentation model.
    """
    model = U2NET(in_ch=3, out_ch=4)
    download_segmentation_model(save_checkpoint_path=checkpoint_path)
    model = load_model_checkpoint(model, checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def get_class_seg_palette(num_cls: int) -> List[int]:
    """
    Returns the color map for visualizing the segmentation mask.

    Args:
        num_cls (int): Number of classes.

    Returns:
        List[int]: The color palette.
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette
