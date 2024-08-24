from typing import Union, Tuple, Dict
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from ..segmentation.segmentation import ImageSegmenter
from ..utils.model_utils import predict
from ..utils.image_utils import preprocess_image
from ..visualization.gradcam import GradCAM


def get_class_mapping(json_path: str) -> Dict[int, str]:
    """
    Load class indices from a JSON file.

    Args:
        json_path (str): The path to the JSON file containing class indices.

    Returns:
        Dict[int, str]: A dictionary mapping class indices to class names.
    """
    with open(json_path, "r") as f:
        class_indices = json.load(f)
    return class_indices


def run_streamlit_inference(
    image: Image.Image,
    segmentation_model: ImageSegmenter,
    classifier_model: tf.keras.Model,
    gradcam_model: GradCAM,
    class_mapping: Dict[str, str] = {"0": "formal", "1": "informal"},
    layer_name: str = "top_activation",
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[str, float, Image.Image, Image.Image]:
    """
    Run inference on the given image.

    Args:
        image (Image.Image): The input image for inference.
        segmentation_model (ImageSegmenter): The segmentation model.
        classifier_model (tf.keras.Model): The classification model.
        gradcam_model (GradCAM): The Grad-CAM visualizer.
        class_indices (Dict[int, str]): The class indices.
        layer_name (str, optional): The name of the layer to use for Grad-CAM, by default "top_activation".

    Returns:
        Tuple[str, float, Image.Image, Image.Image]: The predicted class name, predicted probability, segmented image, and Grad-CAM visualization.
    """
    segmented_image = segmentation_model.segment_image(image)
    processed_img_tensor = preprocess_image(segmented_image, target_size)

    predicted_class_idx, predicted_class_name, predicted_probability = predict(
        classifier_model, processed_img_tensor, class_mapping
    )
    gradcam_fig = gradcam_model.gradcam_single_layer(
        processed_img_tensor, layer_name, predicted_class_idx
    )
    return predicted_class_name, predicted_probability, segmented_image, gradcam_fig
