from typing import Dict
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


def load_classification_model(
    checkpoint_path: str = "models/EfficientNetB0.keras",
    use_cuda: bool = False,
    use_mps: bool = False,
) -> Tuple[tf.keras.Model, str]:
    """
    Load a Keras model from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str, optional
        The path to the Keras model file, by default "models/EfficientNetB0.keras".
    use_cuda : bool, optional
        Whether to use CUDA (GPU) for loading the model, by default False.
    use_mps : bool, optional
        Whether to use MPS (Apple Silicon) for loading the model, by default False.

    Returns
    -------
    Tuple[tf.keras.Model, str]
        The loaded Keras model and the device used.
    """
    if use_cuda:
        device = "gpu"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model from: {checkpoint_path}")

    with tf.device(device):
        model = tf.keras.models.load_model(checkpoint_path)

    return model, device


def predict(
    model,
    img_array: Image,
    class_mapping: Dict[int, str],
) -> Tuple[int, str, float]:
    """
    Predicts the class of an image using a given model.

    Args:
        model: The model used for prediction.
        img_array (Image): The image array to be predicted.
        class_mapping (Dict[int, str]): A dictionary mapping class indices to class names.

    Returns:
        Tuple[int, str, float]: A tuple containing the predicted class index,
                                predicted class name, and predicted probability.
    """
    # Predict the class
    predictions = model.predict(
        img_array
    )  # A list of predictions for each image in the batch
    predicted_class_index = np.argmax(
        predictions[0]
    )  # Index of class with highest probability
    predicted_class_name = class_mapping[
        predicted_class_index
    ]  # Class name corresponding to the index
    predicted_probability = predictions[0][
        predicted_class_index
    ]  # Probability of the predicted class

    return predicted_class_index, predicted_class_name, predicted_probability
