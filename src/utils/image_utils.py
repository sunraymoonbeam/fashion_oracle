from typing import Union, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import os


def preprocess_image(
    img_input: Union[str, Image.Image], target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Load and preprocess an image from a file path or preprocess an image array.

    Args:
        img_input (Union[str, Image.Image]): Path to the image file or an image array.
        target_size (Tuple[int, int]): Target size for the image (height, width).

    Returns:
        np.ndarray: Preprocessed image tensor with shape (1, target_size[0], target_size[1], 3).
    """
    if isinstance(img_input, str):
        # Load image from file path
        img = keras.utils.load_img(img_input, target_size=target_size)
        img_array = keras.utils.img_to_array(img)
    elif isinstance(img_input, Image.Image):
        # Preprocess image array
        img_array = np.array(img_input)  # Convert PIL image to numpy array
        img_array = tf.image.resize(img_array, target_size)
    else:
        raise ValueError(
            "img_input must be a file path (str) or an image array (Image.Image)"
        )

    # Add batch dimension
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor


def create_tf_datasets(
    train_dir_path: str,
    val_dir_path: str | None,
    test_dir_path: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> tuple[tf.data.Dataset, tf.data.Dataset | None, tf.data.Dataset]:
    """
    Converts folders of images into TensorFlow datasets.

    Args:
        train_dir_path (str): Path to the training dataset directory.
        val_dir_path (str | None): Path to the validation dataset directory. None if no validation data.
        test_dir_path (str): Path to the testing dataset directory.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset | None, tf.data.Dataset]: Training, validation (None if val_dir_path is None), and testing datasets.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir_path,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = None
    if val_dir_path:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir_path,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=batch_size,
        )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir_path,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
    )

    # Extract class names
    train_class_names = train_ds.class_names
    val_class_names = val_ds.class_names if val_ds else train_class_names
    test_class_names = test_ds.class_names

    assert (
        train_class_names == val_class_names == test_class_names
    ), "Class names do not match!"

    # Save class names to a JSON file
    class_names_dict = {i: name for i, name in enumerate(train_class_names)}
    os.makedirs("conf", exist_ok=True)
    with open("./conf/class_mapping.json", "w") as f:
        json.dump(class_names_dict, f)

    return train_ds, val_ds, test_ds
