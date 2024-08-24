import logging
from typing import List
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    A factory class for creating and building models using pre-trained base models.

    Attributes:
        models (dict): A dictionary mapping model names to their corresponding pre-trained models.

    Methods:
        list_available_models(): Lists the names of all available models in the factory.
        create_base_model(base_model_name, input_shape, include_top, weights): Creates a base model from the available models in the factory.
        build_model(base_model_name, input_shape, num_classes): Builds a model with a pre-trained base and a classifier head.
    """

    def __init__(self):
        self.models = {
            "VGG16": tf.keras.applications.VGG16,
            "InceptionV3": tf.keras.applications.InceptionV3,
            "ResNet50": tf.keras.applications.ResNet50,
            "MobileNetV2": tf.keras.applications.MobileNetV2,
            "EfficientNetB0": tf.keras.applications.EfficientNetB0,
            "DenseNet121": tf.keras.applications.DenseNet121,
            "NASNetMobile": tf.keras.applications.NASNetMobile,
        }

    def list_available_models(self) -> List[str]:
        """
        Lists the names of all available models in the factory.

        Returns:
            List[str]: A list of available model names.
        """
        list_of_models = list(self.models.keys())
        logger.info(f"Available models in the factory: {list_of_models}")
        return list_of_models

    def create_base_model(
        self,
        base_model_name: str,
        input_shape: Tuple[int, int, int],
        include_top: bool = False,
        weights: str = "imagenet",
    ) -> tf.keras.Model:
        """
        Creates a base model from the available models in the factory.

        Args:
            base_model_name (str): Name of the model to be created.
            input_shape (Tuple[int, int, int]): Shape of the input data.
            include_top (bool, optional): Whether to include the top layer of the model. Defaults to False.
            weights (str, optional): Pre-trained weights to be loaded. Defaults to "imagenet".

        Raises:
            ValueError: If the model name is not available in the factory.

        Returns:
            tf.keras.Model: The base model.
        """
        if base_model_name in self.models:
            base_model = self.models[base_model_name](
                input_shape=input_shape, include_top=include_top, weights=weights
            )
        else:
            raise ValueError(f"Model {base_model_name} not available in the factory.")
        return base_model

    def build_model(self, base_model_name, input_shape, num_classes):
        """
        Builds and returns a Keras model with a specified base model.

        Args:
            base_model_name (str): Name of the base model to use.
            input_shape (tuple): Shape of the input data.
            num_classes (int): Number of output classes.

        Returns:
            tf.keras.Model: The complete model.
        """
        # Build the base model
        base_model = self.create_base_model(
            base_model_name, input_shape, include_top=False
        )

        # Get the output of the base model
        x = base_model.output

        # Add additional layers
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate=0.2)(x)
        outputs = Dense(
            num_classes, activation="softmax" if num_classes > 1 else "sigmoid"
        )(x)

        # Create the final model
        model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)

        logger.info(
            f"Classifier model with pretrained {base_model_name}'s weights successfully initialized."
        )
        return model
