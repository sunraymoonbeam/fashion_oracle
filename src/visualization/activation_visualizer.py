import math
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..utils.image_utils import preprocess_image


class ActivationVisualizer:
    def __init__(self, base_model: tf.keras.Model):
        """
        Initialize the ActivationVisualizer with a base model.

        Args:
            base_model (tf.keras.Model): The pre-trained Keras model.
        """
        self.base_model = base_model

    def activate_single_layer(
        self, img_tensor: np.ndarray, layer_name: str, num_filters: int = 64
    ) -> plt.Figure:
        """
        Visualize the activations of a specific layer.

        Args:
            img_tensor (np.ndarray): Preprocessed image tensor.
            layer_name (str): Name of the layer to visualize.

        Returns:
            plt.Figure: Matplotlib figure with activation visualizations.
        """
        layer = self.base_model.get_layer(layer_name)
        model = keras.Model(inputs=self.base_model.input, outputs=layer.output)
        activations = model.predict(img_tensor)

        if activations.ndim == 4:
            total_num_filters = activations.shape[-1]
            num_filters_to_visualize = min(num_filters, total_num_filters)
            num_columns = 8
            num_rows = math.ceil(num_filters_to_visualize / num_columns)

            fig, axes = plt.subplots(
                num_rows, num_columns, figsize=(num_columns * 1.5, num_rows * 1.5)
            )
            for i in range(num_filters_to_visualize):
                row = i // num_columns
                column = i % num_columns
                ax = axes[row, column] if num_rows > 1 else axes[column]
                ax.imshow(activations[0, :, :, i], cmap="viridis")
                ax.axis("off")

            plt.suptitle(f"Layer: {layer_name}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            return fig
        else:
            print(f"Skipping layer {layer_name} as it does not have 4D activations.")
            return None

    def activate_multi_layer(
        self,
        img_path: str,
        layers_to_visualize: list,
        num_filters: int = 64,
        save_dir: str = "./reports/figures/visuals",
    ):
        """
        Visualize and save the activations.

        Args:
            img_path (str): Path to the image file.
            layers_to_visualize (list): List of layer names to visualize.
            save_dir (str): Directory to save the visualizations.
        """
        img_tensor = preprocess_image(img_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        os.makedirs(save_dir, exist_ok=True)
        for layer_name in layers_to_visualize:
            print(f"Visualizing layer: {layer_name}")
            activation_fig = self.activate_single_layer(
                img_tensor, layer_name, num_filters
            )
            if activation_fig is not None:
                activation_img_path = os.path.join(
                    save_dir, f"{img_name}_{layer_name}_activations.png"
                )
                activation_fig.savefig(activation_img_path)
                plt.close(activation_fig)
                print(
                    f"Saved activation visualization for layer {layer_name} at {activation_img_path}"
                )
