import os
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..utils.image_utils import preprocess_image


class GradCAM:
    def __init__(self, model: tf.keras.Model):
        """
        Initialize the GradCAMVisualizer with a base model.

        Args:
            model (tf.keras.Model): The pre-trained Keras model.
        """
        self.model = model

    def build_grad_model(self, layer_name: str) -> keras.models.Model:
        """
        Build a model that outputs the activations of a specific layer and the model's predictions.

        Args:
            layer_name (str): Name of the convolutional layer.

        Returns:
            keras.models.Model: A model that outputs the layer activations and predictions.
        """
        return keras.models.Model(
            self.model.inputs,
            [
                self.model.get_layer(layer_name).output,
                self.model.output,
            ],
        )

    def generate_heatmap(
        self,
        grad_model: keras.models.Model,
        img_tensor: np.ndarray,
        pred_index: int = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            grad_model (keras.models.Model): Model that outputs layer activations and predictions.
            img_tensor (np.ndarray): Preprocessed image tensor.
            pred_index (int, optional): Index of the predicted class.

        Returns:
            np.ndarray: Grad-CAM heatmap.
        """
        with tf.GradientTape() as tape:
            layer_output, preds = grad_model(img_tensor)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        layer_output = layer_output[0]
        heatmap = layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(
        self, img_tensor: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6
    ) -> plt.Figure:
        """
        Generate and visualize the Grad-CAM heatmap.

        Args:
            img_tensor (np.ndarray): Preprocessed image tensor.
            heatmap (np.ndarray): Grad-CAM heatmap.
            alpha (float): Transparency factor for the heatmap.

        Returns:
            plt.Figure: Matplotlib figure with Grad-CAM visualizations.
        """
        img_tensor = img_tensor[0]
        heatmap = np.uint8(255 * heatmap)

        # Apply the JET colormap to the heatmap
        jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
        original_heatmap = jet_heatmap.copy()

        # Resize the heatmap to match the input image size
        jet_heatmap = cv2.resize(
            jet_heatmap, (img_tensor.shape[1], img_tensor.shape[0])
        )

        # Ensure both images have the same number of channels
        if img_tensor.shape[-1] == 1:
            img_tensor = np.repeat(img_tensor, 3, axis=-1)

        # Convert both images to float32 for blending
        jet_heatmap = np.float32(jet_heatmap)
        img_tensor = np.float32(img_tensor)

        # Superimpose the heatmap on the original image
        superimposed_img = cv2.addWeighted(img_tensor, 1 - alpha, jet_heatmap, alpha, 0)
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_heatmap / 255.0)
        axes[0].set_title("Heatmap")
        axes[0].axis("off")
        axes[1].imshow(superimposed_img)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        plt.close(fig)
        return fig

    def gradcam_single_layer(
        self, img_input: Union[str, np.ndarray], layer_name: str, pred_index: int = None
    ) -> plt.Figure:
        """
        Return Grad-CAM visualization for a single layer.

        Args:
            img_input (Union[str, np.ndarray]): Path to the image file or an image tensor.
            layer_name (str): Name of the convolutional layer.
            pred_index (int): Index of the predicted class.

        Returns:
            plt.Figure: Matplotlib figure with Grad-CAM visualizations.
        """
        if isinstance(img_input, str):
            img_tensor = preprocess_image(img_input)
        elif isinstance(img_input, np.ndarray):
            img_tensor = img_input
        else:
            raise ValueError(
                "img_input must be a file path (str) or an image tensor (np.ndarray)"
            )

        grad_model = self.build_grad_model(layer_name)
        heatmap = self.generate_heatmap(grad_model, img_tensor, pred_index)
        gradcam_fig = self.overlay_heatmap(img_tensor, heatmap)
        return gradcam_fig

    def gradcam_multi_layer(
        self,
        img_path: str,
        layers: List[str],
        save_dir: str = "./reports/figures/visuals",
    ):
        """
        Visualize and save the Grad-CAM visualizations for multiple layers.

        Args:
            img_path (str): Path to the image file.
            layers (List[str]): List of convolutional layer names.
            save_dir (str): Directory to save the visualizations.
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        os.makedirs(save_dir, exist_ok=True)

        for layer_name in layers:
            print(f"Visualizing layer: {layer_name}")
            gradcam_fig = self.gradcam_single_layer(img_path, layer_name)
            save_img_path = os.path.join(
                save_dir, f"{img_name}_{layer_name}_gradcam.jpg"
            )
            gradcam_fig.savefig(save_img_path)
            print(
                f"Saved Grad-CAM visualization for layer {layer_name} at {save_img_path}"
            )
