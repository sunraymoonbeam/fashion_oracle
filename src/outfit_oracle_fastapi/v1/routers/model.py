import os
import logging
import fastapi
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
from tensorflow.keras.models import Model
import math

import outfit_oracle_fastapi as tfcv_fapi
from outfit_oracle.visualization.gradcam import GradCAMVisualizer
from outfit_oracle.visualization.activation_visualizer import ActivationVisualizer
from outfit_oracle.visualization.gradient_ascent_filter import GradientAscentVisualizer

# Set Matplotlib backend to 'Agg'
import matplotlib

matplotlib.use("Agg")

from fastapi import Form

logger = logging.getLogger(__name__)

# Load class mappings from JSON file
with open("../conf/class_names.json", "r") as f:
    CLASS_MAPPINGS = json.load(f)

ROUTER = fastapi.APIRouter()
PRED_MODEL = tfcv_fapi.deps.PRED_MODEL
DEVICE = tfcv_fapi.deps.DEVICE

# Initialize visualizers with the pre-trained model
base_model = tf.keras.applications.EfficientNetB0(weights="imagenet")
gradcam_visualizer = GradCAMVisualizer(base_model)
activation_visualizer = ActivationVisualizer(base_model)
gradient_ascent_visualizer = GradientAscentVisualizer(
    base_model, img_width=224, img_height=224
)


@ROUTER.post("/predict", status_code=fastapi.status.HTTP_200_OK)
def classify_image(image_file: fastapi.UploadFile):
    """Endpoint that returns classification of an image."""
    result_dict = {"data": []}

    try:
        logger.info("Classifying image...")

        contents = image_file.file.read()
        with open(image_file.filename, "wb") as buffer:
            buffer.write(contents)
        image = Image.open(image_file.filename)
        image = image.resize((224, 224))  # Resize image to match model input size
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        with tf.device(DEVICE):
            output = PRED_MODEL.predict(image)
        pred = np.argmax(output, axis=1)
        pred_index = int(pred[0])
        pred_label = CLASS_MAPPINGS[str(pred_index)]
        pred_score = float(output[0][pred_index])

        result_dict["data"].append(
            {
                "image_filename": image_file.filename,
                "prediction": pred_label,
                "score": pred_score,
            }
        )
        logger.info(
            "Prediction for image filename %s: %s with score %f",
            image_file.filename,
            pred_label,
            pred_score,
        )

    except Exception as error:
        logger.error(error)
        raise fastapi.HTTPException(status_code=500, detail="Internal server error.")

    finally:
        image_file.file.close()
        os.remove(image_file.filename)

    return result_dict


@ROUTER.post("/visualize", status_code=fastapi.status.HTTP_200_OK)
def visualize_image(image_file: fastapi.UploadFile, layer_name: str = Form(...)):
    """Endpoint that returns filter visualization, Grad-CAM images, and activation visualizations."""
    try:
        logger.info("Visualizing image...")

        contents = image_file.file.read()
        with open(image_file.filename, "wb") as buffer:
            buffer.write(contents)
        img_tensor = gradcam_visualizer.get_img_tensor(image_file.filename)

        # Generate filter visualization
        filter_fig = gradient_ascent_visualizer.visualize_and_save_filters(
            [layer_name], num_filters=64, save_dir="../reports/figures/gradcam_fastapi"
        )
        if filter_fig is not None:
            filter_img_path = (
                f"../reports/figures/gradcam_fastapi/{layer_name}_filters.png"
            )
            filter_fig.savefig(filter_img_path)
            plt.close(filter_fig)
            logger.info(
                f"Saved filter visualization for layer {layer_name} at {filter_img_path}"
            )

        # Generate Grad-CAM visualization
        heatmap, gradcam_img = gradcam_visualizer.visualize_and_save_gradcam(
            image_file.filename,
            [layer_name],
            save_dir="../reports/figures/gradcam_fastapi",
        )
        gradcam_img_path = (
            f"../reports/figures/gradcam_fastapi/{layer_name}_gradcam.png"
        )
        gradcam_img.save(gradcam_img_path)
        logger.info(
            f"Saved Grad-CAM visualization for layer {layer_name} at {gradcam_img_path}"
        )

        # Generate activation visualization
        activation_fig = activation_visualizer.visualize_and_save_activations(
            img_tensor, [layer_name], save_dir="../reports/figures/gradcam_fastapi"
        )
        if activation_fig is not None:
            activation_img_path = (
                f"../reports/figures/gradcam_fastapi/{layer_name}_activations.png"
            )
            activation_fig.savefig(activation_img_path)
            plt.close(activation_fig)
            logger.info(
                f"Saved activation visualization for layer {layer_name} at {activation_img_path}"
            )

        return {
            "filter_visualization": filter_img_path,
            "gradcam_visualization": gradcam_img_path,
            "activation_visualization": activation_img_path,
        }

    except Exception as error:
        logger.error(error)
        raise fastapi.HTTPException(status_code=500, detail="Internal server error.")

    finally:
        image_file.file.close()
        os.remove(image_file.filename)


@ROUTER.get("/version", status_code=fastapi.status.HTTP_200_OK)
def model_version():
    """Get version (UUID) of predictive model used for the API."""
    return {"data": {"model_uuid": tfcv_fapi.config.SETTINGS.PRED_MODEL_UUID}}
