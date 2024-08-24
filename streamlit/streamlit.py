import logging
import os
import json

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

import streamlit as st
from outfit_oracle.utils import general_utils
from outfit_oracle.visualization.activation_visualizer import ActivationVisualizer
from outfit_oracle.visualization.gradcam import GradCAMVisualizer
from outfit_oracle.visualization.guided_bp import GuidedBackpropVisualizer
from torchvision import transforms
import numpy as np
from outfit_oracle.utils.model_utils import load_model, predict, grad_cam
from segmentation.segmentation import ImageSegmentation


@hydra.main(version_base=None, config_path="../conf", config_name="visualize.yaml")
def main(cfg: DictConfig) -> None:
    st.title("Image Classifier App")

    # Load model from the path specified in config
    model, _ = load_model(cfg.model_path, use_cuda=False, use_mps=False)

    # Load classes from JSON file specified in config
    classes = load_classes(cfg.class_names_path)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_image = preprocess_image(image)
        st.image(processed_image, caption="Segmented Image", use_column_width=True)

        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                predicted_class_idx, predicted_class_name, predicted_probability = (
                    predict(model, processed_image, classes)
                )

            # Display results
            st.subheader("Classification Results:")
            st.write(f"Class: {predicted_class_name}")
            st.write(f"Confidence: {predicted_probability:.2f}")

        if st.button("Generate Grad-CAM"):
            with st.spinner("Generating Grad-CAM..."):
                gradcam_fig = grad_cam(model, image, layer_name=cfg.last_conv_layer)
                st.pyplot(gradcam_fig)


if __name__ == "__main__":
    main()
