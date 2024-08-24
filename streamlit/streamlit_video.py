import time
from datetime import datetime
import base64
import cv2
import hydra
from hydra.core.global_hydra import GlobalHydra
import streamlit as st
from omegaconf import DictConfig
from PIL import Image
import io
import tensorflow as tf
from outfit_oracle.utils.streamlit_utils import (
    get_class_mapping,
    segment_and_preprocess_image,
    load_segmentation_model,
    load_classification_model,
    load_gradcam_model,
    run_inference,
)


@hydra.main(version_base=None, config_path="../conf", config_name="streamlit.yaml")
def main(cfg: DictConfig) -> None:
    st.title("Webcam Live Feed with Inference")

    # Load models and class mappings
    segmentation_model = load_segmentation_model(cfg.segmentation_model_path)
    classifier_model, _ = load_classification_model(cfg.classification_model_path)
    base_model = tf.keras.applications.EfficientNetB0(weights="imagenet")
    gradcam_model = load_gradcam_model(base_model)
    class_indices = get_class_mapping(cfg.class_mapping_path)

    run = st.checkbox("Run")  # Checkbox to start/stop the webcam feed
    camera = cv2.VideoCapture(
        0
    )  # Use the first camera device (0 for iPhone camera, 1 for macbook camera)
    frame_placeholder = st.empty()
    snapshot_counter = 0
    frame_counter = 0

    try:
        while run:
            ret, frame = (
                camera.read()
            )  # ret is a boolean that returns true if the frame is available, and frame is the image
            if not ret:
                st.error(
                    "Failed to capture frame from camera. Please check your camera."
                )
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb)

            # Increment the frame counter
            frame_counter += 1

            # Capture a snapshot every 100 frames
            if frame_counter % 100 == 0:
                snapshot_counter += 1
                snapshot = Image.fromarray(frame_rgb)
                (
                    predicted_class_name,
                    predicted_probability,
                    segmented_img,
                    gradcam_fig,
                ) = run_inference(
                    snapshot,
                    segmentation_model,
                    classifier_model,
                    gradcam_model,
                    class_indices,
                    cfg.layer_to_visualize,
                )

                # Gather meta information
                snapshot_time = datetime.now().strftime("%H:%M:%S")
                resolution = f"{frame.shape[1]}x{frame.shape[0]}"

                # Display inference results
                st.write(f"Inference Results (Snapshot #{snapshot_counter}):")
                st.image(snapshot, caption="Snapshot")
                st.image(segmented_img, caption="Segmented Image")
                st.write(f"Class: {predicted_class_name}")
                st.write(f"Confidence: {predicted_probability:.2f}")
                st.write(f"Time: {snapshot_time}")
                st.write(f"Resolution: {resolution}")
                st.pyplot(gradcam_fig)

            # Release the frame to free memory
            del frame, frame_rgb

    finally:
        camera.release()  # Release the camera resource
        cv2.destroyAllWindows()  # Close any OpenCV windows
        st.write("Stopped")


if __name__ == "__main__":
    # Clear Hydra's global state before initializing
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    main()
