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

    # Initialize HOG descriptor and set the SVM detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

            # Run HOG inference only every 30 frames
            if frame_counter % 30 == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                humans, _ = hog.detectMultiScale(gray_frame, winStride=(8, 8))

                # Draw bounding boxes around detected humans
                for x, y, w, h in humans:
                    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the image with bounding boxes
                frame_placeholder.image(frame_rgb)

                # Capture a snapshot only if humans are detected
                if len(humans) > 0:
                    snapshot_counter += 1
                    for x, y, w, h in humans:
                        human_crop = frame_rgb[y : y + h, x : x + w]
                        snapshot = Image.fromarray(human_crop)
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

                        # Display inference results in a grid
                        st.subheader(
                            f"Inference Results (Snapshot #{snapshot_counter}):"
                        )
                        st.markdown("---")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(frame_rgb, caption="Original with Bounding Box")
                        with col2:
                            st.image(snapshot, caption="Cropped Frame")
                        with col3:
                            st.image(segmented_img, caption="Segmented Image")
                        st.pyplot(gradcam_fig)

                        # Determine the color and label based on the predicted class
                        if predicted_class_name.lower() == "formal":
                            label_color = "green"
                            label_text = "FORMAL"
                        else:
                            label_color = "red"
                            label_text = "INFORMAL"

                        # Create the styled HTML content
                        html_content = f"""
                        <div style="display: flex; justify-content: space-around; align-items: center; margin-top: 20px;">
                            <div style="text-align: center; font-size: 48px; color: {label_color};">
                                {label_text}
                            </div>
                            <div style="text-align: center; font-size: 48px;">
                                {predicted_probability:.2f}
                            </div>
                        </div>
                        """

                        # Display the styled content using st.markdown
                        st.markdown(html_content, unsafe_allow_html=True)
                        st.write(f"Snapshot Time: {snapshot_time}")

            # Release the frame to free memory
            del frame, frame_rgb

    finally:
        camera.release()  # Release the camera resource
        cv2.destroyAllWindows()  # Close any OpenCV windows
        cv2.waitKey(1)  # Wait for a key press to close the window
        st.write("Stopped")


if __name__ == "__main__":
    # Clear Hydra's global state before initializing
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    main()
