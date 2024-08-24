import logging
import os

import hydra
import tensorflow as tf
from omegaconf import DictConfig

from src.visualization.activation_visualizer import ActivationVisualizer
from src.visualization.gradcam import GradCAM
from src.utils import general_utils


@hydra.main(version_base=None, config_path="../conf", config_name="visualize.yaml")
def main(cfg: DictConfig) -> None:
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    general_utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    # Load the pre-trained model (for some reason this only works with pre-trained model backbones)
    base_model = tf.keras.applications.EfficientNetB0(weights="imagenet")

    # Visualize and save Grad-CAM visualizations
    gradcam = GradCAM(base_model)
    logger.info("Visualizing Grad-CAM activations.")
    gradcam.gradcam_multi_layer(cfg.img_path, cfg.layers_to_visualize, cfg.save_dir)

    # Visualize and save filter activation visualizations
    activation_visualizer = ActivationVisualizer(base_model)
    logger.info("Visualizing and saving activation visualizations.")
    activation_visualizer.activate_multi_layer(
        cfg.img_path, cfg.layers_to_visualize, cfg.num_filters, cfg.save_dir
    )


if __name__ == "__main__":
    main()
