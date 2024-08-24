"""This script processes raw tensorfood images and saves them in the processed data directory."""

import logging
import os

import hydra
import tensorflow as tf
from omegaconf import DictConfig

from src.segmentation.segmentation import ImageSegmenter
from src.data_prep.data_augmentation import DataAugmentation
from src.data_prep.data_spilt import DataSpiltter
from src.utils import general_utils


# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf", config_name="process_data.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main function to orchestrate the data preprocessing pipeline from downloading, unzipping, splitting, and augmenting the data.
    """
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    general_utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    tf.random.set_seed(cfg.random_seed)

    # Segmentation
    segmentation_model = ImageSegmenter(
        checkpoint_path=cfg.segmentation.checkpoint_path,
        device=cfg.segmentation.device,
    )
    logger.info("Starting image segmentation.")
    segmentation_model.segment_images_dir(
        input_dir=cfg.segmentation.input_dir,
        output_dir=cfg.segmentation.output_dir,
    )
    logger.info("Image segmentation completed.")

    # Split the segmented folders into train, val, and test folders
    data_preprocessor = DataSpiltter()
    logger.info("Starting train-validation-test split.")
    train_dir_path, val_dir_path, test_dir_path = (
        data_preprocessor.train_val_test_split(
            input_data_dir=cfg.segmentation.output_dir,
            save_dir=cfg.preprocessing.processed_data_dir,
            val_ratio=cfg.preprocessing.val_split,
            test_ratio=cfg.preprocessing.test_split,
        )
    )
    logger.info("Data split completed.")

    # Augment the data
    if cfg.augmentation.augment:
        data_augmentor = DataAugmentation(cfg.augmentation)
        logger.info("Starting data augmentation.")
        # Perform data augmentation
        data_augmentor.augment_data(
            input_data_dir=train_dir_path or cfg.augmentation.processed_train_dir_path,
            save_dir=cfg.augmentation.augmented_train_dir_path,
            minority_threshold=cfg.augmentation.minority_threshold,
            majority_threshold=cfg.augmentation.majority_threshold,
        )
        logger.info("Data augmentation completed.")


if __name__ == "__main__":
    main()
