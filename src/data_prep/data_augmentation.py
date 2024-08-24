import logging
import os
import random
import shutil
from typing import Dict

from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    A class for performing data augmentation on image datasets.

    Attributes:
        transform_steps (torchvision.transforms.Compose): Composed transformations based on the configuration.

    Methods:
        _initialize_transforms(augmentation_cfg: Dict) -> transforms.Compose:
            Initialize transformation steps based on the configuration.
        augment_data(input_data_dir: str, save_dir: str, minority_threshold: float, majority_threshold: float) -> None:
            Perform data augmentation based on the configuration.
    """

    def __init__(self, cfg: Dict):
        """
        Initialize the DataAugmentation class with configuration.

        Args:
            cfg (dict): Configuration dictionary.
        """
        self.transform_steps = self._initialize_transforms(cfg["transformations"])

    def _initialize_transforms(self, augmentation_cfg: Dict) -> transforms.Compose:
        """
        Initialize transformation steps based on the configuration.

        Args:
            augmentation_cfg (dict): Augmentation configuration dictionary.

        Returns:
            torchvision.transforms.Compose: Composed transformations.
        """
        transform_list = [
            transforms.RandomRotation(augmentation_cfg["rotation_range"]),
            (
                transforms.RandomHorizontalFlip()
                if augmentation_cfg["horizontal_flip"]
                else None
            ),
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.ColorJitter(
                brightness=augmentation_cfg.get("brightness", 0.2),
                contrast=augmentation_cfg.get("contrast", 0.2),
                saturation=augmentation_cfg.get("saturation", 0.2),
                hue=augmentation_cfg.get("hue", 0.1),
            ),
        ]
        # Remove None values from the list
        transform_list = [t for t in transform_list if t is not None]
        return transforms.Compose(transform_list)

    def augment_data_dir(
        self,
        input_data_dir: str = "./data/silver/train/",
        save_dir: str = "./data/gold/",
        minority_threshold: float = 0.1,
        majority_threshold: float = 0.5,
    ) -> None:
        """
        Perform data augmentation based on the configuration.

        Args:
            input_data_dir (str): Directory containing the input data.
            save_dir (str): Directory to save the augmented data.
            minority_threshold (float): Threshold to identify minority classes.
            majority_threshold (float): Threshold to identify majority classes.
        """
        # Clear the save_dir if it exists
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        class_folders = [
            folder
            for folder in os.listdir(input_data_dir)
            if os.path.isdir(os.path.join(input_data_dir, folder))
        ]
        class_image_counts = {
            folder: len(
                [
                    image
                    for image in os.listdir(os.path.join(input_data_dir, folder))
                    if image.endswith((".jpg", ".png"))
                ]
            )
            for folder in class_folders
        }
        max_images = max(class_image_counts.values())
        actual_minority_threshold = minority_threshold * max_images
        actual_majority_threshold = majority_threshold * max_images

        # Log the information about all classes
        logger.info(
            "Class distribution report for training set (percentage with respect to the majority class):"
        )

        # Sort class folders by the number of images
        sorted_class_image_counts = sorted(
            class_image_counts.items(), key=lambda item: item[1], reverse=True
        )

        # Identify the majority class
        majority_class = sorted_class_image_counts[0][0]
        logger.info(
            f"Majority class: '{majority_class}' with {class_image_counts[majority_class]} images"
        )

        # Calculate the percentage with respect to the majority class and log the information
        for class_folder, count in sorted_class_image_counts:
            percentage = (count / class_image_counts[majority_class]) * 100
            is_minority = count < actual_minority_threshold
            logger.info(
                f"Class '{class_folder}': {count} images ({percentage:.2f}%), Minority: {is_minority}"
            )

        # Check for minority classes
        minority_classes = [
            class_folder
            for class_folder, count in class_image_counts.items()
            if count < actual_minority_threshold
        ]

        if not minority_classes:
            logger.info("No minority class exists. No data augmentation required.")
            return

        for class_folder in class_folders:
            class_path = os.path.join(input_data_dir, class_folder)
            output_class_path = os.path.join(save_dir, class_folder)
            os.makedirs(output_class_path, exist_ok=True)
            images = [
                image
                for image in os.listdir(class_path)
                if image.endswith((".jpg", ".png"))
            ]

            if class_folder in minority_classes:
                num_augmented_images = int(actual_majority_threshold - len(images))
                for i in range(num_augmented_images):
                    image_path = os.path.join(class_path, random.choice(images))
                    image = Image.open(image_path)
                    augmented_image = self.transform_steps(image)
                    if augmented_image.mode == "RGBA":
                        augmented_image = augmented_image.convert("RGB")

                    # Extract the original file name without the extension
                    original_file_name, file_extension = os.path.splitext(
                        os.path.basename(image_path)
                    )

                    # Create the new file name by appending "augmented" before the file extension
                    new_file_name = (
                        f"{original_file_name}_augmented_{i}{file_extension}"
                    )

                    # Save the augmented image with the new file name
                    augmented_image.save(os.path.join(output_class_path, new_file_name))
                logger.info(
                    f"Augmented {num_augmented_images} images for class '{class_folder}'."
                )

            for image in images:
                shutil.copyfile(
                    os.path.join(class_path, image),
                    os.path.join(output_class_path, image),
                )

        logger.info("Data augmentation completed.")
