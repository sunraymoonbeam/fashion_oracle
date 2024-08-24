import glob
import json
import logging
import os
import random
import shutil

logger = logging.getLogger(__name__)


class DataSpiltter:
    """
    A class to process image data for machine learning models, including splitting datasets
    and converting them into TensorFlow datasets.

    Attributes:
        image_size (tuple[int, int]): The target size of the images (width, height).
        batch_size (int): The size of the batches of data.

    Methods:
        _create_folders(save_dir: str) -> tuple[str, str, str]:
            Creates folders for training, validation, and testing datasets.
        _copy_files(source_files: list[str], destination_folder: str) -> None:
            Copies files from source to destination folder.
        train_val_test_split(input_data_dir: str, save_dir: str, test_ratio: float, val_ratio: float) -> tuple[str, str | None, str]:
            Splits the dataset into training, validation, and testing sets and creates folders for them.
        create_tf_datasets(train_dir_path: str, val_dir_path: str | None, test_dir_path: str) -> tuple[tf.data.Dataset, tf.data.Dataset | None, tf.data.Dataset]:
            Converts folders of images into TensorFlow datasets.
    """

    def __init__(self) -> None:
        pass

    def _create_folders(self, save_dir: str = "./data/silver/") -> tuple[str, str, str]:
        """
        Creates folders for training, validation, and testing datasets.

        Args:
            save_dir (str): The base directory to save the split datasets.

        Returns:
            tuple[str, str, str]: Paths to the training, validation, and testing directories.
        """
        folder_names = ["train", "val", "test"]
        folder_paths = [os.path.join(save_dir, name) for name in folder_names]

        # Different seeds and randomizations lead to different spilts and distributions of images
        # Without removing the folders, the images might acculumate in the folders
        # and the split might not be as expected
        for folder_path in folder_paths:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)

        return tuple(folder_paths)

    def _copy_files(self, source_files: list[str], destination_folder: str) -> None:
        """
        Copies files from source to destination folder.

        Args:
            source_files (list[str]): List of file paths to copy.
            destination_folder (str): The destination folder path.
        """
        os.makedirs(destination_folder, exist_ok=True)
        for file in source_files:
            destination_file = os.path.join(destination_folder, os.path.basename(file))
            if os.path.exists(destination_file):
                os.remove(destination_file)
            shutil.copy(file, destination_file)

    def train_val_test_split(
        self,
        input_data_dir: str,
        save_dir: str = "./data/spilts/",
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> tuple[str, str | None, str]:
        """
        Splits the dataset into training, validation, and testing sets and creates folders for them.

        Args:
            input_data_dir (str): The directory containing the dataset to split.
            save_dir (str): The directory to save the split datasets. Defaults to "data/silver".
            test_ratio (float): Split ratio for the testing set. Defaults to 0.2.
            val_ratio (float): Split ratio for the validation set. Defaults to 0.2.
            random_seed (int): Seed for random number generator. Defaults to 42.

        Returns:
            tuple[str, str | None, str]: Paths to the training, validation (None if val_ratio is 0), and testing directories.
        """
        random.seed(random_seed)

        train_folder, val_folder, test_folder = self._create_folders(save_dir)
        class_folders = glob.glob(os.path.join(input_data_dir, "*"))

        for class_folder in class_folders:
            if os.path.isdir(class_folder):
                class_name = os.path.basename(class_folder)
                image_files = glob.glob(os.path.join(class_folder, "*.jpg"))
                random.shuffle(image_files)

                num_samples = len(image_files)
                num_test = int(num_samples * test_ratio)
                num_val = int(num_samples * val_ratio) if val_ratio > 0 else 0
                num_train = num_samples - num_test - num_val

                self._copy_files(
                    image_files[:num_train], os.path.join(train_folder, class_name)
                )
                if num_val > 0:
                    self._copy_files(
                        image_files[num_train : num_train + num_val],
                        os.path.join(val_folder, class_name),
                    )
                self._copy_files(
                    image_files[num_train + num_val :],
                    os.path.join(test_folder, class_name),
                )

        logger.info("Data successfully split into train, validation, and test sets.")
        val_folder = val_folder if val_ratio > 0 else None
        return train_folder, val_folder, test_folder
