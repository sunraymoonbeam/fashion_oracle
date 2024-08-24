import logging
import os

import requests

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class for downloading and extracting files from a given URL.

    Methods:
    - __init__(): Initializes the DataLoader class.
    - download(url, save_filepath): Downloads a file from a given URL to a specified local file path.
    - unzip(input_file_path, save_dir): Unzips a file to a specified directory.
    """

    def __init__(self) -> None:
        pass

    def download(
        self, url: str, save_filepath: str = "./data/bronze/tensorfood.tar.gz"
    ) -> str:
        """
        Downloads a file from a given URL to a specified local file path.

        Args:
            url (str): URL to download the file from.
            save_filepath (str): Local path to save the downloaded file. Defaults to "data/raw/tensorfood.tar.gz".

        Returns:
            str: The file path where the downloaded file is saved.
        """
        response = requests.get(
            url, timeout=10
        )  # timeout after 10 seconds, if no response from the server
        with open(save_filepath, "wb") as f:
            f.write(response.content)
        logger.info(f"File successfully downloaded from {url} to {save_filepath}")
        return save_filepath

    def unzip(self, input_file_path: str, save_dir: str = "./data/bronze/") -> None:
        """
        Unzips a file to a specified directory.

        Args:
            input_file_path (str): The path to the file to be unzipped.
            save_dir (str): The directory to extract the file to. Defaults to "data/raw/".
        """
        os.system(f"tar -xf {input_file_path} -C {save_dir}")
        logger.info(f"File successfully extracted to {save_dir}")
