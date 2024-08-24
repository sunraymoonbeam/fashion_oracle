import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from .utils import (
    load_segmentation_model,
    get_class_seg_palette,
)


class NormalizeImage:
    """Normalize given tensor into given mean and standard deviation."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        self.normalize_1 = transforms.Normalize([self.mean], [self.std])
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)
        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)
        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)
        else:
            raise ValueError("Normalization implemented only for 1, 3, and 18 channels")


class ImageSegmenter:
    """
    A class for segmenting images using a pre-trained segmentation model.

    Attributes:
        model (torch.nn.Module): The pre-trained segmentation model.
        palette (list): The color palette for the segmentation mask.
        device (str): The device to run the model on (e.g., 'cpu' or 'cuda').

    Methods:
        preprocess_image(img): Preprocesses the input image by resizing, transforming, and unsqueezing it.
        generate_segmentation_mask(image_tensor): Generates the segmentation mask for the input image tensor.
        overlay_mask(input_image, mask_arr, bg_color): Overlays the mask on the input image with a specified background color.
        segment_image(input_image_path, output_image_path): Segments a single image and saves the result.
        segment_folder_images(input_folder, output_folder): Segments all images in a folder and saves the results.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/segmentation/u2net.pth",
        device: str = "cpu",
    ):
        """
        Initializes the ImageSegmenter with a pre-trained model and device.

        Args:
            checkpoint_path (str): Path to the model checkpoint.
            device (str): Device to run the model on (default is 'cpu').
        """
        self.model = load_segmentation_model(checkpoint_path, device=device)
        self.palette = get_class_seg_palette(num_cls=4)
        self.device = device

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocesses the input image by resizing, transforming, and unsqueezing it.

        Args:
            img (Image.Image): The input image.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        img = img.resize((768, 768), Image.BICUBIC)
        transforms_list = [transforms.ToTensor(), NormalizeImage(0.5, 0.5)]
        transform_rgb = transforms.Compose(transforms_list)
        image_tensor = transform_rgb(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        return image_tensor

    def generate_segmentation_mask(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Generates the segmentation mask for the input image tensor.

        Args:
            image_tensor (torch.Tensor): The preprocessed image tensor.

        Returns:
            np.ndarray: The segmentation mask array.
        """
        with torch.no_grad():
            output_tensor = self.model(image_tensor.to(self.device))
            output_tensor = F.log_softmax(output_tensor[0], dim=1)
            output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
            output_tensor = torch.squeeze(output_tensor, dim=0)
            output_arr = output_tensor.cpu().numpy()
        return output_arr

    def overlay_mask(
        self,
        input_image: Image.Image,
        mask_arr: np.ndarray,
        bg_color=(255, 0, 255, 255),
    ) -> Image.Image:
        """
        Overlays the mask on the input image with a specified background color.

        Args:
            input_image (Image.Image): The original input image.
            mask_arr (np.ndarray): The segmentation mask array.
            bg_color (tuple): The background color in RGBA format (default is magenta).

        Returns:
            Image.Image: The composited image with the mask overlay.
        """
        img_size = input_image.size
        mask = Image.fromarray(mask_arr[0].astype(np.uint8), mode="P")
        mask.putpalette(self.palette)
        mask = mask.resize(img_size, Image.BICUBIC).convert("L")

        threshold = 1
        binary_mask = mask.point(lambda p: p > threshold and 255)

        bg_image = Image.new("RGBA", img_size, bg_color)
        original_img = input_image.convert("RGBA")
        segmented_img = Image.composite(original_img, bg_image, binary_mask)
        segmented_img = segmented_img.convert("RGB")
        return segmented_img

    def segment_image(self, input_image: Union[str, Image.Image]) -> Image.Image:
        """
        Segments a single image and returns the result.

        Args:
            input_image (Union[str, Image.Image]): Path to the input image or an image array.

        Returns:
            Image.Image: The segmented image.
        """
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        elif not isinstance(input_image, Image.Image):
            raise ValueError(
                "input_image must be a file path (str) or an image array (Image.Image)"
            )

        image_tensor = self.preprocess_image(input_image)
        mask_arr = self.generate_segmentation_mask(image_tensor)
        segmented_img = self.overlay_mask(input_image, mask_arr)
        return segmented_img

    def segment_images_dir(self, input_dir: str, output_dir: str) -> None:
        """
        Segments all images in a directory and saves the results.

        Args:
            input_dir (str): Path to the input directory containing images.
            output_dir (str): Path to save the segmented images.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    input_image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_dir)
                    output_image_dir = os.path.join(output_dir, relative_path)

                    if not os.path.exists(output_image_dir):
                        os.makedirs(output_image_dir)

                    file_name, file_ext = os.path.splitext(file)
                    output_image_path = os.path.join(
                        output_image_dir, f"{file_name}_segmented{file_ext}"
                    )

                    segmented_img = self.segment_image(input_image_path)
                    segmented_img.save(output_image_path)
                    print(f"Processed and saved: {output_image_path}")


# class ImageSegmenter:
#     def __init__(
#         self,
#         checkpoint_path: str = "models/semgentation/u2net.pth",
#         device: str = "cpu",
#     ):
#         self.model = load_segmentation_model(checkpoint_path, device=device)
#         self.palette = get_palette(num_cls=4)

#     def apply_transform(self, img: Image.Image) -> torch.Tensor:
#         """Apply necessary transformations to the input image."""
#         transforms_list = [transforms.ToTensor(), NormalizeImage(0.5, 0.5)]
#         transform_rgb = transforms.Compose(transforms_list)
#         return transform_rgb(img)

#     def generate_mask(self, input_image: Image.Image, output_path: str) -> Image.Image:
#         """Generate and save the segmentation mask for the input image."""
#         img_size = input_image.size
#         img = input_image.resize((768, 768), Image.BICUBIC)
#         image_tensor = self.apply_transform(img)
#         image_tensor = torch.unsqueeze(image_tensor, 0)

#         with torch.no_grad():
#             output_tensor = self.model(image_tensor.to(self.device))
#             output_tensor = F.log_softmax(output_tensor[0], dim=1)
#             output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
#             output_tensor = torch.squeeze(output_tensor, dim=0)
#             output_arr = output_tensor.cpu().numpy()

#         mask = Image.fromarray(output_arr[0].astype(np.uint8), mode="P")
#         mask.putpalette(self.palette)
#         mask = mask.resize(img_size, Image.BICUBIC).convert("L")

#         threshold = 1
#         binary_mask = mask.point(lambda p: p > threshold and 255)

#         magenta_bg = Image.new("RGBA", img_size, (255, 0, 255, 255))

#         original_img = input_image.convert("RGBA")
#         region = Image.composite(original_img, magenta_bg, binary_mask)

#         region = region.convert("RGB")

#         if output_path is not None:
#             region.save(output_path)

#         return region

#     def segment_image(self, input_image_path: str, output_image_path: str) -> None:
#         """Segment a single image and save the result."""
#         img = Image.open(input_image_path).convert("RGB")
#         self.generate_mask(img, output_image_path)

#     def segment_folder_images(self, input_folder: str, output_folder: str) -> None:
#         """Segment all images in a folder and save the results."""
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)

#         for root, _, files in os.walk(input_folder):
#             for file in files:
#                 if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
#                     input_image_path = os.path.join(root, file)
#                     relative_path = os.path.relpath(root, input_folder)
#                     output_image_dir = os.path.join(output_folder, relative_path)

#                     if not os.path.exists(output_image_dir):
#                         os.makedirs(output_image_dir)

#                     output_image_path = os.path.join(output_image_dir, file)
#                     self.segment_image(input_image_path, output_image_path)
#                     print(f"Processed and saved: {output_image_path}")
