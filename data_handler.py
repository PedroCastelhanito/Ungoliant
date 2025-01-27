
import os
import csv
import numpy as np
from PIL import Image
from typing import List

DATA_EXTENSIONS = [""]
IMAGE_EXTENSIONS = [".png", ".nef", ".tif"]

DEFAULT_MASK_FILENAME = "mask.png"

STAGE_STEPS_PER_IMAGE = 20
STAGE_MM_PER_STEP = 0.00625
CAMERA_MM_PER_PIXEL = 0.0370
IMAGE_SPACING = STAGE_STEPS_PER_IMAGE * STAGE_MM_PER_STEP


class DataLoading:

    def __init__(self) -> None:
        pass

    @staticmethod
    def validate_file_path(filepath: str) -> None:
        """
        Validates if the given file path exists and is accessible.

        Args:
            filepath (str): The full path to the file.

        Raises:
            FileNotFoundError: If the folder or file does not exist.
        """
        file_name = os.path.basename(filepath)
        file_folder = os.path.dirname(filepath)

        if not os.path.isdir(file_folder):
            raise FileNotFoundError(f"The specified folder \
                                    '{file_folder}' does not exist. Please check the folder path.")

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"The file '{file_name}' does not exist in the specified folder \
                                    '{file_folder}'. Please verify the file path.")

    @staticmethod
    def validate_file_extension(filepath: str, valid_extensions: List[str]) -> None:
        """
        Validates if the file has an allowed extension.

        Args:
            filepath (str): The full path to the file.
            valid_extensions (list[str]): A list of supported file extensions.

        Raises:
            ValueError: If the file extension is not supported.
        """
        _, file_extension = os.path.splitext(filepath)
        if file_extension not in valid_extensions:
            raise ValueError(
                f"Unsupported file format: '{file_extension}'. "
                f"Accepted formats are: {', '.join(valid_extensions)}."
            )

    @staticmethod
    def load_raw_image(filepath: str, valid_extensions=IMAGE_EXTENSIONS) -> np.ndarray:
        """
        Loads a raw image file into a numpy array.

        Args:
            filepath (str): The full path to the image file.
            valid_extensions (list[str]): A list of supported file extensions.

        Returns:
            np.ndarray: The image data loaded as a numpy array.

        Raises:
            RuntimeError: If loading for the given extension is not implemented.
        """
        DataLoading.validate_file_path(filepath)
        DataLoading.validate_file_extension(filepath, valid_extensions)

        _, file_extension = os.path.splitext(filepath)

        if file_extension.lower() == ".png":
            img = np.array(Image.open(filepath), dtype=np.uint8)
        else:
            raise RuntimeError(
                f"Loading for '{file_extension}' files has not been implemented yet.")

        return img


class DataProcessing:

    def __init__(self) -> None:
        pass

    @staticmethod
    def extract_cloud_brightness(image: np.ndarray, threshold=.8, mask=1) -> np.ndarray:
        image = mask*image
        max_val = np.nanmax(image)
        threshold_val = max_val * threshold
        points = np.column_stack(np.where(image > threshold_val))

        return points

    @staticmethod
    def extract_cloud_metropolis(image: np.ndarray, t=1) -> np.ndarray:
        probabilities = 1/np.exp((255/(image + 1)) / t)
        randomValues = np.random.uniform(0, 1, size=probabilities.shape)
        points = np.array(np.where((1 - np.isnan(probabilities))
                          & (randomValues < probabilities))).T
        
        return points

    @staticmethod
    def get_image_cloud_points(filepath: str, channel: int, method='max_brightness') -> np.ndarray:
        img = DataLoading.load_raw_image(filepath)
        channel_data = img[:, :, channel]

        if method == "metropolis":
            point_coordinates = DataProcessing.extract_cloud_metropolis(
                channel_data)
        elif method == "max_brightness":
            point_coordinates = DataProcessing.extract_cloud_brightness(
                channel_data)

        return point_coordinates

    @staticmethod
    def get_dataset_cloud_points(input_folder: str, channel: int) -> None:

        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"The specified folder \
                                    '{input_folder}' does not exist. Please check the folder path.")

        files = []
        for f in os.listdir(input_folder):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS and f != DEFAULT_MASK_FILENAME:
                files.append(os.path.join(input_folder, f))

        current_distance = 0
        number_of_imgs = len(files)
        full_dataset_cloud = []

        for file in files:
            point_coordinates = DataProcessing.get_image_cloud_points(file, channel)

            if point_coordinates.size > 0:
                current_distance_column = np.full((point_coordinates.shape[0], 1), current_distance)
                point_coordinates_adjusted = np.hstack((point_coordinates, current_distance_column))
            else:
                point_coordinates_adjusted = np.empty((0, 3))

            full_dataset_cloud.append(point_coordinates_adjusted)

            current_distance += IMAGE_SPACING

        output_coordinates = np.vstack(full_dataset_cloud)