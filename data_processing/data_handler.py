
import os
import csv
import numpy as np
from PIL import Image
from typing import List

import alive_progress

DATA_EXTENSIONS = [".csv", ".h5"]
IMAGE_EXTENSIONS = [".png", ".nef", ".tif"]

DEFAULT_MASK_FILENAME = "mask.png"
DEFAULT_POINT_EXTRACTION_METHOD = {"method": 'metropolis', "temperature": 1}

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
    def load_image(filepath: str, valid_extensions=IMAGE_EXTENSIONS) -> np.ndarray:
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


class DataWriting:

    def __init__(self) -> None:
        pass

    @staticmethod
    def validate_file_path(filepath: str, overwrite=1) -> None:
        """
        Validates that the specified file path can be used for writing output data.

        Args:
            filepath (str): The full path (including file name) for the output.
            overwrite (bool, optional): Whether to overwrite an existing file. Defaults to True.

        Raises:
            RuntimeError: If a file already exists at `filepath` and `overwrite` is False.

        Returns:
            None
        """
        file_folder = os.path.dirname(filepath)

        if not os.path.isfile(filepath):
            if not os.path.isdir(file_folder):
                os.makedirs(file_folder)
        else:
            if overwrite:
                del filepath
            else:
                raise RuntimeError(f"The file '{filepath}' already exists. "
                                   "Use 'overwrite=1' to overwrite it or provide another filepath"
                                   )

    @staticmethod
    def write_HDF5(filepath: str, datasets: dict, attributes=None) -> None:
        pass


class DataProcessing:

    def __init__(self) -> None:
        pass

    @staticmethod
    def load_dataset_mask(input_folder: str) -> np.ndarray:
        """
        Loads the dataset mask from the specified folder (if it exists) and processes it by
        setting any non-zero values to 1. If the mask file does not exist in the folder,
        returns an array of shape (1,).

        Args:
            input_folder (str): The folder containing the mask file.

        Returns:
            np.ndarray:
                - A NumPy array with non-zero values set to 1 if the mask file is found.
                - An array of shape (1,) if no mask file is found.
        """
        folder_files = [f for f in os.listdir(input_folder)]

        if DEFAULT_MASK_FILENAME in folder_files:
            filepath = os.path.join(input_folder, DEFAULT_MASK_FILENAME)
            mask = DataLoading.load_image(filepath)
            mask[mask > 0] = 1
            return mask
        else:
            return np.ndarray(1)

    @staticmethod
    def extract_cloud_brightness(image: np.ndarray, threshold: float) -> np.ndarray:
        """
        Identifies bright cloud regions in an image by applying a threshold based on
        the maximum value in the image (ignoring NaNs).

        Args:
            image (np.ndarray): The input image as a NumPy array.
            threshold (float): A value between 0 and 1 indicating the fraction of the
                maximum pixel intensity used as the threshold.

        Returns:
            np.ndarray: An N x 2 array of (row, column) coordinates where the pixel value
                        exceeds `threshold * max_pixel_value`.
        """
        max_val = np.nanmax(image)
        threshold_val = max_val * threshold
        points = np.column_stack(np.where(image > threshold_val))

        return points

    @staticmethod
    def extract_cloud_metropolis(image: np.ndarray, temperature: float) -> np.ndarray:
        """
        Extracts cloud points from an image using a Metropolis-like probabilistic sampling
        algorithm. The probability for selecting each pixel depends on a computed
        exponential distribution.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            temperature (float): A temperature parameter controlling the probability
                distribution. Higher values yield more uniform sampling.

        Returns:
            np.ndarray: An N x 2 array of (row, column) coordinates indicating the
                        sampled points.
        """
        probabilities = 1/np.exp((255/(image + 1)) / temperature)
        randomValues = np.random.uniform(0, 1, size=probabilities.shape)
        points = np.array(np.where((1 - np.isnan(probabilities))
                          & (randomValues < probabilities))).T

        return points

    @staticmethod
    def get_image_cloud_points(filepath: str, channel: int, method_args: dict, mask: np.ndarray) -> np.ndarray:
        """
        Retrieves cloud points from a specified channel of an image, using a method
        defined in `method_args`. The two supported methods are 'metropolis' and
        'max_brightness'. If a mask is provided, it is element-wise multiplied with
        the channel data before processing.

        Args:
            filepath (str): The full path to the image file.
            channel (int): The index of the channel to process (e.g., 0 for Red,
                        1 for Green, 2 for Blue).
            method_args (dict): A dictionary specifying the extraction method and
                its parameters. Must include "method" key with either "metropolis"
                or "max_brightness". Additional keys may include "temperature" or
                "threshold", depending on the method.
            mask (np.ndarray): A mask array (of the same shape as the image channel)
                that is multiplied with the channel data to exclude unwanted regions.

        Returns:
            np.ndarray: An N x 2 array of coordinates where cloud points were
                        identified, returned as float64.
        """
        img = DataLoading.load_image(filepath)
        channel_data = img[:, :, channel]
        channel_data = channel_data*mask

        if method_args["method"] == "metropolis":
            point_coordinates = DataProcessing.extract_cloud_metropolis(
                channel_data, temperature=method_args["temperature"])
        elif method_args["method"] == "max_brightness":
            point_coordinates = DataProcessing.extract_cloud_brightness(
                channel_data)

        point_coordinates = point_coordinates.astype(np.float64)

        return point_coordinates

    @staticmethod
    def get_dataset_cloud_points(input_folder: str, channel: int, method_args: dict) -> np.ndarray:
        """
        Processes all valid images in an input folder to extract cloud points from
        a specified channel, stacking the extracted points in a 3D point cloud. The
        Z-coordinate (distance) increments by a constant spacing for each image.

        Args:
            input_folder (str): The directory containing the images.
            channel (int): The channel index in each image from which cloud points
                        should be extracted.
            method_args (dict): A dictionary specifying the extraction method ("method")
                and any relevant parameters (e.g., "temperature" or "threshold").

        Returns:
            np.ndarray: A NumPy array of shape (N, 3), where each row contains the
                        (x, y, distance) coordinates of a cloud point. The x and y
                        coordinates are scaled by CAMERA_MM_PER_PIXEL, and the
                        distance is incremented by IMAGE_SPACING for each image.
        """
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"The specified folder \
                                    '{input_folder}' does not exist. Please check the folder path.")

        files = []
        for f in os.listdir(input_folder):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS and f != DEFAULT_MASK_FILENAME:
                files.append(os.path.join(input_folder, f))

        mask = DataProcessing.load_dataset_mask(input_folder)

        current_distance = 0
        number_of_imgs = len(files)
        full_dataset_cloud = []

        with alive_progress.alive_bar(number_of_imgs) as bar:
            for file in files:
                point_coordinates = DataProcessing.get_image_cloud_points(
                    file, channel, method_args, mask)
                point_coordinates *= CAMERA_MM_PER_PIXEL
                if point_coordinates.size > 0:
                    current_distance_column = np.full(
                        (point_coordinates.shape[0], 1), current_distance)
                    point_coordinates_adjusted = np.hstack(
                        (point_coordinates, current_distance_column))
                else:
                    point_coordinates_adjusted = np.empty((0, 3))

                full_dataset_cloud.append(point_coordinates_adjusted)
                current_distance += IMAGE_SPACING
                bar()
        output_coordinates = np.vstack(full_dataset_cloud)

        return output_coordinates

    @staticmethod
    def process_images(input_folder: str, output_filepath: str, channel: int, method_args=None, overwrite=1) -> None:
        if method_args is None:
            method_args = DEFAULT_POINT_EXTRACTION_METHOD

        dataset_points = DataProcessing.get_dataset_cloud_points(
            input_folder, channel, method_args=method_args)

        if not os.path.isfile(output_filepath):
            output_folder = os.path.dirname(output_filepath)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
        else:
            if overwrite:
                del output_filepath

        method_name = method_args['method']
        method_parameter = [p for p in method_args.keys() if p != method_name]

        attributes = {
            "point_extraction_method": method_args["method"],
            method_parameter: method_args[method_parameter],
            "image_spacing":IMAGE_SPACING,
            "mm_per_pixel":CAMERA_MM_PER_PIXEL,
        }

        output_dataset = {
            '\point_cloud\x_coordinate': dataset_points[:, 0],
            '\point_cloud\y_coordinate': dataset_points[:, 1],
            '\point_cloud\z_coordinate': dataset_points[:, 2],
        }


if __name__ == "__main__":

    folder = r"C:\Users\Pedro Castelhanito\Desktop\test_dataset"

    data_points = DataProcessing.get_dataset_cloud_points(folder, 1)

    from data_visualization.data_visualizer import Visualizer

    Visualizer.plot_point_cloud(
        data_points[:, 0], data_points[:, 1], data_points[:, 2])
