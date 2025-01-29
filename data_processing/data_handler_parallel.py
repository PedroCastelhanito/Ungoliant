import os
import cv2
import h5py
import numpy as np
from datetime import datetime
from file_handler import FileReader, FileWriter
from alive_progress import alive_bar
from typing import Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

class ImageProcessing:
    STAGE_STEPS_PER_IMAGE = 20
    STAGE_MM_PER_STEP = 0.00625
    CAMERA_MM_PER_PIXEL = 0.0370
    IMAGE_SPACING = STAGE_STEPS_PER_IMAGE * STAGE_MM_PER_STEP

    DATA_EXTENSIONS = [".h5"]
    IMAGE_EXTENSIONS = [".png", ".nef", ".tif"]

    DEFAULT_MASK_FILENAME = "mask.png"
    DEFAULT_METHOD_ARGS = {"method": "metropolis", "temperature": 1}

    def __init__(self) -> None:
        pass

    @staticmethod
    def preprocess_dataset(input_folder: str,
                           output_folder: str,
                           channel: int,
                           method_args: Optional[Dict] = None,
                           overwrite: bool = True) -> None:
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Input folder '{input_folder}' not found")
        os.makedirs(output_folder, exist_ok=True)
        method_args = method_args or ImageProcessing.DEFAULT_METHOD_ARGS

        folder_name = os.path.basename(input_folder.rstrip(os.sep))
        output_path = os.path.join(output_folder, f"{folder_name}.h5")

        results = ImageProcessing.process_image_batch(
            input_folder, channel, method_args
        )

        # Prepare HDF5 dataset
        attributes = ImageProcessing._get_dataset_attributes(results, method_args)
        output_data = ImageProcessing._create_hdf5_datastructure(input_folder, *results)

        # Write to HDF5
        FileWriter.write_HDF5(output_path, output_data, attributes, overwrite)

    @staticmethod
    def process_image_batch(input_folder: str,
                            channel: int,
                            method_args: Dict) -> Tuple[np.ndarray, ...]:
        """
        Parallel version of processing all images in a folder and returning extracted data.
        """
        # Pre‐load the mask (all workers can re‐load it if desired, but here we load once)
        mask = ImageProcessing.load_dataset_mask(input_folder)
        img_paths = ImageProcessing.get_image_filenames_sorted(input_folder)

        # Helper function to process a single image in parallel
        def _process_single_image(idx: int,
                                  filename: str,
                                  folder: str,
                                  ch: int,
                                  m_args: Dict,
                                  m_mask: Optional[np.ndarray]) -> Tuple:
            """
            This function runs in a worker process for a single image.
            Returns a tuple with (filename, raw_points, adjusted_points, idx).
            If an error occurs, it can return None or raise an exception.
            """
            img_path = os.path.join(folder, filename)
            # Load image
            image = FileReader.load_image(img_path, valid_extensions=ImageProcessing.IMAGE_EXTENSIONS)
            # Check channel
            if image.ndim != 3 or ch >= image.shape[2]:
                raise ValueError(f"Invalid channel {ch} for image shape {image.shape}")

            # Extract points
            points = ImageProcessing.get_image_cloud_points(image, ch, m_args, m_mask)
            # Compute Z‐position directly from index
            z_position = idx * ImageProcessing.IMAGE_SPACING
            z_column = np.full((points.shape[0], 1), z_position)
            adjusted = np.hstack((points * ImageProcessing.CAMERA_MM_PER_PIXEL, z_column))

            return (filename, points, adjusted, idx)

        results = []
        # We will track completion in real time with alive_bar
        with alive_bar(len(img_paths), title='Processing Images', bar='filling') as bar:
            # Create a pool of worker processes
            with ProcessPoolExecutor() as executor:
                # Submit a task for each image
                future_map = {
                    executor.submit(_process_single_image, idx, fn, input_folder, channel, method_args, mask): fn
                    for idx, fn in enumerate(img_paths)
                }

                # As each future completes, append results
                for future in as_completed(future_map):
                    bar()  # update progress bar for each completed image
                    try:
                        res = future.result()  # may raise if the worker had an exception
                        if res is not None:
                            results.append(res)
                    except Exception as e:
                        # Log or handle exception; skip the image
                        print(f"Skipping {future_map[future]} due to error: {e}")

        # Sort results by idx to maintain a consistent order
        # (if order isn't crucial, you can skip this sort)
        results.sort(key=lambda x: x[3])

        # Parse all results
        if not results:
            return (np.array([]),   # image names
                    np.array([]),   # indices
                    np.empty((0, 2)),  # raw xy
                    np.empty((0, 3)),  # adjusted xyz
                    np.empty((0, 1)))  # image indices per point

        # Unpack columns
        image_names, raw_pts, adj_pts, indices = [], [], [], []
        for (filename, points, adjusted, idx) in results:
            image_names.append(filename)
            raw_pts.append(points)
            adj_pts.append(adjusted)
            # For each point, store the corresponding image index
            indices.append(np.full((points.shape[0], 1), idx))

        # Convert to final numpy arrays
        image_names = np.array(image_names)
        image_indices = np.arange(len(results))  # or store actual idx if desired
        raw_points = np.vstack(raw_pts) if raw_pts else np.empty((0, 2))
        adj_points = np.vstack(adj_pts) if adj_pts else np.empty((0, 3))
        point_indices = np.vstack(indices) if indices else np.empty((0, 1))

        return (image_names,       # array of filenames
                image_indices,     # numeric indices for each image
                raw_points,        # stacked raw XY points
                adj_points,        # stacked XYZ points
                point_indices)     # which image each point came from
    
    @staticmethod
    def _metropolis_sampling(data: np.ndarray, temperature: float) -> np.ndarray:
        data = np.clip(data, 1e-6, 255)
        exponent = (255 / data) / temperature
        exponent = np.clip(exponent, -100, 100)
        probabilities = 1 / np.exp(exponent)
        rand_mask = np.random.random(data.shape) < probabilities
        return np.column_stack(np.where(rand_mask))

    @staticmethod
    def _threshold_sampling(data: np.ndarray, threshold: float) -> np.ndarray:
        threshold_val = np.nanmax(data) * threshold
        return np.column_stack(np.where(data > threshold_val))

    @staticmethod
    def _get_dataset_attributes(results: Tuple, method_args: Dict) -> Dict:
        return {
            "image_spacing": ImageProcessing.IMAGE_SPACING,
            "mm_per_pixel": ImageProcessing.CAMERA_MM_PER_PIXEL,
            "number_of_images": results[1].shape[0],
            "number_of_points": results[2].shape[0],
            "extraction_method": method_args["method"],
            "last_updated_in": datetime.now().isoformat(),
            **{k: v for k, v in method_args.items() if k != "method"},
        }

    @staticmethod
    def _create_hdf5_datastructure(input_folder: str,
                                   image_names: np.ndarray,
                                   image_indices: np.ndarray,
                                   raw_points: np.ndarray,
                                   adj_points: np.ndarray,
                                   point_indices: np.ndarray) -> Dict:
        mask = ImageProcessing.load_dataset_mask(input_folder)
        # Convert names to HDF5‐compatible strings
        image_names = np.array(
            [str(name) for name in image_names], dtype=h5py.string_dtype(encoding='utf-8')
        )
        return {
            '/scan_images/image_mask': mask if mask is not None else np.array([]),
            '/scan_images/image_indices': image_indices,
            '/scan_images/image_filenames': image_names,
            '/point_cloud/image_indices': point_indices,
            '/point_cloud/xy_coordinates': raw_points,
            '/point_cloud/xyz_coordinates': adj_points,
        }

    @staticmethod
    def get_image_cloud_points(image: np.ndarray,
                               channel: int,
                               method_args: Dict,
                               mask: Optional[np.ndarray] = None) -> np.ndarray:
        method = method_args.get("method")
        if method not in ["metropolis", "max_brightness"]:
            raise ValueError(f"Invalid method '{method}'")

        if mask is not None:
            if mask.shape != image.shape:
                raise ValueError(f"Mask shape {mask.shape} doesn't match image {image.shape}")
            image *= mask

        channel_data = image[:, :, channel].astype(np.float32)

        if method == "metropolis":
            if "temperature" not in method_args:
                raise ValueError("Missing 'temperature' parameter for metropolis method")
            points = ImageProcessing._metropolis_sampling(channel_data, method_args["temperature"])
        else:
            if "threshold" not in method_args:
                raise ValueError("Missing 'threshold' parameter for max_brightness method")
            points = ImageProcessing._threshold_sampling(channel_data, method_args["threshold"])

        return points.astype(np.float64)

    @staticmethod
    def get_image_filenames_sorted(input_folder: str) -> list:
        return sorted(
            [
                f for f in os.listdir(input_folder)
                if os.path.splitext(f)[1].lower() in ImageProcessing.IMAGE_EXTENSIONS
                and f != ImageProcessing.DEFAULT_MASK_FILENAME
            ],
            key=lambda x: int(''.join(filter(str.isdigit, x)))  # naive numeric sort
        )

    @staticmethod
    def load_dataset_mask(input_folder: str) -> Optional[np.ndarray]:
        mask_path = os.path.join(input_folder, ImageProcessing.DEFAULT_MASK_FILENAME)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            return mask
        return None