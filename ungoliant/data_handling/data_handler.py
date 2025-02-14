import os
from typing import Tuple, Dict, Optional, Union, Any

import cv2
import h5py
import numpy as np
from datetime import datetime
from scipy.signal import convolve
from alive_progress import alive_bar

from file_handler import FileReader, FileWriter


class ImageProcessing:

    """
    A class for processing 2D image data and converting it into point clouds,
    along with various utility functions for sampling, cleaning, and contracting
    point clouds.
    """

    # Class constants
    STAGE_STEPS_PER_IMAGE = 20
    STAGE_MM_PER_STEP = 0.00625
    CAMERA_MM_PER_PIXEL = 0.0370
    IMAGE_SPACING = STAGE_STEPS_PER_IMAGE * STAGE_MM_PER_STEP

    DATA_EXTENSIONS = [".h5"]
    IMAGE_EXTENSIONS = [".png", ".nef", ".tif"]

    DEFAULT_MASK_FILENAME = "mask.png"
    DEFAULT_METHOD_ARGS = {"method": "metropolis", "temperature": 1}

    ###### * Main code * ######

    @staticmethod
    def preprocess_dataset(
        input_folder: str,
        output_folder: str,
        channel: int,
        method_args: Optional[Dict] = None,
        overwrite: bool = True,
        compression_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Process a dataset of images into a 3D point cloud stored in HDF5 format.

        Args:
            input_folder (str): Path to the input images directory.
            output_folder (str): Path to the output directory for the HDF5 file.
            channel (int): Color channel index to process (0-based).
            method_args (dict, optional): Dictionary of parameters for the point
                extraction method. Defaults to None, in which case
                `DEFAULT_METHOD_ARGS` is used.
            overwrite (bool): Whether to overwrite an existing HDF5 file with the
                same name. Defaults to True.

        Raises:
            FileNotFoundError: If the input folder does not exist.
            ValueError: If invalid method parameters or channel index are provided.

        Returns:
            None
        """
        # Validate input path
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Input folder '{input_folder}' not found")

        # Create output directory if needed
        os.makedirs(output_folder, exist_ok=True)

        # Set default method arguments
        method_args = method_args or ImageProcessing.DEFAULT_METHOD_ARGS

        # Generate output path
        folder_name = os.path.basename(input_folder.rstrip(os.sep))
        output_path = os.path.join(output_folder, f"{folder_name}.h5")

        # Process dataset
        (
            image_indices,
            image_names,
            raw_points,
            adj_points,
            adj_point_indices,
        ) = ImageProcessing.process_image_batch(input_folder, channel, method_args)

        # Clean bottom and top cloud points
        # adj_points, adj_point_indices = ImageProcessing.clean_point_cloud(
        #     adj_points, adj_point_indices, (0.05, 0.92))

        # Apply single dimension contraction
        for _ in range(5):
            adj_points = ImageProcessing.contract_points_along_dimension(
                adj_points, 2, latticeSize=300, p=5
            )

        # Prepare HDF5 dataset
        attributes = ImageProcessing._get_dataset_attributes(
            image_names, adj_points, method_args
        )
        output_data = ImageProcessing._create_hdf5_datastructure(
            input_folder,
            image_indices,
            image_names,
            raw_points,
            adj_points,
            adj_point_indices,
        )

        # Write to HDF5
        FileWriter.write_HDF5(
            output_path,
            output_data,
            attributes,
            overwrite,
            compression_args,
        )

    @staticmethod
    def process_image_batch(
        input_folder: str, channel: int, method_args: Dict
    ) -> Tuple[np.ndarray, ...]:
        """
        Process all images in a folder to extract point coordinates.

        Args:
            input_folder (str): Path to the images directory.
            channel (int): Color channel index to process.
            method_args (dict): Parameters for the point-extraction method.

        Returns:
            tuple of np.ndarray:
                - image_indices (1D array): Numeric indices for each image (in sorted order).
                - image_filenames (1D array): Filenames corresponding to each image index.
                - raw_points (2D array): Combined (row, col) pixel coordinates from all images, shape (N, 2).
                - adj_points (2D array): Combined (x, y, z) coordinates in mm, shape (N, 3).
                - point_image_indices (1D or 2D array): Image index for each extracted point, shape (N, 1).

        Raises:
            ValueError: If the specified channel is out of range for any image.
        """
        mask = ImageProcessing.load_dataset_mask(input_folder)
        img_paths = ImageProcessing._get_image_filenames_sorted(input_folder)

        indices = []
        z_position = 0
        image_data = []
        point_data = []
        with alive_bar(len(img_paths), title="Processing Images", bar="filling") as bar:
            for idx, filename in enumerate(img_paths):
                try:
                    # Load and validate image
                    img_path = os.path.join(input_folder, filename)
                    image = FileReader.load_image(
                        img_path, valid_extensions=ImageProcessing.IMAGE_EXTENSIONS
                    )

                    if image.ndim != 3 or channel >= image.shape[2]:
                        raise ValueError(
                            f"Invalid channel \
                                         {channel} for image shape {image.shape}"
                        )

                    # Extract points
                    points = ImageProcessing.get_image_cloud_points(
                        image, channel, method_args, mask
                    )

                    # Store data
                    image_data.append((idx, filename))
                    z_column = np.full((points.shape[0], 1), z_position)
                    adjusted = np.hstack(
                        (points * ImageProcessing.CAMERA_MM_PER_PIXEL, z_column)
                    )

                    point_data.append((points, adjusted))
                    indices.append(np.full((points.shape[0], 1), idx))
                    z_position += ImageProcessing.IMAGE_SPACING

                except Exception as e:
                    print(f"Skipping {filename}: {str(e)}")
                finally:
                    bar()

        # Stack all results
        image_names, image_indices = zip(*image_data) if image_data else ((), ())
        raw_points, adj_points = zip(*point_data) if point_data else ((), ())

        return (
            np.array(image_names),
            np.stack(image_indices),
            np.vstack(raw_points) if raw_points else np.empty((0, 2)),
            np.vstack(adj_points) if adj_points else np.empty((0, 3)),
            np.vstack(indices) if indices else np.empty((0, 1)),
        )

    ###### * Sampling methods * ######

    @staticmethod
    def _metropolis_sampling(data: np.ndarray, temperature: float) -> np.ndarray:
        """
        Perform Metropolis-like sampling on the input intensity data.

        Args:
            data (np.ndarray): 2D array of grayscale intensities (float).
            temperature (float): Temperature parameter controlling sample acceptance probability.

        Returns:
            np.ndarray: A 2D array of shape (M, 2), where each row is (row_index, col_index) of a selected pixel.
        """
        probabilities = 1 / np.exp((255 / (data + 1)) / temperature)
        randomValues = np.random.uniform(0, 1, size=probabilities.shape)
        return np.array(
            np.where((1 - np.isnan(probabilities)) & (randomValues < probabilities))
        ).T

    @staticmethod
    def _threshold_sampling(data: np.ndarray, threshold: float) -> np.ndarray:
        """
        Perform threshold-based sampling on the input intensity data.

        Args:
            data (np.ndarray): 2D array of grayscale intensities (float).
            threshold (float): Fraction of the max intensity to use as a cutoff (e.g. 0.5).

        Returns:
            np.ndarray: A 2D array of shape (M, 2), where each row is (row_index, col_index) of a selected pixel.
        """
        threshold_val = np.nanmax(data) * threshold
        return np.column_stack(np.where(data > threshold_val))

    ###### * Helper code * ######

    @staticmethod
    def _get_dataset_attributes(
        image_names: np.ndarray, points: np.ndarray, method_args: Dict
    ) -> Dict:
        """
        Generate a dictionary of HDF5 attributes for the processed dataset.

        Args:
            image_names (np.ndarray): 1D array of image filenames.
            points (np.ndarray): 2D array of (x, y, z) point coordinates.
            method_args (dict): Dictionary of method parameters used for extraction.

        Returns:
            dict: A dictionary of attribute keys and values to store in the HDF5 file.
        """
        attributes = {
            "image_spacing": ImageProcessing.IMAGE_SPACING,
            "mm_per_pixel": ImageProcessing.CAMERA_MM_PER_PIXEL,
            "number_of_images": image_names.shape[0],
            "number_of_points": points.shape[0],
            "extraction_method": method_args["method"],
            "last_updated_in": datetime.now().isoformat(),
        }

        method_parameters = [k for k in method_args.keys() if k != "method"]
        method_parameter_values = [method_args[k] for k in method_parameters]
        for k, v in zip(method_parameters, method_parameter_values):
            attributes[k] = v

        return attributes

    @staticmethod
    def _create_hdf5_datastructure(
        input_folder: str,
        image_idx: np.ndarray,
        image_names: np.ndarray,
        raw_points: np.ndarray,
        adj_points: np.ndarray,
        point_indices: np.ndarray,
    ) -> Dict:
        """
        Organize data into a nested dictionary matching the desired HDF5 structure.

        Args:
            input_folder (str): Path to the folder containing images (and optional mask).
            image_idx (np.ndarray): 1D array of image indices.
            image_names (np.ndarray): 1D array of corresponding image filenames.
            raw_points (np.ndarray): 2D array of (row, col) pixel coordinates.
            adj_points (np.ndarray): 2D array of (x, y, z) coordinates in mm.
            point_indices (np.ndarray): 1D or 2D array of image indices for each point row.

        Returns:
            dict: A dictionary whose keys represent HDF5 paths and values are the corresponding NumPy arrays.
        """
        # Load dataset mask
        mask = ImageProcessing.load_dataset_mask(input_folder)
        image_names = np.array(
            [str(name) for name in image_names],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        return {
            "/scan_images/image_mask": mask,
            "/scan_images/image_indices": image_idx,
            "/scan_images/image_filenames": image_names,
            "/point_cloud/image_indices": point_indices,
            "/point_cloud/xy_coordinates": raw_points,
            "/point_cloud/xyz_coordinates": adj_points,
        }

    @staticmethod
    def _get_image_filenames_sorted(input_folder: str) -> list:
        """
        Return a list of image filenames from the folder, sorted by numeric substrings.

        Args:
            input_folder (str): Folder containing the images.

        Returns:
            list: Sorted list of valid image filenames.
        """
        return sorted(
            [
                f
                for f in os.listdir(input_folder)
                if os.path.splitext(f)[1].lower() in ImageProcessing.IMAGE_EXTENSIONS
                and f != ImageProcessing.DEFAULT_MASK_FILENAME
            ],
            key=lambda x: int("".join(filter(str.isdigit, x))),
        )

    @staticmethod
    def get_image_cloud_points(
        image: np.ndarray,
        channel: int,
        method_args: Dict,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract pixel coordinates from the specified channel in an image.

        Args:
            image (np.ndarray): The input image array of shape (H, W, C).
            channel (int): Which channel index to process.
            method_args (dict): Dictionary specifying the 'method' ('metropolis' or 'max_brightness')
                                and its required parameters (e.g., 'temperature' or 'threshold').
            mask (np.ndarray, optional): A mask array of shape (H, W) that may zero-out pixels.
                                         Defaults to None.

        Returns:
            np.ndarray: A 2D array of shape (N, 2), where each row is (row_index, col_index)
                        of a selected pixel.

        Raises:
            ValueError: If an unsupported method is provided or necessary parameters are missing.
        """
        # Validate method arguments
        method = method_args.get("method")
        if method not in ["metropolis", "max_brightness"]:
            raise ValueError(f"Invalid method '{method}'")

        if mask is not None:
            if mask.shape != image.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} doesn't match image \
                                 {image.shape}"
                )
            image *= mask

        # Select channel data
        channel_data = image[:, :, channel].astype(np.float32)

        # Apply selected method
        if method == "metropolis":
            if "temperature" not in method_args:
                raise ValueError(
                    "Missing 'temperature' parameter for metropolis method"
                )
            points = ImageProcessing._metropolis_sampling(
                channel_data, method_args["temperature"]
            )
        else:
            if "threshold" not in method_args:
                raise ValueError(
                    "Missing 'threshold' parameter for max_brightness method"
                )
            points = ImageProcessing._threshold_sampling(
                channel_data, method_args["threshold"]
            )

        return points.astype(np.float64)

    @staticmethod
    def load_dataset_mask(input_folder: str) -> Optional[np.ndarray]:
        """
        Load a mask file if present in the directory.

        Args:
            input_folder (str): Folder path where the mask image may reside.

        Returns:
            np.ndarray or None: The mask array if found, otherwise None.
        """
        mask_path = os.path.join(input_folder, ImageProcessing.DEFAULT_MASK_FILENAME)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            return mask
        return None

    @staticmethod
    def clean_point_cloud(
        points: np.ndarray,
        image_indices: np.ndarray,
        zfrac: Union[float, Tuple[float, ...]],
    ) -> np.ndarray:
        """
        Remove points whose Z-coordinate (last dimension) is below certain fractions
        of the total Z-range.

        Args:
            points (np.ndarray): Point cloud of shape (N, D), where D>=3 if Z is the last dimension.
            image_indices (np.ndarray): Array of shape (N, 1) or (N,) indicating which image each point came from.
            zfrac (float or tuple of floats): Fraction(s) of the Z-range. If multiple fractions
                                              are provided, cleaning is applied iteratively.

        Returns:
            tuple of np.ndarray:
                - adjusted_points (np.ndarray): Filtered point cloud.
                - adjusted_imaged_indices (np.ndarray): Filtered image indices.
        """
        if isinstance(zfrac, (float, int)):
            zfrac = (zfrac,)

        adjusted_points = points
        adjusted_imaged_indices = image_indices

        for fraction in zfrac:
            indices = adjusted_points[:, -1] - np.min(adjusted_points[:, -1]) < (
                np.max(adjusted_points[:, -1]) - np.min(adjusted_points[:, -1])
            ) * (1 - fraction)
            adjusted_points = adjusted_points[indices]
            adjusted_imaged_indices = adjusted_imaged_indices[indices]

        return adjusted_points, adjusted_imaged_indices

    @staticmethod
    def course_grain_field(
        points,
        values=None,
        latticeSpacing=None,
        defaultLatticeSize=100,
        fixedBounds=None,
        kernel="gaussian",
        kernelSize=5,
        returnSpacing=False,
        returnCorner=False,
    ):
        """
        Coarse-grain a set of point values into a discretized field.

        This function places a regular grid (lattice) over the bounding region
        of the points (or a user-defined region), accumulates values into the
        corresponding cells, and then optionally convolves the field with a kernel
        to smooth it.

        Args:
            points (np.ndarray): An array of shape (N, d) representing point positions
                                 in d-dimensional space.
            values (np.ndarray, optional): An array of shape (N, k) containing values
                                           at each point. If None, then a density field
                                           is computed.
            latticeSpacing (float, optional): Spacing of the lattice grid. If None,
                                              it is chosen based on 'defaultLatticeSize'.
            defaultLatticeSize (int): Number of lattice points along each dimension if
                                      'latticeSpacing' is not provided.
            fixedBounds (np.ndarray, optional): If given, specifies a (d,2) array of
                                                [min,max] bounds for each dimension.
            kernel (str or np.ndarray): Either 'gaussian' (default) or a custom 2D/3D
                                        kernel array for convolution.
            kernelSize (int): Size of the Gaussian kernel if kernel='gaussian'.
            returnSpacing (bool): If True, return the spacing used for the lattice.
            returnCorner (bool): If True, return the minimum corner coordinates of the lattice.

        Returns:
            np.ndarray or list:
                - If both returnSpacing and returnCorner are False, returns the
                  coarse-grained field as a NumPy array.
                - If either is True, returns a list where the first element is the
                  coarse-grained field, and subsequent elements are the requested
                  spacing or corner information.

        Notes:
            - If the dimension 'd' > 2, the current code extends the same approach
              but the kernel usage for higher dimensions may need careful handling.
            - 'values' can be multi-dimensional, in which case the output field
              will have that many channels in the last axis.
        """
        dim = np.shape(points)[-1] if len(np.shape(points)) > 1 else 1

        if dim == 1:
            points = np.array(points)[:, None]

        if not hasattr(fixedBounds, "__iter__"):
            occupiedVolumeBounds = np.array(
                list(zip(np.min(points, axis=0), np.max(points, axis=0)))
            )
        else:
            occupiedVolumeBounds = np.array(fixedBounds)

        # Create a lattice with the selected scale for that cube
        if latticeSpacing is not None:
            spacing = latticeSpacing
            if hasattr(fixedBounds, "__iter__"):
                occupiedVolumeBounds[:, 1] -= spacing
        else:
            spacing = (occupiedVolumeBounds[:, 1] - occupiedVolumeBounds[:, 0]) / (
                defaultLatticeSize - 1
            )

        if hasattr(spacing, "__iter__"):
            spacing[spacing == 0] = 1
        else:
            spacing = spacing if spacing != 0 else 1

        fieldDims = (
            np.ceil(
                1
                + (occupiedVolumeBounds[:, 1] - occupiedVolumeBounds[:, 0]) / (spacing)
            )
        ).astype(np.int64)
        latticePositions = np.floor(
            (points - occupiedVolumeBounds[:, 0]) / (spacing)
        ).astype(np.int64)

        if hasattr(values, "__iter__"):
            k = np.shape(values)[-1]
            valArr = values
        else:
            k = 1
            valArr = np.ones((np.shape(points)[0], 1))

        fieldArr = np.zeros((*fieldDims, k))
        for i in range(np.shape(points)[0]):
            fieldArr[tuple(latticePositions[i])] += valArr[i]

        if kernel == "gaussian":
            singleAxis = np.arange(kernelSize)
            kernelGrid = np.meshgrid(
                *np.repeat([singleAxis], np.shape(points)[-1], axis=0)
            )
            kernelArr = np.exp(
                -np.sum(
                    [
                        (kernelGrid[i] - (kernelSize - 1) / 2.0) ** 2
                        for i in range(np.shape(points)[-1])
                    ],
                    axis=0,
                )
                / (kernelSize)
            )
        else:
            kernelArr = kernel

        transConvolution = np.zeros_like(fieldArr.T)

        for i in range(k):
            transConvolution[i] = convolve(
                fieldArr.T[i], kernelArr.T, mode="same"
            ) / np.sum(kernelArr)

        convolution = transConvolution.T

        if k == 1:
            convolution = convolution[..., 0]

        returnResult = [convolution]

        if returnSpacing:
            returnResult += [spacing]

        if returnCorner:
            returnResult += [occupiedVolumeBounds[:, 0]]

        return returnResult if len(returnResult) > 1 else convolution

    @staticmethod
    def contract_points_along_dimension(points, dim=0, p=1, latticeSize=100) -> np.ndarray:
        """
        Contract a point cloud along one dimension by "peak-sharpening" its density.

        This function:
        1. Creates a coarse-grained density field of the point cloud (default Gaussian kernel).
        2. Estimates a local gradient along the specified dimension.
        3. Adjusts point positions along that dimension proportionally to the gradient (scaled by p).

        Args:
            points (np.ndarray): Array of shape (N, d) representing the point cloud.
            dim (int): Which dimension (0-based) to contract along.
            p (float): Contraction parameter controlling the strength of contraction.
            latticeSize (int): Grid size for coarse-graining the density field.

        Returns:
            np.ndarray: A new array of the same shape as `points`, with updated coordinates
                        along the specified dimension.
        """
        densityField, fieldSpacing, fieldCorner = ImageProcessing.course_grain_field(
            points,
            defaultLatticeSize=latticeSize,
            returnSpacing=True,
            returnCorner=True,
        )

        dimSizeArr = densityField.shape
        planarDimArr = [i for i in range(len(dimSizeArr)) if i != dim]
        planarDimSizeArr = [dimSizeArr[i] for i in planarDimArr]

        individualAxes = [np.arange(d) for d in planarDimSizeArr]
        planarIndices = np.array(np.meshgrid(*individualAxes)).T
        planarIndices = planarIndices.reshape(
            (np.prod(planarDimSizeArr), len(planarDimSizeArr))
        )

        fullIndices = [
            [slice(None)] * len(dimSizeArr) for _ in range(len(planarIndices))
        ]

        for i in range(len(planarIndices)):
            for j in range(len(planarDimArr)):
                fullIndices[i][planarDimArr[j]] = planarIndices[i][j]

        directionalField = np.zeros_like(densityField)

        for i in range(len(fullIndices)):
            oneDimSliceDensityField = densityField[tuple(fullIndices[i])]

            directionalField[tuple(fullIndices[i])][1:-1] = (
                oneDimSliceDensityField[2:] - oneDimSliceDensityField[:-2]
            ) / (2 * fieldSpacing[dim])

        directionalField = directionalField / np.max(np.abs(directionalField)) * p

        newPoints = np.copy(points)
        normalizedPoints = ((points - fieldCorner) / fieldSpacing).astype(np.int64)

        for i in range(len(points)):
            newPoints[i, dim] += directionalField[tuple(normalizedPoints[i])]

        return newPoints
