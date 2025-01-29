import os
import cv2
import h5py
import numpy as np
from datetime import datetime
from scipy.signal import convolve
from alive_progress import alive_bar
from file_handler import FileReader, FileWriter
from typing import Tuple, Dict, Optional, Union


class ImageProcessing:
    # Class constants
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

    ###### * Main code * ######

    @staticmethod
    def preprocess_dataset(input_folder: str,
                           output_folder: str,
                           channel: int,
                           method_args: Optional[Dict] = None,
                           overwrite: bool = True) -> None:
        """
        Process a dataset of images into a 3D point cloud stored in HDF5 format.

        Args:
            input_folder: Path to input images directory
            output_folder: Path to output directory for HDF5 file
            channel: Color channel index to process (0-based)
            method_args: Dictionary of parameters for point extraction method
            overwrite: Overwrite existing output file if True

        Raises:
            FileNotFoundError: If input folder doesn't exist
            ValueError: For invalid method parameters or channel index
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
        image_indices, image_names, raw_points, adj_points, adj_point_indices = ImageProcessing.process_image_batch(
            input_folder, channel, method_args)

        # Clean bottom and top cloud points
        adj_points, adj_point_indices = ImageProcessing.clean_point_cloud(
            adj_points, adj_point_indices, (0.05, 0.92))

        # Apply single dimension contraction
        # for _ in range(5):
        #     adj_points = ImageProcessing.contractPointsAlongDimension(
        #         adj_points, 2, latticeSize=300, p=5)

        # Prepare HDF5 dataset
        attributes = ImageProcessing._get_dataset_attributes(
            image_names, adj_points, method_args)
        output_data = ImageProcessing._create_hdf5_datastructure(
            input_folder, image_indices, image_names, raw_points, adj_points, adj_point_indices)

        # Write to HDF5
        FileWriter.write_HDF5(output_path, output_data, attributes, overwrite)

    @staticmethod
    def process_image_batch(input_folder: str,
                            channel: int,
                            method_args: Dict) -> Tuple[np.ndarray, ...]:
        """
        Process all images in a folder and return extracted data.

        Args:
            input_folder: Path to images directory
            channel: Color channel index to process
            method_args: Point extraction parameters

        Returns:
            Tuple containing:
            - image_names: Array of filenames
            - images: Stacked image arrays
            - point_coordinates: Raw XY coordinates
            - adjusted_coordinates: XYZ coordinates with Z-offset
            - image_indices: Corresponding image indices
        """
        mask = ImageProcessing.load_dataset_mask(input_folder)
        img_paths = ImageProcessing.get_image_filenames_sorted(input_folder)

        indices = []
        z_position = 0
        image_data = []
        point_data = []

        with alive_bar(len(img_paths), title='Processing Images', bar='filling') as bar:
            for idx, filename in enumerate(img_paths):
                try:
                    # Load and validate image
                    img_path = os.path.join(input_folder, filename)
                    image = FileReader.load_image(
                        img_path, valid_extensions=ImageProcessing.IMAGE_EXTENSIONS)

                    if image.ndim != 3 or channel >= image.shape[2]:
                        raise ValueError(f"Invalid channel \
                                         {channel} for image shape {image.shape}")

                    # Extract points
                    points = ImageProcessing.get_image_cloud_points(
                        image, channel, method_args, mask)

                    # Store data
                    image_data.append((idx, filename))
                    z_column = np.full((points.shape[0], 1), z_position)
                    adjusted = np.hstack(
                        (points * ImageProcessing.CAMERA_MM_PER_PIXEL, z_column))

                    point_data.append((points, adjusted))
                    indices.append(np.full((points.shape[0], 1), idx))
                    z_position += ImageProcessing.IMAGE_SPACING

                except Exception as e:
                    print(f"Skipping {filename}: {str(e)}")
                finally:
                    bar()

        # Stack all results
        image_names, image_indices = zip(
            *image_data) if image_data else ((), ())
        raw_points, adj_points = zip(*point_data) if point_data else ((), ())

        return (
            np.array(image_names),
            np.stack(image_indices),
            np.vstack(raw_points) if raw_points else np.empty((0, 2)),
            np.vstack(adj_points) if adj_points else np.empty((0, 3)),
            np.vstack(indices) if indices else np.empty((0, 1))
        )

    ###### * Sampling methods * ######

    @staticmethod
    def _metropolis_sampling(data: np.ndarray, temperature: float) -> np.ndarray:
        """Metropolis sampling implementation with overflow protection."""
        # Clip data to avoid division by very small numbers
        data = np.clip(data, 1e-6, 255)

        # Rescale the input to the exponential function
        exponent = (255 / data) / temperature
        exponent = np.clip(exponent, -100, 100)  # Prevent overflow

        probabilities = 1 / np.exp(exponent)
        rand_mask = np.random.random(data.shape) < probabilities
        return np.column_stack(np.where(rand_mask))

    @staticmethod
    def _threshold_sampling(data: np.ndarray, threshold: float) -> np.ndarray:
        """Threshold-based sampling with NaN handling."""
        threshold_val = np.nanmax(data) * threshold
        return np.column_stack(np.where(data > threshold_val))

    ###### * Helper code * ######

    @staticmethod
    def _get_dataset_attributes(image_names: np.ndarray, points: np.ndarray, method_args: Dict) -> Dict:
        """Generate HDF5 attributes dictionary from processing results."""
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
    def _create_hdf5_datastructure(input_folder: str,
                                   image_idx: np.ndarray,
                                   image_names: np.ndarray,
                                   raw_points: np.ndarray,
                                   adj_points: np.ndarray,
                                   point_indices: np.ndarray) -> Dict:
        """Organize data into HDF5 group structure."""

        # Load dataset mask
        mask = ImageProcessing.load_dataset_mask(input_folder)
        image_names = np.array(
            [str(name) for name in image_names], dtype=h5py.string_dtype(encoding='utf-8'))

        return {
            '/scan_images/image_mask': mask,
            '/scan_images/image_indices': image_idx,
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
        """
        Extract cloud points from image channel using specified method.

        Args:
            image: Input image array (HxWxC)
            channel: Channel index to process
            method_args: Dictionary with method parameters
            mask: Optional mask array (HxW)

        Returns:
            Array of (row, col) coordinates (Nx2)

        Raises:
            ValueError: For invalid method or parameters
        """
        # Validate method arguments
        method = method_args.get("method")
        if method not in ["metropolis", "max_brightness"]:
            raise ValueError(f"Invalid method '{method}'")

        if mask is not None:
            if mask.shape != image.shape:
                raise ValueError(f"Mask shape {mask.shape} doesn't match image \
                                 {image.shape}")
            image *= mask

        # Select channel data
        channel_data = image[:, :, channel].astype(np.float32)

        # Apply selected method
        if method == "metropolis":
            if "temperature" not in method_args:
                raise ValueError(
                    "Missing 'temperature' parameter for metropolis method")
            points = ImageProcessing._metropolis_sampling(
                channel_data, method_args["temperature"])
        else:
            if "threshold" not in method_args:
                raise ValueError(
                    "Missing 'threshold' parameter for max_brightness method")
            points = ImageProcessing._threshold_sampling(
                channel_data, method_args["threshold"])

        return points.astype(np.float64)

    @staticmethod
    def get_image_filenames_sorted(input_folder: str) -> list:
        """Get sorted list of image filenames with natural number sorting."""
        return sorted(
            [f for f in os.listdir(input_folder)
             if os.path.splitext(f)[1].lower() in ImageProcessing.IMAGE_EXTENSIONS
             and f != ImageProcessing.DEFAULT_MASK_FILENAME],
            key=lambda x: int(''.join(filter(str.isdigit, x)))
        )

    @staticmethod
    def load_dataset_mask(input_folder: str) -> Optional[np.ndarray]:
        """Load and binarize mask file if present in directory."""
        mask_path = os.path.join(
            input_folder, ImageProcessing.DEFAULT_MASK_FILENAME)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            return mask
        return None

    @staticmethod
    def clean_point_cloud(points: np.ndarray, image_indices: np.ndarray, zfrac: Union[float, Tuple[float, ...]]) -> np.ndarray:
        """
        Removes points from `points` whose last dimension is below a certain fraction
        of the range (z-dimension). 

        Args:
            points: A NumPy array of shape (N, D) representing point cloud data.
            zfrac: A single float (or int) or a tuple of floats (or ints). 
            Represents the fractional threshold(s).

        Returns:
            A single NumPy array
        """

        if isinstance(zfrac, (float, int)):
            zfrac = (zfrac,)

        adjusted_points = points
        adjusted_imaged_indices = image_indices

        for fraction in zfrac:
            indices = adjusted_points[:, -1] - np.min(adjusted_points[:, -1]) < (
                np.max(adjusted_points[:, -1]) - np.min(adjusted_points[:, -1]))*(1 - fraction)
            adjusted_points = adjusted_points[indices]
            adjusted_imaged_indices = adjusted_imaged_indices[indices]

        return adjusted_points, adjusted_imaged_indices

    @staticmethod
    def courseGrainField(points,
                         values=None,
                         latticeSpacing=None,
                         defaultLatticeSize=100,
                         fixedBounds=None, kernel='gaussian',
                         kernelSize=5,
                         returnSpacing=False,
                         returnCorner=False):
        """
        Course grains a collection of values at arbitrary points,
        into a discrete field.

        If `values=None`, course-grained field is the point density.

        Parameters
        ----------
        points : numpy.ndarray[N,d]
            Spatial positions of N points in d-dimensional space.

        values : numpy.ndarray[N,[k]] or func(points)->numpy.ndarray[N,[k]] or None
            Field values at each point. Can be k-dimensional vector,
            resulting in k course-grained fields.

            Can also be a (vectorized) function that returns a value given
            a collection of points. eg. neighbor counting function. This
            functionality is provided such that if the function is computationally
            expensive, eg. neighbor counting, the points can be subdivided into
            batches and the course grained fields can be summed at the end. This
            is a way to approximate the course grained field for a huge (>1e6)
            number of points, while still remaining computationally feasible.
            See `subsample`.

            If `None`, returned field will be the point density.

        defaultValue : float or numpy.ndarray[k]
            The default value of the course-grained field;
            probably `0` for most applications.

        latticeSpacing : float or None
            The spacing of lattice points for the course-grained field.

            If `None`, will be chosen such that each axis has
            `defaultLatticeSize` points.

        defaultLatticeSize : int
            The number of lattice points for the course grained field, assuming
            no explicit value for the lattice spacing is given (see `latticeSpacing`).

        fixedBounds : numpy.ndarray[d] or None
            The bounds of the field to define the discretized
            grid over. If None, will be calculated based on the
            extrema of the provided points.

        kernel : str or numpy.ndarray[A,A]
            The kernel to course-grain the field with. 'gaussian'
            option is implemented as default, but a custom matrix
            can be provided. If using default gaussian option,
            kernel size can be set with `kernelSize`.

        kernelSize : int
            The kernel size to use if `kernel='gaussian'`.
            If a custom kernel is provided, this has no effect.

        returnSpacing : bool

        returnCorner : bool
        """
        dim = np.shape(points)[-1] if len(np.shape(points)) > 1 else 1

        if dim == 1:
            points = np.array(points)[:, None]

        if not hasattr(fixedBounds, '__iter__'):
            occupiedVolumeBounds = np.array(
                list(zip(np.min(points, axis=0), np.max(points, axis=0))))
        else:
            occupiedVolumeBounds = np.array(fixedBounds)

        # Create a lattice with the selected scale for that cube
        if latticeSpacing is not None:
            spacing = latticeSpacing
            if hasattr(fixedBounds, '__iter__'):
                occupiedVolumeBounds[:, 1] -= spacing
        else:
            spacing = (
                occupiedVolumeBounds[:, 1] - occupiedVolumeBounds[:, 0]) / (defaultLatticeSize-1)

        if hasattr(spacing, '__iter__'):
            spacing[spacing == 0] = 1
        else:
            spacing = spacing if spacing != 0 else 1

        fieldDims = (np.ceil(
            1 + (occupiedVolumeBounds[:, 1] - occupiedVolumeBounds[:, 0])/(spacing))).astype(np.int64)
        latticePositions = np.floor(
            (points - occupiedVolumeBounds[:, 0])/(spacing)).astype(np.int64)

        if hasattr(values, '__iter__'):
            k = np.shape(values)[-1]
            valArr = values
        else:
            k = 1
            valArr = np.ones((np.shape(points)[0], 1))

        fieldArr = np.zeros((*fieldDims, k))
        for i in range(np.shape(points)[0]):
            fieldArr[tuple(latticePositions[i])] += valArr[i]

        if kernel == 'gaussian':
            singleAxis = np.arange(kernelSize)
            kernelGrid = np.meshgrid(
                *np.repeat([singleAxis], np.shape(points)[-1], axis=0))
            kernelArr = np.exp(-np.sum([(kernelGrid[i] - (kernelSize-1)/2.) **
                               2 for i in range(np.shape(points)[-1])], axis=0) / (kernelSize))
        else:
            kernelArr = kernel

        transConvolution = np.zeros_like(fieldArr.T)

        for i in range(k):
            transConvolution[i] = convolve(
                fieldArr.T[i], kernelArr.T, mode='same') / np.sum(kernelArr)

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
    def contractPointsAlongDimension(points, dim=0, p=1, latticeSize=100) -> np.ndarray:
        """
        Contract a point cloud along a single dimension by performing
        peak sharpening on the point density field.

        This is used to correct for the finite width of the laser beam.

        Parameters
        ----------
        points : numpy.ndarray[N,d]
            The points representing the cloud.

        dim : int
            The index of the dimension along which to perform
            the contraction; should be in [0, 1, ... d).

        p : float
            Contraction parameter controlling the strength of
            contraction. Recommended to start with a small value,
            and then increase it as possible.

            Too large of a value will lead to points actually spreading
            out more because they will overshoot local high density regions.

        latticeSize : int
            The size of the lattice to break the point cloud into for calculating
            the density field. Should be large enough such that points can be found
            in the same lattice cells, but not so large that a single cell contains
            multiple structural features.

        Returns
        -------
        contractedPoints : numpy.ndarray[N,d]
            Contracted points, in the same order as the input.
        """
        densityField, fieldSpacing, fieldCorner = ImageProcessing.courseGrainField(
            points, defaultLatticeSize=latticeSize, returnSpacing=True, returnCorner=True)

        dimSizeArr = densityField.shape
        planarDimArr = [i for i in range(len(dimSizeArr)) if i != dim]
        planarDimSizeArr = [dimSizeArr[i] for i in planarDimArr]

        individualAxes = [np.arange(d) for d in planarDimSizeArr]
        planarIndices = np.array(np.meshgrid(*individualAxes)).T
        planarIndices = planarIndices.reshape(
            (np.prod(planarDimSizeArr), len(planarDimSizeArr)))

        fullIndices = [[slice(None)]*len(dimSizeArr)
                       for _ in range(len(planarIndices))]

        for i in range(len(planarIndices)):
            for j in range(len(planarDimArr)):
                fullIndices[i][planarDimArr[j]] = planarIndices[i][j]

        directionalField = np.zeros_like(densityField)

        for i in range(len(fullIndices)):
            oneDimSliceDensityField = densityField[tuple(fullIndices[i])]

            directionalField[tuple(fullIndices[i])][1:-1] = (
                oneDimSliceDensityField[2:] - oneDimSliceDensityField[:-2]) / (2 * fieldSpacing[dim])

        directionalField = directionalField / \
            np.max(np.abs(directionalField)) * p

        newPoints = np.copy(points)
        normalizedPoints = ((points - fieldCorner) /
                            fieldSpacing).astype(np.int64)

        for i in range(len(points)):
            newPoints[i, dim] += directionalField[tuple(normalizedPoints[i])]

        return newPoints
