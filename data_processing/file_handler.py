import os
import h5py
import numpy as np
from PIL import Image
from typing import Optional, Union, List, Dict, Tuple


class FileReader:

    def __init__(self) -> None:
        pass

    ###### * Main code * ######

    @staticmethod
    def load_image(filepath: str, valid_extensions: Optional[List[str]] = None) -> np.ndarray:
        """
        Loads an image file into a numpy array.

        Args:
            filepath: Full path to the image file.
            valid_extensions: List of allowed file extensions. Defaults to common image formats.

        Returns:
            Numpy array containing image data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file extension is not supported.
            RuntimeError: For unsupported file formats or PIL errors.
        """
        FileReader._validate_file_path(filepath)
        
        if valid_extensions is not None:
            FileReader._validate_file_extension(filepath, valid_extensions)

        try:
            with Image.open(filepath) as img:
                return np.array(img, dtype=np.uint8)
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}") from e

    @staticmethod
    def load_HDF5(
            filepath: str,
            dataset_filter: Optional[Union[str, List[str]]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Loads datasets and root attributes from an HDF5 file with optional filtering.

        Args:
            filepath: Path to the HDF5 file
            dataset_filter: String/list of strings to filter datasets (substring match). 
                     If None, loads all datasets.

        Returns:
            Tuple of (filtered_datasets, attributes)

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: For HDF5 read errors
        """
        datasets = {}
        attributes = {}
        FileReader._validate_file_path(filepath)

        # Normalize filter input
        filter_strings = None
        if dataset_filter is not None:
            filter_strings = [dataset_filter] if isinstance(
                dataset_filter, str) else list(dataset_filter)

        try:
            with h5py.File(filepath, 'r') as h5_file:
                # Collect root attributes
                attributes = dict(h5_file.attrs)

                # Visitor function with filter logic
                def _collect_filtered(name: str, obj: h5py.HLObject) -> None:
                    if isinstance(obj, h5py.Dataset):
                        if not filter_strings or any(fs in name for fs in filter_strings):
                            datasets[name] = np.array(obj)

                h5_file.visititems(_collect_filtered)
                return datasets, attributes

        except Exception as e:
            raise RuntimeError(f"HDF5 read failed: {e}") from e

    ###### * Helper code * ######

    @staticmethod
    def _validate_file_path(filepath: str) -> None:
        """Validates that the file exists."""
        if not os.path.isfile(filepath):
            file_name = os.path.basename(filepath)
            dir_name = os.path.dirname(filepath) or 'current directory'
            raise FileNotFoundError(
                f"File '{file_name}' not found in {dir_name}."
            )

    @staticmethod
    def _validate_file_extension(filepath: str, valid_extensions: List[str]) -> None:
        """Validates the file extension against allowed list."""
        if not valid_extensions:
            return
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in valid_extensions:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Allowed: \
                    {', '.join(valid_extensions)}"
            )


class FileWriter:

    def __init__(self) -> None:
        pass

    ###### * Main code * ######

    @staticmethod
    def write_HDF5(
            filepath: str,
            datasets: Dict[str, np.ndarray],
            attributes: Optional[Dict[str, object]] = None,
            overwrite: bool = True) -> None:
        """
        Writes datasets and attributes to an HDF5 file.

        Args:
            filepath: Target file path.
            datasets: Dictionary of dataset names and their data.
            attributes: Root attributes to store. Defaults to None.
            overwrite: Overwrite existing file. Defaults to True.

        Raises:
            RuntimeError: If file exists and overwrite is False.
        """
        FileWriter._validate_output_path(filepath, overwrite)

        try:
            with h5py.File(filepath, 'w') as hdf_file:
                # Write datasets
                for name, data in datasets.items():
                    hdf_file.create_dataset(
                        name,
                        data=data,
                    )

                # Write attributes
                if attributes:
                    for key, value in attributes.items():
                        hdf_file.attrs[key] = value
        except Exception as e:
            raise RuntimeError(f"Failed to write HDF5 file: {e}") from e

    @staticmethod
    def update_HDF5(
            filepath: str,
            datasets: Optional[Dict[str, np.ndarray]] = None,
            attributes: Optional[Dict[str, object]] = None) -> None:
        """
        Updates existing HDF5 file with new datasets/attributes.

        Args:
            filepath: Path to existing HDF5 file.
            datasets: Datasets to add/update.
            attributes: Root attributes to add/update.

        Raises:
            FileNotFoundError: If file does not exist.
            RuntimeError: If HDF5 update fails.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File '{filepath}' does not exist.")

        try:
            with h5py.File(filepath, 'r+') as hdf_file:
                # Update datasets
                if datasets:
                    for name, data in datasets.items():
                        if name in hdf_file:
                            # Attempt in-place update if shape/dtype match
                            ds = hdf_file[name]
                            if ds.shape == data.shape and ds.dtype == data.dtype:
                                ds[...] = data
                            else:
                                del hdf_file[name]
                                hdf_file.create_dataset(name, data=data)
                        else:
                            hdf_file.create_dataset(name, data=data)

                # Update attributes
                if attributes:
                    for key, value in attributes.items():
                        hdf_file.attrs[key] = value
        except Exception as e:
            raise RuntimeError(f"Failed to update HDF5 file: {e}") from e

    ###### * Helper code * ######

    @staticmethod
    def _validate_output_path(filepath: str, overwrite: bool) -> None:
        """Ensures output directory exists and handles overwrite permissions."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if os.path.isfile(filepath) and not overwrite:
            raise RuntimeError(
                f"File '{filepath}' exists. Set overwrite=True to replace it."
            )
