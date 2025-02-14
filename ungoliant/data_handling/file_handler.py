import os
from typing import Optional, Union, List, Dict, Tuple, Any

import h5py
import numpy as np
from PIL import Image


class FileReader:

    ###### * Main code * ######

    @staticmethod
    def load_image(
        filepath: str, valid_extensions: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Loads an image file into a numpy array.

        Args:
            filepath (str): Full path to the image file.
            valid_extensions (Optional[List[str]]): List of allowed file extensions.
                If None, defaults to common image formats.

        Returns:
            np.ndarray: A numpy array containing the image data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported (when `valid_extensions` is provided).
            RuntimeError: For unsupported file formats or errors encountered by PIL.
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
        filepath: str, dataset_filter: Optional[Union[str, List[str]]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Loads datasets and root attributes from an HDF5 file with optional substring filtering.

        Args:
            filepath (str): Path to the HDF5 file.
            dataset_filter (Optional[Union[str, List[str]]]): A string or list of strings
                used to filter datasets by substring match. If None, loads all datasets.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
                A tuple where the first element is a dictionary of filtered dataset names
                and their corresponding numpy arrays, and the second element is a dictionary
                of file-level attributes.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If there is an error reading the HDF5 file.
        """
        datasets = {}
        attributes = {}
        FileReader._validate_file_path(filepath)

        # Normalize filter input
        filter_strings = None
        if dataset_filter is not None:
            filter_strings = (
                [dataset_filter]
                if isinstance(dataset_filter, str)
                else list(dataset_filter)
            )

        try:
            with h5py.File(filepath, "r") as h5_file:
                # Collect root attributes
                attributes = dict(h5_file.attrs)

                # Visitor function with filter logic
                def _collect_filtered(name: str, obj: h5py.HLObject) -> None:
                    if isinstance(obj, h5py.Dataset):
                        if not filter_strings or any(
                            fs in name for fs in filter_strings
                        ):
                            datasets[name] = np.array(obj)

                h5_file.visititems(_collect_filtered)
                return datasets, attributes

        except Exception as e:
            raise RuntimeError(f"HDF5 read failed: {e}") from e

    ###### * Helper code * ######

    @staticmethod
    def _validate_file_path(filepath: str) -> None:
        """
        Validates that the given file path points to an existing file.

        Args:
            filepath (str): Path to the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(filepath):
            file_name = os.path.basename(filepath)
            dir_name = os.path.dirname(filepath) or "current directory"
            raise FileNotFoundError(f"File '{file_name}' not found in {dir_name}.")

    @staticmethod
    def _validate_file_extension(filepath: str, valid_extensions: List[str]) -> None:
        """
        Validates that the file extension is in the allowed list.

        Args:
            filepath (str): Path to the file.
            valid_extensions (List[str]): List of allowed file extensions.

        Raises:
            ValueError: If the file's extension is not in `valid_extensions`.
        """
        if not valid_extensions:
            return
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in valid_extensions:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Allowed: \
                    {', '.join(valid_extensions)}"
            )

class FileWriter:

    ###### * Main code * ######

    @staticmethod
    def write_HDF5(
        filepath: str,
        datasets: Dict[str, np.ndarray],
        attributes: Optional[Dict[str, Any]] = None,
        overwrite: bool = True,
        compression_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Writes datasets and optional attributes to an HDF5 file with optional compression.

        Args:
            filepath (str): Target file path where the HDF5 file will be written.
            datasets (Dict[str, np.ndarray]): Dictionary mapping dataset names to numpy arrays.
            attributes (Optional[Dict[str, Any]]): Optional dictionary of root-level
                attributes to store in the file. Defaults to None.
            overwrite (bool): If True, overwrites the file if it already exists; otherwise,
                raises an error. Defaults to True.
            compression_args (Optional[Dict[str, Any]]): A dictionary with the following keys:
                - "method" (str or None): "gzip" or None
                - "level" (int): Compression level (0 to 9). Only required if method == "gzip".

                Example:
                    {"method": "gzip", "level": 4}
                    {"method": None}

        Raises:
            RuntimeError: If the file already exists and `overwrite` is False.
            RuntimeError: If there is an error writing the HDF5 file.
        """
        FileWriter._validate_file_path(filepath, overwrite)

        # Parse the compression arguments
        compression, compression_opts = FileWriter._parse_compression_args(
            compression_args
        )

        try:
            with h5py.File(filepath, "w") as hdf_file:
                # Write datasets with optional compression settings
                for name, data in datasets.items():
                    hdf_file.create_dataset(
                        name,
                        data=data,
                        compression=compression,
                        compression_opts=compression_opts,
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
        attributes: Optional[Dict[str, Any]] = None,
        compression_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Updates an existing HDF5 file with new or modified datasets and attributes.
        Allows modifying or adding compression settings for new or replaced datasets.

        Args:
            filepath (str): Path to the existing HDF5 file.
            datasets (Optional[Dict[str, np.ndarray]]): Dictionary of dataset names
                mapped to new data. If a dataset exists, it will be updated in-place if
                possible; otherwise it is replaced. When replaced, new compression
                settings will be applied.
            attributes (Optional[Dict[str, Any]]): Dictionary of attributes to add or update
                at the root level of the file.
            compression_args (Optional[Dict[str, Any]]): A dictionary with:
                - "method" (str or None): "gzip" or None
                - "level" (int): Compression level (0 to 9) if method=="gzip".

        Raises:
            FileNotFoundError: If the HDF5 file does not exist.
            RuntimeError: If there is an error updating the HDF5 file.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File '{filepath}' does not exist.")

        # Parse the compression arguments
        compression, compression_opts = FileWriter._parse_compression_args(
            compression_args
        )

        try:
            with h5py.File(filepath, "r+") as hdf_file:
                # Update datasets
                if datasets:
                    for name, data in datasets.items():
                        if name in hdf_file:
                            ds = hdf_file[name]
                            # Attempt in-place update if shape/dtype match
                            # and compression parameters are unchanged
                            same_shape = ds.shape == data.shape
                            same_dtype = ds.dtype == data.dtype
                            same_compression = ds.compression == compression
                            same_compression_opts = (
                                ds.compression_opts == compression_opts
                            )

                            # If everything is compatible, update in place
                            if all(
                                [
                                    same_shape,
                                    same_dtype,
                                    same_compression,
                                    same_compression_opts,
                                ]
                            ):
                                ds[...] = data
                            else:
                                # Delete and recreate dataset with new compression settings
                                del hdf_file[name]
                                hdf_file.create_dataset(
                                    name,
                                    data=data,
                                    compression=compression,
                                    compression_opts=compression_opts,
                                )
                        else:
                            # Create new dataset with specified compression
                            hdf_file.create_dataset(
                                name,
                                data=data,
                                compression=compression,
                                compression_opts=compression_opts,
                            )

                # Update attributes
                if attributes:
                    for key, value in attributes.items():
                        hdf_file.attrs[key] = value
        except Exception as e:
            raise RuntimeError(f"Failed to update HDF5 file: {e}") from e

    ###### * Helper code * ######

    @staticmethod
    def _validate_file_path(filepath: str, overwrite: bool) -> None:
        """
        Ensures that the output directory exists and handles file overwrite permissions.

        Args:
            filepath (str): Target file path for writing.
            overwrite (bool): If True, allows overwriting an existing file.

        Raises:
            RuntimeError: If the file already exists and `overwrite` is False.
        """
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if os.path.isfile(filepath) and not overwrite:
            raise RuntimeError(
                f"File '{filepath}' exists. Set overwrite=True to replace it."
            )

    @staticmethod
    def _parse_compression_args(
        compression_args: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse the compression_args dictionary to return HDF5-compatible
        compression and compression_opts values.

        Args:
            compression_args (Optional[Dict[str, Any]]): A dictionary that may contain:
                {
                    "method": "gzip" or None,
                    "level": 0-9 (required if method="gzip")
                }

        Returns:
            (compression, compression_opts) suitable for h5py create_dataset().
        """
        if not compression_args:
            # No compression_args provided
            return None, None

        method = compression_args.get("method", None)
        level = compression_args.get("level", None)

        # Validate method
        if method not in (None, "gzip"):
            raise ValueError(
                "Invalid compression_args['method']. Must be either None or 'gzip'."
            )

        # If using gzip, validate level is within 0-9
        if method == "gzip":
            if level is None or not (0 <= level <= 9):
                raise ValueError(
                    "Compression level must be an integer from 0 to 9 for gzip."
                )
            return "gzip", level

        # Otherwise, no compression
        return None, None
