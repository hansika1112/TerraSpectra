
from pathlib import Path

import h5py
import numpy as np


SUPPORTED_FORMATS = {".h5", ".hdf5", ".tif", ".tiff"}


def validate_file(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return path


def inspect_hdf5(file_path):
    path = validate_file(file_path)

    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("Expected an HDF5 file.")

    datasets = []

    with h5py.File(path, "r") as file:

        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append({
                    "name": name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype)
                })

        file.visititems(collect_datasets)

    return datasets


def load_hdf5_cube(file_path, dataset_key):
    path = validate_file(file_path)

    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("Expected an HDF5 file.")

    with h5py.File(path, "r") as file:

        if dataset_key not in file:
            raise KeyError(f"Dataset not found: {dataset_key}")

        cube = file[dataset_key][:]

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D hyperspectral cube, got {cube.ndim} dimensions."
        )

    if not np.issubdtype(cube.dtype, np.number):
        raise TypeError("Hyperspectral cube must contain numeric values.")

    return cube


def load_geotiff_cube(file_path):
    path = validate_file(file_path)

    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Expected a GeoTIFF file.")

    try:
        import rasterio
    except ImportError as error:
        raise ImportError(
            "Rasterio is required to read GeoTIFF files."
        ) from error

    with rasterio.open(path) as dataset:
        band_first_cube = dataset.read()
        cube = np.moveaxis(band_first_cube, 0, -1)

        metadata = {
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "crs": str(dataset.crs),
            "dtype": dataset.dtypes,
            "transform": tuple(dataset.transform),
            "cube_layout": "height_width_bands"
        }

    return cube, metadata


def get_cube_summary(cube):
    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube.")

    finite_values = cube[np.isfinite(cube)]

    return {
        "shape": tuple(cube.shape),
        "height": cube.shape[0],
        "width": cube.shape[1],
        "spectral_bands": cube.shape[2],
        "data_type": str(cube.dtype),
        "total_pixels": int(cube.shape[0] * cube.shape[1]),
        "total_values": int(cube.size),
        "missing_values": int(np.isnan(cube).sum()),
        "infinite_values": int(np.isinf(cube).sum()),
        "finite_min": float(finite_values.min()) if finite_values.size else None,
        "finite_max": float(finite_values.max()) if finite_values.size else None
    }
