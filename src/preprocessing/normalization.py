import numpy as np


def handle_missing_values(cube):
    cube = np.asarray(cube, dtype=np.float32).copy()

    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube.")

    for band_index in range(cube.shape[2]):
        band = cube[:, :, band_index]
        valid_mask = np.isfinite(band)

        if not valid_mask.any():
            raise ValueError(
                f"Band {band_index} contains no valid values."
            )

        band_mean = band[valid_mask].mean()
        band[~valid_mask] = band_mean

    return cube


def normalize_cube(cube):
    cube = handle_missing_values(cube)

    minimum = cube.min(axis=(0, 1), keepdims=True)
    maximum = cube.max(axis=(0, 1), keepdims=True)

    ranges = maximum - minimum
    ranges[ranges == 0] = 1.0

    normalized = (cube - minimum) / ranges

    return normalized.astype(np.float32)
