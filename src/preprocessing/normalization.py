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


def normalize_cube(cube, valid_mask=None):
    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube.")

    if valid_mask is None:
        cube = handle_missing_values(cube)

        minimum = cube.min(axis=(0, 1), keepdims=True)
        maximum = cube.max(axis=(0, 1), keepdims=True)

        ranges = maximum - minimum
        ranges[ranges == 0] = 1.0

        normalized = (cube - minimum) / ranges
        return normalized.astype(np.float32)

    valid_mask = np.asarray(valid_mask, dtype=bool)

    if valid_mask.shape != cube.shape[:2]:
        raise ValueError(
            "Valid mask shape must match the cube's spatial dimensions."
        )

    normalized = np.full(cube.shape, np.nan, dtype=np.float32)

    for band_index in range(cube.shape[2]):
        band = cube[:, :, band_index]
        band_valid = valid_mask & np.isfinite(band)

        if not band_valid.any():
            raise ValueError(
                f"Band {band_index} contains no valid values."
            )

        values = band[band_valid]
        minimum = values.min()
        maximum = values.max()
        value_range = maximum - minimum

        if value_range == 0:
            normalized[:, :, band_index][band_valid] = 0.0
        else:
            normalized[:, :, band_index][band_valid] = (
                values - minimum
            ) / value_range

    return normalized
