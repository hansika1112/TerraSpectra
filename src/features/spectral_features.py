import numpy as np


FEATURE_NAMES = (
    "spectral_mean",
    "spectral_std",
    "spectral_min",
    "spectral_max",
    "spectral_range",
    "spectral_energy"
)


def extract_spectral_features(cube, valid_mask=None):
    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube.")

    height, width, bands = cube.shape

    if bands < 1:
        raise ValueError("Cube must contain at least one spectral band.")

    finite_pixels = np.isfinite(cube).all(axis=2)

    if valid_mask is None:
        valid_mask = finite_pixels
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != (height, width):
            raise ValueError(
                "Valid mask shape must match the cube's spatial dimensions."
            )

        valid_mask = valid_mask & finite_pixels

    if not valid_mask.any():
        raise ValueError("No valid pixels available for feature extraction.")

    pixel_matrix = cube[valid_mask]

    spectral_mean = pixel_matrix.mean(axis=1)
    spectral_std = pixel_matrix.std(axis=1)
    spectral_min = pixel_matrix.min(axis=1)
    spectral_max = pixel_matrix.max(axis=1)
    spectral_range = spectral_max - spectral_min
    spectral_energy = np.sum(np.square(pixel_matrix), axis=1)

    feature_matrix = np.column_stack((
        spectral_mean,
        spectral_std,
        spectral_min,
        spectral_max,
        spectral_range,
        spectral_energy
    )).astype(np.float32)

    return {
        "features": feature_matrix,
        "feature_names": FEATURE_NAMES,
        "valid_mask": valid_mask,
        "valid_pixel_count": int(valid_mask.sum()),
        "total_pixel_count": int(height * width),
        "feature_count": len(FEATURE_NAMES)
    }


def summarize_spectral_features(feature_result):
    features = np.asarray(feature_result["features"], dtype=np.float32)

    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Invalid spectral feature matrix.")

    if features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("Feature matrix must contain finite feature values.")

    return {
        name: {
            "mean": float(features[:, index].mean()),
            "std": float(features[:, index].std()),
            "min": float(features[:, index].min()),
            "max": float(features[:, index].max())
        }
        for index, name in enumerate(FEATURE_NAMES)
    }
