import numpy as np
from sklearn.decomposition import PCA

from src.preprocessing.normalization import handle_missing_values


def apply_pca(cube, n_components=0.95, valid_mask=None):
    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube.")

    height, width, bands = cube.shape

    if valid_mask is None:
        cube = handle_missing_values(cube)
        pixel_matrix = cube.reshape(-1, bands)
        output_mask = np.ones((height, width), dtype=bool)

    else:
        output_mask = np.asarray(valid_mask, dtype=bool)

        if output_mask.shape != (height, width):
            raise ValueError(
                "Valid mask shape must match the cube's spatial dimensions."
            )

        output_mask = output_mask & np.isfinite(cube).all(axis=2)
        pixel_matrix = cube[output_mask]

        if pixel_matrix.shape[0] == 0:
            raise ValueError("No valid pixels available for PCA.")

    if isinstance(n_components, int):
        if n_components < 1 or n_components > min(pixel_matrix.shape):
            raise ValueError("Invalid number of PCA components.")

    elif isinstance(n_components, float):
        if not 0 < n_components <= 1:
            raise ValueError(
                "Float n_components must be between 0 and 1."
            )
    else:
        raise TypeError("n_components must be an integer or float.")

    model = PCA(n_components=n_components)
    reduced_matrix = model.fit_transform(pixel_matrix)

    output_components = model.n_components_

    reduced_cube = np.full(
        (height, width, output_components),
        np.nan,
        dtype=np.float32
    )

    reduced_cube[output_mask] = reduced_matrix.astype(np.float32)

    return reduced_cube, model
