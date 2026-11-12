import numpy as np
from sklearn.decomposition import PCA

from src.preprocessing.normalization import handle_missing_values


def apply_pca(cube, n_components=0.95):
    cube = handle_missing_values(cube)

    height, width, bands = cube.shape
    pixel_matrix = cube.reshape(-1, bands)

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

    reduced_cube = reduced_matrix.reshape(
        height,
        width,
        model.n_components_
    )

    return reduced_cube, model
