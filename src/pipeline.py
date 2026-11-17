import numpy as np

from src.data.ingestion import load_geotiff_cube
from src.preprocessing.normalization import normalize_cube
from src.dimensionality_reduction.pca import apply_pca


def preprocess_hyperspectral_cube(
    cube,
    valid_mask=None,
    n_components=0.95
):
    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube.")

    height, width, _ = cube.shape

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != (height, width):
            raise ValueError(
                "Valid mask shape must match the cube's spatial dimensions."
            )

        valid_mask = valid_mask & np.isfinite(cube).all(axis=2)

        if not valid_mask.any():
            raise ValueError("No valid pixels available for preprocessing.")

    normalized_cube = normalize_cube(
        cube,
        valid_mask=valid_mask
    )

    reduced_cube, pca_model = apply_pca(
        normalized_cube,
        n_components=n_components,
        valid_mask=valid_mask
    )

    return {
        "normalized_cube": normalized_cube,
        "reduced_cube": reduced_cube,
        "pca_model": pca_model,
        "valid_mask": valid_mask,
        "original_shape": cube.shape,
        "reduced_shape": reduced_cube.shape,
        "explained_variance_ratio": pca_model.explained_variance_ratio_,
        "explained_variance": pca_model.explained_variance_ratio_.sum()
    }


def preprocess_geotiff(
    file_path,
    n_components=0.95
):
    cube, metadata = load_geotiff_cube(file_path)

    results = preprocess_hyperspectral_cube(
        cube,
        valid_mask=metadata["valid_mask"],
        n_components=n_components
    )

    results["metadata"] = metadata

    return results
