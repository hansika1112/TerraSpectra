import numpy as np
import pytest

from src.pipeline import (
    preprocess_hyperspectral_cube,
    preprocess_geotiff
)


SAMPLE_PATH = "data/sample/sample_hyperspectral.tif"


def test_preprocess_hyperspectral_cube():
    cube = np.random.default_rng(42).random(
        (10, 10, 4)
    ).astype(np.float32)

    results = preprocess_hyperspectral_cube(
        cube,
        n_components=2
    )

    assert results["normalized_cube"].shape == (10, 10, 4)
    assert results["reduced_cube"].shape == (10, 10, 2)
    assert results["pca_model"].n_components_ == 2
    assert np.isfinite(results["reduced_cube"]).all()


def test_preprocessing_preserves_invalid_pixels():
    cube = np.random.default_rng(42).random(
        (10, 10, 4)
    ).astype(np.float32)

    valid_mask = np.ones((10, 10), dtype=bool)
    valid_mask[0, 0] = False
    valid_mask[3, 4] = False

    results = preprocess_hyperspectral_cube(
        cube,
        valid_mask=valid_mask,
        n_components=2
    )

    assert np.isnan(results["normalized_cube"][0, 0]).all()
    assert np.isnan(results["normalized_cube"][3, 4]).all()
    assert np.isnan(results["reduced_cube"][0, 0]).all()
    assert np.isnan(results["reduced_cube"][3, 4]).all()
    assert np.isfinite(results["reduced_cube"][valid_mask]).all()


def test_preprocessing_rejects_wrong_mask_shape():
    cube = np.ones((10, 10, 3), dtype=np.float32)
    valid_mask = np.ones((9, 10), dtype=bool)

    with pytest.raises(ValueError, match="Valid mask shape"):
        preprocess_hyperspectral_cube(
            cube,
            valid_mask=valid_mask
        )


def test_preprocessing_rejects_no_valid_pixels():
    cube = np.ones((10, 10, 3), dtype=np.float32)
    valid_mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError, match="No valid pixels"):
        preprocess_hyperspectral_cube(
            cube,
            valid_mask=valid_mask
        )


def test_preprocess_geotiff_end_to_end():
    results = preprocess_geotiff(
        SAMPLE_PATH,
        n_components=2
    )

    assert results["original_shape"] == (100, 100, 3)
    assert results["reduced_shape"] == (100, 100, 2)
    assert results["metadata"]["crs"] == "EPSG:4326"
    assert results["valid_mask"].shape == (100, 100)
    assert np.isfinite(results["reduced_cube"]).all()
    assert results["explained_variance"] > 0
