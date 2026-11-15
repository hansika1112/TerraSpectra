import numpy as np
import pytest

from src.preprocessing.normalization import (
    handle_missing_values,
    normalize_cube
)


def test_normalize_cube_range():
    cube = np.random.default_rng(42).random(
        (20, 30, 5)
    ).astype(np.float32)

    normalized = normalize_cube(cube)

    assert normalized.shape == cube.shape
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_each_band_independently():
    cube = np.zeros((10, 10, 2), dtype=np.float32)
    cube[:, :, 0] = np.arange(100).reshape(10, 10)
    cube[:, :, 1] = np.arange(100).reshape(10, 10) * 10

    normalized = normalize_cube(cube)

    assert normalized[:, :, 0].min() == 0.0
    assert normalized[:, :, 0].max() == 1.0
    assert normalized[:, :, 1].min() == 0.0
    assert normalized[:, :, 1].max() == 1.0


def test_handle_missing_values():
    cube = np.ones((5, 5, 2), dtype=np.float32)
    cube[0, 0, 0] = np.nan
    cube[1, 1, 1] = np.inf

    processed = handle_missing_values(cube)

    assert np.isfinite(processed).all()
    assert processed[0, 0, 0] == 1.0
    assert processed[1, 1, 1] == 1.0


def test_constant_band_normalization():
    cube = np.ones((5, 5, 2), dtype=np.float32) * 7

    normalized = normalize_cube(cube)

    assert np.isfinite(normalized).all()
    assert np.all(normalized == 0.0)


def test_reject_invalid_dimensions():
    cube = np.ones((10, 10))

    with pytest.raises(ValueError):
        normalize_cube(cube)


def test_reject_completely_invalid_band():
    cube = np.ones((5, 5, 2), dtype=np.float32)
    cube[:, :, 1] = np.nan

    with pytest.raises(ValueError):
        normalize_cube(cube)


def test_mask_aware_normalization_excludes_invalid_pixels():
    cube = np.array(
        [
            [[1, 10], [3, 30]],
            [[5, 50], [9999, 9999]]
        ],
        dtype=np.float32
    )
    valid_mask = np.array(
        [[True, True], [True, False]]
    )

    normalized = normalize_cube(cube, valid_mask=valid_mask)

    assert normalized[0, 0, 0] == 0.0
    assert normalized[1, 0, 0] == 1.0
    assert np.isnan(normalized[1, 1]).all()
    assert normalized[0, 0, 1] == 0.0
    assert normalized[1, 0, 1] == 1.0


def test_mask_aware_normalization_preserves_nan_pixels():
    cube = np.array(
        [[[1.0], [np.nan]], [[3.0], [5.0]]],
        dtype=np.float32
    )
    valid_mask = np.ones((2, 2), dtype=bool)

    normalized = normalize_cube(cube, valid_mask=valid_mask)

    assert np.isnan(normalized[0, 1, 0])
    assert normalized[0, 0, 0] == 0.0
    assert normalized[1, 1, 0] == 1.0


def test_mask_aware_normalization_rejects_wrong_mask_shape():
    cube = np.ones((5, 5, 2), dtype=np.float32)
    valid_mask = np.ones((4, 5), dtype=bool)

    with pytest.raises(ValueError, match="Valid mask shape"):
        normalize_cube(cube, valid_mask=valid_mask)


def test_mask_aware_normalization_rejects_empty_valid_band():
    cube = np.ones((3, 3, 2), dtype=np.float32)
    valid_mask = np.zeros((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="no valid values"):
        normalize_cube(cube, valid_mask=valid_mask)
