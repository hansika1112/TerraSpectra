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
