import numpy as np
import pytest

from src.dimensionality_reduction.pca import apply_pca


def test_pca_reduces_spectral_bands():
    cube = np.random.default_rng(42).random(
        (20, 20, 10)
    ).astype(np.float32)

    reduced, model = apply_pca(cube, n_components=3)

    assert reduced.shape == (20, 20, 3)
    assert model.n_components_ == 3


def test_pca_preserves_spatial_dimensions():
    cube = np.random.default_rng(42).random(
        (15, 25, 8)
    ).astype(np.float32)

    reduced, _ = apply_pca(cube, n_components=2)

    assert reduced.shape[:2] == (15, 25)


def test_pca_variance_threshold():
    cube = np.random.default_rng(42).random(
        (20, 20, 6)
    ).astype(np.float32)

    reduced, model = apply_pca(cube, n_components=0.95)

    assert reduced.shape[2] == model.n_components_
    assert model.explained_variance_ratio_.sum() >= 0.95


def test_pca_rejects_invalid_components():
    cube = np.ones((10, 10, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        apply_pca(cube, n_components=10)


def test_pca_rejects_invalid_dimensions():
    cube = np.ones((10, 10))

    with pytest.raises(ValueError):
        apply_pca(cube)


def test_pca_handles_missing_values():
    cube = np.random.default_rng(42).random(
        (10, 10, 5)
    ).astype(np.float32)

    cube[0, 0, 0] = np.nan

    reduced, _ = apply_pca(cube, n_components=3)

    assert np.isfinite(reduced).all()
    assert reduced.shape == (10, 10, 3)
