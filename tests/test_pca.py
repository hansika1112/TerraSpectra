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


def test_mask_aware_pca_excludes_invalid_pixels():
    cube = np.random.default_rng(42).random(
        (4, 4, 3)
    ).astype(np.float32)

    valid_mask = np.ones((4, 4), dtype=bool)
    valid_mask[0, 0] = False
    cube[0, 0] = 1000000.0

    valid_pixels = cube[valid_mask].copy()

    reduced, model = apply_pca(
        cube,
        n_components=2,
        valid_mask=valid_mask
    )

    np.testing.assert_allclose(
        model.mean_,
        valid_pixels.mean(axis=0),
        rtol=1e-5
    )
    assert reduced.shape == (4, 4, 2)


def test_mask_aware_pca_preserves_invalid_pixels():
    cube = np.random.default_rng(42).random(
        (5, 5, 4)
    ).astype(np.float32)

    valid_mask = np.ones((5, 5), dtype=bool)
    valid_mask[0, 0] = False
    valid_mask[2, 3] = False

    reduced, _ = apply_pca(
        cube,
        n_components=2,
        valid_mask=valid_mask
    )

    assert np.isnan(reduced[0, 0]).all()
    assert np.isnan(reduced[2, 3]).all()
    assert np.isfinite(reduced[valid_mask]).all()


def test_mask_aware_pca_rejects_wrong_mask_shape():
    cube = np.ones((5, 5, 3), dtype=np.float32)
    valid_mask = np.ones((4, 5), dtype=bool)

    with pytest.raises(ValueError, match="Valid mask shape"):
        apply_pca(cube, valid_mask=valid_mask)


def test_mask_aware_pca_rejects_no_valid_pixels():
    cube = np.ones((5, 5, 3), dtype=np.float32)
    valid_mask = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError, match="No valid pixels"):
        apply_pca(cube, valid_mask=valid_mask)


def test_mask_aware_pca_rejects_insufficient_valid_samples():
    cube = np.random.default_rng(42).random(
        (3, 3, 3)
    ).astype(np.float32)

    valid_mask = np.zeros((3, 3), dtype=bool)
    valid_mask[0, 0] = True

    with pytest.raises(ValueError, match="Invalid number"):
        apply_pca(
            cube,
            n_components=2,
            valid_mask=valid_mask
        )
