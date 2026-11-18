import numpy as np
import pytest

from src.features.spectral_features import (
    FEATURE_NAMES,
    extract_spectral_features,
    summarize_spectral_features
)


def test_extract_spectral_features_calculations():
    cube = np.array(
        [[[1, 2, 3], [4, 5, 6]]],
        dtype=np.float32
    )

    result = extract_spectral_features(cube)

    expected = np.array([
        [2.0, np.std([1, 2, 3]), 1.0, 3.0, 2.0, 14.0],
        [5.0, np.std([4, 5, 6]), 4.0, 6.0, 2.0, 77.0]
    ])

    np.testing.assert_allclose(
        result["features"],
        expected,
        rtol=1e-5
    )


def test_feature_output_metadata():
    cube = np.ones((3, 4, 5), dtype=np.float32)

    result = extract_spectral_features(cube)

    assert result["features"].shape == (12, 6)
    assert result["feature_names"] == FEATURE_NAMES
    assert result["feature_count"] == 6
    assert result["valid_pixel_count"] == 12
    assert result["total_pixel_count"] == 12


def test_feature_extraction_excludes_invalid_pixels():
    cube = np.ones((3, 3, 4), dtype=np.float32)
    cube[0, 0] = np.nan

    result = extract_spectral_features(cube)

    assert result["valid_pixel_count"] == 8
    assert result["features"].shape == (8, 6)
    assert np.isfinite(result["features"]).all()


def test_feature_extraction_respects_valid_mask():
    cube = np.random.default_rng(42).random(
        (4, 4, 3)
    ).astype(np.float32)

    valid_mask = np.ones((4, 4), dtype=bool)
    valid_mask[0, 0] = False
    valid_mask[2, 2] = False

    result = extract_spectral_features(
        cube,
        valid_mask=valid_mask
    )

    assert result["valid_pixel_count"] == 14
    assert result["features"].shape == (14, 6)
    assert not result["valid_mask"][0, 0]
    assert not result["valid_mask"][2, 2]


def test_feature_extraction_rejects_wrong_mask_shape():
    cube = np.ones((4, 4, 3), dtype=np.float32)
    valid_mask = np.ones((3, 4), dtype=bool)

    with pytest.raises(ValueError, match="Valid mask shape"):
        extract_spectral_features(cube, valid_mask)


def test_feature_extraction_rejects_no_valid_pixels():
    cube = np.ones((4, 4, 3), dtype=np.float32)
    valid_mask = np.zeros((4, 4), dtype=bool)

    with pytest.raises(ValueError, match="No valid pixels"):
        extract_spectral_features(cube, valid_mask)


def test_summarize_spectral_features():
    cube = np.array(
        [[[1, 2, 3], [4, 5, 6]]],
        dtype=np.float32
    )

    result = extract_spectral_features(cube)
    summary = summarize_spectral_features(result)

    assert set(summary) == set(FEATURE_NAMES)
    assert summary["spectral_mean"]["mean"] == pytest.approx(3.5)
    assert summary["spectral_energy"]["mean"] == pytest.approx(45.5)


def test_summary_rejects_invalid_feature_matrix():
    result = {
        "features": np.array([[1, 2, 3]], dtype=np.float32)
    }

    with pytest.raises(ValueError, match="Invalid spectral feature matrix"):
        summarize_spectral_features(result)
