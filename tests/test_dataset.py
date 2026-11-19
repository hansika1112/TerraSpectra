import numpy as np
import pandas as pd
import pytest

from src.features.dataset import (
    build_feature_dataframe,
    export_feature_dataset,
)


def create_feature_result():
    valid_mask = np.array(
        [
            [True, False, True],
            [False, True, True],
        ]
    )

    features = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )

    return {
        "spectral_features": features,
        "feature_names": ["feature_1", "feature_2", "feature_3"],
        "valid_mask": valid_mask,
    }


def test_build_feature_dataframe_shape():
    dataframe = build_feature_dataframe(create_feature_result())

    assert dataframe.shape == (4, 5)


def test_build_feature_dataframe_coordinates():
    dataframe = build_feature_dataframe(create_feature_result())

    expected_coordinates = [
        (0, 0),
        (0, 2),
        (1, 1),
        (1, 2),
    ]

    actual_coordinates = list(
        zip(dataframe["row"], dataframe["column"])
    )

    assert actual_coordinates == expected_coordinates


def test_build_feature_dataframe_columns():
    dataframe = build_feature_dataframe(create_feature_result())

    expected_columns = [
        "row",
        "column",
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    assert list(dataframe.columns) == expected_columns


def test_build_feature_dataframe_values():
    dataframe = build_feature_dataframe(create_feature_result())

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )

    np.testing.assert_array_equal(
        dataframe.iloc[:, 2:].to_numpy(),
        expected,
    )


def test_export_feature_dataset(tmp_path):
    output_path = tmp_path / "output" / "features.csv"

    dataframe = export_feature_dataset(
        create_feature_result(),
        output_path,
    )

    assert output_path.exists()
    assert dataframe.shape == (4, 5)


def test_export_feature_dataset_csv_roundtrip(tmp_path):
    output_path = tmp_path / "features.csv"

    export_feature_dataset(create_feature_result(), output_path)

    loaded_dataframe = pd.read_csv(output_path)
    expected_dataframe = build_feature_dataframe(create_feature_result())

    pd.testing.assert_frame_equal(
        loaded_dataframe,
        expected_dataframe,
    )


def test_invalid_feature_row_count():
    feature_result = create_feature_result()
    feature_result["spectral_features"] = np.array(
        [[1.0, 2.0, 3.0]]
    )

    with pytest.raises(ValueError):
        build_feature_dataframe(feature_result)


def test_feature_name_mismatch():
    feature_result = create_feature_result()
    feature_result["feature_names"] = ["feature_1"]

    with pytest.raises(ValueError):
        build_feature_dataframe(feature_result)


def test_invalid_mask_dimensions():
    feature_result = create_feature_result()
    feature_result["valid_mask"] = np.array([True, False])

    with pytest.raises(ValueError):
        build_feature_dataframe(feature_result)


def test_non_boolean_mask():
    feature_result = create_feature_result()
    feature_result["valid_mask"] = np.array(
        [[1, 0, 1], [0, 1, 1]]
    )

    with pytest.raises(TypeError):
        build_feature_dataframe(feature_result)


def test_non_finite_features():
    feature_result = create_feature_result()
    feature_result["spectral_features"][0, 0] = np.nan

    with pytest.raises(ValueError):
        build_feature_dataframe(feature_result)


def test_missing_required_keys():
    with pytest.raises(ValueError):
        build_feature_dataframe({})