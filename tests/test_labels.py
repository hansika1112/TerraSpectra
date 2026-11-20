import numpy as np
import pandas as pd
import pytest

from src.features.labels import integrate_ground_truth_labels


def create_feature_dataframe():
    return pd.DataFrame(
        {
            "row": [0, 0, 1, 1],
            "column": [0, 1, 0, 1],
            "spectral_mean": [0.1, 0.2, 0.3, 0.4],
            "spectral_std": [0.01, 0.02, 0.03, 0.04],
        }
    )


def create_label_map():
    return np.array(
        [
            [1, 0],
            [2, 3],
        ],
        dtype=int,
    )


def test_integrate_ground_truth_labels():
    result = integrate_ground_truth_labels(
        create_feature_dataframe(),
        create_label_map(),
    )

    assert result.shape[0] == 3
    assert "label" in result.columns


def test_unlabeled_pixels_are_excluded():
    result = integrate_ground_truth_labels(
        create_feature_dataframe(),
        create_label_map(),
    )

    assert len(result) == 3
    assert not ((result["row"] == 0) & (result["column"] == 1)).any()


def test_labels_match_pixel_coordinates():
    result = integrate_ground_truth_labels(
        create_feature_dataframe(),
        create_label_map(),
    )

    expected_labels = [1, 2, 3]

    assert result["label"].tolist() == expected_labels


def test_feature_columns_are_preserved():
    result = integrate_ground_truth_labels(
        create_feature_dataframe(),
        create_label_map(),
    )

    expected_columns = [
        "row",
        "column",
        "spectral_mean",
        "spectral_std",
        "label",
    ]

    assert list(result.columns) == expected_columns


def test_result_index_is_reset():
    dataframe = create_feature_dataframe()
    dataframe.index = [10, 20, 30, 40]

    result = integrate_ground_truth_labels(
        dataframe,
        create_label_map(),
    )

    assert result.index.tolist() == [0, 1, 2]


def test_reject_non_dataframe():
    with pytest.raises(TypeError):
        integrate_ground_truth_labels(
            np.array([[0, 0]]),
            create_label_map(),
        )


def test_reject_missing_coordinates():
    dataframe = pd.DataFrame({"spectral_mean": [0.1]})

    with pytest.raises(ValueError):
        integrate_ground_truth_labels(
            dataframe,
            create_label_map(),
        )


def test_reject_wrong_label_map_dimensions():
    with pytest.raises(ValueError):
        integrate_ground_truth_labels(
            create_feature_dataframe(),
            np.array([1, 2, 3]),
        )


def test_reject_non_integer_labels():
    label_map = np.array(
        [
            [1.5, 0.0],
            [2.0, 3.0],
        ]
    )

    with pytest.raises(TypeError):
        integrate_ground_truth_labels(
            create_feature_dataframe(),
            label_map,
        )


def test_reject_negative_labels():
    label_map = np.array(
        [
            [1, -1],
            [2, 3],
        ]
    )

    with pytest.raises(ValueError):
        integrate_ground_truth_labels(
            create_feature_dataframe(),
            label_map,
        )


def test_reject_out_of_bounds_coordinates():
    dataframe = create_feature_dataframe()
    dataframe.loc[0, "row"] = 5

    with pytest.raises(ValueError):
        integrate_ground_truth_labels(
            dataframe,
            create_label_map(),
        )


def test_reject_duplicate_coordinates():
    dataframe = create_feature_dataframe()
    dataframe.loc[1, ["row", "column"]] = [0, 0]

    with pytest.raises(ValueError):
        integrate_ground_truth_labels(
            dataframe,
            create_label_map(),
        )


def test_reject_no_labeled_pixels():
    label_map = np.zeros((2, 2), dtype=int)

    with pytest.raises(ValueError):
        integrate_ground_truth_labels(
            create_feature_dataframe(),
            label_map,
        )


def test_reject_non_integer_row_coordinates():
    dataframe = create_feature_dataframe()
    dataframe["row"] = dataframe["row"].astype(float)

    with pytest.raises(TypeError):
        integrate_ground_truth_labels(
            dataframe,
            create_label_map(),
        )