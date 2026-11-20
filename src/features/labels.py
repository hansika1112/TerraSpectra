import numpy as np
import pandas as pd


def integrate_ground_truth_labels(feature_dataframe, label_map):
    if not isinstance(feature_dataframe, pd.DataFrame):
        raise TypeError("Feature dataset must be a Pandas DataFrame.")

    required_columns = {"row", "column"}

    if not required_columns.issubset(feature_dataframe.columns):
        raise ValueError("Feature dataset must contain row and column columns.")

    label_map = np.asarray(label_map)

    if label_map.ndim != 2:
        raise ValueError("Ground-truth label map must be a 2D array.")

    if not np.issubdtype(label_map.dtype, np.integer):
        raise TypeError("Ground-truth labels must be integers.")

    if (label_map < 0).any():
        raise ValueError("Ground-truth labels cannot be negative.")

    rows = feature_dataframe["row"].to_numpy()
    columns = feature_dataframe["column"].to_numpy()

    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("Pixel row coordinates must be integers.")

    if not np.issubdtype(columns.dtype, np.integer):
        raise TypeError("Pixel column coordinates must be integers.")

    if (
        (rows < 0).any()
        or (columns < 0).any()
        or (rows >= label_map.shape[0]).any()
        or (columns >= label_map.shape[1]).any()
    ):
        raise ValueError("Feature coordinates exceed label map boundaries.")

    if feature_dataframe.duplicated(["row", "column"]).any():
        raise ValueError("Feature dataset contains duplicate pixel coordinates.")

    labels = label_map[rows, columns]

    labeled_mask = labels > 0

    labeled_dataframe = feature_dataframe.loc[labeled_mask].copy()

    labeled_dataframe["label"] = labels[labeled_mask]

    if labeled_dataframe.empty:
        raise ValueError("No labeled pixels found in the dataset.")

    return labeled_dataframe.reset_index(drop=True)