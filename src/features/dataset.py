from pathlib import Path

import numpy as np
import pandas as pd


def build_feature_dataframe(feature_result):
    if not isinstance(feature_result, dict):
        raise TypeError("feature_result must be a dictionary.")

    required_keys = {"spectral_features", "feature_names", "valid_mask"}
    missing_keys = required_keys - feature_result.keys()

    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    features = np.asarray(feature_result["spectral_features"])
    feature_names = list(feature_result["feature_names"])
    valid_mask = np.asarray(feature_result["valid_mask"])

    if features.ndim != 2:
        raise ValueError("Spectral features must be a 2D matrix.")

    if valid_mask.ndim != 2:
        raise ValueError("Valid mask must be a 2D array.")

    if len(feature_names) != features.shape[1]:
        raise ValueError("Feature names must match the number of feature columns.")

    if not np.issubdtype(valid_mask.dtype, np.bool_):
        raise TypeError("Valid mask must contain boolean values.")

    valid_coordinates = np.argwhere(valid_mask)

    if features.shape[0] != len(valid_coordinates):
        raise ValueError(
            "Feature row count must match the number of valid pixels."
        )

    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains non-finite values.")

    dataframe = pd.DataFrame(features, columns=feature_names)

    dataframe.insert(0, "column", valid_coordinates[:, 1])
    dataframe.insert(0, "row", valid_coordinates[:, 0])

    return dataframe


def export_feature_dataset(feature_result, output_path):
    dataframe = build_feature_dataframe(feature_result)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_path, index=False)

    return dataframe