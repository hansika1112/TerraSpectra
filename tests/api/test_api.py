import io

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def create_hdf5_file():
    buffer = io.BytesIO()

    with h5py.File(buffer, "w") as file:
        data = np.random.default_rng(42).random(
            (10, 10, 5)
        ).astype(np.float32)
        file.create_dataset("cube", data=data)

    buffer.seek(0)
    return buffer


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_reject_unsupported_file():
    response = client.post(
        "/api/process",
        files={"file": ("data.txt", b"invalid data", "text/plain")},
    )
    assert response.status_code == 400


def test_hdf5_dataset_selection():
    response = client.post(
        "/api/process",
        files={
            "file": (
                "sample.h5",
                create_hdf5_file(),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "dataset_selection_required"
    assert response.json()["datasets"][0]["name"] == "cube"


def test_process_hdf5_dataset():
    response = client.post(
        "/api/process",
        files={
            "file": (
                "sample.h5",
                create_hdf5_file(),
                "application/octet-stream",
            )
        },
        data={"dataset_key": "cube"},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "success"
    assert result["original_cube"]["shape"] == [10, 10, 5]
    assert result["feature_count"] == 6
    assert result["pca_components"] >= 1
    assert result["valid_pixel_count"] == 100
    assert len(result["explained_variance_ratio"]) == result["pca_components"]
    assert len(result["cumulative_variance_ratio"]) == result["pca_components"]
    assert result["cumulative_variance_ratio"][-1] == pytest.approx(
        result["explained_variance"], abs=1e-6
    )
    assert all(
        current >= previous
        for previous, current in zip(
            result["cumulative_variance_ratio"],
            result["cumulative_variance_ratio"][1:]
        )
    )


def test_export_hdf5_features_csv():
    response = client.post(
        "/api/export/features",
        files={
            "file": (
                "sample.h5",
                create_hdf5_file(),
                "application/octet-stream",
            )
        },
        data={"dataset_key": "cube"},
    )

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "terraspectra_features.csv" in response.headers[
        "content-disposition"
    ]

    import csv

    rows = list(csv.DictReader(io.StringIO(response.text)))

    assert len(rows) == 100
    assert {"row", "column"}.issubset(rows[0].keys())
    assert len(rows[0]) == 8
    assert all(
        row["row"] != "" and row["column"] != ""
        for row in rows
    )


def test_export_rejects_unsupported_file():
    response = client.post(
        "/api/export/features",
        files={
            "file": (
                "data.txt",
                b"invalid data",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_preview_hdf5_band():
    response = client.post(
        "/api/preview/band",
        files={
            "file": (
                "sample.h5",
                create_hdf5_file(),
                "application/octet-stream",
            )
        },
        data={
            "dataset_key": "cube",
            "band_index": "2",
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "success"
    assert result["band_index"] == 2
    assert result["band_count"] == 5
    assert result["original_shape"] == [10, 10]
    assert result["preview_shape"] == [10, 10]
    assert len(result["preview_data"]) == 10
    assert len(result["preview_data"][0]) == 10
    assert result["statistics"]["valid_pixels"] == 100
    assert result["statistics"]["missing_pixels"] == 0
    assert result["statistics"]["min"] <= result["statistics"]["mean"]
    assert result["statistics"]["mean"] <= result["statistics"]["max"]


def test_preview_hdf5_invalid_band_index():
    response = client.post(
        "/api/preview/band",
        files={
            "file": (
                "sample.h5",
                create_hdf5_file(),
                "application/octet-stream",
            )
        },
        data={
            "dataset_key": "cube",
            "band_index": "5",
        },
    )

    assert response.status_code == 400
    assert "Band index must be between 0 and 4" in response.json()["detail"]


def test_preview_hdf5_requires_dataset_selection():
    response = client.post(
        "/api/preview/band",
        files={
            "file": (
                "sample.h5",
                create_hdf5_file(),
                "application/octet-stream",
            )
        },
        data={"band_index": "0"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Select an HDF5 dataset first."
