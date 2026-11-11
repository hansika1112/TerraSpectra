import h5py
import numpy as np
import pytest

from src.data.ingestion import (
    inspect_hdf5,
    load_hdf5_cube,
    get_cube_summary,
    validate_file
)


def test_load_hdf5_cube(tmp_path):
    file_path = tmp_path / "sample.h5"
    expected_cube = np.random.rand(20, 30, 10)

    with h5py.File(file_path, "w") as file:
        file.create_dataset("hyperspectral_cube", data=expected_cube)

    actual_cube = load_hdf5_cube(file_path, "hyperspectral_cube")

    assert actual_cube.shape == (20, 30, 10)
    np.testing.assert_array_equal(actual_cube, expected_cube)


def test_inspect_hdf5(tmp_path):
    file_path = tmp_path / "sample.h5"

    with h5py.File(file_path, "w") as file:
        file.create_dataset("cube", data=np.zeros((10, 12, 5)))

    datasets = inspect_hdf5(file_path)

    assert len(datasets) == 1
    assert datasets[0]["name"] == "cube"
    assert datasets[0]["shape"] == (10, 12, 5)


def test_missing_dataset(tmp_path):
    file_path = tmp_path / "sample.h5"

    with h5py.File(file_path, "w") as file:
        file.create_dataset("cube", data=np.zeros((10, 12, 5)))

    with pytest.raises(KeyError):
        load_hdf5_cube(file_path, "missing_cube")


def test_invalid_dimensions(tmp_path):
    file_path = tmp_path / "sample.h5"

    with h5py.File(file_path, "w") as file:
        file.create_dataset("cube", data=np.zeros((10, 12)))

    with pytest.raises(ValueError):
        load_hdf5_cube(file_path, "cube")


def test_cube_summary():
    cube = np.ones((20, 30, 10))
    summary = get_cube_summary(cube)

    assert summary["height"] == 20
    assert summary["width"] == 30
    assert summary["spectral_bands"] == 10
    assert summary["total_values"] == 6000


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        validate_file("missing.h5")
