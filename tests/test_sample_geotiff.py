from pathlib import Path

import numpy as np
import rasterio


SAMPLE_PATH = Path("data/sample/sample_hyperspectral.tif")


def test_sample_geotiff_exists():
    assert SAMPLE_PATH.is_file()


def test_sample_geotiff_dimensions_and_bands():
    with rasterio.open(SAMPLE_PATH) as dataset:
        assert dataset.count == 3
        assert dataset.height == 100
        assert dataset.width == 100


def test_sample_geotiff_crs():
    with rasterio.open(SAMPLE_PATH) as dataset:
        assert dataset.crs.to_string() == "EPSG:4326"


def test_sample_geotiff_dtype():
    with rasterio.open(SAMPLE_PATH) as dataset:
        assert all(dtype == "float32" for dtype in dataset.dtypes)


def test_sample_geotiff_pixel_values_are_finite():
    with rasterio.open(SAMPLE_PATH) as dataset:
        cube = dataset.read()

    assert np.isfinite(cube).all()
