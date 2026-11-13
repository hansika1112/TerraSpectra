from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

output_path = Path("data/sample/sample_hyperspectral.tif")
output_path.parent.mkdir(parents=True, exist_ok=True)

height = 100
width = 100
bands = 3

cube = np.random.default_rng(42).random(
    (bands, height, width)
).astype("float32")

transform = from_origin(75.0, 32.0, 0.0001, 0.0001)

with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=bands,
    dtype="float32",
    crs="EPSG:4326",
    transform=transform
) as dataset:
    dataset.write(cube)

print(f"Created: {output_path}")
print(f"Shape: {cube.shape}")
