from pathlib import Path

import numpy as np
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from src.data.ingestion import (
    inspect_hdf5,
    load_hdf5_cube,
    load_geotiff_cube,
    get_cube_summary,
)
from src.pipeline import preprocess_hyperspectral_cube
from src.features.dataset import export_feature_dataset

app = FastAPI(
    title="TerraSpectra API",
    description="Hyperspectral data ingestion and processing API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_FORMATS = {".h5", ".hdf5", ".tif", ".tiff"}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024


@app.get("/")
def root():
    return {
        "message": "TerraSpectra API is running",
        "status": "healthy",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/process")
async def process_dataset(
    file: UploadFile = File(...),
    dataset_key: str = Form(default=""),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload TIFF or HDF5.",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            size = 0

            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 100 MB upload limit.",
                    )

                temp_file.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if extension in {".h5", ".hdf5"}:
            datasets = inspect_hdf5(temp_path)

            if not datasets:
                raise HTTPException(
                    status_code=400,
                    detail="No datasets found in the HDF5 file.",
                )

            if not dataset_key:
                return {
                    "status": "dataset_selection_required",
                    "filename": file.filename,
                    "datasets": [
                        {
                            "name": item["name"],
                            "shape": list(item["shape"]),
                            "dtype": item["dtype"],
                        }
                        for item in datasets
                    ],
                }

            cube = load_hdf5_cube(temp_path, dataset_key)
            valid_mask = None
            metadata = None

        else:
            cube, metadata = load_geotiff_cube(temp_path)
            valid_mask = metadata["valid_mask"]

        original_summary = get_cube_summary(cube)

        results = preprocess_hyperspectral_cube(
            cube,
            valid_mask=valid_mask,
        )

        feature_summary = results["feature_summary"]

        response = {
            "status": "success",
            "filename": file.filename,
            "file_size_bytes": size,
            "dataset_key": dataset_key or None,
            "original_cube": original_summary,
            "reduced_shape": list(results["reduced_shape"]),
            "pca_components": int(
                results["pca_model"].n_components_
            ),
            "explained_variance": float(
                results["explained_variance"]
            ),
            "explained_variance_ratio": [
                float(value)
                for value in results["explained_variance_ratio"]
            ],
            "cumulative_variance_ratio": [
                float(value)
                for value in np.cumsum(results["explained_variance_ratio"])
            ],
            "feature_names": results["feature_names"],
            "feature_count": results["feature_count"],
            "valid_pixel_count": results["valid_pixel_count"],
            "feature_summary": feature_summary,
        }

        if metadata is not None:
            response["metadata"] = {
                "width": metadata["width"],
                "height": metadata["height"],
                "bands": metadata["bands"],
                "crs": metadata["crs"],
                "dtype": list(metadata["dtype"]),
                "nodata": metadata["nodata"],
            }

        return response

    except HTTPException:
        raise

    except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(error)}",
        ) from error

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()


@app.post("/api/export/features")
async def export_features(
    file: UploadFile = File(...),
    dataset_key: str = Form(default=""),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload TIFF or HDF5.",
        )

    temp_path = None
    output_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            size = 0

            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 100 MB upload limit.",
                    )

                temp_file.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if extension in {".h5", ".hdf5"}:
            datasets = inspect_hdf5(temp_path)

            if not datasets:
                raise HTTPException(
                    status_code=400,
                    detail="No datasets found in the HDF5 file.",
                )

            if not dataset_key:
                return {
                    "status": "dataset_selection_required",
                    "datasets": [
                        {
                            "name": item["name"],
                            "shape": list(item["shape"]),
                            "dtype": item["dtype"],
                        }
                        for item in datasets
                    ],
                }

            cube = load_hdf5_cube(temp_path, dataset_key)
            valid_mask = None

        else:
            cube, metadata = load_geotiff_cube(temp_path)
            valid_mask = metadata["valid_mask"]

        results = preprocess_hyperspectral_cube(
            cube,
            valid_mask=valid_mask,
        )

        feature_result = {
            "spectral_features": results["spectral_features"],
            "feature_names": results["feature_names"],
            "valid_mask": results["valid_mask"],
        }

        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)

        export_feature_dataset(feature_result, output_path)

        return FileResponse(
            path=output_path,
            media_type="text/csv",
            filename="terraspectra_features.csv",
            background=BackgroundTask(
                output_path.unlink,
                missing_ok=True,
            ),
        )

    except HTTPException:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise

    except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error)) from error

    except Exception as error:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Feature export failed: {str(error)}",
        ) from error

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()


@app.post("/api/preview/band")
async def preview_band(
    file: UploadFile = File(...),
    band_index: int = Form(...),
    dataset_key: str = Form(default=""),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload TIFF or HDF5.",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            size = 0

            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 100 MB upload limit.",
                    )

                temp_file.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if extension in {".h5", ".hdf5"}:
            if not dataset_key:
                raise HTTPException(
                    status_code=400,
                    detail="Select an HDF5 dataset first.",
                )

            cube = load_hdf5_cube(temp_path, dataset_key)
        else:
            cube, _ = load_geotiff_cube(temp_path)

        band_count = cube.shape[2]

        if band_index < 0 or band_index >= band_count:
            raise HTTPException(
                status_code=400,
                detail=f"Band index must be between 0 and {band_count - 1}.",
            )

        band = np.asarray(cube[:, :, band_index], dtype=np.float64)
        finite_mask = np.isfinite(band)
        finite_values = band[finite_mask]

        if finite_values.size:
            band_min = float(finite_values.min())
            band_max = float(finite_values.max())
            band_mean = float(finite_values.mean())
            band_std = float(finite_values.std())
        else:
            band_min = None
            band_max = None
            band_mean = None
            band_std = None

        height, width = band.shape
        row_step = max(1, int(np.ceil(height / 256)))
        column_step = max(1, int(np.ceil(width / 256)))

        preview = band[::row_step, ::column_step]
        preview_finite = np.isfinite(preview)

        normalized_preview = np.zeros(preview.shape, dtype=np.uint8)

        if finite_values.size and band_max > band_min:
            normalized_preview[preview_finite] = (
                (preview[preview_finite] - band_min)
                / (band_max - band_min)
                * 255
            ).astype(np.uint8)
        elif finite_values.size:
            normalized_preview[preview_finite] = 127

        return {
            "status": "success",
            "filename": file.filename,
            "band_index": band_index,
            "band_count": band_count,
            "original_shape": [height, width],
            "preview_shape": list(normalized_preview.shape),
            "preview_data": normalized_preview.tolist(),
            "statistics": {
                "min": band_min,
                "max": band_max,
                "mean": band_mean,
                "std": band_std,
                "valid_pixels": int(finite_values.size),
                "missing_pixels": int(band.size - finite_values.size),
            },
        }

    except HTTPException:
        raise

    except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Band preview failed: {str(error)}",
        ) from error

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()
