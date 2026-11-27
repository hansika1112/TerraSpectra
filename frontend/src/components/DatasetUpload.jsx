import { useEffect, useRef, useState } from 'react'
import {
  Upload,
  FileImage,
  X,
  CheckCircle2,
  AlertCircle,
  LoaderCircle,
  Database,
  Download,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/api/process'
const EXPORT_API_URL = 'http://127.0.0.1:8000/api/export/features'
const BAND_API_URL = 'http://127.0.0.1:8000/api/preview/band'
const supportedFormats = ['.tif', '.tiff', '.h5', '.hdf5']
const MAX_FILE_SIZE = 100 * 1024 * 1024

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export default function DatasetUpload() {
  const inputRef = useRef(null)

  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [datasets, setDatasets] = useState([])
  const [datasetKey, setDatasetKey] = useState('')
  const [result, setResult] = useState(null)
  const [bandIndex, setBandIndex] = useState(0)
  const [bandPreview, setBandPreview] = useState(null)
  const [bandLoading, setBandLoading] = useState(false)
  const [bandError, setBandError] = useState('')
  const bandCanvasRef = useRef(null)
  const bandRequestRef = useRef(0)

  useEffect(() => {
    if (!bandPreview?.preview_data?.length || !bandCanvasRef.current) {
      return
    }

    const pixels = bandPreview.preview_data
    const height = pixels.length
    const width = pixels[0].length
    const canvas = bandCanvasRef.current
    const context = canvas.getContext('2d')

    canvas.width = width
    canvas.height = height

    const imageData = context.createImageData(width, height)

    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        const offset = (row * width + column) * 4
        const value = pixels[row][column]

        imageData.data[offset] = value
        imageData.data[offset + 1] = value
        imageData.data[offset + 2] = value
        imageData.data[offset + 3] = 255
      }
    }

    context.putImageData(imageData, 0, 0)
  }, [bandPreview])

  function handleFile(selectedFile) {
    if (!selectedFile) return

    const extension = selectedFile.name
      .slice(selectedFile.name.lastIndexOf('.'))
      .toLowerCase()

    if (!supportedFormats.includes(extension)) {
      setError('Unsupported format. Please select a TIFF or HDF5 file.')
      return
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('File exceeds the 100 MB upload limit.')
      return
    }

    setFile(selectedFile)
    setError('')
    setDatasets([])
    setDatasetKey('')
    setResult(null)
    setBandPreview(null)
    setBandIndex(0)
    setBandError('')
    bandRequestRef.current += 1
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragging(false)
    handleFile(event.dataTransfer.files[0])
  }

  function clearFile() {
    setFile(null)
    setError('')
    setDatasets([])
    setDatasetKey('')
    setResult(null)
    setBandPreview(null)
    setBandIndex(0)
    setBandError('')
    bandRequestRef.current += 1

    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  async function fetchBandPreview(index, selectedKey = datasetKey) {
    if (!file) return

    const requestId = ++bandRequestRef.current
    setBandLoading(true)
    setBandError('')

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('band_index', String(index))

      if (selectedKey) {
        formData.append('dataset_key', selectedKey)
      }

      const response = await fetch(BAND_API_URL, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Band preview failed.')
      }

      if (requestId === bandRequestRef.current) {
        setBandPreview(data)
      }
    } catch (err) {
      if (requestId === bandRequestRef.current) {
        setBandError(
          err.message.includes('Failed to fetch')
            ? 'Cannot connect to TerraSpectra API.'
            : err.message
        )
        setBandPreview(null)
      }
    } finally {
      if (requestId === bandRequestRef.current) {
        setBandLoading(false)
      }
    }
  }

  async function processDataset(selectedKey = '') {
    if (!file || processing) return

    setProcessing(true)
    setError('')
    setResult(null)
    setBandPreview(null)
    setBandError('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      if (selectedKey) {
        formData.append('dataset_key', selectedKey)
      }

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Dataset processing failed.')
      }

      if (data.status === 'dataset_selection_required') {
        setDatasets(data.datasets || [])
        setDatasetKey(data.datasets?.[0]?.name || '')
        return
      }

      if (data.status === 'success') {
        setResult(data)
        setDatasets([])
        setBandIndex(0)
        await fetchBandPreview(0, selectedKey || datasetKey)
      } else {
        throw new Error('Unexpected response from the processing server.')
      }
    } catch (err) {
      setError(
        err.message.includes('Failed to fetch')
          ? 'Cannot connect to TerraSpectra API. Make sure the backend server is running.'
          : err.message
      )
    } finally {
      setProcessing(false)
    }
  }


  function downloadBlob(blob, filename) {
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
  }

  async function downloadFeatureCsv() {
    if (!file || !result || exporting) return

    setExporting(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      if (datasetKey) {
        formData.append('dataset_key', datasetKey)
      }

      const response = await fetch(EXPORT_API_URL, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Feature CSV export failed.')
      }

      const blob = await response.blob()
      downloadBlob(blob, 'terraspectra_features.csv')
    } catch (err) {
      setError(
        err.message.includes('Failed to fetch')
          ? 'Cannot connect to TerraSpectra API. Make sure the backend server is running.'
          : err.message
      )
    } finally {
      setExporting(false)
    }
  }

  function downloadPcaReport() {
    if (!result) return

    const report = {
      project: 'TerraSpectra',
      report_type: 'PCA Analysis Report',
      generated_at: new Date().toISOString(),
      filename: result.filename,
      dataset_key: result.dataset_key,
      original_cube: result.original_cube,
      reduced_shape: result.reduced_shape,
      pca_components: result.pca_components,
      explained_variance: result.explained_variance,
      explained_variance_ratio: result.explained_variance_ratio,
      cumulative_variance_ratio: result.cumulative_variance_ratio,
      feature_names: result.feature_names,
      feature_count: result.feature_count,
      valid_pixel_count: result.valid_pixel_count,
      feature_summary: result.feature_summary,
      metadata: result.metadata || null,
    }

    const blob = new Blob(
      [JSON.stringify(report, null, 2)],
      { type: 'application/json' }
    )

    downloadBlob(blob, 'terraspectra_pca_report.json')
  }

  return (
    <section className="panel upload-panel">
      <div className="panel-header">
        <div>
          <h3>Hyperspectral Dataset Upload</h3>
          <p>Import satellite and remote sensing imagery</p>
        </div>
        <div className="upload-format-label">TIFF / HDF5</div>
      </div>

      <div
        className={`upload-dropzone ${dragging ? 'dragging' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".tif,.tiff,.h5,.hdf5"
          hidden
          onChange={(event) => handleFile(event.target.files[0])}
        />

        <div className="upload-icon">
          <Upload size={23} />
        </div>

        <h4>Drop your hyperspectral file here</h4>
        <p>or browse files from your computer</p>

        <button
          className="upload-browse-button"
          onClick={() => inputRef.current?.click()}
          type="button"
          disabled={processing}
        >
          <Upload size={15} /> Browse Files
        </button>

        <span className="upload-hint">
          Supported formats: .TIF, .TIFF, .H5, .HDF5 · Maximum 100 MB
        </span>
      </div>

      {error && (
        <div className="upload-error">
          <AlertCircle size={17} />
          <span>{error}</span>
        </div>
      )}

      {file && (
        <div className="selected-file">
          <div className="selected-file-icon">
            <FileImage size={20} />
          </div>

          <div className="selected-file-info">
            <strong>{file.name}</strong>
            <span>
              {formatSize(file.size)} · {file.name.split('.').pop().toUpperCase()}
            </span>
          </div>

          <div className="selected-file-status">
            <CheckCircle2 size={17} />
            <span>Validated</span>
          </div>

          <button
            className="remove-file-button"
            onClick={clearFile}
            type="button"
            aria-label="Remove selected file"
            disabled={processing}
          >
            <X size={17} />
          </button>
        </div>
      )}

      {file && (
        <button
          className="upload-browse-button"
          type="button"
          onClick={() => processDataset()}
          disabled={processing}
        >
          {processing ? (
            <>
              <LoaderCircle size={16} className="upload-spinner" />
              Uploading and Processing...
            </>
          ) : (
            <>
              <Database size={16} />
              Process Dataset
            </>
          )}
        </button>
      )}

      {datasets.length > 0 && (
        <div className="dataset-selection">
          <h4>Choose HDF5 Dataset</h4>
          <p>Multiple datasets were found. Select the hyperspectral cube to process.</p>

          <select
            value={datasetKey}
            onChange={(event) => setDatasetKey(event.target.value)}
            disabled={processing}
          >
            {datasets.map((dataset) => (
              <option key={dataset.name} value={dataset.name}>
                {dataset.name} — {dataset.shape.join(' × ')} ({dataset.dtype})
              </option>
            ))}
          </select>

          <button
            className="upload-browse-button"
            type="button"
            onClick={() => processDataset(datasetKey)}
            disabled={processing || !datasetKey}
          >
            {processing ? 'Processing...' : 'Process Selected Dataset'}
          </button>
        </div>
      )}

      {result && (
        <div className="processing-results">
          <div className="processing-success">
            <CheckCircle2 size={19} />
            <strong>Dataset processed successfully</strong>
          </div>

          <h4>Processing Summary</h4>

          <div className="processing-result-grid">
            <div>
              <span>Original Cube</span>
              <strong>{result.original_cube.shape.join(' × ')}</strong>
            </div>

            <div>
              <span>Reduced Cube</span>
              <strong>{result.reduced_shape.join(' × ')}</strong>
            </div>

            <div>
              <span>PCA Components</span>
              <strong>{result.pca_components}</strong>
            </div>

            <div>
              <span>Explained Variance</span>
              <strong>
                {(result.explained_variance * 100).toFixed(2)}%
              </strong>
            </div>

            <div>
              <span>Valid Pixels</span>
              <strong>{result.valid_pixel_count}</strong>
            </div>

            <div>
              <span>Spectral Features</span>
              <strong>{result.feature_count}</strong>
            </div>
          </div>

          {bandPreview && (
            <div className="band-explorer">
              <div className="band-explorer-header">
                <div>
                  <h4>Interactive Spectral Band Explorer</h4>
                  <p>Explore individual wavelengths across your hyperspectral cube</p>
                </div>
                <span>
                  Band {bandPreview.band_index + 1} / {bandPreview.band_count}
                </span>
              </div>

              <div className="band-preview-layout">
                <div className="band-image-frame">
                  <canvas
                    ref={bandCanvasRef}
                    aria-label={`Grayscale preview of spectral band ${bandIndex + 1}`}
                  />
                </div>

                <div className="band-stat-grid">
                  <div>
                    <span>Minimum</span>
                    <strong>{bandPreview.statistics.min?.toFixed(4) ?? 'N/A'}</strong>
                  </div>
                  <div>
                    <span>Maximum</span>
                    <strong>{bandPreview.statistics.max?.toFixed(4) ?? 'N/A'}</strong>
                  </div>
                  <div>
                    <span>Mean</span>
                    <strong>{bandPreview.statistics.mean?.toFixed(4) ?? 'N/A'}</strong>
                  </div>
                  <div>
                    <span>Std. Deviation</span>
                    <strong>{bandPreview.statistics.std?.toFixed(4) ?? 'N/A'}</strong>
                  </div>
                  <div>
                    <span>Valid Pixels</span>
                    <strong>{bandPreview.statistics.valid_pixels}</strong>
                  </div>
                  <div>
                    <span>Missing Pixels</span>
                    <strong>{bandPreview.statistics.missing_pixels}</strong>
                  </div>
                </div>
              </div>

              <div className="band-controls">
                <button
                  type="button"
                  aria-label="Previous spectral band"
                  onClick={() => {
                    const next = Math.max(0, bandIndex - 1)
                    setBandIndex(next)
                    fetchBandPreview(next)
                  }}
                  disabled={bandLoading || bandIndex === 0}
                >
                  <ChevronLeft size={17} />
                </button>

                <div className="band-slider-container">
                  <input
                    type="range"
                    min="0"
                    max={bandPreview.band_count - 1}
                    value={bandIndex}
                    onChange={(event) => {
                      const next = Number(event.target.value)
                      setBandIndex(next)
                      fetchBandPreview(next)
                    }}
                    aria-label="Select spectral band"
                    disabled={bandLoading}
                  />
                  <div className="band-slider-labels">
                    <span>Band 1</span>
                    <span>Band {bandPreview.band_count}</span>
                  </div>
                </div>

                <button
                  type="button"
                  aria-label="Next spectral band"
                  onClick={() => {
                    const next = Math.min(
                      bandPreview.band_count - 1,
                      bandIndex + 1
                    )
                    setBandIndex(next)
                    fetchBandPreview(next)
                  }}
                  disabled={bandLoading || bandIndex === bandPreview.band_count - 1}
                >
                  <ChevronRight size={17} />
                </button>
              </div>

              {bandLoading && (
                <p className="band-loading">
                  <LoaderCircle size={15} className="upload-spinner" />
                  Loading spectral band...
                </p>
              )}

              {bandError && (
                <p className="band-error">
                  <AlertCircle size={15} />
                  {bandError}
                </p>
              )}

              <p className="band-preview-caption">
                Preview: {bandPreview.preview_shape.join(' × ')} pixels
                {' · '}
                Original: {bandPreview.original_shape.join(' × ')} pixels
              </p>
            </div>
          )}

          {bandError && !bandPreview && (
            <div className="upload-error">
              <AlertCircle size={17} />
              <span>{bandError}</span>
            </div>
          )}

          <div className="variance-visualization">
            <div className="variance-heading">
              <span>PCA Information Retained</span>
              <strong>{(result.explained_variance * 100).toFixed(2)}%</strong>
            </div>
            <div className="variance-track">
              <div
                className="variance-fill"
                style={{
                  width: `${Math.min(Math.max(result.explained_variance * 100, 0), 100)}%`,
                }}
              />
            </div>
            <span className="variance-caption">
              Explained variance captured by the reduced spectral representation
            </span>
          </div>

          <div className="pca-component-analysis">
            <div className="pca-component-header">
              <h4>Component-wise Variance</h4>
              <span>
                Cumulative: {(result.cumulative_variance_ratio?.at(-1) * 100 || 0).toFixed(2)}%
              </span>
            </div>

            <div className="pca-component-chart">
              {result.explained_variance_ratio?.map((variance, index) => (
                <div className="pca-component-row" key={index}>
                  <div className="pca-component-label">
                    <strong>PC{index + 1}</strong>
                    <span>{(variance * 100).toFixed(2)}%</span>
                  </div>

                  <div className="pca-component-track">
                    <div
                      className="pca-component-fill"
                      style={{ width: `${Math.min(variance * 100, 100)}%` }}
                    />
                  </div>

                  <div className="pca-cumulative-label">
                    Cumulative: {(result.cumulative_variance_ratio?.[index] * 100 || 0).toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>
          </div>

          <h4>Spectral Feature Analysis</h4>

          <div className="feature-summary">
            {Object.entries(result.feature_summary).map(([name, values]) => (
              <div className="feature-summary-item" key={name}>
                <strong>{name.replaceAll('_', ' ')}</strong>
                <span>
                  Mean: {values.mean.toFixed(4)} · Min: {values.min.toFixed(4)} · Max: {values.max.toFixed(4)}
                </span>
              </div>
            ))}
          </div>

          <div className="dataset-download-actions">
            <button
              className="upload-browse-button"
              type="button"
              onClick={downloadFeatureCsv}
              disabled={exporting}
            >
              {exporting ? (
                <>
                  <LoaderCircle size={16} className="upload-spinner" />
                  Exporting CSV...
                </>
              ) : (
                <>
                  <Download size={16} />
                  Download Feature CSV
                </>
              )}
            </button>

            <button
              className="upload-browse-button"
              type="button"
              onClick={downloadPcaReport}
            >
              <Download size={16} />
              Download PCA Report
            </button>
          </div>
        </div>
      )}

      <div className="upload-footer">
        <span>
          <CheckCircle2 size={14} /> Backend processing enabled
        </span>
        <span>FastAPI · PCA · Spectral Analysis</span>
      </div>
    </section>
  )
}
