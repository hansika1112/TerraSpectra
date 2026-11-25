import { useRef, useState } from 'react'
import {
  Upload,
  FileImage,
  X,
  CheckCircle2,
  AlertCircle,
  LoaderCircle,
  Database,
} from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/api/process'
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
  const [datasets, setDatasets] = useState([])
  const [datasetKey, setDatasetKey] = useState('')
  const [result, setResult] = useState(null)

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

    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  async function processDataset(selectedKey = '') {
    if (!file || processing) return

    setProcessing(true)
    setError('')
    setResult(null)

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
