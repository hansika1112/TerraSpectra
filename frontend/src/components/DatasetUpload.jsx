import { useRef, useState } from 'react'
import { Upload, FileImage, X, CheckCircle2, AlertCircle } from 'lucide-react'

const supportedFormats = ['.tif', '.tiff', '.h5', '.hdf5']

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

  function handleFile(selectedFile) {
    if (!selectedFile) return

    const extension = selectedFile.name
      .slice(selectedFile.name.lastIndexOf('.'))
      .toLowerCase()

    if (!supportedFormats.includes(extension)) {
      setFile(null)
      setError('Unsupported format. Please select a TIFF or HDF5 file.')
      return
    }

    setError('')
    setFile(selectedFile)
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragging(false)
    handleFile(event.dataTransfer.files[0])
  }

  function clearFile() {
    setFile(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
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
        >
          <Upload size={15} /> Browse Files
        </button>

        <span className="upload-hint">
          Supported formats: .TIF, .TIFF, .H5, .HDF5
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
            <span>{formatSize(file.size)} · {file.name.split('.').pop().toUpperCase()}</span>
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
          >
            <X size={17} />
          </button>
        </div>
      )}

      <div className="upload-footer">
        <span> <CheckCircle2 size={14} /> Local file validation enabled</span>
        <span>Backend integration pending</span>
      </div>
    </section>
  )
}
