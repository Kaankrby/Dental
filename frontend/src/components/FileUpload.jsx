import { useState } from 'react'
import axios from 'axios'
import { Upload, FileUp, Play } from 'lucide-react'

export default function FileUpload({ onAnalysisStart, settings }) {
    const [refFile, setRefFile] = useState(null)
    const [testFiles, setTestFiles] = useState([])
    const [uploading, setUploading] = useState(false)

    const handleUpload = async () => {
        if (!refFile || testFiles.length === 0) return alert("Please select both reference and test files")

        setUploading(true)
        try {
            const refFormData = new FormData()
            refFormData.append('file', refFile)
            await axios.post('http://localhost:8000/api/upload/reference', refFormData)

            const testFormData = new FormData()
            for (let i = 0; i < testFiles.length; i++) {
                testFormData.append('files', testFiles[i])
            }
            await axios.post('http://localhost:8000/api/upload/test', testFormData)

            // Pass settings to analyze endpoint
            const res = await axios.post('http://localhost:8000/api/analyze', settings)
            onAnalysisStart(res.data.job_id)
        } catch (e) {
            alert("Error: " + (e.response?.data?.detail || e.message))
        } finally {
            setUploading(false)
        }
    }

    return (
        <div className="file-upload-container">
            <div className="upload-section">
                <h3>Reference (.3dm)</h3>
                <div className="file-input-wrapper">
                    <label htmlFor="ref-upload" className="custom-file-upload">
                        <Upload size={16} /> Choose File
                    </label>
                    <input
                        id="ref-upload"
                        type="file"
                        accept=".3dm"
                        onChange={(e) => setRefFile(e.target.files[0])}
                    />
                    <span className="file-name">{refFile ? refFile.name : "No file chosen"}</span>
                </div>
            </div>

            <div className="upload-section">
                <h3>Test Files (.stl)</h3>
                <div className="file-input-wrapper">
                    <label htmlFor="test-upload" className="custom-file-upload">
                        <FileUp size={16} /> Choose Files
                    </label>
                    <input
                        id="test-upload"
                        type="file"
                        accept=".stl"
                        multiple
                        onChange={(e) => setTestFiles(Array.from(e.target.files))}
                    />
                    <span className="file-name">{testFiles.length > 0 ? `${testFiles.length} files selected` : "No files chosen"}</span>
                </div>
            </div>

            <button className="analyze-btn" onClick={handleUpload} disabled={uploading}>
                <Play size={16} /> {uploading ? "Uploading..." : "Start Analysis"}
            </button>
        </div>
    )
}
