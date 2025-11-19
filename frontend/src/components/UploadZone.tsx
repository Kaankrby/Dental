import React, { useState } from 'react';
import { uploadReference, uploadTest } from '../api/client';

export const UploadZone: React.FC = () => {
    const [refFile, setRefFile] = useState<File | null>(null);
    const [testFiles, setTestFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState('');

    const handleRefChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setRefFile(e.target.files[0]);
        }
    };

    const handleTestChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setTestFiles(Array.from(e.target.files));
        }
    };

    const handleUpload = async () => {
        setUploading(true);
        setMessage('');
        try {
            if (refFile) {
                await uploadReference(refFile);
            }
            if (testFiles.length > 0) {
                await uploadTest(testFiles);
            }
            setMessage('Upload successful!');
        } catch (error) {
            console.error(error);
            setMessage('Upload failed.');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="card upload-zone">
            <h2 className="card-title">Upload Files</h2>

            <div className="form-group">
                <label>Reference (.3dm)</label>
                <input
                    type="file"
                    accept=".3dm"
                    onChange={handleRefChange}
                    className="file-input"
                />
            </div>

            <div className="form-group">
                <label>Test Scans (.stl)</label>
                <input
                    type="file"
                    accept=".stl"
                    multiple
                    onChange={handleTestChange}
                    className="file-input"
                />
            </div>

            <button
                onClick={handleUpload}
                disabled={uploading || (!refFile && testFiles.length === 0)}
                className="btn btn-primary"
            >
                {uploading ? 'Uploading...' : 'Start Analysis'}
            </button>

            {message && <p className="status-message">{message}</p>}
        </div>
    );
};
