import React from 'react';
import { UploadZone } from './UploadZone';

export const Dashboard: React.FC = () => {
    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <h1>Dental STL Analyzer Pro</h1>
                <p>Modern Stack Edition</p>
            </header>

            <main className="dashboard-grid">
                <div className="grid-item">
                    <UploadZone />
                </div>

                <div className="grid-item viewer-placeholder">
                    <p>3D Viewer Placeholder</p>
                    {/* Future: <Viewer3D /> */}
                </div>
            </main>
        </div>
    );
};
