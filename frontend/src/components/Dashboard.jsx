import React, { useState } from 'react'

export default function Dashboard({ results }) {
    const [activeTab, setActiveTab] = useState('summary')

    if (!results || results.length === 0) return null

    // For simplicity, showing first result or iterating
    // Streamlit app shows results per file in expanders. 
    // Here we'll show a list of cards.

    return (
        <div className="dashboard">
            <h3>Analysis Results</h3>

            {results.map((res, idx) => (
                <div key={idx} className="result-card">
                    <h4>{res.file}</h4>

                    <div className="tabs">
                        <button
                            className={`tab-btn ${activeTab === 'summary' ? 'active' : ''}`}
                            onClick={() => setActiveTab('summary')}
                        >
                            Summary
                        </button>
                        <button
                            className={`tab-btn ${activeTab === 'deviation' ? 'active' : ''}`}
                            onClick={() => setActiveTab('deviation')}
                        >
                            Deviation
                        </button>
                        <button
                            className={`tab-btn ${activeTab === 'volumes' ? 'active' : ''}`}
                            onClick={() => setActiveTab('volumes')}
                        >
                            Volumes
                        </button>
                    </div>

                    {activeTab === 'summary' && (
                        <div className="metrics-grid">
                            <div className="metric-item">
                                <span className="label">Mean Dev:</span>
                                <span className="value">{res.metrics.mean_deviation?.toFixed(3)} mm</span>
                            </div>
                            <div className="metric-item">
                                <span className="label">Max Dev:</span>
                                <span className="value">{res.metrics.max_deviation?.toFixed(3)} mm</span>
                            </div>
                            <div className="metric-item">
                                <span className="label">Mean Weighted:</span>
                                <span className="value">{res.metrics.mean_weighted_deviation?.toFixed(3)} mm</span>
                            </div>
                            <div className="metric-item">
                                <span className="label">Max Weighted:</span>
                                <span className="value">{res.metrics.max_weighted_deviation?.toFixed(3)} mm</span>
                            </div>
                            <div className="metric-item">
                                <span className="label">RMS:</span>
                                <span className="value">{res.metrics.inlier_rmse?.toFixed(3)}</span>
                            </div>
                            <div className="metric-item">
                                <span className="label">Fitness:</span>
                                <span className="value">{res.metrics.fitness?.toFixed(3)}</span>
                            </div>
                        </div>
                    )}

                    {activeTab === 'deviation' && (
                        <div className="metrics-grid">
                            {/* Placeholder for histograms or more detailed stats */}
                            <div className="metric-item">
                                <span className="label">Ref &rarr; Test Mean:</span>
                                <span className="value">{res.metrics.mean_ref_deviation?.toFixed(3)} mm</span>
                            </div>
                            <div className="metric-item">
                                <span className="label">Ref &rarr; Test Max:</span>
                                <span className="value">{res.metrics.max_ref_deviation?.toFixed(3)} mm</span>
                            </div>
                        </div>
                    )}

                    {activeTab === 'volumes' && (
                        <div className="metrics-grid">
                            <div className="metric-item">
                                <span className="label">Overlap (Jaccard):</span>
                                <span className="value">{(res.metrics.volume_overlap_jaccard * 100)?.toFixed(1)}%</span>
                            </div>
                            <div className="metric-item">
                                <span className="label">Intersect Vol:</span>
                                <span className="value">{res.metrics.volume_intersection_vox?.toFixed(3)} mm³</span>
                            </div>
                            <div className="metric-item">
                                <span className="label">Ref Coverage:</span>
                                <span className="value">{res.metrics.coverage_ref_pct?.toFixed(1)}%</span>
                            </div>
                            <div className="metric-item">
                                <span className="label">Test Coverage:</span>
                                <span className="value">{res.metrics.coverage_test_pct?.toFixed(1)}%</span>
                            </div>
                        </div>
                    )}
                </div>
            ))}
        </div>
    )
}
