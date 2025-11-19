import React from 'react'
import { Settings, Sliders, Activity, Box, Layers } from 'lucide-react'

export default function Sidebar({ settings, setSettings }) {
    const handleChange = (key, value) => {
        setSettings(prev => ({ ...prev, [key]: value }))
    }

    const handleModeChange = (mode) => {
        let newSettings = { ...settings, processing_mode: mode }
        if (mode === 'Balanced') {
            newSettings = { ...newSettings, num_points: 15000, voxel_size: 1.5, icp_threshold: 0.3, icp_max_iter: 200, volume_voxel_size: 0.5 }
        } else if (mode === 'Precision') {
            newSettings = { ...newSettings, num_points: 30000, voxel_size: 0.5, icp_threshold: 0.1, icp_max_iter: 500, volume_voxel_size: 0.25 }
        } else if (mode === 'Speed') {
            newSettings = { ...newSettings, num_points: 5000, voxel_size: 3.0, icp_threshold: 0.5, icp_max_iter: 100, volume_voxel_size: 1.0, use_global_reg: false }
        } else if (mode === 'Adaptive') {
            newSettings = { ...newSettings, num_points: 20000, icp_threshold: 0.25 }
        }
        setSettings(newSettings)
    }

    return (
        <div className="sidebar-settings">
            <div className="section-header">
                <Settings size={16} />
                <h3>Analysis Parameters</h3>
            </div>

            <div className="control-group">
                <label>Processing Mode</label>
                <div className="radio-group">
                    {['Balanced', 'Precision', 'Speed', 'Adaptive'].map(mode => (
                        <label key={mode} className="radio-label">
                            <input
                                type="radio"
                                name="processing_mode"
                                checked={settings.processing_mode === mode}
                                onChange={() => handleModeChange(mode)}
                            />
                            {mode}
                        </label>
                    ))}
                </div>
            </div>

            <div className="section-header">
                <Activity size={16} />
                <h3>Point Cloud</h3>
            </div>
            <div className="control-group">
                <label>Sample Points: {settings.num_points}</label>
                <input
                    type="range"
                    min="1000"
                    max="100000"
                    step="1000"
                    value={settings.num_points}
                    onChange={(e) => handleChange('num_points', parseInt(e.target.value))}
                />
            </div>

            <div className="section-header">
                <Sliders size={16} />
                <h3>Registration</h3>
            </div>
            <div className="control-group checkbox-group">
                <label>
                    <input
                        type="checkbox"
                        checked={settings.use_global_reg}
                        onChange={(e) => handleChange('use_global_reg', e.target.checked)}
                        disabled={settings.processing_mode === 'Speed'}
                    />
                    Enable Global Registration
                </label>
            </div>

            <div className="control-group">
                <label>Global Voxel Size (mm): {settings.voxel_size}</label>
                <input
                    type="range"
                    min="0.1"
                    max="5.0"
                    step="0.1"
                    value={settings.voxel_size}
                    onChange={(e) => handleChange('voxel_size', parseFloat(e.target.value))}
                    disabled={!settings.use_global_reg}
                />
            </div>

            <div className="control-group checkbox-group">
                <label>
                    <input
                        type="checkbox"
                        checked={settings.use_full_ref_global}
                        onChange={(e) => handleChange('use_full_ref_global', e.target.checked)}
                    />
                    Use full reference for global
                </label>
            </div>

            <div className="section-header">
                <h3>ICP Parameters</h3>
            </div>
            <div className="control-group">
                <label>ICP Threshold (mm): {settings.icp_threshold}</label>
                <input
                    type="range"
                    min="0.01"
                    max="2.0"
                    step="0.01"
                    value={settings.icp_threshold}
                    onChange={(e) => handleChange('icp_threshold', parseFloat(e.target.value))}
                    disabled={settings.processing_mode === 'Adaptive'}
                />
            </div>
            <div className="control-group">
                <label>ICP Max Iterations: {settings.icp_max_iter}</label>
                <input
                    type="range"
                    min="10"
                    max="2000"
                    step="10"
                    value={settings.icp_max_iter}
                    onChange={(e) => handleChange('icp_max_iter', parseInt(e.target.value))}
                />
            </div>
            <div className="control-group">
                <label>ICP Mode</label>
                <select
                    value={settings.icp_mode}
                    onChange={(e) => handleChange('icp_mode', e.target.value)}
                >
                    <option value="auto">Auto</option>
                    <option value="point_to_plane">Point-to-Plane</option>
                    <option value="point_to_point">Point-to-Point</option>
                </select>
            </div>

            <div className="section-header">
                <Box size={16} />
                <h3>Units & Volume</h3>
            </div>
            <div className="control-group">
                <label>Test STL Units</label>
                <select
                    value={settings.stl_scale}
                    onChange={(e) => handleChange('stl_scale', parseFloat(e.target.value))}
                >
                    <option value="1.0">Millimeters (mm)</option>
                    <option value="10.0">Centimeters (cm)</option>
                    <option value="1000.0">Meters (m)</option>
                    <option value="25.4">Inches (in)</option>
                    <option value="0.001">Microns (um)</option>
                </select>
            </div>
            <div className="control-group">
                <label>Volume Voxel Size (mm): {settings.volume_voxel_size}</label>
                <input
                    type="range"
                    min="0.05"
                    max="2.0"
                    step="0.05"
                    value={settings.volume_voxel_size}
                    onChange={(e) => handleChange('volume_voxel_size', parseFloat(e.target.value))}
                />
            </div>

            <div className="control-group checkbox-group">
                <label>
                    <input
                        type="checkbox"
                        checked={settings.include_notimportant_metrics}
                        onChange={(e) => handleChange('include_notimportant_metrics', e.target.checked)}
                    />
                    Include NOTIMPORTANT metrics
                </label>
            </div>

        </div>
    )
}
