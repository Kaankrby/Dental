import { useState, useEffect } from 'react'
import './App.css'
import FileUpload from './components/FileUpload'
import Dashboard from './components/Dashboard'
import Viewer3D from './components/Viewer3D'
import Sidebar from './components/Sidebar'
import axios from 'axios'

function App() {
  const [jobId, setJobId] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)

  // Default settings matching "Balanced" mode
  const [settings, setSettings] = useState({
    processing_mode: 'Balanced',
    num_points: 15000,
    use_global_reg: true,
    voxel_size: 1.5,
    icp_threshold: 0.3,
    icp_max_iter: 200,
    icp_mode: 'auto',
    stl_scale: 1.0,
    ignore_outside_bbox: false,
    use_full_ref_global: false,
    include_notimportant_metrics: false,
    volume_voxel_size: 0.5
  })

  useEffect(() => {
    let interval
    if (jobId && loading) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`http://localhost:8000/api/results/${jobId}`)
          if (res.data.status === 'completed') {
            setResults(res.data.results)
            setLoading(false)
            clearInterval(interval)
          } else if (res.data.status === 'failed') {
            alert("Analysis failed: " + res.data.error)
            setLoading(false)
            clearInterval(interval)
          }
        } catch (e) {
          console.error(e)
        }
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [jobId, loading])

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Dental STL Analyzer Pro</h1>
      </header>
      <main className="app-main">
        <div className="sidebar">
          <FileUpload
            onAnalysisStart={(id) => { setJobId(id); setLoading(true); setResults(null); }}
            settings={settings}
          />
          <hr style={{ borderColor: 'var(--border-color)', width: '100%' }} />
          <Sidebar settings={settings} setSettings={setSettings} />
        </div>

        <div className="viewer-area">
          <Viewer3D results={results} />
          {loading && <div className="loading-overlay">Analyzing...</div>}
          {results && <Dashboard results={results} />}
        </div>
      </main>
    </div>
  )
}

export default App
