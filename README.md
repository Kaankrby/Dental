# Dental STL Analyzer Pro

Dental STL Analyzer Pro is a Streamlit application for comparing dental scans against Rhino `.3dm` reference models. It combines robust registration, weighted deviation metrics, and interactive visualizations tailored for fissure-level analysis.

![App Screenshot](https://via.placeholder.com/1200x420.png?text=3D+Visualization+and+Metrics+Dashboard)

## Features

- **Registration Pipeline** – RANSAC global registration, multi-iteration ICP (point-to-plane or point-to-point), adaptive voxel sizing.
- **Layer-Aware Metrics** – Reference layers inherit weights and can be filtered interactively (e.g., emphasize inner fissures while ignoring `NOTIMPORTANT` surfaces).
- **Comparator Modes** – Choose between the legacy “Test anchored” comparator or the new dual Reference/Test analysis without rerunning uploads.
- **Persistent Results** – Latest analysis payloads remain in `st.session_state`, so toggling radios or changing tabs does not reset the workflow.
- **Interactive Visuals** – Combined deviation histograms, 3‑D heatmaps (raw/weighted), volume overlap metrics, and downloadable CSV/3DM exports.

## Quick Start

### Prerequisites

- Python 3.8+
- Recommended OS: Ubuntu/Debian (for easier Open3D dependencies)

```bash
sudo apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1
```

### Install & Run

```bash
git clone https://github.com/yourusername/dental-stl-analyzer.git
cd dental-stl-analyzer

python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
streamlit run stl_analyzer.py
```

## Processing Modes

| Mode      | Sample Points | Global Voxel | Primary Use Case            |
|-----------|---------------|--------------|-----------------------------|
| Speed     | 5,000         | 3.0 mm       | Quick chairside checks      |
| Balanced  | 15,000        | 1.5 mm       | Default clinical analysis   |
| Precision | 30,000        | 0.5 mm       | Lab-grade inspections       |
| Adaptive  | Auto          | Auto         | Scale-aware, auto ICP limit |

## File Requirements

| Parameter     | Requirement                    |
|---------------|--------------------------------|
| Reference     | Rhino `.3dm` with mesh layers  |
| Test          | STL (ASCII/Binary)             |
| Max size      | ~250 MB per file               |
| Mesh quality  | Watertight / manifold preferred |

## Comparator Modes & Layer Focus

- **Legacy (Test anchored)** – Matches the original UI: metrics and heatmaps are based solely on test→reference deviations. Best when you only need to inspect the scanned surface.
- **Dual Reference/Test** – Adds reference→test evaluation so missing anatomy (e.g., inner fissures) lights up immediately. A radio toggle in the sidebar switches modes without reprocessing.
- **Layer Focus Multiselect** – Pick which reference layers drive stats, histograms, and heatmaps. The app respects layer weights and hides `NOTIMPORTANT` layers by default.

## Persistent Results

- Every analysis run writes a compact payload to `st.session_state["analysis_payloads"]`.
- The UI renders these cached entries via `render_analysis_entries(...)`, so interacting with radios/tabs does not trigger a restart.
- Download buttons reuse the stored CSV/3DM artifacts per test file.

## Sample Metrics Payload

```json
{
  "mean_deviation": 0.12,
  "max_deviation": 1.45,
  "mean_weighted_deviation": 0.08,
  "volume_overlap_jaccard": 0.987,
  "ref_distances": [...],
  "weighted_distances": [...],
  "volume_ref_gap_vox": 12.5,
  "fitness": 0.94,
  "inlier_rmse": 0.21
}
```

## Project Structure

```
Dental/
├─ stl_analyzer.py    # Streamlit UI & workflow orchestration
├─ processing.py      # RhinoAnalyzer, ICP, metrics
├─ visualization.py   # Histograms, 3D scatter/heatmaps
├─ utils.py           # Helpers: mesh I/O, voxels, validation
├─ requirements.txt   # Python dependencies
└─ README.md
```

## Clinical Disclaimer

This tool is intended for professional dental operators. Automated measurements should always be reviewed by a qualified clinician before guiding treatment decisions.

## License

MIT License – see [LICENSE](LICENSE) for details.

---

Need help or want to report a bug? Open an issue in the repository or reach out to your development contact. Replace the placeholder screenshot above with real app images when available.
