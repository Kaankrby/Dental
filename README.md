
# 🦷 Dental STL Analyzer Pro

![App Screenshot](https://via.placeholder.com/800x400.png?text=3D+Visualization+and+Metrics+Dashboard)

A professional-grade web application for comparing dental STL files with advanced 3D visualization and quantitative analysis.

## 🌟 Features

- **3D Registration & Alignment**
  - RANSAC-based global registration
  - ICP refinement with normal constraints
  - Multi-resolution processing
- **Advanced Metrics**
  - Surface deviation analysis (Hausdorff distance, RMSE)
  - Volume similarity comparison
  - Normal vector angle analysis
  - Statistical distribution metrics
- **Visualization Tools**
  - Interactive 3D heatmaps
  - Normal angle distribution plots
  - Comparative histograms
- **Clinical-Grade Processing**
  - Watertight mesh validation
  - Adaptive point cloud sampling
  - Outlier detection and removal

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ubuntu/Debian recommended
```bash
# System dependencies
sudo apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/dental-stl-analyzer.git
cd dental-stl-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### Usage
```bash
streamlit run stl_analyzer.py
```

## 📊 Processing Modes

| Mode | Points | Voxel Size | Use Case |
|------|--------|------------|----------|
| 🚀 Speed | 5,000 | 3.0 mm | Quick preliminary checks |
| ⚖️ Balanced | 15,000 | 1.5 mm | Standard clinical analysis |
| 🔍 Precision | 30,000 | 0.5 mm | Detailed lab-grade inspection |

## 📂 File Requirements

| Parameter | Requirement |
|-----------|-------------|
| File Format | STL (ASCII/Binary) |
| Max Size | 250 MB |
| Mesh Quality | Watertight, manifold |
| Triangle Count | >1,000 faces |

## 📈 Key Metrics

```python
{
  "mean_deviation": "0.12 mm",      # Average surface deviation
  "max_deviation": "1.45 mm",       # Maximum localized deviation
  "volume_similarity": "98.7%",     # Volume matching score
  "normal_alignment": "8.2°",       # Average normal vector difference
  "hausdorff_distance": "1.78 mm",  # Maximum surface mismatch
}
```

## 📦 Project Structure
```
dental-stl-analyzer/
├── stl_analyzer.py        # Main application
├── processing.py          # Core analysis algorithms
├── visualization.py       # 3D plotting components
├── utils.py               # Helper functions
├── requirements.txt       # Python dependencies
└── packages.txt           # System dependencies
```

## 📄 License
MIT License - See [LICENSE](LICENSE) for full text

## 🩺 Clinical Disclaimer
This software is intended for professional use by qualified dental practitioners. Automated measurements should always be verified by clinical experts. The developers assume no responsibility for treatment decisions made using this tool.

---

**Note**: Replace the placeholder image with actual screenshots of your application. For technical questions, please open an issue in the repository.
```

This README includes:
1. Feature overview with emoji visual cues
2. System requirements and installation instructions
3. Processing mode comparison table
4. File specifications
5. Sample metrics output
6. Project structure visualization
7. Important clinical disclaimer
8. License information

The document uses modern markdown formatting with clear section separation and emoji icons for improved readability. You can customize the placeholder image and repository URL as needed.