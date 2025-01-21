# 🦷 Dental Cavity Preparation Analyzer

A sophisticated web application for analyzing and evaluating dental cavity preparations by comparing student-created models with reference standards.

## 🎯 Features

- **3D Visualization**: Interactive 3D visualization of cavity preparations with deviation maps
- **Real-time Analysis**: Instant feedback on preparation quality
- **Comprehensive Metrics**: 
  - Maximum and average deviations
  - Under/over-preparation analysis
  - Point-wise deviation distribution
  - Cavity region segmentation
- **Export Capabilities**: Download detailed analysis results in CSV format

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd dental-cavity-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run stl_analyzer.py
```

## 📊 Usage

1. **Upload Reference Model**:
   - Select your pre-cropped reference cavity model (STL format)
   - This model represents the ideal cavity preparation

2. **Upload Student Models**:
   - Upload one or more student cavity preparations (STL format)
   - Multiple files can be analyzed in sequence

3. **Adjust Parameters** (optional):
   - Point Cloud Generation:
     - Number of points (default: 10,000)
     - Cleaning strength (default: 2.0)
   - Analysis Settings:
     - Tolerance range (default: 0.5mm)
     - Registration precision

4. **View Results**:
   - 3D Deviation Map
   - Preparation Quality Analysis
   - Statistical Distribution
   - Detailed Metrics

## 📈 Analysis Features

### Visualization Types

1. **3D Deviation Map**:
   - Color-coded visualization of deviations
   - Interactive 3D view
   - Tolerance planes for reference

2. **Quality Analysis**:
   - Percentage of points within tolerance
   - Under-prepared regions
   - Over-prepared regions

3. **Statistical Analysis**:
   - Deviation distribution histogram
   - Mean and standard deviation
   - Tolerance range indicators

### Metrics

- **Maximum Deviation**: Largest difference from reference
- **Average Deviation**: Mean difference across all points
- **Points Within Tolerance**: Percentage of acceptable deviations
- **Under/Over-prepared Points**: Count and percentage
- **Registration Quality**: Alignment fitness and RMSE

## 🛠️ Technical Details

### File Structure

```
dental-cavity-analyzer/
├── stl_analyzer.py     # Main application
├── processing.py       # Core processing logic
├── visualization.py    # Visualization functions
├── utils.py           # Utility functions
└── requirements.txt   # Dependencies
```

### Components

- **STL Processing**: Using Open3D for mesh handling
- **Point Cloud Analysis**: Custom algorithms for dental-specific metrics
- **Visualization**: Plotly for interactive 3D visualization
- **UI**: Streamlit for web interface

## 🔒 Security

- File validation for STL uploads
- Size limits on uploads
- Sanitized file handling

## 📝 Notes

- Reference models should be pre-cropped to the cavity region
- Recommended point cloud size: 10,000-50,000 points
- Typical processing time: 2-5 seconds per model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- ENES KAAN KARABAY

## 🙏 Acknowledgments

- Open3D team for the 3D processing library
- Streamlit team for the web framework
- Plotly team for visualization capabilities

## Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your forked repository
5. Select the main branch and `stl_analyzer.py` as the main file
6. Click "Deploy"

The app will be automatically deployed and available at a public URL.

### Important Notes for Cloud Deployment

- Maximum file upload size is set to 50MB
- Temporary files are automatically cleaned up
- Session state is maintained per user
- All computations are performed in-memory
