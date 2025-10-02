# 🏸 Badminton Analysis System

A comprehensive AI system for badminton stroke analysis with **dual interface options**. Choose between single-method BST analysis or advanced dual-method comparison.

## 🎯 Two Analysis Methods Available

### 🤖 **BST Transformer** (Original)
- **35 Stroke Types**: Detailed classification including smashes, clears, drops, drives, and serves
- **Shuttlecock Tracking**: TrackNetV3-based trajectory analysis
- **Multi-modal Input**: Human poses + shuttlecock trajectories
- **Gradio Interface**: Easy-to-use web interface
- **High Accuracy**: 60%+ accuracy with 77%+ top-2 accuracy

### 🧠 **Dual-Method Analysis** (New)
- **BST Transformer + LSTM**: Side-by-side comparison
- **Performance Grading**: A-D scale assessment
- **Movement Analysis**: Court coverage and positioning
- **Streamlit Interface**: Rich interactive dashboard
- **Comparative Results**: Top-3 from both methods

## 🚀 Quick Launch

### Option 1: Interactive Launcher (Recommended)
```bash
python launch.py
```
Choose your preferred interface from the menu.

### Option 2: Direct Launch
```bash
# BST Transformer (Gradio)
python app.py

# Dual-Method Analysis (Streamlit)
streamlit run badminton-analysis-suite/main_app.py
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/badminton-stroke-classifier.git
cd badminton-stroke-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model Weights

Download the pre-trained model weights from Google Drive and place them in the `weights/` folder:

- **BST Model** (~7.5 MB): [Download from Google Drive](https://drive.google.com/drive/folders/1jTlfcXD50FtxcMNjoY_UxEvoInzm8F66) → `weights/bst_model.pt`
- **TrackNetV3 Model** (~45 MB): [Download from Google Drive](https://drive.google.com/drive/folders/1jTlfcXD50FtxcMNjoY_UxEvoInzm8F66) → `weights/tracknet_model.pt`

**Note**: Replace `YOUR_BST_WEIGHT_ID` and `YOUR_TRACKNET_WEIGHT_ID` with actual Google Drive file IDs.

### 3. Launch Web Interface

```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser to access the web interface.

## Usage

### Web Interface
1. Upload a badminton video (0.5-30 seconds, MP4/AVI/MOV/MKV)
2. Adjust confidence threshold if needed
3. Click "Analyze Stroke"
4. View results: prediction, confidence scores, and technical details
5. Download raw data if needed

### Python API
```python
from pipeline.file_based_pipeline import FileBased_Pipeline

# Initialize pipeline
pipeline = FileBased_Pipeline()

# Process video
result = pipeline.process_video("path/to/video.mp4")

if result.success:
    print(f"Prediction: {result.stroke_prediction}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Top 3: {result.top3_predictions}")
else:
    print(f"Error: {result.error_message}")
```

## Architecture

The system uses a multi-stage pipeline:

```
Video Input → MMPose → TrackNetV3 → BST → Stroke Classification
              ↓         ↓           ↓
           Pose Data  Shuttle Data  Feature Fusion
```

### Components

1. **MMPose**: Human pose estimation using RTMPose
2. **TrackNetV3**: Shuttlecock trajectory tracking
3. **BST (Badminton Stroke Transformer)**: Multi-modal transformer for stroke classification

### Stroke Types

The system classifies 35 stroke types:

**Basic Strokes** (both Top and Bottom court):
- Net shots, Return nets, Smashes, Wrist smashes
- Lobs, Clears, Drives, Back-court drives
- Drops, Passive drops, Pushes, Rush shots
- Cross-court net shots, Short/Long serves
- Defensive returns (lobs/drives)

## Model Performance

- **Accuracy**: 60.1%
- **Top-2 Accuracy**: 77.5%
- **F1-Score**: 0.48 (macro average)
- **Processing Time**: ~2-5 seconds per video

Best performing strokes:
- Clear shots: 81-86% F1
- Rush shots: 80% F1
- Service shots: 74-76% F1

## 🔧 Development

### Project Structure
```
badminton-stroke-classifier/
├── models/          # BST architectures and utilities
├── pipeline/        # Processing pipeline
├── tracknet/        # TrackNetV3 components
├── ui/              # Gradio web interface
├── weights/         # Model weights (downloaded)
├── examples/        # Sample videos
├── scripts/         # Utility scripts
└── docs/           # Documentation
```

### Training Your Own Model

The BST model can be retrained with your own data:

```python
from models.bst import BST_8
from models.dataset import prepare_npy_collated_loaders

# See stroke_classification/bst_main.py for full training script
```

## Citation

If you use this project in your research, please cite:

```bibtex
@article{badminton-stroke-transformer,
  title={BST: Badminton Stroke Transformer for Stroke Classification},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- [MMPose](https://github.com/open-mmlab/mmpose) for human pose estimation
- [TrackNetV3](https://github.com/alenzenx/TracknetV3) for shuttlecock tracking
- The badminton community for data and feedback