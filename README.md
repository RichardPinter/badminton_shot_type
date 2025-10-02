# Badminton Stroke Classifier

A comparative AI system for badminton shot classification featuring two distinct deep learning approaches: LSTM (pose-only) and BST Transformer (pose + shuttlecock trajectory). Upload a video and compare predictions from both models side-by-side.

## Overview

This project provides a web-based interface for classifying badminton shots using two different AI models. Both models classify the same **6 shot types** but use different architectures and input features:

**Shot Types:**
- **Clear**: High defensive shot to backcourt
- **Drive**: Fast flat shot at mid-height
- **Drop**: Soft shot to frontcourt
- **Lob**: High lifting shot
- **Net**: Shot close to the net
- **Smash**: Powerful downward attacking shot

## Two Classification Approaches

### 1. LSTM Classifier (Pose-Only)
- **Input**: Player pose keypoints only (13 keypoints)
- **Pose Detection**: YOLO11x-pose
- **Architecture**: LSTM neural network with masking
- **Training**: 15 professional badminton matches
- **Speed**: ~5-10 seconds per video
- **Use Case**: Fast classification when shuttlecock tracking not needed

### 2. BST Transformer (Pose + Shuttlecock)
- **Input**: Player pose + shuttlecock trajectory + bone vectors
- **Pose Detection**: MMPose (RTMPose)
- **Shuttlecock Tracking**: TrackNetV3
- **Architecture**: 8-layer Transformer with joint & bone features
- **Speed**: ~30-60 seconds per video
- **Use Case**: Higher accuracy with full trajectory analysis

## Quick Start

### Installation

```bash
# Navigate to project directory
cd badminton-stroke-classifier

# Install dependencies
pip install -r requirements.txt
```

**Dependencies Include:**
- PyTorch 2.0+ (for BST model)
- TensorFlow 2.12+ (for LSTM model)
- Ultralytics (for YOLO pose detection)
- MMPose (for BST pose detection)
- OpenCV, Gradio, NumPy, Pandas

### Model Weights

The following model weights should be in the `weights/` directory:

- **LSTM Model**: `15Matches_LSTM.h5` or `15Matches_LSTM.keras` (~1.6 MB)
- **BST Model**: `bst_8_JnB_bone_bottom_frontier_6class.pt` (~7.5 MB)
- **TrackNet Model**: `tracknet_model.pt` (~174 MB)

All weights are included in the repository.

### Launch Application

```bash
python app.py
```

The Gradio interface will open automatically in your browser (typically http://127.0.0.1:7860).

## Usage

### Web Interface

1. **Upload Video**: Click to upload a badminton video (MP4, AVI, MOV, MKV)
   - Or select a demo video from the provided examples

2. **For BST Classifier Only**: Calibrate court corners
   - Click "Extract Frame for Calibration"
   - Click 4 corners in order: Back-Left → Back-Right → Front-Right → Front-Left
   - LSTM classifier does not require calibration

3. **Run Classification**:
   - Click "Classify with LSTM" for fast pose-only classification
   - Click "Classify with BST" for full trajectory-based classification
   - Or run both to compare results

4. **View Results**: Each classifier returns:
   - Predicted shot type
   - Confidence score
   - Processing details
   - Top-3 predictions (BST only)

### Demo Videos

Pre-labeled demo videos are included in `demo_videos/` for testing:
- Smash, Net, Lob, Clear, Drop, Drive examples
- Pre-calibrated for immediate BST classification

## Project Structure

```
badminton-stroke-classifier/
├── app.py                          # Main Gradio web application
├── models/                         # Model implementations
│   ├── bst/                        # BST Transformer system
│   │   ├── models/                 # BST model architecture
│   │   ├── pipeline/               # Video processing pipeline
│   │   ├── ui/                     # BST-specific UI components
│   │   └── app.py                  # Standalone BST app
│   ├── lstm/                       # LSTM classifier system
│   │   ├── run_video_classifier.py # Main LSTM pipeline
│   │   ├── shot_classifier.py      # LSTM model wrapper
│   │   ├── match_loader.py         # CSV data loader
│   │   └── gradio_app.py           # Standalone LSTM app
│   ├── tracknet/                   # TrackNetV3 shuttlecock detection
│   │   ├── model.py                # TrackNetV2 architecture
│   │   ├── predict.py              # Inference script
│   │   └── utils.py                # Helper functions
│   └── preprocessing/              # Video preprocessing utilities
│       ├── process_single_video.py # End-to-end preprocessing
│       └── prepare_train.py        # Training data preparation
├── weights/                        # Model weights
│   ├── 15Matches_LSTM.keras        # LSTM model
│   ├── bst_8_JnB_bone_bottom_frontier_6class.pt  # BST model
│   └── tracknet_model.pt           # TrackNet model
├── demo_videos/                    # Sample videos for testing
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Technical Details

### LSTM Classifier Pipeline

```
Video Input
    ↓
YOLO11x-Pose Detection (17 keypoints)
    ↓
Extract 13 keypoints (exclude eyes/ears)
    ↓
Normalize & Create CSV (setN_shots.csv, setN_wireframe.csv)
    ↓
LSTM Model (Masked LSTM, 256 hidden units)
    ↓
Shot Prediction (6 classes)
```

**Key Details:**
- Input: 41 frames × 13 keypoints × 2 coordinates (26 features)
- Masking layer handles variable-length sequences
- Framework: TensorFlow 2.12 / Keras 2
- Model file: `weights/15Matches_LSTM.keras`

### BST Transformer Pipeline

```
Video Input
    ↓
Court Calibration (Homography matrix from 4 corners)
    ↓
TrackNetV3 Shuttlecock Detection
    ↓
MMPose Player Pose Estimation (RTMPose)
    ↓
Data Collation (joints, positions, shuttle trajectory)
    ↓
BST-8 Transformer (Joint + Bone features)
    ↓
Shot Prediction (6 classes + confidence + top-3)
```

**Key Details:**
- Input: Player joints (17 keypoints) + bone vectors + shuttlecock trajectory
- Homography: Maps pixel coordinates to real court positions (6.1m × 13.4m)
- Architecture: 8-layer Transformer with attention mechanism
- Framework: PyTorch
- Model file: `weights/bst_8_JnB_bone_bottom_frontier_6class.pt`

### TrackNetV3 Details

- **Architecture**: TrackNetV2 with CBAM attention mechanism
- **Input**: 3 consecutive frames (RGB)
- **Output**: Heatmap predicting shuttlecock location
- **Checkpoint Format**: Supports both old (param_dict) and new (state_dict) formats
- **Model file**: `weights/tracknet_model.pt`

## System Requirements

- **Python**: 3.8 - 3.12
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended for faster processing
  - CUDA-capable GPU significantly speeds up both models
  - CPU-only mode supported but slower
- **Storage**: ~5GB for models and temporary processing files

## Performance

| Metric | LSTM Classifier | BST Transformer |
|--------|----------------|-----------------|
| Processing Time | 5-10 seconds | 30-60 seconds |
| Input Features | Pose only | Pose + Shuttlecock + Bone |
| Accuracy | Good | Higher (with trajectory context) |
| Court Calibration | Not required | Required |
| GPU Acceleration | Optional | Recommended |

## Standalone Apps

Each classifier can also run independently:

### LSTM Only
```bash
cd models/lstm
python gradio_app.py
```

### BST Only
```bash
cd models/bst
python app.py
```

### Command-Line LSTM
```bash
cd models/lstm
python run_video_classifier.py <video_path>
```

## Troubleshooting

### LSTM Classifier Issues

**"No players detected in video"**
- Ensure video shows clear view of player
- Check video quality (>720p recommended)
- Player should be visible throughout the clip

**"Module not found" errors**
- Install dependencies: `pip install -r requirements.txt`
- Ensure using Python 3.8+

### BST Classifier Issues

**"Please calibrate court corners first"**
- Must click 4 court corners before running BST
- Corners must be in order: Back-Left, Back-Right, Front-Right, Front-Left

**"Preprocessing failed"**
- Check TrackNet model exists: `weights/tracknet_model.pt`
- Verify MMPose installation: `pip install mmpose mmcv`
- Ensure sufficient GPU memory or use CPU mode

**"BST inference failed"**
- Check BST model exists: `weights/bst_8_JnB_bone_bottom_frontier_6class.pt`
- Verify intermediate files generated in temp directory
- Use `--keep-intermediates` flag for debugging

### General Issues

**Out of Memory**
- Close other applications
- Reduce video resolution
- Use CPU mode if GPU memory insufficient

**Slow Processing**
- GPU significantly speeds up processing
- Shorter videos process faster (<10 seconds recommended)
- LSTM is faster than BST if speed is priority

## Development

### Adding New Shot Types

To extend beyond the 6 current shot types:

1. Retrain both models with new labeled data
2. Update `SHOT_TYPES` in `app.py`
3. Update config files in `models/lstm/config.json` and BST config
4. Retrain and replace model weights

### Testing with Custom Videos

```bash
# Test LSTM standalone
cd models/lstm
python run_video_classifier.py /path/to/video.mp4 --keep-temp

# Test BST standalone
cd models/bst/pipeline
python single_video_inference.py /path/to/video.mp4
```

### Model Training

- **LSTM Training**: See `models/lstm/README.md` for training instructions
- **BST Training**: See `models/bst/README.md` for training pipeline

## Credits

- **BST Transformer**: Badminton Stroke Transformer architecture
- **LSTM Classifier**: Trained on 15 professional matches
- **TrackNetV3**: Shuttlecock detection from TrackNet paper
- **Pose Detection**: MMPose (Alibaba), Ultralytics YOLO
- **Dataset**: ShuttleSet badminton dataset

## License

See individual component licenses for specific terms.

## Citation

If you use this code in your research, please cite the original BST paper and dataset:

```
[Add appropriate citations here]
```

---

**Note**: This is a research tool. For production use, consider additional validation and error handling.
