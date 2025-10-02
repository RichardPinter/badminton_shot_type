# LSTM Shot Classifier - Standalone Pipeline

End-to-end badminton shot classification using YOLO11x-pose + LSTM.

## Overview

This standalone pipeline classifies badminton shots into **6 categories**:
- **clear** - High defensive shot to backcourt
- **drive** - Fast flat shot at mid-height
- **drop** - Soft shot to frontcourt
- **lob** - High lifting shot
- **net** - Shot close to the net
- **smash** - Powerful downward attacking shot

## Pipeline

```
Video Input
    ↓
[YOLO11x-Pose] → Extract player poses (17 keypoints per frame)
    ↓
[CSV Generation] → Create setN_shots.csv + setN_wireframe.csv
    ↓
[LSTM Classifier] → Load model & predict shot type
    ↓
Result: Shot type + confidence
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Install Dependencies

```bash
pip install tensorflow>=2.12.0
pip install ultralytics>=8.0.35
pip install torch==2.2.0
pip install opencv-python
pip install numpy pandas joblib
```

Or install from the root visualisation directory:
```bash
cd /path/to/badminton-stroke-classifier/visualisation
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
cd /path/to/visualisation/lstm
python run_video_classifier.py <video_path>
```

Example:
```bash
python run_video_classifier.py sample_match.mp4
```

### With Court Corner Points

If you know the court boundary coordinates:

```bash
python run_video_classifier.py video.mp4 --corner-points 170,142,465,142,550,358,89,356
```

Format: `x1,y1,x2,y2,x3,y3,x4,y4` (4 corners: top-left, top-right, bottom-right, bottom-left)

### Specify Custom Model

```bash
python run_video_classifier.py video.mp4 --model /path/to/custom_model.keras
```

### Process Specific Frame Range

```bash
python run_video_classifier.py video.mp4 --start-frame 100 --end-frame 500
```

### Keep Temporary Files for Debugging

```bash
python run_video_classifier.py video.mp4 --keep-temp
```

This will preserve the generated CSV files in a temporary directory for inspection.

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `video_path` | Path to input video (required) | - |
| `--corner-points` | Court corners as `x1,y1,x2,y2,x3,y3,x4,y4` | Interactive prompt |
| `--model` | Path to LSTM .keras model | `../weights/15Matches_LSTM.keras` |
| `--start-frame` | Starting frame number | `0` |
| `--end-frame` | Ending frame number | End of video |
| `--keep-temp` | Keep temporary CSV files | `False` |

## Model

- **Model File**: `../weights/15Matches_LSTM.keras`
- **Architecture**: LSTM with Masking layer for variable-length sequences
- **Input**: Pose keypoint sequences (17 keypoints × 2 coordinates per frame)
- **Output**: 6-class softmax (clear, drive, drop, lob, net, smash)
- **Training**: Trained on 15 professional badminton matches

## File Structure

```
lstm/
├── run_video_classifier.py       # Main script
├── shot_classifier.py             # LSTM classifier
├── match_loader.py                # Data loading utilities
├── model.py                       # ShotClassification dataclass
├── exceptions.py                  # Custom exceptions
├── badmintonplayeranalysis_main.py # Pose extraction
├── config.py                      # Configuration
├── main_helper_functions.py       # Helper functions
├── config.json                    # Shot types config
└── README.md                      # This file
```

## Output Format

```
🏸 SHOT CLASSIFICATION RESULT
============================================================
Predicted Shot Type: SMASH
Confidence: 0.92
All Predictions: ['smash']
============================================================
```

## Troubleshooting

### "Video file not found"
Ensure the video path is correct and the file exists.

### "No valid player data found"
The YOLO pose model couldn't detect players. Try:
- Providing explicit court corner points with `--corner-points`
- Using a higher quality video
- Checking video resolution is sufficient (>720p recommended)

### "No matches loaded from CSV"
CSV generation failed. Check:
- Temporary directory has write permissions
- Pose data contains valid keypoints
- Use `--keep-temp` to inspect generated CSV files

### CUDA out of memory
- Use CPU instead: Edit `config.py` and set `COMPUTATION_DEVICE = 'cpu'`
- Process fewer frames: Use `--start-frame` and `--end-frame`

## Development

### Adding New Shot Types

1. Edit `config.json` to add new shot type
2. Retrain LSTM model with new classes
3. Update `SHOT_TYPES` list in `run_video_classifier.py`

### Testing

```bash
# Run with sample video (keep intermediates for inspection)
python run_video_classifier.py test_video.mp4 --keep-temp

# Check generated CSV files
ls /tmp/lstm_shot_classifier_*/
```

## Credits

- **Pose Extraction**: YOLO11x-pose (Ultralytics)
- **Shot Classifier**: LSTM architecture by Drew
- **Dataset**: ShuttleSet badminton dataset

## License

Part of the BST-Badminton-Stroke-type-Transformer project.
