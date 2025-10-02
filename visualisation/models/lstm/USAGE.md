# Quick Start Guide - LSTM Shot Classifier

## ✅ Setup Complete!

All files have been copied and configured. The standalone LSTM shot classifier is ready to use.

## 🚀 Quick Test

```bash
cd /home/richard/Desktop/Projects/Personal/UNE/BST-Badminton-Stroke-type-Transformer/badminton-stroke-classifier/visualisation/lstm

# Test the help message
micromamba run -n bst311 python run_video_classifier.py --help

# Run on a video
micromamba run -n bst311 python run_video_classifier.py <your_video.mp4>
```

## 📁 Files Copied

### Core Shot Classifier (from `Badminton/src/`)
✅ `shot_classifier.py` - LSTM classifier
✅ `match_loader.py` - CSV data loader
✅ `model.py` - ShotClassification dataclass (simplified)
✅ `exceptions.py` - Error handling

### Pose Extraction Pipeline (from `Badminton/`)
✅ `badmintonplayeranalysis_main.py` - YOLO pose extraction
✅ `config.py` - Configuration & pose definitions
✅ `main_helper_functions.py` - Helper utilities
✅ `feature_extraction.py` - Feature extraction utilities

### New Files Created
✅ `run_video_classifier.py` - Main executable script
✅ `config.json` - 6 shot types configuration
✅ `README.md` - Full documentation
✅ `requirements.txt` - Dependencies
✅ `USAGE.md` - This file

## 🎯 Shot Types (6 Classes)

The model predicts one of these 6 shot types:
1. **clear** - High defensive shot to backcourt
2. **drive** - Fast flat shot at mid-height
3. **drop** - Soft shot to frontcourt
4. **lob** - High lifting shot
5. **net** - Shot close to the net
6. **smash** - Powerful downward attacking shot

## 🔧 Model Information

- **Weight File**: `../weights/15Matches_LSTM.keras` (1.6 MB)
- **Architecture**: LSTM with masking for variable-length sequences
- **Input**: Pose keypoints (17 keypoints × 2 coords per frame)
- **Training Data**: 15 professional badminton matches

## 📖 Example Usage

### Basic (with interactive court selection)
```bash
micromamba run -n bst311 python run_video_classifier.py video.mp4
```

### With pre-defined court corners
```bash
micromamba run -n bst311 python run_video_classifier.py video.mp4 \
  --corner-points 170,142,465,142,550,358,89,356
```

### Process specific frames
```bash
micromamba run -n bst311 python run_video_classifier.py video.mp4 \
  --start-frame 100 \
  --end-frame 500
```

### Keep intermediate files for debugging
```bash
micromamba run -n bst311 python run_video_classifier.py video.mp4 --keep-temp
```

## 📊 Expected Output

```
============================================================
STEP 1: Extracting player poses using YOLO
============================================================
Extracting poses from video: video.mp4
Frame range: 0 to end
Pose extraction complete. Players detected: ['Player_A', 'Player_B']

============================================================
STEP 2: Converting pose data to CSV format
============================================================
Converting pose data to CSV format in /tmp/lstm_shot_classifier_xyz/
Created shots CSV: /tmp/lstm_shot_classifier_xyz/set1_shots.csv
Created wireframe CSV: /tmp/lstm_shot_classifier_xyz/set1_wireframe.csv with 250 frames

============================================================
STEP 3: Running LSTM shot classification
============================================================
Loading match data from: /tmp/lstm_shot_classifier_xyz/
Loaded 1 match(es)
Found 1 shot(s) to classify
Running LSTM prediction with model: ../weights/15Matches_LSTM.keras
Prediction complete: smash

============================================================
🏸 SHOT CLASSIFICATION RESULT
============================================================
Predicted Shot Type: SMASH
Confidence: 0.92
All Predictions: ['smash']
============================================================
```

## 🐛 Troubleshooting

### Import Errors
Make sure you're using the correct environment:
```bash
micromamba run -n bst311 python run_video_classifier.py --help
```

### CUDA Warnings
The warnings about cuDNN/cuBLAS are normal TensorFlow initialization messages and can be ignored.

### Missing Dependencies
If you see "ModuleNotFoundError", install dependencies:
```bash
micromamba run -n bst311 pip install -r requirements.txt
```

## 📝 Next Steps

1. **Test with a sample video** from your dataset
2. **Verify predictions** against ground truth labels
3. **Adjust parameters** (frame range, court points) as needed
4. **Integrate** into your larger pipeline if needed

## 💡 Tips

- **First run is slower** due to YOLO model download
- **Use `--keep-temp`** to inspect intermediate CSV files
- **Court corners** can be saved and reused for videos from same source
- **GPU recommended** for faster pose extraction (CPU works but slower)
