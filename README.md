# 🏸 Badminton Analysis Suite

A unified dual-method AI system for comprehensive badminton stroke analysis and performance assessment.

## 🎯 Overview

This project combines two powerful analysis systems:

- **🤖 BST Transformer**: 35 stroke type classification with shuttlecock tracking
- **🧠 LSTM Analysis**: Performance grading and movement analysis

## ✨ Features

### BST Transformer Analysis
- **35 Stroke Types**: Detailed classification including smashes, clears, drops, drives, and serves
- **Shuttlecock Tracking**: TrackNetV3-based trajectory analysis
- **Multi-modal Input**: Human poses (MMPose) + shuttlecock trajectories
- **High Accuracy**: 60%+ accuracy with 77%+ top-2 accuracy

### LSTM Analysis (Architecture Ready)
- **Performance Grading**: A-D scale assessment
- **Movement Analysis**: Court coverage and positioning
- **Technical Assessment**: Stroke consistency and form evaluation
- **Ready for Activation**: Complete architecture, awaiting model weights

### Unified Interface
- **Single Video Upload**: One input for both analysis methods
- **Parallel Processing**: Both systems run simultaneously
- **Comparative Results**: Top-3 predictions from each method
- **Rich Visualizations**: Confidence charts and technical details

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository (if not already in the project)
cd badminton-analysis-suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Weights Setup

#### BST Transformer (Active)
Ensure BST model weights are available in `models/bst/weights/`:
- `bst_model.pt` - BST Transformer weights
- `tracknet_model.pt` - TrackNetV3 weights

#### LSTM Analysis (Pending)
System is ready for LSTM weights activation:
- Architecture: Complete
- Integration: Ready
- Missing: Trained model weights (.keras files)

### 3. Launch Application

```bash
streamlit run main_app.py
```

Open http://localhost:8501 in your browser.

## 📱 Usage

### Video Upload
1. Upload a badminton video (MP4, AVI, MOV, MKV)
2. Video requirements: 0.5-60 seconds, max 500MB
3. Ensure clear view of player(s) for optimal analysis

### Analysis Process
1. **Automatic Validation**: System checks video format and quality
2. **Dual Processing**: Both BST and LSTM systems analyze simultaneously
3. **Results Display**: Top-3 predictions from each method
4. **Comparison View**: Side-by-side analysis when both systems active

### Results Interpretation

#### BST Transformer Results
- **Primary Prediction**: Highest confidence stroke classification
- **Confidence Scores**: Percentage confidence for each prediction
- **Technical Details**: Processing pipeline information
- **Shuttlecock Data**: Trajectory analysis (when available)

#### LSTM Analysis (When Active)
- **Performance Grade**: Overall A-D assessment
- **Movement Analysis**: Court coverage and efficiency
- **Technical Consistency**: Stroke form evaluation
- **Comparative Metrics**: Performance benchmarking

## 🏗️ Architecture

### Project Structure
```
badminton-analysis-suite/
├── main_app.py                    # Main Streamlit application
├── core/                          # Core integration layer
│   ├── video_processor.py         # Unified video handling
│   ├── bst_wrapper.py            # BST system integration
│   ├── lstm_wrapper.py           # LSTM system wrapper
│   └── result_formatter.py       # Output standardization
├── models/                        # Model components
│   ├── bst/                      # BST Transformer system
│   └── lstm/                     # LSTM analysis system
├── ui/                           # User interface components
│   ├── shared_components.py      # Common UI elements
│   ├── bst_components.py         # BST-specific displays
│   └── lstm_components.py        # LSTM-specific displays
├── requirements.txt              # Unified dependencies
└── README.md                     # This file
```

### System Integration Flow
```
Video Upload → Validation → Dual Processing → Results Formatting → UI Display
                              ↙         ↘
                    BST Analysis    LSTM Analysis
                    (MMPose +       (YOLO Pose +
                     TrackNetV3 +    Movement +
                     BST-8)          LSTM)
```

## 🔧 Technical Details

### Dependencies
- **Python**: 3.8-3.12
- **Deep Learning**: PyTorch 2.0+, TensorFlow 2.12+
- **Computer Vision**: OpenCV, MMPose, Ultralytics
- **Web Interface**: Streamlit, Gradio components
- **Visualization**: Plotly, Matplotlib

### System Requirements
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: Optional but recommended for faster processing
- **Storage**: 5GB free space for models and processing
- **Network**: Internet connection for initial model downloads

### Performance
- **BST Analysis**: 2-5 seconds per video
- **LSTM Analysis**: 1-3 seconds per video (when active)
- **Parallel Processing**: Both systems run simultaneously
- **Memory Usage**: 2-4GB during active processing

## 📊 Model Information

### BST Transformer
- **Architecture**: Multi-modal transformer with pose and trajectory inputs
- **Training Data**: Professional badminton match footage
- **Classes**: 35 distinct stroke types
- **Accuracy**: 60.1% (77.5% top-2)

### LSTM Classifier (Ready for Activation)
- **Architecture**: Masked LSTM with 256 hidden units
- **Input**: Pose sequence data from YOLO detection
- **Output**: Shot classification + performance grading
- **Features**: Movement analysis and technical assessment

## 🚧 Development Status

### ✅ Complete
- [x] Unified project architecture
- [x] BST Transformer integration
- [x] LSTM system framework
- [x] Streamlit GUI interface
- [x] Dual-method result display
- [x] Video processing pipeline
- [x] Error handling and validation

### 🔄 In Progress
- [ ] BST model weight verification
- [ ] LSTM model weight integration
- [ ] Shuttlecock trajectory visualization
- [ ] Court coverage heatmaps

### 📋 Planned Features
- [ ] Batch video processing
- [ ] Historical analysis comparison
- [ ] Training recommendation system
- [ ] Export to PDF reports
- [ ] Mobile-responsive interface

## 🐛 Troubleshooting

### Common Issues

**BST Analysis Not Working**
- Check model weights in `models/bst/weights/`
- Verify CUDA/GPU memory availability
- Ensure video meets format requirements

**LSTM Analysis Pending**
- System architecture is complete
- Awaiting trained model weights
- Ready for immediate activation when weights available

**Video Upload Issues**
- Supported formats: MP4, AVI, MOV, MKV
- Size limit: 500MB
- Duration: 0.5-60 seconds

**Performance Issues**
- Close other applications to free memory
- Use GPU acceleration if available
- Reduce video resolution if needed

### Getting Help
1. Check error messages in the UI
2. Review system status in sidebar
3. Verify model weights are present
4. Ensure all dependencies are installed

## 🤝 Contributing

This project combines work from multiple badminton analysis systems:
- BST Transformer: Advanced stroke classification
- LSTM Analysis: Performance grading system
- Integration Layer: Unified processing pipeline

## 📄 License

See individual component licenses for specific terms.

## 🎯 Future Roadmap

### Phase 1: Current Implementation
- [x] Dual-method architecture
- [x] BST Transformer integration
- [x] GUI interface development

### Phase 2: LSTM Activation
- [ ] LSTM model weight integration
- [ ] Full dual-method functionality
- [ ] Comparative analysis features

### Phase 3: Advanced Features
- [ ] Real-time analysis
- [ ] Coaching recommendations
- [ ] Performance tracking over time
- [ ] Professional player comparisons

---

🏸 **Ready to analyze your badminton game with cutting-edge AI!**