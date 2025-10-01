#!/usr/bin/env python3
# ABOUTME: Combined Gradio app for both LSTM and BST badminton stroke classifiers
# ABOUTME: Two tabs: LSTM (pose-only) and BST Transformer (pose + shuttlecock)

"""
Combined Gradio App for Badminton Stroke Classification

This provides a unified web interface with two classification models:
1. LSTM Classifier - 6 shot types using player pose only (clear, drive, drop, lob, net, smash)
2. BST Transformer - 6 shot types using pose + shuttlecock trajectory (clear, drive, drop, lob, net, smash)

Both models classify the same 6 shot types but use different architectures and features.

Usage:
    python combined_gradio_app.py

Then open the provided URL in your browser.
"""

import gradio as gr
import subprocess
import sys
import tempfile
import shutil
import os
import logging
from pathlib import Path
from typing import Tuple

# Import LSTM classifier components
sys.path.insert(0, str(Path(__file__).parent / "lstm"))
from lstm.run_video_classifier import (
    extract_poses_from_video,
    convert_pose_data_to_csv,
    predict_shot_from_csv,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Both LSTM and BST classify the same 6 shot types
SHOT_TYPES = ['clear', 'drive', 'drop', 'lob', 'net', 'smash']


# ============================================================================
# LSTM CLASSIFIER FUNCTIONS
# ============================================================================

def classify_shot_lstm(video_file, progress=gr.Progress()) -> str:
    """
    Process uploaded video with LSTM classifier (6 shot types).

    Args:
        video_file: Uploaded video file from Gradio

    Returns:
        Formatted markdown result string
    """
    if video_file is None:
        return "❌ **Error:** No video uploaded"

    try:
        progress(0.1, desc="Creating temporary directory...")
        temp_dir = tempfile.mkdtemp(prefix='lstm_shot_classifier_')
        logger.info(f"Processing video: {video_file}")

        # Step 1: Extract poses from video
        progress(0.2, desc="Extracting player poses with YOLO...")
        pose_data = extract_poses_from_video(
            video_file,
            corner_points=None,
            start_frame=0,
            end_frame=None
        )

        # Count frames
        frame_counts = {player: len(frames) for player, frames in pose_data.items()}
        total_frames = max(frame_counts.values()) if frame_counts else 0

        if total_frames == 0:
            return "❌ **Error:** No players detected in video. Please ensure the video contains visible badminton players."

        # Step 2: Convert to CSV format
        progress(0.5, desc="Converting pose data to CSV...")
        convert_pose_data_to_csv(
            pose_data,
            temp_dir,
            set_id=1,
            video_path=video_file
        )

        # Step 3: Run LSTM prediction
        progress(0.8, desc="Running LSTM classification...")
        model_path = 'weights/15Matches_LSTM.h5'
        if not os.path.exists(model_path):
            model_path = '../weights/15Matches_LSTM.h5'

        result = predict_shot_from_csv(temp_dir, model_path)

        # Format results
        shot_type = result['shot'].upper()
        confidence = result['confidence']

        # Create result with emoji
        shot_emoji = {
            'clear': '🏸',
            'drive': '➡️',
            'drop': '⬇️',
            'lob': '⬆️',
            'net': '🎾',
            'smash': '💥'
        }

        emoji = shot_emoji.get(result['shot'].lower(), '🏸')

        result_md = f"""
# {emoji} LSTM Classification Result

## Predicted Shot Type: **{shot_type}**

**Confidence:** {confidence:.1%}

---

### Processing Summary:
- **Total frames processed:** {total_frames}
- **Players detected:** {', '.join(frame_counts.keys())}
- **Frame counts:** {', '.join(f'{k}: {v}' for k, v in frame_counts.items())}

### Model Information:
- **Model:** 15Matches_LSTM (LSTM Neural Network)
- **Shot classes:** clear, drive, drop, lob, net, smash
- **Input:** 41 frames × 13 keypoints (26 features)
- **Framework:** TensorFlow 2.12 / Keras 2

---

✅ Classification complete!
"""

        # Cleanup
        shutil.rmtree(temp_dir)
        progress(1.0, desc="Complete!")

        return result_md

    except Exception as e:
        logger.error(f"Error during LSTM classification: {e}", exc_info=True)
        return f"❌ **Error occurred during processing:**\n\n```\n{str(e)}\n```"


# ============================================================================
# BST CLASSIFIER FUNCTIONS
# ============================================================================

def classify_shot_bst(video_file, progress=gr.Progress()) -> str:
    """
    Process uploaded video with BST Transformer (6 shot types with shuttlecock tracking).

    Args:
        video_file: Uploaded video file from Gradio

    Returns:
        Formatted markdown result string
    """
    if video_file is None:
        return "❌ **Error:** No video uploaded"

    try:
        progress(0.1, desc="Creating temporary directory...")
        temp_dir = Path(tempfile.mkdtemp(prefix="bst_classifier_"))
        npy_dir = temp_dir / "npy"
        npy_dir.mkdir(exist_ok=True)

        # Step 1: Run preprocessing (TrackNet + MMPose + collation)
        progress(0.3, desc="Running TrackNet and MMPose preprocessing...")

        process_script = Path("src/process_single_video.py")
        cmd = [
            sys.executable,
            str(process_script),
            str(video_file),
            "-o", str(npy_dir),
            "--seq-len", "100",
            "--keep-intermediates"
        ]

        logger.info(f"Running preprocessing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            error_msg = f"❌ **Preprocessing failed:**\n\n```\n{result.stderr}\n```"
            return error_msg

        progress(0.6, desc="Preprocessing complete, running BST inference...")

        # Step 2: Prepare files for BST inference
        intermediates_dir = npy_dir / "intermediates"
        video_stem = Path(video_file).stem

        base_name = npy_dir / video_stem
        joints_src = intermediates_dir / f"{video_stem}_joints.npy"
        pos_src = intermediates_dir / f"{video_stem}_pos.npy"
        shuttle_src = intermediates_dir / f"{video_stem}_shuttle.npy"

        joints_dst = Path(str(base_name) + "_joints.npy")
        pos_dst = Path(str(base_name) + "_pos.npy")
        shuttle_dst = Path(str(base_name) + "_shuttle.npy")

        if not joints_src.exists():
            return f"❌ **Error:** Expected intermediate file not found: {joints_src}"

        shutil.copy2(joints_src, joints_dst)
        shutil.copy2(pos_src, pos_dst)
        shutil.copy2(shuttle_src, shuttle_dst)

        # Step 3: Run BST inference
        progress(0.8, desc="Running BST Transformer inference...")

        bst_script = Path("models/bst/pipeline/run_bst_on_triplet.py")
        bst_cmd = [
            sys.executable,
            str(bst_script),
            str(base_name),
            "--weights", "weights/bst_8_JnB_bone_bottom_frontier_6class.pt",
            "--device", "cuda"
        ]

        logger.info(f"Running BST: {' '.join(bst_cmd)}")
        bst_result = subprocess.run(bst_cmd, capture_output=True, text=True, timeout=60)

        if bst_result.returncode != 0:
            error_msg = f"❌ **BST inference failed:**\n\n```\n{bst_result.stderr}\n```"
            return error_msg

        # Parse the output
        output_lines = bst_result.stdout.strip().split('\n')
        predicted_class = None
        predicted_conf = None
        top3_results = []

        for i, line in enumerate(output_lines):
            if line.startswith("Class:"):
                predicted_class = line.split(":")[-1].strip()
            elif line.startswith("Confidence:"):
                predicted_conf = line.split(":")[-1].strip()
            elif line.startswith("Top-3:"):
                for j in range(i + 1, min(i + 4, len(output_lines))):
                    if output_lines[j].strip() and output_lines[j][0].isdigit():
                        top3_results.append(output_lines[j])

        if predicted_class is None:
            return f"❌ **Could not parse BST output:**\n\n```\n{bst_result.stdout}\n```"

        # Format top-3 results
        top3_display = "\n".join(top3_results) if top3_results else "N/A"

        # Create result with emoji
        shot_emoji = {
            'clear': '🏸',
            'drive': '➡️',
            'drop': '⬇️',
            'lob': '⬆️',
            'net': '🎾',
            'smash': '💥'
        }
        emoji = shot_emoji.get(predicted_class.lower(), '🏸')

        result_md = f"""
# {emoji} BST Transformer Classification Result

## Predicted Shot Type: **{predicted_class.upper()}**

**Confidence:** {predicted_conf}

---

### Top-3 Predictions:
```
{top3_display}
```

---

### Processing Summary:
✅ Video processed successfully
✅ TrackNet shuttlecock detection
✅ MMPose player pose estimation
✅ BST Transformer inference

### Model Information:
- **Model:** BST Transformer (8 layers, Joint & Bone features)
- **Shot classes:** 6 types (clear, drive, drop, lob, net, smash)
- **Features:** Player joints, shuttlecock trajectory, bone vectors
- **Device:** CUDA (GPU)

Temporary files: `{temp_dir}`

---

✅ Classification complete!
"""

        progress(1.0, desc="Complete!")
        return result_md

    except subprocess.TimeoutExpired:
        return "❌ **Error:** Processing timed out (>10 minutes)"
    except Exception as e:
        import traceback
        error_msg = f"❌ **Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        logger.error(error_msg)
        return error_msg


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="Badminton Stroke Classifiers", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏸 Badminton Stroke Classification System

    Compare **two different classification models** side-by-side. Both classify the same 6 shot types:

    **🏸 Clear** • **➡️ Drive** • **⬇️ Drop** • **⬆️ Lob** • **🎾 Net** • **💥 Smash**
    """)

    # Single shared video input at the top
    with gr.Row():
        video_input = gr.Video(
            label="Upload Badminton Video",
            format="mp4",
            scale=2
        )

    gr.Markdown("---")

    # Side-by-side classifiers
    with gr.Row(equal_height=True):
        # ====================================================================
        # LEFT COLUMN: LSTM CLASSIFIER
        # ====================================================================
        with gr.Column(scale=1):
            gr.Markdown("""
            ## 🎯 LSTM Classifier

            **Fast pose-only classification**
            - ⚡ Processing: ~5-10 seconds
            - 🎭 Input: Player pose only (13 keypoints)
            - 🧠 Architecture: LSTM Neural Network
            - 📊 Trained on 15 matches
            """)

            lstm_classify_btn = gr.Button(
                "Classify with LSTM 🎯",
                variant="primary",
                size="lg",
                scale=1
            )

            lstm_result_output = gr.Markdown(
                label="LSTM Results",
                value="Click 'Classify with LSTM' to get results.",
                container=True
            )

        # ====================================================================
        # RIGHT COLUMN: BST TRANSFORMER
        # ====================================================================
        with gr.Column(scale=1):
            gr.Markdown("""
            ## 🚀 BST Transformer

            **Advanced pose + shuttlecock classification**
            - ⏱️ Processing: ~30-60 seconds
            - 🎾 Input: Pose + shuttlecock trajectory
            - 🤖 Architecture: Transformer (8 layers)
            - 📈 Higher accuracy with context
            """)

            bst_classify_btn = gr.Button(
                "Classify with BST 🚀",
                variant="secondary",
                size="lg",
                scale=1
            )

            bst_result_output = gr.Markdown(
                label="BST Results",
                value="Click 'Classify with BST' to get results.",
                container=True
            )

    # Connect buttons to functions
    lstm_classify_btn.click(
        fn=classify_shot_lstm,
        inputs=[video_input],
        outputs=[lstm_result_output]
    )

    bst_classify_btn.click(
        fn=classify_shot_bst,
        inputs=[video_input],
        outputs=[bst_result_output]
    )

    # ========================================================================
    # FOOTER
    # ========================================================================
    gr.Markdown("""
    ---

    ## Model Comparison

    | Feature | LSTM Classifier | BST Transformer |
    |---------|----------------|-----------------|
    | Shot classes | 6 types | 6 types (same as LSTM) |
    | Processing time | ~5-10 seconds | ~30-60 seconds |
    | Input features | Player pose only | Pose + shuttlecock trajectory |
    | Shuttlecock tracking | No | Yes (TrackNetV3) |
    | Architecture | LSTM Neural Network | Transformer (8 layers) |
    | Best for | Quick analysis | Higher accuracy |

    ## Technical Details

    ### LSTM Classifier
    - **Architecture:** LSTM Neural Network
    - **Training:** 15 professional badminton matches
    - **Input:** 41 frames × 13 keypoints (26 features)
    - **Pose extraction:** YOLO11x-pose
    - **Framework:** TensorFlow 2.12 / Keras 2

    ### BST Transformer
    - **Architecture:** Transformer with 8 layers
    - **Features:** Joint positions, bone vectors, shuttlecock trajectory
    - **Shuttlecock detection:** TrackNetV3
    - **Pose extraction:** MMPose (RTMPose)
    - **Framework:** PyTorch

    ---

    **Note:** For best results, ensure videos show clear view of the player performing the shot.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True
    )
