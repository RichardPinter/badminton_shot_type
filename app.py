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
    python app.py

Then open the provided URL in your browser.
"""

import gradio as gr
import subprocess
import sys
import tempfile
import shutil
import os
import logging
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Import LSTM classifier components
sys.path.insert(0, str(Path(__file__).parent / "models" / "lstm"))
from models.lstm.run_video_classifier import (
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

# Standard badminton court dimensions (doubles, in meters)
COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4


# ============================================================================
# COURT CALIBRATION HELPER FUNCTIONS
# ============================================================================

def extract_first_frame(video_path: str) -> Optional[np.ndarray]:
    """Extract first frame from video for court calibration."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def draw_court_corners(image: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
    """Draw clicked corner points on image with labels."""
    img = image.copy()
    corner_labels = ["1: Back-Left", "2: Back-Right", "3: Front-Right", "4: Front-Left"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i, (x, y) in enumerate(corners):
        cv2.circle(img, (x, y), 10, colors[i], -1)
        cv2.putText(img, corner_labels[i], (x + 15, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

    # Draw court outline if we have all 4 points
    if len(corners) == 4:
        pts = np.array(corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)

    return img


def compute_homography_from_corners(
    corners: List[Tuple[int, int]],
    video_width: int,
    video_height: int
) -> np.ndarray:
    """
    Compute homography matrix from 4 clicked court corners.

    Args:
        corners: List of 4 (x, y) tuples in order: back-left, back-right, front-right, front-left
        video_width: Video frame width in pixels
        video_height: Video frame height in pixels

    Returns:
        3x3 homography matrix H
    """
    # Source points (camera coordinates, normalized to 1280x720)
    src_points = np.array([
        [corners[0][0] * 1280 / video_width, corners[0][1] * 720 / video_height], # back-left
        [corners[1][0] * 1280 / video_width, corners[1][1] * 720 / video_height], # back-right
        [corners[2][0] * 1280 / video_width, corners[2][1] * 720 / video_height], # front-right
        [corners[3][0] * 1280 / video_width, corners[3][1] * 720 / video_height], # front-left
    ], dtype=np.float32)

    # Destination points (court coordinates in meters)
    dst_points = np.array([
        [0.0, 0.0], # back-left
        [COURT_WIDTH_M, 0.0], # back-right
        [COURT_WIDTH_M, COURT_LENGTH_M], # front-right
        [0.0, COURT_LENGTH_M], # front-left
    ], dtype=np.float32)

    # Compute homography
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    return H


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
        return " **Error:**No video uploaded"

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
            return " **Error:**No players detected in video. Please ensure the video contains visible badminton players."

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
            'clear': '',
            'drive': '',
            'drop': '',
            'lob': '',
            'net': '',
            'smash': ''
        }

        emoji = shot_emoji.get(result['shot'].lower(), '')

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
- **Shot classes:**clear, drive, drop, lob, net, smash
- **Input:** 41 frames × 13 keypoints (26 features)
- **Framework:**TensorFlow 2.12 / Keras 2

---

 Classification complete!
"""

        # Cleanup
        shutil.rmtree(temp_dir)
        progress(1.0, desc="Complete!")

        return result_md

    except Exception as e:
        logger.error(f"Error during LSTM classification: {e}", exc_info=True)
        return f" **Error occurred during processing:**\n\n```\n{str(e)}\n```"


# ============================================================================
# BST CLASSIFIER FUNCTIONS
# ============================================================================

def classify_shot_bst(video_file, homography_matrix, progress=gr.Progress()) -> str:
    """
    Process uploaded video with BST Transformer (6 shot types with shuttlecock tracking).

    Args:
        video_file: Uploaded video file from Gradio
        homography_matrix: 3x3 homography matrix (None for demo videos)

    Returns:
        Formatted markdown result string
    """
    if video_file is None:
        return " **Error:**No video uploaded"

    if homography_matrix is None:
        return " **Error:**Please calibrate court corners first by clicking 4 points on the court image above"

    try:
        progress(0.1, desc="Creating temporary directory...")
        temp_dir = Path(tempfile.mkdtemp(prefix="bst_classifier_"))
        npy_dir = temp_dir / "npy"
        npy_dir.mkdir(exist_ok=True)

        # Step 1: Run preprocessing (TrackNet + MMPose + collation)
        progress(0.3, desc="Running TrackNet and MMPose preprocessing...")

        process_script = Path("models/preprocessing/process_single_video.py")

        # Serialize homography matrix as JSON
        homography_json = json.dumps(homography_matrix.tolist())

        cmd = [
            sys.executable,
            str(process_script),
            str(video_file),
            "-o", str(npy_dir),
            "--seq-len", "100",
            "--keep-intermediates",
            "--homography", homography_json
        ]

        logger.info(f"Running preprocessing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            error_msg = f" **Preprocessing failed:**\n\n```\n{result.stderr}\n```"
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
            return f" **Error:**Expected intermediate file not found: {joints_src}"

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
            error_msg = f" **BST inference failed:**\n\n```\n{bst_result.stderr}\n```"
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
            return f" **Could not parse BST output:**\n\n```\n{bst_result.stdout}\n```"

        # Format top-3 results
        top3_display = "\n".join(top3_results) if top3_results else "N/A"

        # Create result with emoji
        shot_emoji = {
            'clear': '',
            'drive': '',
            'drop': '',
            'lob': '',
            'net': '',
            'smash': ''
        }
        emoji = shot_emoji.get(predicted_class.lower(), '')

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
 Video processed successfully
 TrackNet shuttlecock detection
 MMPose player pose estimation
 BST Transformer inference

### Model Information:
- **Model:**BST Transformer (8 layers, Joint & Bone features)
- **Shot classes:** 6 types (clear, drive, drop, lob, net, smash)
- **Features:**Player joints, shuttlecock trajectory, bone vectors
- **Device:**CUDA (GPU)

Temporary files: `{temp_dir}`

---

 Classification complete!
"""

        progress(1.0, desc="Complete!")
        return result_md

    except subprocess.TimeoutExpired:
        return " **Error:**Processing timed out (>10 minutes)"
    except Exception as e:
        import traceback
        error_msg = f" **Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        logger.error(error_msg)
        return error_msg


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="Badminton Stroke Classifiers", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Badminton Stroke Classification System

    Compare **two different classification models**side-by-side. Both classify the same 6 shot types:

    **Clear** • **Drive** • **Drop** • **Lob** • **Net** • **Smash**
    """)

    # Single shared video input at the top
    with gr.Row():
        video_input = gr.Video(
            label="Upload Badminton Video",
            format="mp4",
            scale=2
        )

    # Demo video examples
    gr.Markdown("### Or try these demo videos from the validation set:")
    demo_videos_path = Path("demo_videos")
    if demo_videos_path.exists():
        demo_videos = [
            ("01_smash_32_1_16_16.mp4", " SMASH"),
            ("02_net_32_1_10_2.mp4", " NET"),
            ("03_lob_32_1_10_10.mp4", " LOB"),
            ("04_clear_32_1_10_4.mp4", " CLEAR"),
            ("05_drop_32_1_10_18.mp4", " DROP"),
            ("06_drive_32_1_13_8.mp4", " DRIVE"),
        ]

        with gr.Row():
            for video_file, label in demo_videos:
                video_path = str(demo_videos_path / video_file)
                if Path(video_path).exists():
                    gr.Button(label).click(
                        lambda p=video_path: p,
                        outputs=video_input
                    )

    # Court calibration section for BST
    gr.Markdown("### BST Court Calibration (Required for uploaded videos)")
    gr.Markdown("Click 4 court corners in order: **1) Back-Left** → **2) Back-Right** → **3) Front-Right** → **4) Front-Left**")

    with gr.Row():
        with gr.Column(scale=1):
            calibration_image = gr.Image(
                label="Click 4 Court Corners",
                type="numpy",
                interactive=False
            )
            extract_frame_btn = gr.Button(" Extract Frame for Calibration", variant="primary")

        with gr.Column(scale=1):
            calibration_status = gr.Markdown("**Status:**Upload a video and click 'Extract Frame' to begin calibration")
            reset_calibration_btn = gr.Button(" Reset Calibration", size="sm")

    # Hidden state to store corner points and homography
    corner_points_state = gr.State([])
    homography_state = gr.State(None)
    video_dims_state = gr.State((1280, 720)) # width, height
    original_frame_state = gr.State(None) # Store original frame

    # Calibration callback functions
    def on_extract_frame(video_file):
        """Extract first frame from video for calibration."""
        if video_file is None:
            return None, "**Status:** No video uploaded", [], None, (1280, 720), None

        frame = extract_first_frame(video_file)
        if frame is None:
            return None, "**Status:** Failed to extract frame from video", [], None, (1280, 720), None

        h, w = frame.shape[:2]
        return frame, f"**Status:** Frame extracted ({w}×{h}). Click 4 court corners in order.", [], None, (w, h), frame

    def on_corner_click(evt: gr.SelectData, corners, video_dims, original_frame):
        """Handle click on calibration image to add corner point."""
        if original_frame is None:
            return None, corners, "**Status:** No frame loaded. Please extract frame first.", None

        x, y = evt.index
        corners = corners + [(x, y)]

        # Debug: Print clicked corner
        print(f" Corner {len(corners)} clicked: ({x}, {y})")

        # Draw corners on original frame
        annotated = draw_court_corners(original_frame, corners)

        if len(corners) < 4:
            status = f"**Status:** {len(corners)}/4 corners marked. Click corner #{len(corners)+1}"
            return annotated, corners, status, None
        elif len(corners) == 4:
            # Compute homography
            print(f" Computing homography from corners: {corners}")
            print(f" Video dimensions: {video_dims[0]}×{video_dims[1]}")
            H = compute_homography_from_corners(corners, video_dims[0], video_dims[1])
            print(f" Homography matrix:\n{H}")
            status = "**Status:** Calibration complete! You can now run BST classification."
            return annotated, corners, status, H
        else:
            # Too many clicks - reset
            status = "**Status:** Too many clicks. Please reset and try again."
            return annotated, corners, status, None

    def on_reset_calibration(original_frame):
        """Reset calibration state."""
        if original_frame is None:
            return None, [], "**Status:** No frame loaded", None
        return original_frame, [], "**Status:** Calibration reset. Click 4 corners again.", None

    # Wire up calibration events
    extract_frame_btn.click(
        fn=on_extract_frame,
        inputs=[video_input],
        outputs=[calibration_image, calibration_status, corner_points_state, homography_state, video_dims_state, original_frame_state]
    )

    calibration_image.select(
        fn=on_corner_click,
        inputs=[corner_points_state, video_dims_state, original_frame_state],
        outputs=[calibration_image, corner_points_state, calibration_status, homography_state]
    )

    reset_calibration_btn.click(
        fn=on_reset_calibration,
        inputs=[original_frame_state],
        outputs=[calibration_image, corner_points_state, calibration_status, homography_state]
    )

    gr.Markdown("---")

    # Demo videos section
    gr.Markdown("### Try Demo Videos (Bottom Player)")
    gr.Markdown("**Note:**Demo videos are pre-calibrated and don't require court corner selection.")

    demo_base_path = "demo_videos"
    demo_videos = {
        " Clear": f"{demo_base_path}/34_1_1_7.mp4",
        " Drive": f"{demo_base_path}/36_1_20_16.mp4",
        " Drop": f"{demo_base_path}/34_1_26_10.mp4",
        " Lob": f"{demo_base_path}/32_2_13_7.mp4",
        " Net": f"{demo_base_path}/33_2_19_4.mp4",
        " Smash": f"{demo_base_path}/35_2_33_11.mp4",
    }

    def load_demo_video(demo_path):
        """Load demo video and set dummy homography (demo videos are pre-calibrated)."""
        # Demo videos don't need calibration - use identity homography as marker
        demo_homography = np.eye(3)
        return demo_path, demo_homography, "**Status:** Demo video loaded (pre-calibrated)"

    with gr.Row():
        for label, path in demo_videos.items():
            gr.Button(label, size="sm").click(
                lambda p=path: load_demo_video(p),
                outputs=[video_input, homography_state, calibration_status]
            )

    gr.Markdown("---")

    # Side-by-side classifiers
    with gr.Row(equal_height=True):
        # ====================================================================
        # LEFT COLUMN: LSTM CLASSIFIER
        # ====================================================================
        with gr.Column(scale=1):
            gr.Markdown("""
            ## LSTM Classifier

            **Fast pose-only classification**
            - Processing: ~5-10 seconds
            - Input: Player pose only (13 keypoints)
            - Architecture: LSTM Neural Network
            - Trained on 15 matches
            """)

            lstm_classify_btn = gr.Button(
                "Classify with LSTM ",
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
            ## BST Transformer

            **Advanced pose + shuttlecock classification**
            - ⏱ Processing: ~30-60 seconds
            - Input: Pose + shuttlecock trajectory
            - Architecture: Transformer (8 layers)
            - Higher accuracy with context
            """)

            bst_classify_btn = gr.Button(
                "Classify with BST ",
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
        inputs=[video_input, homography_state],
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
    - **Architecture:**LSTM Neural Network
    - **Training:** 15 professional badminton matches
    - **Input:** 41 frames × 13 keypoints (26 features)
    - **Pose extraction:**YOLO11x-pose
    - **Framework:**TensorFlow 2.12 / Keras 2

    ### BST Transformer
    - **Architecture:**Transformer with 8 layers
    - **Features:**Joint positions, bone vectors, shuttlecock trajectory
    - **Shuttlecock detection:**TrackNetV3
    - **Pose extraction:**MMPose (RTMPose)
    - **Framework:**PyTorch

    ---

    **Note:**For best results, ensure videos show clear view of the player performing the shot.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True
    )
