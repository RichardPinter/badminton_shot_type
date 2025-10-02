#!/usr/bin/env python3
# ABOUTME: Gradio web interface for LSTM badminton shot classifier
# ABOUTME: Provides video upload interface and displays predicted shot type

"""
Gradio App for LSTM Badminton Shot Classifier

This provides a web interface where users can:
1. Upload a badminton video
2. Get predicted shot type (clear, drive, drop, lob, net, smash)
3. View confidence scores and processing details

Usage:
    python gradio_app.py

Then open the provided URL in your browser.
"""

import gradio as gr
import tempfile
import shutil
import os
import logging
from pathlib import Path
from typing import Tuple, Dict

# Import the classifier pipeline
from run_video_classifier import (
    extract_poses_from_video,
    convert_pose_data_to_csv,
    predict_shot_from_csv,
    SHOT_TYPES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def classify_shot(video_file) -> Tuple[str, str, str]:
    """
    Process uploaded video and classify the badminton shot.

    Args:
        video_file: Uploaded video file from Gradio

    Returns:
        Tuple of (shot_type, confidence, details)
    """
    if video_file is None:
        return " Error", "No video uploaded", ""

    try:
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix='lstm_shot_classifier_')
        logger.info(f"Processing video: {video_file}")
        logger.info(f"Temporary directory: {temp_dir}")

        # Step 1: Extract poses from video
        logger.info("Step 1: Extracting player poses...")
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
            return " Error", "No players detected in video", "Could not extract pose data from the video. Please ensure the video contains visible badminton players."

        # Step 2: Convert to CSV format
        logger.info("Step 2: Converting pose data to CSV...")
        shots_csv, wireframe_csv = convert_pose_data_to_csv(
            pose_data,
            temp_dir,
            set_id=1,
            video_path=video_file
        )

        # Step 3: Run LSTM prediction
        logger.info("Step 3: Running LSTM classification...")
        model_path = '../weights/15Matches_LSTM.h5'
        if not os.path.exists(model_path):
            model_path = './weights/15Matches_LSTM.h5'

        result = predict_shot_from_csv(temp_dir, model_path)

        # Format results
        shot_type = result['shot'].upper()
        confidence = result['confidence']

        # Create detailed output
        details = f"""
**Processing Summary:**
- Total frames processed: {total_frames}
- Players detected: {', '.join(frame_counts.keys())}
- Frame counts: {', '.join(f'{k}: {v}' for k, v in frame_counts.items())}

**Classification:**
- Predicted shot type: **{shot_type}**
- Confidence: {confidence:.1%}

**Model Information:**
- Model: 15Matches_LSTM
- Shot classes: clear, drive, drop, lob, net, smash
- Input: 41 frames × 13 keypoints (26 features)
"""

        # Cleanup
        shutil.rmtree(temp_dir)
        logger.info("Processing complete!")

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
        shot_display = f"{emoji} {shot_type}"
        confidence_display = f"{confidence:.1%}"

        return shot_display, confidence_display, details

    except Exception as e:
        logger.error(f"Error during classification: {e}", exc_info=True)
        return " Error", "0%", f"**Error occurred during processing:**\n\n{str(e)}"


# Create Gradio interface
with gr.Blocks(title="Badminton Shot Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Badminton Shot Classifier

    Upload a badminton video to classify the shot type using LSTM deep learning model.

    **Supported shot types:**
    - **Clear** - High defensive shot to the back
    - **Drive** - Fast flat shot
    - **Drop** - Gentle shot that falls close to net
    - **Lob** - High underhand shot
    - **Net** - Shot played at the net
    - **Smash** - Powerful downward shot
    """)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="Upload Badminton Video",
                format="mp4"
            )
            classify_btn = gr.Button("Classify Shot ", variant="primary", size="lg")

            gr.Markdown("""
            ### Instructions:
            1. Upload a video showing a badminton shot
            2. Click "Classify Shot" button
            3. Wait for processing (usually 5-10 seconds)
            4. View the predicted shot type and details

            **Note:**Best results with clear view of the player performing the shot.
            """)

        with gr.Column():
            shot_output = gr.Textbox(
                label="Predicted Shot Type",
                placeholder="Upload video and click classify...",
                lines=1,
                interactive=False
            )
            confidence_output = gr.Textbox(
                label="Confidence",
                placeholder="",
                lines=1,
                interactive=False
            )
            details_output = gr.Markdown(
                label="Details",
                value=""
            )

    # Set up the button click event
    classify_btn.click(
        fn=classify_shot,
        inputs=[video_input],
        outputs=[shot_output, confidence_output, details_output]
    )

    gr.Markdown("""
    ---
    ### About
    This classifier uses:
    - **YOLO11x-pose**for pose extraction
    - **LSTM neural network**for shot classification
    - Trained on 15 professional badminton matches

    Model trained with TensorFlow 2.12 / Keras 2
    """)


if __name__ == "__main__":
    # Auto-detect available port
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True
    )
