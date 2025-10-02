#!/usr/bin/env python3
# ABOUTME: Interactive GUI for debugging pose detection with court boundary selection
# ABOUTME: Click 4 corners, see pose detection results overlaid on video

import gradio as gr
import cv2
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
from mmpose.apis import MMPoseInferencer
import sys

# Import court checking functions
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from prepare_train import check_pos_in_court

# Constants
COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4

# COCO skeleton
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def extract_first_frame(video_path: str) -> np.ndarray:
    """Extract first frame from video."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def compute_homography(corners, video_width, video_height):
    """Compute homography from 4 clicked corners."""
    # Normalize to 1280x720
    src_points = np.array([
        [corners[0][0] * 1280 / video_width, corners[0][1] * 720 / video_height],
        [corners[1][0] * 1280 / video_width, corners[1][1] * 720 / video_height],
        [corners[2][0] * 1280 / video_width, corners[2][1] * 720 / video_height],
        [corners[3][0] * 1280 / video_width, corners[3][1] * 720 / video_height],
    ], dtype=np.float32)

    dst_points = np.array([
        [0.0, 0.0],
        [COURT_WIDTH_M, 0.0],
        [COURT_WIDTH_M, COURT_LENGTH_M],
        [0.0, COURT_LENGTH_M],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_points, dst_points)
    return H

def draw_skeleton(frame, keypoints, color, thickness=2):
    """Draw skeleton for one person."""
    for (start_idx, end_idx) in SKELETON:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            pt1 = tuple(keypoints[start_idx].astype(int))
            pt2 = tuple(keypoints[end_idx].astype(int))
            if (0 <= pt1[0] < frame.shape[1] and 0 <= pt1[1] < frame.shape[0] and
                0 <= pt2[0] < frame.shape[1] and 0 <= pt2[1] < frame.shape[0]):
                cv2.line(frame, pt1, pt2, color, thickness)

    for pt in keypoints:
        pt_int = tuple(pt.astype(int))
        if 0 <= pt_int[0] < frame.shape[1] and 0 <= pt_int[1] < frame.shape[0]:
            cv2.circle(frame, pt_int, 4, color, -1)

def draw_corners(frame, corners):
    """Draw clicked corners on frame."""
    annotated = frame.copy()
    labels = ['BL (Back-Left)', 'BR (Back-Right)', 'FR (Front-Right)', 'FL (Front-Left)']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i, (corner, label, color) in enumerate(zip(corners, labels, colors)):
        cv2.circle(annotated, corner, 10, color, -1)
        cv2.putText(annotated, f"{i+1}. {label}", (corner[0] + 15, corner[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if len(corners) == 4:
        pts = np.array(corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], True, (0, 255, 255), 3)

    return annotated

def process_video_with_pose_debug(video_path, corners, video_dims):
    """Process video and overlay pose detections."""
    if not video_path:
        return None, "❌ No video loaded"

    if len(corners) != 4:
        return None, f"❌ Need 4 corners, currently have {len(corners)}"

    try:
        # Compute homography
        width, height = video_dims
        H = compute_homography(corners, width, height)

        # Setup court info
        vid = 0
        all_court_info = {
            vid: {
                'H': H,
                'border_L': 0.0,
                'border_R': COURT_WIDTH_M,
                'border_U': 0.0,
                'border_D': COURT_LENGTH_M,
            }
        }
        res_df = pd.DataFrame({'width': [width], 'height': [height]}, index=[vid])
        res_df.index.name = 'id'

        # Initialize MMPose
        inferencer = MMPoseInferencer('human')

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create output video
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_with_2_players = 0

        for result in inferencer(video_path, show=False):
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections
            keypoints_list = [person['keypoints'] for person in result['predictions'][0]]
            keypoints = np.array(keypoints_list) if keypoints_list else np.array([])

            num_detected = len(keypoints)

            # Draw court boundary
            pts = np.array(corners, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            # Check court filtering
            if num_detected > 0:
                # Get bounding boxes
                bboxes = np.array([person['bbox'][0] for person in result['predictions'][0]])

                # Use bbox-based selection (2 largest people)
                in_court, pos_normalized = check_pos_in_court(keypoints, 0, all_court_info, res_df, bboxes)
                in_court_indices = np.nonzero(in_court)[0]
                num_in_court = len(in_court_indices)

                if num_in_court == 2:
                    frames_with_2_players += 1

                # ONLY draw the 2 selected players (no gray/red skeletons)
                if num_in_court >= 1:
                    draw_skeleton(frame, keypoints[in_court_indices[0]], (0, 255, 0), thickness=3)  # Green - Player 1
                if num_in_court >= 2:
                    draw_skeleton(frame, keypoints[in_court_indices[1]], (255, 0, 0), thickness=3)  # Blue - Player 2

                # Stats overlay
                cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Detected: {num_detected}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                color = (0, 255, 0) if num_in_court == 2 else (0, 0, 255)
                cv2.putText(frame, f"In Court: {num_in_court}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if num_in_court != 2:
                    cv2.putText(frame, "WILL BE ZEROED!", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Frame {frame_idx}: NO DETECTIONS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        success_rate = 100 * frames_with_2_players / frame_idx if frame_idx > 0 else 0
        status = f"✅ Processed {frame_idx} frames\n"
        status += f"Frames with exactly 2 players: {frames_with_2_players}/{frame_idx} ({success_rate:.1f}%)\n"
        status += f"Output: {output_path}"

        return str(output_path), status

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Global state
corner_points = []
video_dimensions = (1280, 720)
original_frame = None
current_video_path = None

def on_extract_frame(video):
    """Extract first frame from video."""
    global original_frame, current_video_path, corner_points, video_dimensions

    if video is None:
        return None, [], "❌ No video uploaded"

    current_video_path = video
    corner_points = []

    # Get video dimensions
    cap = cv2.VideoCapture(video)
    video_dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    original_frame = extract_first_frame(video)
    if original_frame is None:
        return None, [], "❌ Failed to extract frame"

    return original_frame, [], f"✅ Frame extracted ({video_dimensions[0]}×{video_dimensions[1]}). Click 4 corners: back-left, back-right, front-right, front-left"

def on_corner_click(evt: gr.SelectData):
    """Handle click on calibration image."""
    global corner_points, original_frame

    if original_frame is None:
        return None, corner_points, "❌ No frame loaded"

    x, y = evt.index
    corner_points.append((x, y))

    print(f"🎯 Corner {len(corner_points)}: ({x}, {y})")

    annotated = draw_corners(original_frame, corner_points)

    if len(corner_points) == 4:
        status = "✅ All 4 corners selected! Click 'Process Video' to see pose detection."
    else:
        labels = ['back-left', 'back-right', 'front-right', 'front-left']
        status = f"📍 Click corner {len(corner_points)+1}/4: {labels[len(corner_points)]}"

    return annotated, corner_points, status

def on_reset_corners():
    """Reset corner selection."""
    global corner_points, original_frame
    corner_points = []

    if original_frame is None:
        return None, [], "❌ No frame loaded"

    return original_frame, [], "🔄 Corners reset. Click 4 corners again."

def on_process_video():
    """Process video with current corners."""
    global corner_points, current_video_path, video_dimensions

    return process_video_with_pose_debug(current_video_path, corner_points, video_dimensions)

# Build Gradio interface
with gr.Blocks(title="Pose Detection Debugger") as app:
    gr.Markdown("# 🏸 Pose Detection Debugger")
    gr.Markdown("Upload a badminton video, click the 4 court corners, and see pose detection results!")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")
            extract_btn = gr.Button("1. Extract Frame", variant="primary")

            gr.Markdown("### Click 4 corners in order:")
            gr.Markdown("1. **Back-Left** (top-left of court)\n2. **Back-Right** (top-right)\n3. **Front-Right** (bottom-right)\n4. **Front-Left** (bottom-left)")

            reset_btn = gr.Button("Reset Corners", variant="secondary")
            process_btn = gr.Button("2. Process Video", variant="primary")

            status_text = gr.Textbox(label="Status", lines=3)

        with gr.Column(scale=2):
            calibration_image = gr.Image(label="Click 4 Court Corners", interactive=True)

            # Hidden state
            corners_state = gr.State([])

    with gr.Row():
        output_video = gr.Video(label="Pose Detection Debug Output")

    # Event handlers
    extract_btn.click(
        on_extract_frame,
        inputs=[video_input],
        outputs=[calibration_image, corners_state, status_text]
    )

    calibration_image.select(
        on_corner_click,
        inputs=[],
        outputs=[calibration_image, corners_state, status_text]
    )

    reset_btn.click(
        on_reset_corners,
        inputs=[],
        outputs=[calibration_image, corners_state, status_text]
    )

    process_btn.click(
        on_process_video,
        inputs=[],
        outputs=[output_video, status_text]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7861)
