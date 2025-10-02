#!/usr/bin/env python3
# ABOUTME: Debug pose detection with MMPose - visualize raw detections and court filtering
# ABOUTME: Shows what MMPose detects vs what survives court boundary filtering

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from mmpose.apis import MMPoseInferencer

# Import court checking functions
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from prepare_train import check_pos_in_court, to_court_coordinate, normalize_position

# COCO skeleton
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def draw_skeleton(frame, keypoints, color, label="", thickness=2):
    """Draw skeleton for one person."""
    # Draw bones
    for (start_idx, end_idx) in SKELETON:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            pt1 = tuple(keypoints[start_idx].astype(int))
            pt2 = tuple(keypoints[end_idx].astype(int))

            if (0 <= pt1[0] < frame.shape[1] and 0 <= pt1[1] < frame.shape[0] and
                0 <= pt2[0] < frame.shape[1] and 0 <= pt2[1] < frame.shape[0]):
                cv2.line(frame, pt1, pt2, color, thickness)

    # Draw keypoints
    for pt in keypoints:
        pt_int = tuple(pt.astype(int))
        if 0 <= pt_int[0] < frame.shape[1] and 0 <= pt_int[1] < frame.shape[0]:
            cv2.circle(frame, pt_int, 4, color, -1)

    # Draw label if provided
    if label and len(keypoints) > 0:
        pt = keypoints[0].astype(int)
        if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
            cv2.putText(frame, label, tuple(pt), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)

    return frame

def draw_court_boundary(frame, corners, color=(0, 255, 255)):
    """Draw court boundary from clicked corners."""
    if len(corners) == 4:
        pts = np.array(corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, 3)

        # Label corners
        labels = ['BL', 'BR', 'FR', 'FL']  # back-left, back-right, front-right, front-left
        for i, (corner, label) in enumerate(zip(corners, labels)):
            cv2.circle(frame, tuple(corner), 8, color, -1)
            cv2.putText(frame, label, (corner[0] + 10, corner[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def debug_pose_detection(video_path, output_path, homography_matrix=None, max_frames=None):
    """
    Debug pose detection by showing:
    1. All detected people (gray)
    2. People passing court check (green/blue)
    3. People failing court check (red)
    4. Court boundaries
    """
    video_path = Path(video_path)

    # Initialize MMPose
    print("Initializing MMPose...")
    inferencer = MMPoseInferencer('human')

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps}fps")

    # Setup court info if homography provided
    if homography_matrix is not None:
        vid = 0
        all_court_info = {
            vid: {
                'H': homography_matrix,
                'border_L': 0.0,
                'border_R': 6.1,
                'border_U': 0.0,
                'border_D': 13.4,
            }
        }
        res_df = pd.DataFrame({'width': [width], 'height': [height]}, index=[vid])
        res_df.index.name = 'id'
        print(f"Using homography for court filtering")
        print(f"Court boundaries: L=0.0, R=6.1, U=0.0, D=13.4")

        # Compute court corners in pixel space
        # Map court corners back to camera space
        court_corners_meters = np.array([
            [0.0, 0.0],              # back-left
            [6.1, 0.0],              # back-right
            [6.1, 13.4],             # front-right
            [0.0, 13.4],             # front-left
        ])
        # Transform to camera coordinates (inverse homography)
        H_inv = np.linalg.inv(homography_matrix)
        court_corners_camera = []
        for pt_m in court_corners_meters:
            pt_h = np.array([pt_m[0], pt_m[1], 1.0])
            pt_cam_h = H_inv @ pt_h
            pt_cam = pt_cam_h[:2] / pt_cam_h[2]
            # Scale from normalized 1280x720 to actual resolution
            pt_cam_scaled = (pt_cam * [width/1280, height/720]).astype(int)
            court_corners_camera.append(tuple(pt_cam_scaled))
        print(f"Court corners (pixels): {court_corners_camera}")
    else:
        all_court_info = None
        res_df = None
        court_corners_camera = []
        print("No homography - showing all detections")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    total_detected = 0
    total_in_court = 0

    for result in inferencer(str(video_path), show=False):
        ret, frame = cap.read()
        if not ret:
            break

        # Get all detected keypoints
        keypoints_list = [person['keypoints'] for person in result['predictions'][0]]
        keypoints = np.array(keypoints_list) if keypoints_list else np.array([])

        num_detected = len(keypoints)
        total_detected += num_detected

        # Draw court boundary
        if court_corners_camera:
            frame = draw_court_boundary(frame, court_corners_camera)

        # Process detections
        if num_detected > 0:
            # Draw all detected people in gray first
            for i, kp in enumerate(keypoints):
                frame = draw_skeleton(frame, kp, (128, 128, 128), f"Person {i}")

            # Check court filtering if homography available
            if all_court_info is not None:
                in_court, pos_normalized = check_pos_in_court(keypoints, 0, all_court_info, res_df)
                in_court_indices = np.nonzero(in_court)[0]
                num_in_court = len(in_court_indices)
                total_in_court += num_in_court

                # Highlight in-court players
                for i in in_court_indices:
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for player 1, Blue for player 2
                    frame = draw_skeleton(frame, keypoints[i], color, f"In Court {i}", thickness=3)

                # Show out-of-court players
                out_court_indices = np.nonzero(~in_court)[0]
                for i in out_court_indices:
                    frame = draw_skeleton(frame, keypoints[i], (0, 0, 255), f"Out {i}", thickness=2)

                # Display stats
                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Detected: {num_detected}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"In Court: {num_in_court}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if num_in_court == 2 else (0, 0, 255), 2)

                if num_in_court != 2:
                    cv2.putText(frame, "FRAME WILL BE ZEROED!", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Frame: {frame_idx} - NO DETECTIONS", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}: {num_detected} detected, " +
                  (f"{num_in_court} in court" if all_court_info else ""))

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    out.release()

    print(f"\n{'='*60}")
    print(f"Processed {frame_idx} frames")
    print(f"Total detections: {total_detected}")
    if all_court_info:
        print(f"Total in-court: {total_in_court}")
        print(f"Average detections per frame: {total_detected/frame_idx:.2f}")
        print(f"Average in-court per frame: {total_in_court/frame_idx:.2f}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Debug pose detection with court filtering")
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output video")
    parser.add_argument("--homography", type=str, help="Homography matrix as JSON string")
    parser.add_argument("--max-frames", type=int, help="Max frames to process")

    args = parser.parse_args()

    homography_matrix = None
    if args.homography:
        import json
        H_data = json.loads(args.homography)
        homography_matrix = np.array(H_data, dtype=np.float32)
        if homography_matrix.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got {homography_matrix.shape}")

    debug_pose_detection(args.video, args.output, homography_matrix, args.max_frames)

if __name__ == "__main__":
    main()
