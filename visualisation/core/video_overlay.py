#!/usr/bin/env python3
"""
ABOUTME: Video overlay module for combining pose estimation, shuttlecock tracking, and court detection visualizations
ABOUTME: Provides functions to draw multiple analysis overlays on badminton video frames
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple

# COCO keypoint connections for skeleton drawing (MMPose format)
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # Body
    (5, 11), (6, 12), (5, 6),  # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # Head
]

# Colors for visualization (BGR format)
PLAYER_COLORS = [
    (0, 255, 0),    # Player 1: Green
    (255, 0, 255),  # Player 2: Magenta
]
SHUTTLECOCK_COLOR = (0, 0, 255)  # Red
COURT_COLOR = (255, 255, 0)  # Cyan
SKELETON_THICKNESS = 2
KEYPOINT_RADIUS = 4
SHUTTLECOCK_RADIUS = 5
COURT_LINE_THICKNESS = 2


def draw_pose_skeleton(
    frame: np.ndarray,
    poses: np.ndarray,
    frame_idx: int,
    confidence_threshold: float = 0.3
) -> np.ndarray:
    """
    Draw pose estimation skeleton on frame for both players.

    Args:
        frame: Video frame (H, W, 3) BGR image
        poses: Pose data array of shape (T, 2, 17, 2) from MMPose
               T=frames, 2=players, 17=keypoints, 2=(x,y)
        frame_idx: Current frame index
        confidence_threshold: Minimum confidence to draw keypoint (not used with current data)

    Returns:
        Frame with pose skeleton overlay
    """
    if poses is None or frame_idx >= len(poses):
        return frame

    frame_copy = frame.copy()
    current_poses = poses[frame_idx]  # Shape: (2, 17, 2)

    for player_idx in range(min(2, current_poses.shape[0])):
        keypoints = current_poses[player_idx]  # Shape: (17, 2)
        color = PLAYER_COLORS[player_idx]

        # Draw skeleton connections
        for bone in COCO_SKELETON:
            pt1_idx, pt2_idx = bone
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]

                # Only draw if both points are valid (non-zero)
                if not (np.allclose(pt1, 0) or np.allclose(pt2, 0)):
                    pt1_int = (int(pt1[0]), int(pt1[1]))
                    pt2_int = (int(pt2[0]), int(pt2[1]))
                    cv2.line(frame_copy, pt1_int, pt2_int, color, SKELETON_THICKNESS)

        # Draw keypoints on top of skeleton
        for kp_idx, kp in enumerate(keypoints):
            if not np.allclose(kp, 0):
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(frame_copy, pt, KEYPOINT_RADIUS, color, -1)

    return frame_copy


def draw_shuttlecock_position(
    frame: np.ndarray,
    shuttlecock_df: pd.DataFrame,
    frame_idx: int,
    show_trajectory: bool = False,
    trajectory_length: int = 10
) -> np.ndarray:
    """
    Draw shuttlecock position on frame.

    Args:
        frame: Video frame (H, W, 3) BGR image
        shuttlecock_df: DataFrame with columns ['Frame', 'Visibility', 'X', 'Y']
        frame_idx: Current frame index
        show_trajectory: Whether to show trajectory trail
        trajectory_length: Number of past frames to show in trajectory

    Returns:
        Frame with shuttlecock marker overlay
    """
    if shuttlecock_df is None or shuttlecock_df.empty:
        return frame

    frame_copy = frame.copy()

    # Get current frame shuttlecock position
    current_row = shuttlecock_df[shuttlecock_df['Frame'] == frame_idx]

    if not current_row.empty and current_row.iloc[0]['Visibility'] == 1:
        x = int(current_row.iloc[0]['X'])
        y = int(current_row.iloc[0]['Y'])

        # Draw trajectory if requested
        if show_trajectory and frame_idx > 0:
            start_frame = max(0, frame_idx - trajectory_length)
            trajectory_data = shuttlecock_df[
                (shuttlecock_df['Frame'] >= start_frame) &
                (shuttlecock_df['Frame'] < frame_idx) &
                (shuttlecock_df['Visibility'] == 1)
            ]

            if len(trajectory_data) > 1:
                points = trajectory_data[['X', 'Y']].values.astype(int)
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i + 1])
                    # Fade trajectory from dark to bright
                    alpha = (i + 1) / len(points)
                    color = tuple(int(c * alpha) for c in SHUTTLECOCK_COLOR)
                    cv2.line(frame_copy, pt1, pt2, color, 1)

        # Draw current shuttlecock position
        cv2.circle(frame_copy, (x, y), SHUTTLECOCK_RADIUS, SHUTTLECOCK_COLOR, -1)
        # Add outer ring for better visibility
        cv2.circle(frame_copy, (x, y), SHUTTLECOCK_RADIUS + 2, (255, 255, 255), 1)

    return frame_copy


def draw_court_boundaries(
    frame: np.ndarray,
    corner_points: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """
    Draw court boundary lines on frame.

    Args:
        frame: Video frame (H, W, 3) BGR image
        corner_points: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                      If None, no court boundaries are drawn

    Returns:
        Frame with court boundary overlay
    """
    if corner_points is None or len(corner_points) != 4:
        return frame

    frame_copy = frame.copy()
    points = np.array(corner_points, dtype=np.int32)

    # Draw court boundary as closed polygon
    cv2.polylines(frame_copy, [points], isClosed=True,
                  color=COURT_COLOR, thickness=COURT_LINE_THICKNESS)

    # Draw corner markers
    for point in points:
        cv2.circle(frame_copy, tuple(point), 6, COURT_COLOR, -1)
        cv2.circle(frame_copy, tuple(point), 8, (255, 255, 255), 2)

    return frame_copy


def create_combined_visualization(
    input_video_path: Path,
    output_video_path: Path,
    poses_npy_path: Path,
    shuttlecock_csv_path: Path,
    court_corner_points: Optional[List[Tuple[int, int]]] = None,
    show_trajectory: bool = True,
    progress_callback: Optional[callable] = None
) -> bool:
    """
    Create combined visualization video with pose, shuttlecock, and court overlays.

    Args:
        input_video_path: Path to input video file
        output_video_path: Path to save output video
        poses_npy_path: Path to poses.npy file (T, 2, 17, 2)
        shuttlecock_csv_path: Path to shuttlecock CSV file
        court_corner_points: Optional list of 4 court corner points
        show_trajectory: Whether to show shuttlecock trajectory
        progress_callback: Optional callback function(message: str, progress: float)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load data
        poses = np.load(poses_npy_path) if poses_npy_path.exists() else None
        shuttlecock_df = pd.read_csv(shuttlecock_csv_path) if shuttlecock_csv_path.exists() else None

        # Open video
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_video_path}")
            return False

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        if progress_callback:
            progress_callback("Creating combined visualization video...", 0.0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply overlays in order: Court -> Pose -> Shuttlecock
            # This ensures shuttlecock is always visible on top

            # 1. Draw court boundaries
            frame = draw_court_boundaries(frame, court_corner_points)

            # 2. Draw pose skeletons
            if poses is not None:
                frame = draw_pose_skeleton(frame, poses, frame_idx)

            # 3. Draw shuttlecock (on top)
            if shuttlecock_df is not None:
                frame = draw_shuttlecock_position(
                    frame, shuttlecock_df, frame_idx,
                    show_trajectory=show_trajectory
                )

            # Write frame
            out.write(frame)

            frame_idx += 1

            # Progress update
            if progress_callback and frame_idx % 10 == 0:
                progress = frame_idx / total_frames
                progress_callback(f"Processing frame {frame_idx}/{total_frames}", progress)

        # Cleanup
        cap.release()
        out.release()

        if progress_callback:
            progress_callback("Combined visualization complete!", 1.0)

        return True

    except Exception as e:
        print(f"Error creating combined visualization: {e}")
        return False