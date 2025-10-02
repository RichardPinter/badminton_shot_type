#!/usr/bin/env python3
# ABOUTME: Standalone LSTM shot classifier - takes video, runs YOLO pose, predicts shot type
# ABOUTME: End-to-end pipeline: Video → Pose Extraction → CSV → LSTM → Shot Prediction

"""
Standalone LSTM Shot Classifier for Badminton

This script provides a complete pipeline:
1. Takes a video file as input
2. Extracts player poses using YOLO11x-pose
3. Generates intermediate CSV files (setN_shots.csv, setN_wireframe.csv)
4. Runs LSTM shot classifier
5. Returns predicted shot type (one of: clear, drive, drop, lob, net, smash)

Usage:
    python run_video_classifier.py <video_path> [--corner-points x1,y1,x2,y2,x3,y3,x4,y4] [--model weights/15Matches_LSTM.keras]

Example:
    python run_video_classifier.py sample_video.mp4
    python run_video_classifier.py sample_video.mp4 --corner-points 170,142,465,142,550,358,89,356
"""

import sys
import os
import logging
import argparse
import tempfile
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import cv2

# Import pose extraction pipeline
from badmintonplayeranalysis_main import analyze_badminton_video_with_pose
import config as cfg

# Import LSTM classifier components
from shot_classifier import ShotClassifier
from match_loader import MatchLoader, ShotUtils, ShotType, SetShot, PlayerSet, Match

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the 6 shot types for classification
SHOT_TYPES = [
    ShotType(shotClassId=0, shotClass='clear', shotGroupId=0, shotGroup='clear'),
    ShotType(shotClassId=1, shotClass='drive', shotGroupId=1, shotGroup='drive'),
    ShotType(shotClassId=2, shotClass='drop', shotGroupId=2, shotGroup='drop'),
    ShotType(shotClassId=3, shotClass='lob', shotGroupId=3, shotGroup='lob'),
    ShotType(shotClassId=4, shotClass='net', shotGroupId=4, shotGroup='net'),
    ShotType(shotClassId=5, shotClass='smash', shotGroupId=5, shotGroup='smash'),
]


def extract_poses_from_video(
    video_path: str,
    corner_points: Optional[List[List[int]]] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None
) -> Dict[str, List[Dict]]:
    """
    Extract player poses from video using YOLO11x-pose.

    Args:
        video_path: Path to input video file
        corner_points: Court boundary points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        start_frame: Starting frame number (default: 0)
        end_frame: Ending frame number (default: None = end of video)

    Returns:
        Dictionary with 'Player_A' and 'Player_B' keys containing pose data per frame
    """
    logger.info(f"Extracting poses from video: {video_path}")
    logger.info(f"Frame range: {start_frame} to {end_frame or 'end'}")

    # Run the pose analysis pipeline
    # Use entire frame as court boundary to ensure all players are tracked
    # This avoids filtering out players based on court position
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Create court boundary that covers entire frame
    full_frame_court = [[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]]
    logger.info(f"Using full frame as court boundary: {frame_width}x{frame_height}")

    result = analyze_badminton_video_with_pose(
        video_path=video_path,
        court_points=full_frame_court,
        start_frame=start_frame,
        end_frame=end_frame,
        export_to_csv=False,
        return_player_data=True
    )

    if result is None:
        raise RuntimeError("Pose extraction failed - no data returned")

    logger.info(f"Pose extraction complete. Players detected: {list(result.keys())}")

    # Debug: Check if data is actually populated
    for player_key, frames in result.items():
        logger.info(f" {player_key}: {len(frames)} frames")
        if len(frames) > 0:
            logger.info(f" Sample frame keys: {list(frames[0].keys())}")

    return result


def convert_pose_data_to_csv(
    pose_data: Dict[str, List[Dict]],
    output_dir: str,
    set_id: int = 1,
    video_path: str = "video"
) -> Tuple[str, str]:
    """
    Convert pose data to CSV files in the format expected by MatchLoader.

    Creates two files:
    - set{set_id}_shots.csv: Shot metadata
    - set{set_id}_wireframe.csv: Per-frame pose keypoints

    Args:
        pose_data: Dict with Player_A/Player_B keys containing frame data
        output_dir: Directory to save CSV files
        set_id: Set number for filename (default: 1)
        video_path: Video path for metadata

    Returns:
        Tuple of (shots_csv_path, wireframe_csv_path)
    """
    logger.info(f"Converting pose data to CSV format in {output_dir}")

    # MatchLoader expects directory structure: basedir/matchname/setN_shots.csv
    # Create a subdirectory for the match
    match_dir = os.path.join(output_dir, 'match1')
    os.makedirs(match_dir, exist_ok=True)

    # Get player with most data (highest Y position = bottom player)
    player_key = None
    max_frames = 0
    for key, frames in pose_data.items():
        if len(frames) > max_frames:
            max_frames = len(frames)
            player_key = key

    if player_key is None or max_frames == 0:
        raise RuntimeError("No valid player data found in pose_data")

    logger.info(f"Using player: {player_key} with {max_frames} frames")
    player_data = pose_data[player_key]

    # Create shots CSV (one shot for the entire video)
    # Use 'clear' as default since we don't know the actual shot type yet
    shots_data = [{
        'player': 'A',
        'shot': 0,
        'time': '00:00:00',
        'start_frame': 0,
        'total_frames': max_frames,
        'shot_class': 'clear',
        'shot_class_grouped': 'clear',
        'WristSpeed': 0.0,
        'ElbowSpeed': 0.0
    }]

    shots_df = pd.DataFrame(shots_data)
    shots_path = os.path.join(match_dir, f'set{set_id}_shots.csv')
    shots_df.to_csv(shots_path, index=False)
    logger.info(f"Created shots CSV: {shots_path}")

    # Create wireframe CSV (pose keypoints per frame)
    # Model expects 13 keypoints (26 features): exclude eye and ear keypoints
    MODEL_KEYPOINTS = [
        'nose', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]

    wireframe_rows = []
    for frame_idx, frame_data in enumerate(player_data):
        if 'keypoints_xyc' not in frame_data:
            logger.warning(f"Frame {frame_idx} missing keypoints, skipping")
            continue

        row = {
            'Player': 'A',
            'shot': 0,
            'index': frame_idx
        }

        # Extract only the 13 keypoints the model was trained on
        keypoints = frame_data['keypoints_xyc']
        for keypoint_name in MODEL_KEYPOINTS:
            if keypoint_name in keypoints:
                row[f'{keypoint_name}_X'] = keypoints[keypoint_name][0]
                row[f'{keypoint_name}_Y'] = keypoints[keypoint_name][1]
            else:
                row[f'{keypoint_name}_X'] = 0.0
                row[f'{keypoint_name}_Y'] = 0.0

        wireframe_rows.append(row)

    wireframe_df = pd.DataFrame(wireframe_rows)
    wireframe_path = os.path.join(match_dir, f'set{set_id}_wireframe.csv')
    wireframe_df.to_csv(wireframe_path, index=False)
    logger.info(f"Created wireframe CSV: {wireframe_path} with {len(wireframe_rows)} frames")

    return shots_path, wireframe_path


def predict_shot_from_csv(
    csv_dir: str,
    model_path: str = '../weights/15Matches_LSTM.keras'
) -> Dict:
    """
    Load CSV data and run LSTM shot classifier.

    Args:
        csv_dir: Directory containing set*_shots.csv and set*_wireframe.csv files
        model_path: Path to trained LSTM model (.keras file)

    Returns:
        Dictionary with prediction results:
        {
            'shot': 'smash',
            'confidence': 0.92,
            'all_predictions': [...]
        }
    """
    logger.info(f"Loading match data from: {csv_dir}")

    # Create MatchLoader and load data
    loader = MatchLoader(config_path='config.json', logger=logger)
    matches = loader.loadMatches(csv_dir)

    if not matches or len(matches) == 0:
        raise RuntimeError(f"No matches loaded from {csv_dir}")

    logger.info(f"Loaded {len(matches)} match(es)")

    # Get shots from first match
    match = matches[0]
    all_shots: List[SetShot] = []

    for player_sets in match.sets.values():
        for player_set in player_sets:
            all_shots.extend(player_set.shots)

    if not all_shots:
        raise RuntimeError("No shots found in loaded match data")

    logger.info(f"Found {len(all_shots)} shot(s) to classify")

    # Initialize classifier and predict
    classifier = ShotClassifier(
        logger=logger,
        shot_types=SHOT_TYPES,
        model_persist_path=os.path.dirname(model_path)
    )

    logger.info(f"Running LSTM prediction with model: {model_path}")
    predictions = classifier.predict(model_path, all_shots)

    # For now, return the first prediction
    # TODO: Handle multiple shots if needed
    result = {
        'shot': predictions[0] if predictions else 'unknown',
        'confidence': 0.0, # TODO: Extract confidence from model
        'all_predictions': predictions
    }

    logger.info(f"Prediction complete: {result['shot']}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='LSTM Shot Classifier - Classify badminton shots from video'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--corner-points',
        type=str,
        default=None,
        help='Court corner points as: x1,y1,x2,y2,x3,y3,x4,y4 (optional, will prompt if not provided)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to LSTM model file (default: auto-detect in ../weights/ or ./weights/)'
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Starting frame number (default: 0)'
    )
    parser.add_argument(
        '--end-frame',
        type=int,
        default=None,
        help='Ending frame number (default: process entire video)'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary CSV files for debugging'
    )

    args = parser.parse_args()

    # Validate video exists
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)

    # Auto-detect model path if not provided
    if args.model is None:
        # Try multiple possible locations (prefer .h5 format for compatibility with Keras 3)
        possible_paths = [
            '../weights/15Matches_LSTM.h5',
            './weights/15Matches_LSTM.h5',
            os.path.join(os.path.dirname(__file__), '../weights/15Matches_LSTM.h5'),
            '../weights/15Matches_LSTM.keras',
            './weights/15Matches_LSTM.keras',
            '../../weights/15Matches_LSTM.h5',
            os.path.join(os.path.dirname(__file__), 'weights/15Matches_LSTM.h5'),
        ]
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                args.model = abs_path
                logger.info(f"Auto-detected model at: {args.model}")
                break

        if args.model is None:
            logger.error("Could not find LSTM model file. Please specify with --model")
            sys.exit(1)

    # Parse corner points if provided
    corner_points = None
    if args.corner_points:
        try:
            coords = [int(x) for x in args.corner_points.split(',')]
            if len(coords) != 8:
                raise ValueError("Expected 8 coordinates")
            corner_points = [
                [coords[0], coords[1]],
                [coords[2], coords[3]],
                [coords[4], coords[5]],
                [coords[6], coords[7]]
            ]
            logger.info(f"Using provided corner points: {corner_points}")
        except Exception as e:
            logger.error(f"Invalid corner points format: {e}")
            sys.exit(1)

    try:
        # Create temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix='lstm_shot_classifier_')
        logger.info(f"Created temporary directory: {temp_dir}")

        # Step 1: Extract poses from video
        logger.info("=" * 60)
        logger.info("STEP 1: Extracting player poses using YOLO")
        logger.info("=" * 60)
        pose_data = extract_poses_from_video(
            args.video_path,
            corner_points=corner_points,
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )

        # Step 2: Convert to CSV format
        logger.info("=" * 60)
        logger.info("STEP 2: Converting pose data to CSV format")
        logger.info("=" * 60)
        shots_csv, wireframe_csv = convert_pose_data_to_csv(
            pose_data,
            temp_dir,
            set_id=1,
            video_path=args.video_path
        )

        # Step 3: Run LSTM prediction
        logger.info("=" * 60)
        logger.info("STEP 3: Running LSTM shot classification")
        logger.info("=" * 60)
        result = predict_shot_from_csv(temp_dir, args.model)

        # Print results
        print("\n" + "=" * 60)
        print(" SHOT CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"Predicted Shot Type: {result['shot'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"All Predictions: {result['all_predictions']}")
        print("=" * 60)

        # Cleanup or keep temp files
        if args.keep_temp:
            logger.info(f"Temporary files saved in: {temp_dir}")
            print(f"\nTemporary files saved in: {temp_dir}")
        else:
            shutil.rmtree(temp_dir)
            logger.info("Temporary files cleaned up")

        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
