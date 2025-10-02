#!/usr/bin/env python3
# ABOUTME: Converts BST .npy format to LSTM CSV format for evaluation
# ABOUTME: Handles keypoint filtering (17→13) and frame selection (100→41)

"""
Convert BST Dataset to LSTM CSV Format

Converts BST preprocessed .npy files to LSTM-compatible CSV format:
- Filters 17 COCO keypoints → 13 keypoints (removes eyes/ears)
- Selects center 41 frames from 100-frame sequences
- Creates proper directory structure with setN_shots.csv and setN_wireframe.csv
- Preserves ground truth labels

Usage:
    python convert_bst_to_lstm.py <bst_npy_dir> <output_dir> [--split val]

Example:
    python convert_bst_to_lstm.py \\
        /path/to/dataset_npy_collated_between_2_hits_with_max_limits_seq_100 \\
        /path/to/lstm_evaluation_data \\
        --split val
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# COCO keypoint mapping (17 keypoints)
COCO_KEYPOINTS = [
    'nose', # 0
    'left_eye', # 1 - EXCLUDE
    'right_eye', # 2 - EXCLUDE
    'left_ear', # 3 - EXCLUDE
    'right_ear', # 4 - EXCLUDE
    'left_shoulder', # 5
    'right_shoulder', # 6
    'left_elbow', # 7
    'right_elbow', # 8
    'left_wrist', # 9
    'right_wrist', # 10
    'left_hip', # 11
    'right_hip', # 12
    'left_knee', # 13
    'right_knee', # 14
    'left_ankle', # 15
    'right_ankle' # 16
]

# LSTM model keypoints (13 keypoints) - indices in COCO format
LSTM_KEYPOINT_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

LSTM_KEYPOINT_NAMES = [
    'nose',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

# Shot type mapping (0-5)
# Frontier classes order: net, smash, lob, clear, drive, drop
FRONTIER_CLASSES = ['net', 'smash', 'lob', 'clear', 'drive', 'drop']
SHOT_TYPES = ['clear', 'drive', 'drop', 'lob', 'net', 'smash']


def get_38_to_6_class_mapping():
    """
    Create mapping from 38-class labels to 6 frontier classes.

    Based on ShuttleSetMapping.csv and dataset.py get_stroke_types('Both')

    Returns:
        dict: {38_class_id: frontier_class_id} or None for unknown classes
    """
    # Base stroke types (19 classes)
    base_strokes = [
        'drop', 'net_shot', 'smash', 'passive_drop', 'lob', 'defensive_return_lob',
        'clear', 'drive', 'back-court_drive', 'drive_variant', 'cross-court_net_shot', 'push',
        'rush', 'defensive_return_drive', 'return_net', 'short_service', 'long_service', 'wrist_smash', 'none'
    ]

    # Mapping from English stroke name to frontier class (from ShuttleSetMapping.csv)
    stroke_to_frontier = {
        'drop': 'net', #
        'net_shot': 'net', #
        'smash': 'smash', #
        'passive_drop': 'drop', #
        'lob': 'lob', #
        'defensive_return_lob': 'lob', #
        'clear': 'clear', #
        'drive': 'drive', #
        'back-court_drive': 'drive', #
        'drive_variant': 'drop', #
        'cross-court_net_shot': 'drop', #
        'push': 'drive', #
        'rush': 'drive', #
        'defensive_return_drive': 'drive', #
        'return_net': 'net', #
        'short_service': 'net', #
        'long_service': 'net', #
        'wrist_smash': 'smash', # wrist smash
        'none': None # (unknown)
    }

    # Create 38-class mapping (Top + Bottom for each stroke)
    mapping_38_to_6 = {}

    for i, stroke in enumerate(base_strokes):
        frontier = stroke_to_frontier.get(stroke)

        if frontier is None:
            # Unknown class - map to None
            mapping_38_to_6[i] = None # Top_none
            mapping_38_to_6[i + 19] = None # Bottom_none
        else:
            # Map to frontier class index
            frontier_idx = FRONTIER_CLASSES.index(frontier)
            mapping_38_to_6[i] = frontier_idx # Top_*
            mapping_38_to_6[i + 19] = frontier_idx # Bottom_*

    return mapping_38_to_6


def load_bst_data(npy_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load BST preprocessed data.

    Args:
        npy_dir: Directory containing BST .npy files
        split: 'train', 'val', or 'test'

    Returns:
        Tuple of (joints_data, labels)
        - joints_data: shape (N, 100, 2, 17, 2)
        - labels: shape (N,)
    """
    split_dir = os.path.join(npy_dir, split)

    joints_path = os.path.join(split_dir, 'J_only.npy')
    labels_path = os.path.join(split_dir, 'labels.npy')

    if not os.path.exists(joints_path):
        raise FileNotFoundError(f"Joints file not found: {joints_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    print(f"Loading joints from: {joints_path}")
    joints_data = np.load(joints_path)
    print(f" Shape: {joints_data.shape}")

    print(f"Loading labels from: {labels_path}")
    labels = np.load(labels_path)
    print(f" Shape: {labels.shape}")

    return joints_data, labels


def filter_keypoints(joints_data: np.ndarray) -> np.ndarray:
    """
    Filter from 17 COCO keypoints to 13 LSTM keypoints.

    Args:
        joints_data: shape (N, frames, 2_players, 17_keypoints, 2_xy)

    Returns:
        Filtered data: shape (N, frames, 2_players, 13_keypoints, 2_xy)
    """
    # Select only the keypoints LSTM uses (exclude eyes and ears)
    filtered = joints_data[:, :, :, LSTM_KEYPOINT_INDICES, :]
    print(f"Filtered keypoints: {joints_data.shape} → {filtered.shape}")
    return filtered


def select_center_frames(joints_data: np.ndarray, target_frames: int = 41) -> np.ndarray:
    """
    Select center frames from sequence.

    Args:
        joints_data: shape (N, 100, players, keypoints, xy)
        target_frames: Number of frames to select (default: 41)

    Returns:
        Selected frames: shape (N, 41, players, keypoints, xy)
    """
    total_frames = joints_data.shape[1]

    if total_frames < target_frames:
        raise ValueError(f"Not enough frames: {total_frames} < {target_frames}")

    # Calculate center window
    start_frame = (total_frames - target_frames) // 2
    end_frame = start_frame + target_frames

    selected = joints_data[:, start_frame:end_frame, :, :, :]
    print(f"Selected center frames [{start_frame}:{end_frame}]: {joints_data.shape} → {selected.shape}")
    return selected


def create_lstm_csv_structure(
    joints_data: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    split: str,
    use_38_class_mapping: bool = True
) -> None:
    """
    Create LSTM-compatible CSV structure.

    Args:
        joints_data: shape (N, 41, 2, 13, 2)
        labels: shape (N,) - can be 38-class or 6-class labels
        output_dir: Output directory
        split: 'train', 'val', or 'test'
        use_38_class_mapping: If True, map 38 classes to 6 frontier classes
    """
    # Create output directory structure
    # LSTM expects: output_dir/matchname/setN_shots.csv and setN_wireframe.csv

    n_samples = joints_data.shape[0]
    n_frames = joints_data.shape[1]

    print(f"\nCreating LSTM CSV structure for {n_samples} samples...")

    # Get class mapping if using 38-class labels
    if use_38_class_mapping:
        class_mapping = get_38_to_6_class_mapping()
        print("Using 38→6 class mapping")
    else:
        print("Using direct 6-class labels")

    # Group samples by shot type for organization
    samples_by_type = {shot_type: [] for shot_type in SHOT_TYPES}
    unknown_count = 0

    for i in range(n_samples):
        label_38 = int(labels[i])

        if use_38_class_mapping:
            # Map 38-class to 6-class
            frontier_idx = class_mapping.get(label_38)

            if frontier_idx is None:
                # Unknown class - skip this sample
                unknown_count += 1
                continue

            # Convert frontier index to SHOT_TYPES index
            frontier_class = FRONTIER_CLASSES[frontier_idx]
            shot_type = frontier_class
        else:
            # Direct 6-class label
            if 0 <= label_38 < len(SHOT_TYPES):
                shot_type = SHOT_TYPES[label_38]
            else:
                unknown_count += 1
                continue

        samples_by_type[shot_type].append(i)

    if unknown_count > 0:
        print(f"Skipped {unknown_count} samples with unknown class")

    # Print distribution
    print("\nDataset distribution:")
    for shot_type, indices in samples_by_type.items():
        print(f" {shot_type}: {len(indices)} samples")

    # Create one "match" directory per shot type
    sample_idx = 0
    for shot_type, sample_indices in tqdm(samples_by_type.items(), desc="Creating CSV files"):
        if len(sample_indices) == 0:
            continue

        # Create match directory
        match_dir = os.path.join(output_dir, f"{split}_{shot_type}")
        os.makedirs(match_dir, exist_ok=True)

        # Process each sample as a separate "set"
        for set_id, global_idx in enumerate(sample_indices, start=1):
            # shot_class is already determined from samples_by_type grouping
            shot_class = shot_type

            # Get joint data for this sample
            sample_joints = joints_data[global_idx] # shape: (41, 2, 13, 2)

            # Create shots CSV (metadata)
            shots_data = {
                'player': 'A',
                'shot': 0,
                'time': '00:00:00',
                'start_frame': 0,
                'total_frames': n_frames,
                'shot_class': shot_class,
                'shot_class_grouped': shot_class,
                'WristSpeed': 0.0,
                'ElbowSpeed': 0.0
            }
            shots_df = pd.DataFrame([shots_data])
            shots_path = os.path.join(match_dir, f'set{set_id}_shots.csv')
            shots_df.to_csv(shots_path, index=False)

            # Create wireframe CSV (keypoint data)
            # LSTM expects one player per shot, use player 0 (bottom player)
            player_data = sample_joints[:, 0, :, :] # shape: (41, 13, 2)

            wireframe_rows = []
            for frame_idx in range(n_frames):
                row = {
                    'Player': 'A',
                    'shot': 0,
                    'index': frame_idx
                }

                # Add keypoint coordinates
                for kp_idx, kp_name in enumerate(LSTM_KEYPOINT_NAMES):
                    row[f'{kp_name}_X'] = float(player_data[frame_idx, kp_idx, 0])
                    row[f'{kp_name}_Y'] = float(player_data[frame_idx, kp_idx, 1])

                wireframe_rows.append(row)

            wireframe_df = pd.DataFrame(wireframe_rows)
            wireframe_path = os.path.join(match_dir, f'set{set_id}_wireframe.csv')
            wireframe_df.to_csv(wireframe_path, index=False)

            sample_idx += 1

    print(f"\nCreated {sample_idx} sample sets in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert BST .npy format to LSTM CSV format'
    )
    parser.add_argument(
        'bst_npy_dir',
        type=str,
        help='Directory containing BST .npy files (with train/val/test subdirs)'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for LSTM CSV files'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Which split to convert (default: val)'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=41,
        help='Number of frames to select (default: 41)'
    )
    parser.add_argument(
        '--use-all-classes',
        action='store_true',
        help='Use 38→6 class mapping to include all samples (default: only use native 6-class samples)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.bst_npy_dir):
        print(f"Error: BST directory not found: {args.bst_npy_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("BST → LSTM Dataset Conversion")
    print("=" * 60)
    print(f"Input: {args.bst_npy_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Split: {args.split}")
    print(f"Frames: {args.frames}")
    print("=" * 60)

    # Step 1: Load BST data
    print("\n[1/4] Loading BST data...")
    joints_data, labels = load_bst_data(args.bst_npy_dir, args.split)

    # Step 2: Filter keypoints (17 → 13)
    print("\n[2/4] Filtering keypoints (17 → 13)...")
    joints_data = filter_keypoints(joints_data)

    # Step 3: Select center frames (100 → 41)
    print("\n[3/4] Selecting center frames (100 → 41)...")
    joints_data = select_center_frames(joints_data, args.frames)

    # Step 4: Create CSV structure
    print("\n[4/4] Creating LSTM CSV structure...")
    create_lstm_csv_structure(joints_data, labels, args.output_dir, args.split, use_38_class_mapping=args.use_all_classes)

    print("\n" + "=" * 60)
    print(" Conversion complete!")
    print("=" * 60)
    print(f"\nYou can now evaluate LSTM model using:")
    print(f" python evaluate_lstm.py {args.output_dir}")


if __name__ == '__main__':
    main()
