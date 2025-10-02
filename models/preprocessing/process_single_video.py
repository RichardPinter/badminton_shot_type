#!/usr/bin/env python3
"""
ABOUTME: Single video processor for BSTM evaluation - converts video to .npy format
ABOUTME: Runs TrackNet, MMPose, and collation pipeline on a single video file

This script provides a simplified interface to process a single video through
the full BSTM data preparation pipeline:
1. TrackNet shuttlecock detection
2. MMPose player pose estimation
3. Data collation and augmentation (bones, interpolations)

Author: Richard
"""

import sys
import json
import shutil
import tempfile
import subprocess
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

# Import from prepare_train
from prepare_train import (
    prepare_trajectory,
    prepare_2d_dataset_npy_from_raw_video,
    collate_single_video,
    get_court_info,
)


def get_video_resolution(video_path: Path) -> Tuple[int, int]:
    """
    Auto-detect video resolution using ffprobe.

    Returns:
        (width, height) tuple
    """
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)
        video_stream = next(
            (s for s in data['streams'] if s['codec_type'] == 'video'),
            None
        )

        if not video_stream:
            raise ValueError("No video stream found")

        width = int(video_stream['width'])
        height = int(video_stream['height'])

        print(f"Auto-detected resolution: {width}x{height}")
        return width, height

    except Exception as e:
        print(f"Warning: Could not auto-detect resolution: {e}")
        print("Using default 1280x720")
        return 1280, 720


def create_single_video_dataset_structure(
    video_path: Path,
    temp_base_dir: Path,
    set_name: str = "test",
    stroke_type: str = "unknown"
) -> Tuple[Path, Path]:
    """
    Create a temporary directory structure that mimics the expected dataset format:

    temp_dir/
      dataset/
        test/
          unknown/
            video.mp4

    Returns:
        (dataset_root, video_copy_path) tuple
    """
    dataset_root = temp_base_dir / "dataset"
    set_dir = dataset_root / set_name
    stroke_dir = set_dir / stroke_type
    stroke_dir.mkdir(parents=True, exist_ok=True)

    # Copy video to the structure
    video_copy = stroke_dir / video_path.name
    if not video_copy.exists():
        shutil.copy2(video_path, video_copy)

    return dataset_root, video_copy


def create_resolution_dataframe(video_path: Path, width: int, height: int) -> pd.DataFrame:
    """
    Create a resolution DataFrame for a single video.
    The video ID is extracted from the filename (expects format: <id>_*.mp4)
    If no ID prefix exists, uses 0 as the ID.
    """
    # Try to extract ID from filename
    stem = video_path.stem
    if '_' in stem:
        try:
            vid = int(stem.split('_')[0])
        except ValueError:
            vid = 0
    else:
        vid = 0

    df = pd.DataFrame({
        'width': [width],
        'height': [height]
    }, index=[vid])
    df.index.name = 'id'

    return df


def create_dummy_court_info(video_id: int) -> Dict:
    """
    Create a dummy court info dictionary for cases where homography is unavailable.
    This provides normalized court boundaries assuming the full frame is the court.
    """
    return {
        'H': np.eye(3),  # Identity homography (no transformation)
        'border_L': 0.0,
        'border_R': 1.0,
        'border_U': 0.0,
        'border_D': 1.0,
    }


def create_court_info_from_homography(H: np.ndarray, video_id: int, width: int, height: int) -> Dict:
    """
    Create court info dictionary from a homography matrix.

    Args:
        H: 3x3 homography matrix mapping camera coords (normalized to 1280x720) → court coords (meters)
        video_id: Video identifier
        width: Video width in pixels
        height: Video height in pixels

    Returns:
        Dict with keys: H, border_L, border_R, border_U, border_D
    """
    # Since homography maps clicked corners to standard court dimensions,
    # the court boundaries are fixed and known (not computed)
    # Standard badminton court (doubles): 6.1m wide × 13.4m long
    return {
        'H': H,
        'border_L': 0.0,
        'border_R': 6.1,   # COURT_WIDTH_M
        'border_U': 0.0,
        'border_D': 13.4,  # COURT_LENGTH_M
    }


def process_single_video(
    video_path: Path,
    output_dir: Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
    homography_csv: Optional[Path] = None,
    homography_matrix: Optional[np.ndarray] = None,
    seq_len: int = 100,
    joints_center_align: bool = True,
    set_name: str = "test",
    stroke_type: str = "unknown",
    keep_intermediates: bool = False
) -> Dict[str, Path]:
    """
    Process a single video through the full BSTM preparation pipeline.

    Args:
        video_path: Path to the input video file
        output_dir: Directory where final .npy files will be saved
        width: Video width (auto-detected if None)
        height: Video height (auto-detected if None)
        homography_csv: Optional path to homography CSV file
        homography_matrix: Optional 3x3 homography matrix (takes precedence over CSV)
        seq_len: Sequence length for collation (default: 100)
        joints_center_align: Whether to center-align joints (default: True)
        set_name: Dataset split name (default: "test")
        stroke_type: Stroke type label (default: "unknown")
        keep_intermediates: Keep intermediate files (default: False)

    Returns:
        Dictionary mapping file types to their paths:
        {
            'J_only': Path,
            'JnB_interp': Path,
            'JnB_bone': Path,
            'Jn2B': Path,
            'pos': Path,
            'shuttle': Path,
            'videos_len': Path,
            'labels': Path
        }
    """
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"\n{'='*60}")
    print(f"Processing video: {video_path.name}")
    print(f"{'='*60}\n")

    # Auto-detect resolution if not provided
    if width is None or height is None:
        width, height = get_video_resolution(video_path)

    # Create temporary working directory
    with tempfile.TemporaryDirectory(prefix="bstm_video_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"Working directory: {temp_dir}\n")

        # Set up directory structure
        dataset_root, video_copy = create_single_video_dataset_structure(
            video_path, temp_dir, set_name, stroke_type
        )

        # Create resolution DataFrame
        resolution_df = create_resolution_dataframe(video_path, width, height)
        video_id = resolution_df.index[0]
        print(f"Video ID: {video_id}")
        print(f"Resolution: {width}x{height}\n")

        # Create or load court info (priority: matrix > CSV > dummy)
        if homography_matrix is not None:
            print("🔢 Using provided homography matrix")
            print(f"Homography matrix:\n{homography_matrix}")
            all_court_info = {video_id: create_court_info_from_homography(
                homography_matrix, video_id, width, height
            )}
            print(f"✅ Court boundaries: L={all_court_info[video_id]['border_L']:.2f}, "
                  f"R={all_court_info[video_id]['border_R']:.2f}, "
                  f"U={all_court_info[video_id]['border_U']:.2f}, "
                  f"D={all_court_info[video_id]['border_D']:.2f}\n")
        elif homography_csv and Path(homography_csv).exists():
            print(f"Loading homography from: {homography_csv}")
            homo_df = pd.read_csv(homography_csv).set_index('id')
            if video_id in homo_df.index:
                all_court_info = {video_id: get_court_info(homo_df, video_id)}
                print("Using provided homography data\n")
            else:
                print(f"Warning: Video ID {video_id} not found in homography CSV")
                print("Using dummy court info\n")
                all_court_info = {video_id: create_dummy_court_info(video_id)}
        else:
            print("No homography data provided - using dummy court info\n")
            all_court_info = {video_id: create_dummy_court_info(video_id)}

        # Prepare directories for intermediates
        save_shuttle_dir = temp_dir / "shuttlecock_temp"
        save_shuttle_dir.mkdir(exist_ok=True)

        save_root_dir_raw = temp_dir / "dataset_npy_raw"
        save_root_dir_collate = output_dir

        # STEP 1: Trajectory detection (TrackNet)
        print("="*60)
        print("STEP 1: Shuttlecock trajectory detection (TrackNet)")
        print("="*60)
        prepare_trajectory(dataset_root, save_shuttle_dir)
        print("✓ Trajectory detection complete\n")

        # STEP 2: Player pose detection (MMPose) + create raw .npy files
        print("="*60)
        print("STEP 2: Player pose estimation (MMPose)")
        print("="*60)
        prepare_2d_dataset_npy_from_raw_video(
            dataset_root,
            save_shuttle_dir,
            save_root_dir_raw,
            resolution_df,
            all_court_info,
            joints_normalized_by_v_height=False,
            joints_center_align=joints_center_align
        )
        print("✓ Pose estimation complete\n")

        # STEP 3: Collate and augment data
        print("="*60)
        print("STEP 3: Data collation and augmentation")
        print("="*60)

        # Find the raw .npy files
        raw_npy_dir = save_root_dir_raw / set_name / stroke_type
        joints_file = raw_npy_dir / f"{video_path.stem}_joints.npy"
        pos_file = raw_npy_dir / f"{video_path.stem}_pos.npy"
        shuttle_file = raw_npy_dir / f"{video_path.stem}_shuttle.npy"

        # Use collate_single_video instead of collate_npy
        output_files = collate_single_video(
            joints_path=joints_file,
            pos_path=pos_file,
            shuttle_path=shuttle_file,
            output_dir=save_root_dir_collate,
            seq_len=seq_len,
            label=0  # Default label for unknown stroke type
        )
        print("✓ Collation complete\n")

        # Keep intermediates if requested
        if keep_intermediates:
            intermediates_dir = output_dir / "intermediates"
            intermediates_dir.mkdir(exist_ok=True)

            # Copy raw npy files
            for raw_file in [joints_file, pos_file, shuttle_file]:
                if raw_file.exists():
                    shutil.copy2(raw_file, intermediates_dir / raw_file.name)

            # Copy shuttlecock CSV
            shuttle_csv = save_shuttle_dir / f"{video_path.stem}_ball.csv"
            if shuttle_csv.exists():
                shutil.copy2(shuttle_csv, intermediates_dir / shuttle_csv.name)

            print(f"Intermediate files saved to: {intermediates_dir}\n")

    # Verify outputs (output_files already returned from collate_single_video)
    missing = [k for k, v in output_files.items() if not v.exists()]
    if missing:
        raise RuntimeError(f"Missing output files: {missing}")

    print("="*60)
    print("✓ Processing complete!")
    print("="*60)
    print(f"\nOutput files saved to: {save_root_dir_collate}")
    print("\nGenerated files:")
    for name, path in output_files.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {name:12s}: {path.name} ({size_mb:.2f} MB)")
    print()

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Process a single video for BSTM evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect resolution)
  python process_single_video.py video.mp4 -o output/

  # With manual resolution
  python process_single_video.py video.mp4 -o output/ --width 1920 --height 1080

  # With homography data
  python process_single_video.py video.mp4 -o output/ --homography homography.csv

  # Custom sequence length
  python process_single_video.py video.mp4 -o output/ --seq-len 50

  # Keep intermediate files
  python process_single_video.py video.mp4 -o output/ --keep-intermediates
        """
    )

    parser.add_argument(
        'video',
        type=Path,
        help='Path to input video file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output directory for .npy files'
    )
    parser.add_argument(
        '--width',
        type=int,
        help='Video width (auto-detected if not specified)'
    )
    parser.add_argument(
        '--height',
        type=int,
        help='Video height (auto-detected if not specified)'
    )
    parser.add_argument(
        '--homography',
        type=str,
        help='Homography matrix as JSON string (3x3 array) OR path to homography CSV file'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=100,
        help='Sequence length for collation (default: 100)'
    )
    parser.add_argument(
        '--no-center-align',
        action='store_true',
        help='Disable joint center alignment'
    )
    parser.add_argument(
        '--set-name',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split name (default: test)'
    )
    parser.add_argument(
        '--stroke-type',
        type=str,
        default='unknown',
        help='Stroke type label (default: unknown)'
    )
    parser.add_argument(
        '--keep-intermediates',
        action='store_true',
        help='Keep intermediate files (raw .npy, CSV)'
    )

    args = parser.parse_args()

    # Parse homography argument (can be JSON matrix or CSV path)
    homography_csv = None
    homography_matrix = None

    if args.homography:
        # Try to parse as JSON first
        try:
            homography_data = json.loads(args.homography)
            homography_matrix = np.array(homography_data, dtype=np.float32)
            if homography_matrix.shape != (3, 3):
                raise ValueError(f"Homography matrix must be 3x3, got {homography_matrix.shape}")
            print(f"Parsed homography matrix from JSON")
        except (json.JSONDecodeError, ValueError):
            # Not JSON, treat as file path
            homography_csv = Path(args.homography)
            if not homography_csv.exists():
                print(f"Warning: Homography file not found: {homography_csv}")
                homography_csv = None

    try:
        output_files = process_single_video(
            video_path=args.video,
            output_dir=args.output,
            width=args.width,
            height=args.height,
            homography_csv=homography_csv,
            homography_matrix=homography_matrix,
            seq_len=args.seq_len,
            joints_center_align=not args.no_center_align,
            set_name=args.set_name,
            stroke_type=args.stroke_type,
            keep_intermediates=args.keep_intermediates
        )

        print("SUCCESS! Video processed successfully.")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
