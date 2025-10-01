#!/usr/bin/env python3
# ABOUTME: LSTM model evaluation script for bottom court test videos
# ABOUTME: Processes ShuttleSet test videos through YOLO pose extraction + LSTM classification

import sys
import csv
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model

# YOLO imports
from ultralytics import YOLO

# Mapping from English class names to Frontier classes (6 classes)
ENGLISH_TO_FRONTIER = {
    'drop': 'net',
    'net_shot': 'net',
    'smash': 'smash',
    'passive_drop': 'drop',
    'lob': 'lob',
    'defensive_return_lob': 'lob',
    'clear': 'clear',
    'drive': 'drive',
    'driven_flight': 'drive',
    'back-court_drive': 'drive',
    'drive_variant': 'drop',
    'cross-court_net_shot': 'drop',
    'push': 'drive',
    'rush': 'drive',
    'defensive_return_drive': 'drive',
    'return_net': 'net',
    'short_service': 'net',
    'long_service': 'net',
    'wrist_smash': 'smash',
    'none': 'unknown',
    'unmapped_overcut': 'unknown',
}

# Frontier classes (must match LSTM training order)
FRONTIER_CLASSES = ['clear', 'drive', 'drop', 'lob', 'net', 'smash']

class LSTMEvaluator:
    def __init__(self, results_dir: Path, model_path: Path):
        self.results_dir = results_dir
        self.results_file = results_dir / "lstm_results.csv"

        # Load LSTM model
        print("Loading LSTM model...")
        self.model = load_model(str(model_path), compile=False)
        print(f"✓ LSTM model loaded: {model_path}")

        # Load YOLO pose model
        print("Loading YOLOv8 pose model...")
        self.yolo_model = YOLO('yolov8x-pose-p6.pt')
        print("✓ YOLOv8 pose model loaded")

        # Configuration
        self.max_frames = 100  # Maximum sequence length (pad/crop)
        self.num_keypoints = 17  # COCO 17 keypoints
        self.confidence_threshold = 0.3

    def extract_label_from_folder(self, folder_name: str, english_label: str) -> str:
        """Map English label to frontier class"""
        frontier_label = ENGLISH_TO_FRONTIER.get(english_label, english_label)
        return frontier_label

    def get_test_videos(self, test_dir: Path) -> List[Tuple[Path, str, str]]:
        """Get all bottom court videos with their labels

        Returns:
            List of tuples: (video_path, english_label, frontier_label)
        """
        videos = []

        # Get all Bottom_* directories
        for folder in test_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("Bottom_"):
                english_label = folder.name[7:]  # Remove "Bottom_" prefix
                frontier_label = ENGLISH_TO_FRONTIER.get(english_label, english_label)

                # Get all mp4 files in this folder
                for video_file in folder.glob("*.mp4"):
                    videos.append((video_file, english_label, frontier_label))

        # Shuffle videos for random sampling
        random.shuffle(videos)

        english_folders = len([f for f in test_dir.iterdir() if f.name.startswith('Bottom_')])
        frontier_classes = set([frontier for _, _, frontier in videos])
        print(f"Found {len(videos)} bottom court videos")
        print(f"  English folders: {english_folders}")
        print(f"  Mapped to {len(frontier_classes)} frontier classes: {sorted(frontier_classes)}")
        print(f"  Videos shuffled for random sampling")
        return videos

    def extract_poses_from_video(self, video_path: Path) -> np.ndarray:
        """Extract pose keypoints from video using YOLOv8

        Returns:
            np.ndarray: Shape (num_frames, 17, 2) - COCO 17 keypoints
        """
        try:
            # Run YOLOv8 pose detection
            results = self.yolo_model.track(
                str(video_path),
                conf=self.confidence_threshold,
                verbose=False,
                persist=True
            )

            poses = []
            for result in results:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    # Get first person's keypoints (bottom player)
                    kpts = result.keypoints[0].xy.cpu().numpy()  # Shape: (17, 2)
                    poses.append(kpts)

            if len(poses) == 0:
                return None

            poses_array = np.array(poses)  # Shape: (T, 17, 2)
            return poses_array

        except Exception as e:
            print(f"Error extracting poses: {e}")
            return None

    def preprocess_poses(self, poses: np.ndarray) -> np.ndarray:
        """Preprocess poses to match LSTM training format

        Args:
            poses: Shape (T, 17, 2)

        Returns:
            Preprocessed poses: Shape (1, max_frames, 34)
        """
        # Flatten keypoints: (T, 17, 2) -> (T, 34)
        T = poses.shape[0]
        flattened = poses.reshape(T, -1)  # (T, 34)

        # Pad or crop to max_frames
        if T < self.max_frames:
            # Pad with zeros (will be masked)
            padding = np.zeros((self.max_frames - T, 34))
            processed = np.vstack([flattened, padding])
        else:
            # Crop from center
            start = (T - self.max_frames) // 2
            processed = flattened[start:start + self.max_frames]

        # Normalize to [0, 1] range
        processed = processed / np.max(processed + 1e-6)

        # Add batch dimension
        processed = processed[np.newaxis, ...]  # (1, max_frames, 34)

        return processed

    def predict_video(self, video_path: Path) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Predict stroke class for a video

        Returns:
            (predicted_class, confidence, top3_predictions)
        """
        # Extract poses
        poses = self.extract_poses_from_video(video_path)

        if poses is None or len(poses) == 0:
            return None, 0.0, []

        # Preprocess
        processed = self.preprocess_poses(poses)

        # Predict
        predictions = self.model.predict(processed, verbose=0)  # Shape: (1, num_classes)
        probs = predictions[0]  # Shape: (num_classes,)

        # Get top 3
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3 = [(FRONTIER_CLASSES[idx], float(probs[idx])) for idx in top3_indices]

        # Get top prediction
        pred_class = top3[0][0]
        pred_conf = top3[0][1]

        return pred_class, pred_conf, top3

    def process_video(self, video_path: Path, english_label: str, frontier_label: str) -> Dict:
        """Process a single video through LSTM pipeline"""
        try:
            start_time = time.time()

            pred_class, pred_conf, top3 = self.predict_video(video_path)

            processing_time = time.time() - start_time

            if pred_class is None:
                return {
                    'video_path': str(video_path),
                    'video_name': video_path.name,
                    'english_label': english_label,
                    'true_label_frontier': frontier_label,
                    'predicted_label': None,
                    'confidence': None,
                    'pred_2': None,
                    'pred_3': None,
                    'conf_2': None,
                    'conf_3': None,
                    'accuracy_1': 0,
                    'accuracy_2': 0,
                    'processing_time': processing_time,
                    'success': False,
                    'error_message': "Failed to extract poses"
                }

            # Calculate accuracy
            accuracy_1 = 1 if pred_class == frontier_label else 0
            pred_2 = top3[1][0] if len(top3) > 1 else None
            accuracy_2 = 1 if (pred_class == frontier_label or (pred_2 and pred_2 == frontier_label)) else 0

            return {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'english_label': english_label,
                'true_label_frontier': frontier_label,
                'predicted_label': pred_class,
                'confidence': pred_conf,
                'pred_2': top3[1][0] if len(top3) > 1 else None,
                'pred_3': top3[2][0] if len(top3) > 2 else None,
                'conf_2': top3[1][1] if len(top3) > 1 else None,
                'conf_3': top3[2][1] if len(top3) > 2 else None,
                'accuracy_1': accuracy_1,
                'accuracy_2': accuracy_2,
                'processing_time': processing_time,
                'success': True,
                'error_message': None
            }

        except Exception as e:
            return {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'english_label': english_label,
                'true_label_frontier': frontier_label,
                'predicted_label': None,
                'confidence': None,
                'pred_2': None,
                'pred_3': None,
                'conf_2': None,
                'conf_3': None,
                'accuracy_1': 0,
                'accuracy_2': 0,
                'processing_time': 0.0,
                'success': False,
                'error_message': str(e)
            }

    def evaluate(self, test_dir: Path, limit: int = None) -> None:
        """Run evaluation on all test videos"""
        print(f"Starting LSTM evaluation on test directory: {test_dir}")

        # Get all test videos (shuffled)
        videos = self.get_test_videos(test_dir)
        if limit:
            videos = videos[:limit]
            print(f"\nLimited to {limit} videos (randomly sampled)")

            # Show distribution of sampled videos
            frontier_dist = Counter([frontier for _, _, frontier in videos])
            english_dist = Counter([english for _, english, _ in videos])

            print(f"\nSampled distribution by frontier class:")
            for cls, count in sorted(frontier_dist.items()):
                print(f"  {cls}: {count} videos")

            print(f"\nSampled distribution by English class (top 10):")
            for cls, count in sorted(english_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {cls}: {count} videos")

        # Prepare CSV file
        fieldnames = ['video_path', 'video_name', 'english_label', 'true_label_frontier', 'predicted_label',
                     'confidence', 'pred_2', 'pred_3', 'conf_2', 'conf_3',
                     'accuracy_1', 'accuracy_2', 'processing_time', 'success', 'error_message']

        with open(self.results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            total_videos = len(videos)
            successful = 0
            failed = 0
            total_acc_1 = 0
            total_acc_2 = 0

            for i, (video_path, english_label, frontier_label) in enumerate(videos, 1):
                print(f"\nProcessing {i}/{total_videos}: {video_path.name}")
                print(f"English label: {english_label} → Frontier: {frontier_label}")

                result = self.process_video(video_path, english_label, frontier_label)

                if result['success']:
                    successful += 1
                    total_acc_1 += result['accuracy_1']
                    total_acc_2 += result['accuracy_2']

                    # Clear comparison display
                    print(f"📊 ACTUAL (frontier): {frontier_label} | PREDICTED: {result['predicted_label']} (conf: {result['confidence']:.3f})")
                    if result['pred_2']:
                        print(f"   2nd choice: {result['pred_2']} (conf: {result['conf_2']:.3f})")
                    if result['pred_3']:
                        print(f"   3rd choice: {result['pred_3']} (conf: {result['conf_3']:.3f})")

                    # Match indicators
                    match_symbol = "✅" if result['accuracy_1'] else "❌"
                    top2_symbol = "✅" if result['accuracy_2'] else "❌"
                    print(f"   Top-1 Match: {match_symbol} | Top-2 Match: {top2_symbol}")
                else:
                    failed += 1
                    print(f"✗ Failed: {result['error_message']}")

                print(f"  Processing time: {result['processing_time']:.2f}s")

                # Write result to CSV
                writer.writerow(result)
                csvfile.flush()  # Ensure data is written immediately

                # Progress update
                if i % 10 == 0:
                    processed_successful = successful
                    acc_1 = total_acc_1 / processed_successful if processed_successful > 0 else 0
                    acc_2 = total_acc_2 / processed_successful if processed_successful > 0 else 0
                    print(f"\n--- Progress: {i}/{total_videos} ({i/total_videos*100:.1f}%) ---")
                    print(f"Successful: {successful}, Failed: {failed}")
                    print(f"Current Top-1 Accuracy: {acc_1:.3f}")
                    print(f"Current Top-2 Accuracy: {acc_2:.3f}")

        # Final results
        processed_successful = successful
        final_acc_1 = total_acc_1 / processed_successful if processed_successful > 0 else 0
        final_acc_2 = total_acc_2 / processed_successful if processed_successful > 0 else 0

        print(f"\n=== LSTM Evaluation Complete ===")
        print(f"Total videos: {total_videos}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Final Top-1 Accuracy: {final_acc_1:.3f} ({total_acc_1}/{processed_successful})")
        print(f"Final Top-2 Accuracy: {final_acc_2:.3f} ({total_acc_2}/{processed_successful})")
        print(f"Results saved to: {self.results_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LSTM model on test videos")
    parser.add_argument("--test_dir",
                       default="/home/richard/Desktop/Projects/Personal/UNE/BST-Badminton-Stroke-type-Transformer/ShuttleSet/shuttle_set_between_2_hits_with_max_limits/test",
                       help="Path to test directory")
    parser.add_argument("--results_dir",
                       default="/home/richard/Desktop/Projects/Personal/UNE/BST-Badminton-Stroke-type-Transformer/badminton-stroke-classifier/analysis/results",
                       help="Directory to save results")
    parser.add_argument("--model_path",
                       default="/home/richard/Desktop/Projects/Personal/UNE/BST-Badminton-Stroke-type-Transformer/badminton-stroke-classifier/visualisation/weights/15Matches_LSTM.h5",
                       help="Path to LSTM model")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of videos for testing (default: all)")

    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    results_dir = Path(args.results_dir)
    model_path = Path(args.model_path)

    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    evaluator = LSTMEvaluator(results_dir, model_path)
    evaluator.evaluate(test_dir, limit=args.limit)

if __name__ == "__main__":
    main()
