#!/usr/bin/env python3
# ABOUTME: BST model evaluation script for bottom court test videos
# ABOUTME: Processes ShuttleSet test videos through BST pipeline and saves results with top-1 and top-2 accuracy

import sys
import csv
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Add the visualization directory to path for imports
# Script is in: /visualisation/analysis/evaluation_scripts/
# We need to get to: /visualisation/
vis_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(vis_dir))

try:
    from models.bst.pipeline.file_based_pipeline import FileBased_Pipeline
except ImportError as e:
    print(f"Failed to import BST pipeline: {e}")
    print(f"Make sure you're running from the correct directory and the visualization folder exists")
    sys.exit(1)

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

class BSTEvaluator:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_file = results_dir / "bst_results.csv"
        self.pipeline = None

        # Initialize BST pipeline
        try:
            # Path relative to visualisation folder
            bst_weight_path = vis_dir / "weights" / "bst_8_JnB_bone_bottom_frontier_6class.pt"
            tracknet_weight_path = vis_dir / "models" / "bst" / "weights" / "tracknet_model.pt"

            if not bst_weight_path.exists():
                raise FileNotFoundError(f"BST weights not found: {bst_weight_path}")
            if not tracknet_weight_path.exists():
                raise FileNotFoundError(f"TrackNet weights not found: {tracknet_weight_path}")

            self.pipeline = FileBased_Pipeline(
                bst_weight_path=str(bst_weight_path),
                tracknet_model_path=str(tracknet_weight_path)
            )
            print("BST Pipeline initialized successfully")
        except Exception as e:
            print(f"Failed to initialize BST pipeline: {e}")
            raise

    def extract_label_from_folder(self, folder_name: str) -> str:
        """Extract stroke type from folder name and map to frontier class

        e.g., 'Bottom_back-court_drive' -> 'back-court_drive' -> 'drive'
        """
        if folder_name.startswith("Bottom_"):
            english_label = folder_name[7:]  # Remove "Bottom_" prefix
            # Map to frontier class
            frontier_label = ENGLISH_TO_FRONTIER.get(english_label, english_label)
            return frontier_label
        return folder_name

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

        # Shuffle videos for random sampling across all classes
        random.shuffle(videos)

        english_folders = len([f for f in test_dir.iterdir() if f.name.startswith('Bottom_')])
        frontier_classes = set([frontier for _, _, frontier in videos])
        print(f"Found {len(videos)} bottom court videos")
        print(f"  English folders: {english_folders}")
        print(f"  Mapped to {len(frontier_classes)} frontier classes: {sorted(frontier_classes)}")
        print(f"  Videos shuffled for random sampling")
        return videos

    def process_video(self, video_path: Path, english_label: str, frontier_label: str) -> Dict:
        """Process a single video through BST pipeline

        Args:
            video_path: Path to video file
            english_label: Original English class name (e.g., 'back-court_drive')
            frontier_label: Mapped frontier class (e.g., 'drive')
        """
        def progress_callback(msg: str, progress: float):
            pass  # Silent processing for batch evaluation

        try:
            result = self.pipeline.process_video(str(video_path), progress_callback)

            if result.success:
                # Extract top predictions for accuracy calculations
                top3_predictions = result.top3_predictions or []

                # Get top-1 and top-2 predictions
                pred_1 = result.stroke_prediction
                pred_2 = top3_predictions[1]['class'] if len(top3_predictions) > 1 else None

                # Calculate accuracy metrics (using frontier labels)
                accuracy_1 = 1 if pred_1 == frontier_label else 0
                accuracy_2 = 1 if (pred_1 == frontier_label or (pred_2 and pred_2 == frontier_label)) else 0

                return {
                    'video_path': str(video_path),
                    'video_name': video_path.name,
                    'english_label': english_label,
                    'true_label_frontier': frontier_label,
                    'predicted_label': pred_1,
                    'confidence': result.confidence,
                    'pred_2': pred_2,
                    'pred_3': top3_predictions[2]['class'] if len(top3_predictions) > 2 else None,
                    'conf_2': top3_predictions[1]['confidence'] if len(top3_predictions) > 1 else None,
                    'conf_3': top3_predictions[2]['confidence'] if len(top3_predictions) > 2 else None,
                    'accuracy_1': accuracy_1,
                    'accuracy_2': accuracy_2,
                    'processing_time': result.processing_time,
                    'success': True,
                    'error_message': None
                }
            else:
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
                    'processing_time': None,
                    'success': False,
                    'error_message': result.error_message
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
                'processing_time': None,
                'success': False,
                'error_message': str(e)
            }
        finally:
            # Cleanup temporary files
            try:
                if hasattr(result, 'temp_dir') and result.temp_dir:
                    self.pipeline.cleanup(result)
            except:
                pass

    def evaluate(self, test_dir: Path, limit: int = None) -> None:
        """Run evaluation on all test videos"""
        print(f"Starting BST evaluation on test directory: {test_dir}")

        # Get all test videos (shuffled)
        videos = self.get_test_videos(test_dir)
        if limit:
            videos = videos[:limit]
            print(f"\nLimited to {limit} videos (randomly sampled)")

            # Show distribution of sampled videos
            from collections import Counter
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

                start_time = time.time()
                result = self.process_video(video_path, english_label, frontier_label)
                process_time = time.time() - start_time

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

                print(f"  Processing time: {process_time:.2f}s")

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

        print(f"\n=== BST Evaluation Complete ===")
        print(f"Total videos: {total_videos}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Final Top-1 Accuracy: {final_acc_1:.3f} ({total_acc_1}/{processed_successful})")
        print(f"Final Top-2 Accuracy: {final_acc_2:.3f} ({total_acc_2}/{processed_successful})")
        print(f"Results saved to: {self.results_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate BST model on test videos")
    parser.add_argument("--test_dir",
                       default="/home/richard/Desktop/Projects/Personal/UNE/BST-Badminton-Stroke-type-Transformer/ShuttleSet/shuttle_set_between_2_hits_with_max_limits/test",
                       help="Path to test directory")
    parser.add_argument("--results_dir",
                       default="/home/richard/Desktop/Projects/Personal/UNE/BST-Badminton-Stroke-type-Transformer/badminton-stroke-classifier/analysis/results",
                       help="Directory to save results")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of videos for testing (default: all)")

    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    results_dir = Path(args.results_dir)

    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    evaluator = BSTEvaluator(results_dir)
    evaluator.evaluate(test_dir, limit=args.limit)

if __name__ == "__main__":
    main()