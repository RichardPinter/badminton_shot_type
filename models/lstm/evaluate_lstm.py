#!/usr/bin/env python3
# ABOUTME: Evaluation script for LSTM shot classifier
# ABOUTME: Computes accuracy and F1-scores per class on test dataset

"""
LSTM Shot Classifier Evaluation

Evaluates LSTM model on test dataset and reports:
- Overall accuracy
- Per-class F1-scores
- Confusion matrix

Usage:
    python evaluate_lstm.py <test_data_dir> [--model path/to/model.h5]

Example:
    python evaluate_lstm.py /tmp/lstm_eval_data_full --model ../weights/15Matches_LSTM.h5
"""

import sys
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time

# Import LSTM classifier components
from match_loader import MatchLoader, SetShot
from shot_classifier import ShotClassifier, ShotType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Shot types (must match model training)
SHOT_TYPES = [
    ShotType(shotClassId=0, shotClass='clear', shotGroupId=0, shotGroup='clear'),
    ShotType(shotClassId=1, shotClass='drive', shotGroupId=1, shotGroup='drive'),
    ShotType(shotClassId=2, shotClass='drop', shotGroupId=2, shotGroup='drop'),
    ShotType(shotClassId=3, shotClass='lob', shotGroupId=3, shotGroup='lob'),
    ShotType(shotClassId=4, shotClass='net', shotGroupId=4, shotGroup='net'),
    ShotType(shotClassId=5, shotClass='smash', shotGroupId=5, shotGroup='smash'),
]

# Frontier class order for display (matching BST output)
FRONTIER_ORDER = ['net', 'smash', 'lob', 'clear', 'drive', 'drop']


def load_test_data(test_dir: str) -> tuple[List[SetShot], List[str]]:
    """
    Load test data from CSV directory.

    Args:
        test_dir: Directory containing test CSV files

    Returns:
        Tuple of (shots, ground_truth_labels)
    """
    logger.info(f"Loading test data from: {test_dir}")

    # Create MatchLoader
    loader = MatchLoader(config_path='config.json', logger=logger)
    matches = loader.loadMatches(test_dir)

    if not matches:
        raise RuntimeError(f"No matches loaded from {test_dir}")

    logger.info(f"Loaded {len(matches)} match(es)")

    # Collect all shots with ground truth labels
    all_shots = []
    ground_truth = []

    for match in matches:
        for player_sets in match.sets.values():
            for player_set in player_sets:
                for shot in player_set.shots:
                    all_shots.append(shot)
                    # Ground truth is in the CSV shot_class field
                    ground_truth.append(shot.shotClass)

    logger.info(f"Loaded {len(all_shots)} shots for evaluation")

    return all_shots, ground_truth


def evaluate_model(
    model_path: str,
    shots: List[SetShot],
    ground_truth: List[str]
) -> Dict:
    """
    Evaluate LSTM model on test shots.

    Args:
        model_path: Path to trained LSTM model
        shots: List of shots to evaluate
        ground_truth: Ground truth labels

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating model: {model_path}")

    # Initialize classifier
    classifier = ShotClassifier(
        logger=logger,
        shot_types=SHOT_TYPES,
        model_persist_path=os.path.dirname(model_path)
    )

    # Run predictions
    start_time = time.time()
    predictions = classifier.predict(model_path, shots)
    eval_time = time.time() - start_time

    logger.info(f"Evaluation complete in {eval_time:.2f}s")

    # Convert to class indices for sklearn metrics
    class_to_idx = {st.shotClass: st.shotClassId for st in SHOT_TYPES}

    y_true = [class_to_idx[gt] for gt in ground_truth]
    y_pred = [class_to_idx[pred] for pred in predictions]

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    f1_avg = np.mean(f1_per_class)
    f1_min = np.min(f1_per_class)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'f1_per_class': f1_per_class,
        'f1_avg': f1_avg,
        'f1_min': f1_min,
        'confusion_matrix': conf_matrix,
        'eval_time': eval_time,
        'num_samples': len(shots)
    }


def print_results(results: Dict, model_name: str = '15Matches_LSTM'):
    """
    Print evaluation results in BST-style format.

    Args:
        results: Results dictionary from evaluate_model
        model_name: Name of the model for display
    """
    # Format evaluation time
    eval_time_min = int(results['eval_time'] // 60)
    eval_time_sec = int(results['eval_time'] % 60)

    print(f"\nTotal evaluation time: {eval_time_min}:{eval_time_sec:02d}")
    print(f"Test (num_strokes: {results['num_samples']}) =>")
    print(f"    {model_name:<42} F1-score")
    print(f"{'avg':<42} {'Avg':<9} {results['f1_avg']:>4.2f}")
    print(f"{'min':<42} {'Min':<9} {results['f1_min']:>4.2f}")

    # Print per-class F1 scores in frontier order
    class_to_idx = {st.shotClass: st.shotClassId for st in SHOT_TYPES}

    for i, class_name in enumerate(FRONTIER_ORDER):
        class_idx = class_to_idx[class_name]
        f1 = results['f1_per_class'][class_idx]
        print(f"{i:<42} {class_name:<9} {f1:>4.2f}")

    print(f"Accuracy: {results['accuracy']:.3f}")

    # Optional: Print confusion matrix
    print("\nConfusion Matrix:")
    print("Predicted →")
    print("True ↓       ", end="")
    for class_name in FRONTIER_ORDER:
        print(f"{class_name:>6}", end=" ")
    print()

    for i, true_class in enumerate(FRONTIER_ORDER):
        true_idx = class_to_idx[true_class]
        print(f"{true_class:<12} ", end="")
        for pred_class in FRONTIER_ORDER:
            pred_idx = class_to_idx[pred_class]
            count = results['confusion_matrix'][true_idx][pred_idx]
            print(f"{count:>6}", end=" ")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LSTM shot classifier on test data'
    )
    parser.add_argument(
        'test_dir',
        type=str,
        help='Directory containing test CSV files'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to LSTM model file (default: auto-detect)'
    )

    args = parser.parse_args()

    # Validate test directory
    if not os.path.exists(args.test_dir):
        logger.error(f"Test directory not found: {args.test_dir}")
        sys.exit(1)

    # Auto-detect model path if not provided
    if args.model is None:
        possible_paths = [
            '../weights/15Matches_LSTM.h5',
            './weights/15Matches_LSTM.h5',
            '../../weights/15Matches_LSTM.h5',
            os.path.join(os.path.dirname(__file__), '../weights/15Matches_LSTM.h5'),
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

    try:
        print("=" * 60)
        print("LSTM Shot Classifier Evaluation")
        print("=" * 60)
        print(f"Test data: {args.test_dir}")
        print(f"Model: {args.model}")
        print("=" * 60)

        # Load test data
        shots, ground_truth = load_test_data(args.test_dir)

        # Evaluate model
        results = evaluate_model(args.model, shots, ground_truth)

        # Print results
        model_name = Path(args.model).stem
        print_results(results, model_name=model_name)

        return 0

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
