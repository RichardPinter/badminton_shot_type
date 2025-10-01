#!/usr/bin/env python3
"""
ABOUTME: Result formatting utilities for unified badminton analysis
ABOUTME: Standardizes outputs from BST and LSTM systems for consistent UI display
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

# Import using absolute file paths since we're using importlib
import sys
from pathlib import Path
import importlib.util

def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load dependencies
current_dir = Path(__file__).parent
bst_wrapper_module = _load_module_from_path("bst_wrapper", current_dir / "bst_wrapper.py")
lstm_wrapper_module = _load_module_from_path("lstm_wrapper", current_dir / "lstm_wrapper.py")

BSTResult = bst_wrapper_module.BSTResult
LSTMResult = lstm_wrapper_module.LSTMResult

@dataclass
class UnifiedAnalysisResult:
    """Combined results from both analysis systems"""
    # Meta information
    video_filename: str
    total_processing_time: float
    timestamp: str

    # BST Transformer results
    bst_result: BSTResult
    bst_available: bool

    # LSTM results
    lstm_result: LSTMResult
    lstm_available: bool

    # Combined status
    overall_success: bool
    has_active_results: bool

class ResultFormatter:
    """
    Formats and standardizes results from both analysis systems.

    Provides consistent data structures for UI components and
    handles display formatting for both active and placeholder results.
    """

    def __init__(self):
        pass

    def combine_results(
        self,
        video_filename: str,
        bst_result: BSTResult,
        lstm_result: LSTMResult,
        total_time: float
    ) -> UnifiedAnalysisResult:
        """
        Combine results from both analysis systems.

        Args:
            video_filename: Name of analyzed video
            bst_result: Results from BST Transformer
            lstm_result: Results from LSTM system
            total_time: Total processing time

        Returns:
            UnifiedAnalysisResult with combined data
        """
        return UnifiedAnalysisResult(
            video_filename=video_filename,
            total_processing_time=total_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            bst_result=bst_result,
            bst_available=bst_result.success,
            lstm_result=lstm_result,
            lstm_available=lstm_result.success,
            overall_success=bst_result.success or lstm_result.success,
            has_active_results=bst_result.success or lstm_result.success
        )

    def format_predictions_for_display(
        self,
        predictions: List[Dict[str, Any]],
        system_name: str,
        is_active: bool = True
    ) -> Dict[str, Any]:
        """
        Format prediction results for UI display.

        Args:
            predictions: List of predictions from system
            system_name: Name of the system (BST/LSTM)
            is_active: Whether this system is actively running

        Returns:
            Formatted data for UI components
        """
        if not predictions or not is_active:
            return self._get_placeholder_display(system_name)

        # Ensure we have exactly 3 predictions for consistent display
        display_predictions = predictions[:3]
        while len(display_predictions) < 3:
            display_predictions.append({
                'rank': len(display_predictions) + 1,
                'class': 'N/A',
                'confidence': 0.0,
                'percentage': 0.0
            })

        # Format for confidence bars
        confidence_bars = []
        for pred in display_predictions:
            confidence = pred['confidence']
            percentage = pred['percentage']

            # Color coding based on rank and confidence
            if pred['rank'] == 1 and confidence > 0.5:
                color = "#28a745"  # Green for high confidence primary
            elif pred['rank'] == 1:
                color = "#ffc107"  # Yellow for low confidence primary
            else:
                color = "#6c757d"  # Gray for secondary predictions

            confidence_bars.append({
                'rank': pred['rank'],
                'class': pred['class'],
                'confidence': confidence,
                'percentage': percentage,
                'bar_width': min(100, max(1, percentage)),  # Ensure visible bar
                'color': color,
                'display_text': f"{percentage:.1f}%" if percentage > 0 else "0%"
            })

        return {
            'system_name': system_name,
            'is_active': is_active,
            'predictions': display_predictions,
            'confidence_bars': confidence_bars,
            'primary_prediction': display_predictions[0],
            'has_results': True
        }

    def _get_placeholder_display(self, system_name: str) -> Dict[str, Any]:
        """Get placeholder display for inactive systems"""
        placeholder_predictions = [
            {'rank': 1, 'class': 'Model Weights', 'confidence': 0.0, 'percentage': 0.0},
            {'rank': 2, 'class': 'Required For', 'confidence': 0.0, 'percentage': 0.0},
            {'rank': 3, 'class': 'Analysis', 'confidence': 0.0, 'percentage': 0.0}
        ]

        confidence_bars = []
        for pred in placeholder_predictions:
            confidence_bars.append({
                'rank': pred['rank'],
                'class': pred['class'],
                'confidence': 0.0,
                'percentage': 0.0,
                'bar_width': 1,  # Minimal visible bar
                'color': "#dee2e6",  # Light gray for placeholder
                'display_text': "N/A"
            })

        return {
            'system_name': system_name,
            'is_active': False,
            'predictions': placeholder_predictions,
            'confidence_bars': confidence_bars,
            'primary_prediction': placeholder_predictions[0],
            'has_results': False
        }

    def format_processing_info(self, result: UnifiedAnalysisResult) -> Dict[str, Any]:
        """Format processing information for display"""
        return {
            'video_file': result.video_filename,
            'total_time': f"{result.total_processing_time:.2f}s",
            'timestamp': result.timestamp,
            'bst_time': f"{result.bst_result.processing_time:.2f}s" if result.bst_result else "N/A",
            'lstm_time': f"{result.lstm_result.processing_time:.2f}s" if result.lstm_result else "N/A",
            'systems_active': {
                'bst': result.bst_available,
                'lstm': result.lstm_available
            }
        }

    def format_technical_details(self, result: UnifiedAnalysisResult) -> Dict[str, Any]:
        """Format technical analysis details"""
        bst_details = {}
        if result.bst_result and result.bst_result.technical_details:
            bst_details = result.bst_result.technical_details

        lstm_details = {}
        if result.lstm_result and result.lstm_result.movement_analysis:
            lstm_details = result.lstm_result.movement_analysis

        return {
            'bst_technical': {
                'available': result.bst_available,
                'components': ['MMPose', 'TrackNetV3', 'BST-8 Transformer'],
                'details': bst_details
            },
            'lstm_technical': {
                'available': result.lstm_available,
                'components': ['YOLO Pose', 'Movement Analysis', 'LSTM Classification'],
                'details': lstm_details
            }
        }

    def format_comparison_data(self, result: UnifiedAnalysisResult) -> Optional[Dict[str, Any]]:
        """Format data for comparing both systems' results"""
        if not (result.bst_available and result.lstm_available):
            return None

        # Extract top predictions from both systems
        bst_top = result.bst_result.primary_prediction
        lstm_top = result.lstm_result.primary_prediction

        return {
            'bst_prediction': bst_top,
            'bst_confidence': result.bst_result.confidence,
            'lstm_prediction': lstm_top,
            'lstm_confidence': result.lstm_result.confidence,
            'agreement': bst_top.lower() == lstm_top.lower(),
            'confidence_diff': abs(result.bst_result.confidence - result.lstm_result.confidence)
        }

    def export_results_json(self, result: UnifiedAnalysisResult) -> Dict[str, Any]:
        """Export results in JSON format for download/sharing"""
        return {
            'analysis_metadata': {
                'video_filename': result.video_filename,
                'timestamp': result.timestamp,
                'total_processing_time': result.total_processing_time,
                'systems_used': {
                    'bst_transformer': result.bst_available,
                    'lstm_classifier': result.lstm_available
                }
            },
            'bst_results': {
                'success': result.bst_result.success,
                'primary_prediction': result.bst_result.primary_prediction,
                'confidence': result.bst_result.confidence,
                'top_3_predictions': result.bst_result.predictions,
                'processing_time': result.bst_result.processing_time,
                'error_message': result.bst_result.error_message
            } if result.bst_result else None,
            'lstm_results': {
                'success': result.lstm_result.success,
                'primary_prediction': result.lstm_result.primary_prediction,
                'confidence': result.lstm_result.confidence,
                'performance_grade': result.lstm_result.performance_grade,
                'grade_score': result.lstm_result.grade_score,
                'top_3_predictions': result.lstm_result.predictions,
                'processing_time': result.lstm_result.processing_time,
                'movement_analysis': result.lstm_result.movement_analysis,
                'error_message': result.lstm_result.error_message
            } if result.lstm_result else None
        }