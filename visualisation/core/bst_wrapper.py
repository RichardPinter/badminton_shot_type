#!/usr/bin/env python3
"""
ABOUTME: BST Transformer wrapper for unified badminton analysis
ABOUTME: Integrates existing BST pipeline into the dual-method system
"""

import sys
import os
import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

# Add BST model path to Python path (now points to parent badminton-stroke-classifier)
BST_MODEL_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BST_MODEL_PATH))

try:
    from pipeline.file_based_pipeline import FileBased_Pipeline, PipelineResult
    BST_AVAILABLE = True
except ImportError as e:
    BST_AVAILABLE = False
    BST_IMPORT_ERROR = str(e)

@dataclass
class BSTResult:
    """Standardized BST analysis result"""
    success: bool
    predictions: List[Dict[str, Any]]  # Top 3 predictions
    primary_prediction: str
    confidence: float
    processing_time: float
    error_message: Optional[str] = None
    technical_details: Optional[Dict] = None

class BSTWrapper:
    """
    Wrapper for BST Transformer stroke classification system.

    Integrates the existing FileBased_Pipeline into the unified interface
    while maintaining compatibility with the original BST system.
    """

    def __init__(self):
        """Initialize BST wrapper"""
        self.logger = logging.getLogger(__name__)
        self.pipeline = None
        self.is_initialized = False

        if not BST_AVAILABLE:
            self.logger.error(f"BST system not available: {BST_IMPORT_ERROR}")
            return

        try:
            self.pipeline = FileBased_Pipeline()
            self.is_initialized = True
            self.logger.info("BST Transformer pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize BST pipeline: {e}")
            self.is_initialized = False

    def is_ready(self) -> bool:
        """Check if BST system is ready for analysis"""
        return self.is_initialized and BST_AVAILABLE

    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information"""
        if not BST_AVAILABLE:
            return {
                'available': False,
                'error': f"BST import failed: {BST_IMPORT_ERROR}",
                'ready': False
            }

        return {
            'available': BST_AVAILABLE,
            'initialized': self.is_initialized,
            'ready': self.is_ready(),
            'model_components': [
                'MMPose (Pose Detection)',
                'TrackNetV3 (Shuttlecock Tracking)',
                'BST-8 Transformer (Stroke Classification)'
            ]
        }

    async def analyze_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> BSTResult:
        """
        Analyze video using BST Transformer pipeline.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates

        Returns:
            BSTResult with analysis results
        """
        if not self.is_ready():
            return BSTResult(
                success=False,
                predictions=[],
                primary_prediction="N/A",
                confidence=0.0,
                processing_time=0.0,
                error_message="BST system not ready"
            )

        try:
            start_time = time.time()

            # Update progress
            if progress_callback:
                progress_callback("🤖 Starting BST Transformer analysis...", 0.1)

            # Run BST pipeline
            if progress_callback:
                progress_callback("🔍 Running MMPose pose detection...", 0.3)

            # Execute the pipeline (runs synchronously)
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_pipeline_sync, video_path, progress_callback
            )

            processing_time = time.time() - start_time

            if result.success:
                # Convert to standardized format
                predictions = self._format_predictions(result)

                return BSTResult(
                    success=True,
                    predictions=predictions,
                    primary_prediction=result.stroke_prediction or "Unknown",
                    confidence=result.confidence or 0.0,
                    processing_time=processing_time,
                    technical_details={
                        'pipeline_time': result.processing_time,
                        'pose_file': str(result.pose_file) if result.pose_file else None,
                        'tracknet_csv': str(result.tracknet_csv) if result.tracknet_csv else None,
                        'tracknet_video': str(result.tracknet_video) if result.tracknet_video else None
                    }
                )
            else:
                return BSTResult(
                    success=False,
                    predictions=[],
                    primary_prediction="N/A",
                    confidence=0.0,
                    processing_time=processing_time,
                    error_message=result.error_message or "Unknown error"
                )

        except Exception as e:
            self.logger.error(f"BST analysis error: {e}")
            return BSTResult(
                success=False,
                predictions=[],
                primary_prediction="N/A",
                confidence=0.0,
                processing_time=0.0,
                error_message=f"Analysis failed: {str(e)}"
            )

    def _run_pipeline_sync(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> PipelineResult:
        """Run the BST pipeline synchronously"""
        try:
            # Create progress callback wrapper
            def progress_wrapper(message: str, progress: float):
                if progress_callback:
                    progress_callback(f"🚀 {message}", progress)

            # Run the pipeline
            result = self.pipeline.run_single_video(
                video_path=video_path,
                progress_cb=progress_wrapper
            )

            return result

        except Exception as e:
            self.logger.error(f"Pipeline execution error: {e}")
            return PipelineResult(
                success=False,
                error_message=str(e)
            )

    def _format_predictions(self, result: PipelineResult) -> List[Dict[str, Any]]:
        """Format BST predictions for unified display"""
        predictions = []

        if result.top3_predictions:
            for i, pred in enumerate(result.top3_predictions):
                predictions.append({
                    'rank': i + 1,
                    'class': pred['class'],
                    'confidence': float(pred['confidence']),
                    'percentage': float(pred['confidence']) * 100
                })
        elif result.stroke_prediction and result.confidence:
            # Fallback if only primary prediction available
            predictions.append({
                'rank': 1,
                'class': result.stroke_prediction,
                'confidence': result.confidence,
                'percentage': result.confidence * 100
            })

        return predictions

    def get_placeholder_result(self) -> BSTResult:
        """Get placeholder result when BST is not available"""
        return BSTResult(
            success=False,
            predictions=[
                {'rank': 1, 'class': 'Model Loading...', 'confidence': 0.0, 'percentage': 0.0},
                {'rank': 2, 'class': 'Weights Required', 'confidence': 0.0, 'percentage': 0.0},
                {'rank': 3, 'class': 'Please Wait...', 'confidence': 0.0, 'percentage': 0.0}
            ],
            primary_prediction="System Loading",
            confidence=0.0,
            processing_time=0.0,
            error_message="BST Transformer weights not loaded"
        )