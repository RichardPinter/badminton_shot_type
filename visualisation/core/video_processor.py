#!/usr/bin/env python3
"""
ABOUTME: Unified video processor for dual-method badminton analysis
ABOUTME: Handles video upload, validation, and preparation for both BST and LSTM systems
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import cv2
import streamlit as st
from dataclasses import dataclass

@dataclass
class VideoInfo:
    """Information about processed video"""
    path: str
    duration: float
    fps: int
    frame_count: int
    width: int
    height: int
    size_mb: float
    is_valid: bool
    error_message: Optional[str] = None

class UnifiedVideoProcessor:
    """
    Unified video processing for both BST Transformer and LSTM analysis systems.

    Handles video upload, validation, format conversion, and preparation
    for downstream analysis pipelines.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize video processor with optional temporary directory"""
        self.logger = logging.getLogger(__name__)
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.m4v']

        # Video validation constraints
        self.min_duration = 0.5  # seconds
        self.max_duration = 60.0  # seconds
        self.max_file_size = 500  # MB

    def validate_video(self, video_path: str) -> VideoInfo:
        """
        Validate video file for analysis compatibility.

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo object with validation results
        """
        try:
            if not os.path.exists(video_path):
                return VideoInfo(
                    path=video_path, duration=0, fps=0, frame_count=0,
                    width=0, height=0, size_mb=0, is_valid=False,
                    error_message="Video file does not exist"
                )

            # Check file size
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            if size_mb > self.max_file_size:
                return VideoInfo(
                    path=video_path, duration=0, fps=0, frame_count=0,
                    width=0, height=0, size_mb=size_mb, is_valid=False,
                    error_message=f"File too large: {size_mb:.1f}MB (max {self.max_file_size}MB)"
                )

            # Check file extension
            ext = Path(video_path).suffix.lower()
            if ext not in self.supported_formats:
                return VideoInfo(
                    path=video_path, duration=0, fps=0, frame_count=0,
                    width=0, height=0, size_mb=size_mb, is_valid=False,
                    error_message=f"Unsupported format: {ext}. Supported: {self.supported_formats}"
                )

            # Open video and get properties
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return VideoInfo(
                    path=video_path, duration=0, fps=0, frame_count=0,
                    width=0, height=0, size_mb=size_mb, is_valid=False,
                    error_message="Cannot open video file"
                )

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            # Validate duration
            if duration < self.min_duration:
                return VideoInfo(
                    path=video_path, duration=duration, fps=fps, frame_count=frame_count,
                    width=width, height=height, size_mb=size_mb, is_valid=False,
                    error_message=f"Video too short: {duration:.1f}s (min {self.min_duration}s)"
                )

            if duration > self.max_duration:
                return VideoInfo(
                    path=video_path, duration=duration, fps=fps, frame_count=frame_count,
                    width=width, height=height, size_mb=size_mb, is_valid=False,
                    error_message=f"Video too long: {duration:.1f}s (max {self.max_duration}s)"
                )

            # All validations passed
            return VideoInfo(
                path=video_path, duration=duration, fps=fps, frame_count=frame_count,
                width=width, height=height, size_mb=size_mb, is_valid=True
            )

        except Exception as e:
            self.logger.error(f"Video validation error: {e}")
            return VideoInfo(
                path=video_path, duration=0, fps=0, frame_count=0,
                width=0, height=0, size_mb=0, is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )

    def save_uploaded_file(self, uploaded_file) -> str:
        """
        Save Streamlit uploaded file to temporary location.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Path to saved file
        """
        # Create unique filename
        temp_filename = f"video_{uploaded_file.name}"
        temp_path = os.path.join(self.temp_dir, temp_filename)

        # Save file
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        self.logger.info(f"Saved uploaded file to: {temp_path}")
        return temp_path

    def prepare_for_bst(self, video_path: str) -> Dict[str, Any]:
        """
        Prepare video for BST Transformer analysis.

        Args:
            video_path: Path to validated video file

        Returns:
            Dictionary with BST-specific preparation info
        """
        return {
            'video_path': video_path,
            'ready_for_bst': True,
            'preprocessing_needed': False,  # BST handles its own preprocessing
            'notes': 'Video ready for BST pipeline (MMPose + TrackNetV3 + BST-8)'
        }

    def prepare_for_lstm(self, video_path: str) -> Dict[str, Any]:
        """
        Prepare video for LSTM analysis (placeholder for now).

        Args:
            video_path: Path to validated video file

        Returns:
            Dictionary with LSTM preparation info (placeholder)
        """
        return {
            'video_path': video_path,
            'ready_for_lstm': False,  # Placeholder - weights not available
            'preprocessing_needed': True,  # Would need YOLO pose extraction
            'notes': 'LSTM model weights required for analysis'
        }

    def process_video(self, uploaded_file) -> Tuple[VideoInfo, Dict[str, Any]]:
        """
        Main processing pipeline for uploaded video.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Tuple of (VideoInfo, preparation_results)
        """
        try:
            # Save uploaded file
            video_path = self.save_uploaded_file(uploaded_file)

            # Validate video
            video_info = self.validate_video(video_path)

            if not video_info.is_valid:
                return video_info, {'error': video_info.error_message}

            # Prepare for both systems
            bst_prep = self.prepare_for_bst(video_path)
            lstm_prep = self.prepare_for_lstm(video_path)

            preparation_results = {
                'bst': bst_prep,
                'lstm': lstm_prep,
                'video_path': video_path,
                'success': True
            }

            return video_info, preparation_results

        except Exception as e:
            self.logger.error(f"Video processing error: {e}")
            error_info = VideoInfo(
                path="", duration=0, fps=0, frame_count=0,
                width=0, height=0, size_mb=0, is_valid=False,
                error_message=f"Processing error: {str(e)}"
            )
            return error_info, {'error': str(e)}

    def cleanup_temp_files(self, video_path: str):
        """Clean up temporary video files"""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                self.logger.info(f"Cleaned up temporary file: {video_path}")
        except Exception as e:
            self.logger.warning(f"Could not clean up {video_path}: {e}")