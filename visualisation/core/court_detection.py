#!/usr/bin/env python3
"""
ABOUTME: Court detection utilities for badminton analysis
ABOUTME: Provides functions to detect court boundaries and check if points are inside the court
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from shapely.geometry import Point, Polygon


def is_point_inside_court(
    point: Tuple[float, float],
    court_corners: List[Tuple[int, int]]
) -> bool:
    """
    Check if a point is inside the court boundary.

    Args:
        point: (x, y) coordinates to check
        court_corners: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    Returns:
        True if point is inside court, False otherwise
    """
    if not court_corners or len(court_corners) != 4:
        return True  # If no court defined, accept all points

    try:
        court_polygon = Polygon(court_corners)
        pt = Point(point[0], point[1])
        return court_polygon.contains(pt)
    except Exception:
        return True  # On error, accept the point


def extract_frame_for_court_selection(
    video_path: Path,
    frame_number: int = 30
) -> Optional[np.ndarray]:
    """
    Extract a frame from video for court corner selection.

    Args:
        video_path: Path to video file
        frame_number: Frame index to extract (default: 30, ~1 second in)

    Returns:
        Frame as numpy array (H, W, 3) or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    # Jump to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    return frame


def select_court_corners_interactive(
    video_path: Path,
    frame_number: int = 30
) -> Optional[List[Tuple[int, int]]]:
    """
    Interactive court corner selection using OpenCV window.
    User clicks 4 corners of the court in order.

    Args:
        video_path: Path to video file
        frame_number: Frame to use for selection

    Returns:
        List of 4 corner points or None if cancelled
    """
    frame = extract_frame_for_court_selection(video_path, frame_number)
    if frame is None:
        print("Error: Could not extract frame for court selection")
        return None

    corners = []
    display_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal corners, display_frame

        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            # Draw the clicked point
            cv2.circle(display_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display_frame, str(len(corners)), (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw lines between corners
            if len(corners) > 1:
                cv2.line(display_frame, corners[-2], corners[-1], (0, 255, 0), 2)

            # Close the polygon when 4 corners are selected
            if len(corners) == 4:
                cv2.line(display_frame, corners[-1], corners[0], (0, 255, 0), 2)
                cv2.polylines(display_frame, [np.array(corners)], True, (0, 255, 0), 2)

            cv2.imshow('Court Corner Selection', display_frame)

    cv2.namedWindow('Court Corner Selection')
    cv2.setMouseCallback('Court Corner Selection', mouse_callback)

    print("\n=== Court Corner Selection ===")
    print("Click on the 4 corners of the court in order (e.g., top-left, top-right, bottom-right, bottom-left)")
    print("Press 'r' to reset")
    print("Press 'q' to cancel")
    print("Press ENTER when done (after selecting 4 corners)")

    while True:
        cv2.imshow('Court Corner Selection', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Cancelled
            cv2.destroyAllWindows()
            return None
        elif key == ord('r'):
            # Reset
            corners = []
            display_frame = frame.copy()
        elif key == 13 or key == 10:  # ENTER key
            if len(corners) == 4:
                cv2.destroyAllWindows()
                return corners
            else:
                print(f"Please select 4 corners (currently {len(corners)} selected)")

    cv2.destroyAllWindows()
    return None


def auto_detect_court(frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    Attempt to automatically detect court boundaries using line detection.
    This is a basic implementation and may not work for all videos.

    Args:
        frame: Video frame

    Returns:
        List of 4 corner points or None if detection failed
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) < 4:
        return None

    # This is a simplified approach - would need more sophisticated logic
    # to identify the actual court boundary lines
    # For now, return None to indicate auto-detection not implemented
    return None


def validate_court_corners(corners: List[Tuple[int, int]]) -> bool:
    """
    Validate that court corners form a reasonable quadrilateral.

    Args:
        corners: List of 4 corner points

    Returns:
        True if valid, False otherwise
    """
    if not corners or len(corners) != 4:
        return False

    # Check that all points are different
    unique_points = set(corners)
    if len(unique_points) != 4:
        return False

    # Check that polygon has reasonable area (not too small)
    try:
        poly = Polygon(corners)
        area = poly.area
        return area > 1000  # Minimum area threshold
    except Exception:
        return False


def get_court_center(corners: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    Calculate the center point of the court.

    Args:
        corners: List of 4 corner points

    Returns:
        (x, y) coordinates of court center
    """
    if not corners or len(corners) != 4:
        return (0, 0)

    x_coords = [c[0] for c in corners]
    y_coords = [c[1] for c in corners]

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    return (center_x, center_y)