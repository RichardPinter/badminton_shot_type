#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extraction Module for Badminton Analysis

This module contains comprehensive functionality for extracting, calculating, and analyzing 
features from badminton player poses detected in video frames. It processes raw keypoint
data to generate meaningful metrics related to player movement, technique, and performance.

Key components:
- Spatial feature extraction (joint angles, relative distances)
- Temporal feature calculation (velocities, accelerations)
- Movement pattern recognition 
- Court positioning and zone analysis

- Stroke classification
- Statistical aggregation of features

The features extracted by this module form the foundation for player performance analysis
and skill grading in the badminton analysis system.

Authors: Sujit Dahal, Kabir G C, Utsab Gyawali
Version: 3.0.0
"""

import cv2
import numpy as np
import math
import logging
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.stats import entropy
from collections import defaultdict, deque
import config as cfg

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Basic Helper Functions ---
def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points in degrees.
    
    Args:
        p1 (np.array): First point coordinates
        p2 (np.array): Second point coordinates (vertex of the angle)
        p3 (np.array): Third point coordinates
        
    Returns:
        float or None: Angle in degrees, or None if any point is invalid or calculation fails
    """
    if any(v is None for v in [p1, p2, p3]):
        return None
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot = np.dot(v1, v2)
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    if n_v1 == 0 or n_v2 == 0:
        return None
    return np.degrees(np.arccos(np.clip(dot / (n_v1 * n_v2), -1.0, 1.0)))

def calculate_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1 (np.array): First point coordinates
        p2 (np.array): Second point coordinates
        
    Returns:
        float or None: Distance between points, or None if any point is invalid
    """
    if p1 is None or p2 is None:
        return None
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_valid_keypoint(keypoints_data, kp_name_or_idx, min_conf=0.3):
    """
    Extract valid keypoint coordinates from pose data if confidence exceeds threshold.
    
    Args:
        keypoints_data (np.array): Array of keypoint data with shape (num_keypoints, 3)
                                  where each row is [x, y, confidence]
        kp_name_or_idx (str or int): Keypoint name or index
        min_conf (float): Minimum confidence threshold (0-1)
        
    Returns:
        np.array or None: [x, y] coordinates if valid, None otherwise
    """
    idx = kp_name_or_idx if isinstance(kp_name_or_idx, int) else cfg.PoseDefinitions.KP_N[kp_name_or_idx]
    if idx < len(keypoints_data) and keypoints_data[idx, 2] >= min_conf and \
       not (keypoints_data[idx, 0] == 0 and keypoints_data[idx, 1] == 0):
        return keypoints_data[idx, :2]
    return None

def get_torso_height_proxy(keypoints_data):
    """
    Calculate an estimate of the player's torso height for normalization.
    Tries multiple approaches based on available keypoints.
    
    Args:
        keypoints_data (np.array): Array of keypoint data
        
    Returns:
        float or None: Estimated torso height, or None if calculation is not possible
    """
    ls = get_valid_keypoint(keypoints_data, "left_shoulder")
    rs = get_valid_keypoint(keypoints_data, "right_shoulder")
    lh = get_valid_keypoint(keypoints_data, "left_hip")
    rh = get_valid_keypoint(keypoints_data, "right_hip")
    
    # Try multiple approaches to get a stable torso height
    if ls is not None and rs is not None and lh is not None and rh is not None:
        # Best case: all torso keypoints are visible
        mid_s = (np.array(ls) + np.array(rs)) / 2
        mid_h = (np.array(lh) + np.array(rh)) / 2
        dist = calculate_distance(mid_s, mid_h)
        return dist if dist and dist > 10 else None
    elif ls is not None and lh is not None:
        # Left side of torso visible
        dist = calculate_distance(ls, lh)
        return dist if dist and dist > 10 else None
    elif rs is not None and rh is not None:
        # Right side of torso visible
        dist = calculate_distance(rs, rh)
        return dist if dist and dist > 10 else None
    return None

def get_player_position_on_court(keypoints_data):
    """
    Determine the player's position on the court based on ankle positions.
    Falls back to hip positions if ankles are not visible.
    
    Args:
        keypoints_data (np.array): Array of keypoint data
        
    Returns:
        np.array or None: [x, y] coordinates of player's position, or None if calculation is not possible
    """
    # Try ankles first - best indicator of ground position
    la = get_valid_keypoint(keypoints_data, "left_ankle")
    ra = get_valid_keypoint(keypoints_data, "right_ankle")
    if la is not None and ra is not None:
        return (np.array(la) + np.array(ra)) / 2
    if la is not None:
        return np.array(la)
    if ra is not None:
        return np.array(ra)
    
    # Fall back to hips if ankles not visible
    lh = get_valid_keypoint(keypoints_data, "left_hip")
    rh = get_valid_keypoint(keypoints_data, "right_hip")
    if lh is not None and rh is not None:
        return (np.array(lh) + np.array(rh)) / 2
    if lh is not None:
        return np.array(lh)
    if rh is not None:
        return np.array(rh)
    return None

def get_body_center_for_speed(keypoints_data):
    """
    Get the body center position for speed calculations.
    Prefers hip center but falls back to general position.
    
    Args:
        keypoints_data (np.array): Array of keypoint data
        
    Returns:
        np.array or None: [x, y] coordinates of body center, or None if calculation is not possible
    """
    # Hip center is preferred for speed calculations
    lh = get_valid_keypoint(keypoints_data, "left_hip")
    rh = get_valid_keypoint(keypoints_data, "right_hip")
    if lh is not None and rh is not None:
        return (np.array(lh) + np.array(rh)) / 2
    # Fall back to general position
    return get_player_position_on_court(keypoints_data)

# --- Enhanced Temporal Feature Functions with Unit Scaling ---
def calculate_temporal_velocities(pose_buffer, timestamp_buffer, target_joints):
    """
    Calculate velocities for specified joints using a temporal window with unit scaling.
    
    Args:
        pose_buffer (list): Buffer of pose keypoint data for consecutive frames
        timestamp_buffer (list): Buffer of timestamps corresponding to each pose frame
        target_joints (list): List of joint names for which to calculate velocities
        
    Returns:
        list: Average velocities for each target joint, with appropriate scaling applied.
             Returns np.nan for joints with insufficient data.
    
    Notes:
        - Velocities are calculated between consecutive frames
        - Final values are scaled by VELOCITY_SCALE_FACTOR for appropriate units
        - Requires at least 2 frames of data
    """
    if len(pose_buffer) < 2:
        return [np.nan] * len(target_joints)
    
    velocities = {joint: [] for joint in target_joints}
    buffer_list = list(pose_buffer)
    time_list = list(timestamp_buffer)
    
    for i in range(len(buffer_list) - 1):
        kp_curr = buffer_list[i]
        kp_next = buffer_list[i + 1]
        time_curr = time_list[i]
        time_next = time_list[i + 1]
        
        delta_t = time_next - time_curr
        if delta_t <= 1e-6:
            continue  # Skip if time difference is too small
            
        for joint_name in target_joints:
            if joint_name in ['wrist_L', 'wrist_R', 'elbow_L', 'elbow_R', 
                             'ankle_L', 'ankle_R', 'knee_L', 'knee_R']:
                joint_idx = cfg.PoseDefinitions.TARGET_VEL_ACC_KEYPOINTS[joint_name]
            else:
                continue
                
            pt_curr = get_valid_keypoint(kp_curr, joint_idx)
            pt_next = get_valid_keypoint(kp_next, joint_idx)
            
            if pt_curr is not None and pt_next is not None:
                dist = calculate_distance(pt_curr, pt_next)
                if dist is not None:
                    # Apply scaling to normalize values
                    velocity = (dist / delta_t) * cfg.TemporalFeatureBuffer.VELOCITY_SCALE_FACTOR
                    velocities[joint_name].append(velocity)
    
    # Return average velocities for each joint
    avg_velocities = []
    for joint in target_joints:
        joint_vels = velocities[joint]
        if joint_vels:
            avg_velocities.append(np.mean(joint_vels))
        else:
            avg_velocities.append(np.nan)
    
    return avg_velocities

def calculate_temporal_accelerations(pose_buffer, timestamp_buffer, target_joints):
    """
    Calculate accelerations for specified joints using a temporal window with unit scaling.
    
    Args:
        pose_buffer (list): Buffer of pose keypoint data for consecutive frames
        timestamp_buffer (list): Buffer of timestamps corresponding to each pose frame
        target_joints (list): List of joint names for which to calculate accelerations
        
    Returns:
        list: Average accelerations for each target joint, with appropriate scaling applied.
             Returns np.nan for joints with insufficient data.
    
    Notes:
        - Accelerations are calculated by measuring velocity changes between frames
        - Final values are scaled by ACCELERATION_SCALE_FACTOR for appropriate units
        - Requires at least 3 frames of data for meaningful acceleration calculation
    """
    if len(pose_buffer) < 3:
        return [np.nan] * len(target_joints)
    
    accelerations = {joint: [] for joint in target_joints}
    buffer_list = list(pose_buffer)
    time_list = list(timestamp_buffer)
    
    # Calculate instantaneous velocities first
    inst_velocities = {joint: [] for joint in target_joints}
    vel_timestamps = []
    
    for i in range(len(buffer_list) - 1):
        kp_curr = buffer_list[i]
        kp_next = buffer_list[i + 1]
        time_curr = time_list[i]
        time_next = time_list[i + 1]
        
        delta_t = time_next - time_curr
        if delta_t <= 1e-6:
            continue
            
        mid_time = (time_curr + time_next) / 2.0
        vel_timestamps.append(mid_time)
        
        for joint_name in target_joints:
            if joint_name in ['wrist_L', 'wrist_R', 'ankle_L', 'ankle_R']:
                joint_idx = cfg.PoseDefinitions.TARGET_VEL_ACC_KEYPOINTS[joint_name]
            else:
                continue
                
            pt_curr = get_valid_keypoint(kp_curr, joint_idx)
            pt_next = get_valid_keypoint(kp_next, joint_idx)
            velocity = np.nan
            if pt_curr is not None and pt_next is not None:
                dist = calculate_distance(pt_curr, pt_next)
                if dist is not None:
                    velocity = (dist / delta_t) * cfg.TemporalFeatureBuffer.VELOCITY_SCALE_FACTOR
            inst_velocities[joint_name].append(velocity)
    
    # Calculate accelerations from consecutive velocities
    if len(vel_timestamps) < 2:
        return [np.nan] * len(target_joints)
    
    for i in range(len(vel_timestamps) - 1):
        time_vel_curr = vel_timestamps[i]
        time_vel_next = vel_timestamps[i + 1]
        delta_t_vel = time_vel_next - time_vel_curr
        
        if delta_t_vel <= 1e-6:
            continue
            
        for joint in target_joints:
            if joint not in inst_velocities:
                continue
                
            vel_curr = inst_velocities[joint][i]
            vel_next = inst_velocities[joint][i + 1]
            
            if not np.isnan(vel_curr) and not np.isnan(vel_next):
                delta_v = vel_next - vel_curr
                acceleration = (delta_v / delta_t_vel) * cfg.TemporalFeatureBuffer.ACCELERATION_SCALE_FACTOR
                accelerations[joint].append(acceleration)
    
    # Return max and mean_positive like sample JSON
    results = []
    for joint in target_joints:
        joint_accels = accelerations.get(joint, [])
        if joint_accels:
            valid_accels = [a for a in joint_accels if not np.isnan(a)]
            if valid_accels:
                positive_accels = [a for a in valid_accels if a > 0]
                results.append({
                    'max': np.max(np.abs(valid_accels)),
                    'mean_positive': np.mean(positive_accels) if positive_accels else np.nan
                })
            else:
                results.append({'max': np.nan, 'mean_positive': np.nan})
        else:
            results.append({'max': np.nan, 'mean_positive': np.nan})
    
    return results

def calculate_dtw_features(spatial_feature_buffer, stride=cfg.TemporalFeatureBuffer.DEFAULT_TEMPORAL_STRIDE):
    """Calculate DTW features from spatial feature buffer with scaling"""
    num_expected = cfg.TemporalFeatureBuffer.NUM_DTW_FEATURES
    if len(spatial_feature_buffer) < 2:
        return [np.nan] * num_expected
    
    buffer_list = list(spatial_feature_buffer)
    # Ensure we have 1D arrays for FastDTW
    # reference_features = np.array(buffer_list[0], dtype=float).flatten()
    # reference_features = np.nan_to_num(reference_features, nan=0.0)
    # # Make sure array is contiguous in memory
    # reference_features = np.ascontiguousarray(reference_features)

    # If comparing single feature vectors, treat them as sequences of length 1
    # Each feature vector itself is 1D, e.g., [x, y] coordinates.
    # fastdtw expects a sequence of such vectors, so for a single vector,
    # it should be np.array([[x, y]])
    ref_feat_vector = np.array(buffer_list[0], dtype=float)
    ref_feat_vector = np.nan_to_num(ref_feat_vector, nan=0.0)

    if ref_feat_vector.ndim == 1:
        reference_features_for_dtw = np.ascontiguousarray(ref_feat_vector[np.newaxis, :])
    else: # Should not happen if extract_spatial_features is correct
        logger.warning("Reference feature vector is not 1D, using as is for DTW.")
        reference_features_for_dtw = np.ascontiguousarray(ref_feat_vector)
    
    if reference_features_for_dtw.shape[1] == 0: # Check if inner dimension is zero
        return [np.nan] * num_expected
    
    dtw_distances = []
    num_calculated = 0
    
    for i in range(stride, len(buffer_list), stride):
        if num_calculated >= num_expected:
            break
            
        # Ensure current features are also 1D and contiguous
        # current_features = np.array(buffer_list[i], dtype=float).flatten()
        # current_features = np.nan_to_num(current_features, nan=0.0)
        # current_features = np.ascontiguousarray(current_features)

        curr_feat_vector = np.array(buffer_list[i], dtype=float)
        curr_feat_vector = np.nan_to_num(curr_feat_vector, nan=0.0)

        if curr_feat_vector.ndim == 1:
            current_features_for_dtw = np.ascontiguousarray(curr_feat_vector[np.newaxis, :])
        else: # Should not happen
            logger.warning("Current feature vector is not 1D, using as is for DTW.")
            current_features_for_dtw = np.ascontiguousarray(curr_feat_vector)

        if current_features_for_dtw.shape[1] == 0 or reference_features_for_dtw.shape[1] != current_features_for_dtw.shape[1]:
            dtw_distances.append(np.nan)
        else:
            try:
                # Verify arrays are 2D (sequence of 1D vectors) before passing to fastdtw
                if reference_features_for_dtw.ndim != 2 or current_features_for_dtw.ndim != 2:
                    logger.error(f"DTW input not 2D: ref_dim={reference_features_for_dtw.ndim}, cur_dim={current_features_for_dtw.ndim}")
                    dtw_distances.append(np.nan)
                else:
                    distance, _ = fastdtw(reference_features_for_dtw, current_features_for_dtw, dist=euclidean)
                    # Scale DTW distance to match sample
                    dtw_distances.append(float(distance) * 0.01)
            except Exception as e:
                logger.error(f"Error calculating FastDTW: {e}")
                # Try simple Euclidean distance as fallback
                # This fallback should compare the original 1D vectors directly
                try:
                    fallback_distance = np.linalg.norm(ref_feat_vector - curr_feat_vector)
                    dtw_distances.append(float(fallback_distance) * 0.01)
                    logger.info("Using Euclidean distance as fallback for DTW")
                except:
                    dtw_distances.append(np.nan)
        
        num_calculated += 1
    
    # Pad with NaN if needed
    while len(dtw_distances) < num_expected:
        dtw_distances.append(np.nan)
    
    return dtw_distances[:num_expected]

def calculate_movement_patterns(pose_buffer):
    """Calculate movement patterns for key joints"""
    if len(pose_buffer) < 3:
        return []
    
    pattern_features = []
    
    for joint_name in cfg.PoseDefinitions.MOVEMENT_PATTERN_JOINTS:
        joint_idx = cfg.PoseDefinitions.KP_N[joint_name]
        trajectory = []
        
        for pose_data in pose_buffer:
            kp = get_valid_keypoint(pose_data, joint_idx)
            if kp is not None:
                trajectory.append(kp)
        
        valid_points = trajectory
        
        # Calculate total distance and direction changes
        joint_dist = 0.0
        joint_dir_changes = 0
        
        if len(valid_points) >= 2:
            try:
                for i in range(len(valid_points) - 1):
                    dist = calculate_distance(valid_points[i], valid_points[i+1])
                    if dist is not None:
                        joint_dist += dist
                
                # Scale distance to match sample
                joint_dist *= cfg.TemporalFeatureBuffer.VELOCITY_SCALE_FACTOR
                
                # Calculate direction changes
                if len(valid_points) >= 3:
                    for i in range(len(valid_points) - 2):
                        vec1 = np.array(valid_points[i+1]) - np.array(valid_points[i])
                        vec2 = np.array(valid_points[i+2]) - np.array(valid_points[i+1])
                        
                        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                            angle = np.arccos(np.clip(np.dot(vec1, vec2) / 
                                            (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1, 1))
                            if np.degrees(angle) > 45:
                                joint_dir_changes += 1
                
            except Exception as e:
                logger.error(f"Error calculating pattern features for {joint_name}: {e}")
        
        pattern_features.extend([joint_dist, joint_dir_changes])
    
    return pattern_features

def calculate_body_speed_and_acceleration(center_position_buffer, timestamp_buffer):
    """Calculate body speed and acceleration using temporal window with scaling"""
    if len(center_position_buffer) < 2:
        return [np.nan, np.nan]
    
    center_list = list(center_position_buffer)
    time_list = list(timestamp_buffer)
    
    # Filter out invalid center positions
    valid_centers = []
    valid_times = []
    for i, center in enumerate(center_list):
        if center is not None and not np.isnan(center).any():
            valid_centers.append(center)
            valid_times.append(time_list[i])
    
    if len(valid_centers) < 2:
        return [np.nan, np.nan]
    
    speeds = []
    mid_times = []
    
    # Calculate instantaneous speeds
    for i in range(len(valid_centers) - 1):
        pos_curr = valid_centers[i]
        pos_next = valid_centers[i + 1]
        time_curr = valid_times[i]
        time_next = valid_times[i + 1]
        
        delta_t = time_next - time_curr
        if delta_t <= 1e-6:
            continue
        
        dist = calculate_distance(pos_curr, pos_next)
        if dist is not None:
            speed = (dist / delta_t) * cfg.TemporalFeatureBuffer.VELOCITY_SCALE_FACTOR
            speeds.append(speed)
            mid_times.append((time_curr + time_next) / 2.0)
    
    avg_speed = np.mean(speeds) if speeds else np.nan
    max_speed = np.max(speeds) if speeds else np.nan
    
    # Calculate acceleration
    accelerations = []
    if len(speeds) >= 2:
        for i in range(len(speeds) - 1):
            speed_curr = speeds[i]
            speed_next = speeds[i + 1]
            time_curr = mid_times[i]
            time_next = mid_times[i + 1]
            
            delta_t_speed = time_next - time_curr
            if delta_t_speed <= 1e-6:
                continue
            
            delta_v = speed_next - speed_curr
            acceleration = (delta_v / delta_t_speed) * cfg.TemporalFeatureBuffer.ACCELERATION_SCALE_FACTOR
            accelerations.append(acceleration)
    
    avg_acceleration = np.mean(accelerations) if accelerations else np.nan
    
    # Return in format matching sample JSON (avg, max)
    return [avg_speed, max_speed, avg_acceleration]

def get_court_zone(position, court_polygon, num_zones=9):
    """Get court zone (1-9) based on position"""
    if position is None or court_polygon is None:
        return None
    
    # Get court bounds
    x_coords = court_polygon[:, 0]
    y_coords = court_polygon[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Calculate zone dimensions (3x3 grid)
    zone_width = (x_max - x_min) / 3
    zone_height = (y_max - y_min) / 3
    
    # Get zone indices
    x_idx = int((position[0] - x_min) / zone_width)
    y_idx = int((position[1] - y_min) / zone_height)
    
    # Clamp to valid range
    x_idx = max(0, min(2, x_idx))
    y_idx = max(0, min(2, y_idx))
    
    # Return zone number (1-9)
    return y_idx * 3 + x_idx + 1

def extract_spatial_features(keypoints_data, frame_width=None, frame_height=None):
    """Extract all spatial features"""
    features = []
    
    # For normalized court position, we need frame dimensions
    pos = get_player_position_on_court(keypoints_data)
    if pos is not None and frame_width and frame_height:
        # Normalize to 0-1 range
        norm_x = pos[0] / frame_width
        norm_y = pos[1] / frame_height
        features.extend([norm_x, norm_y])
    else:
        features.extend([np.nan, np.nan])
    
    return features

# --- Stroke Classification (Mock) ---
def classify_stroke(wrist_velocity, wrist_height, court_zone):
    """Mock stroke classification based on velocity and position"""
    if np.isnan(wrist_velocity):
        return 'unknown'
    
    # Simple heuristic-based classification
    if wrist_velocity > 5.0:  # High velocity
        return 'smash'
    elif wrist_velocity > 3.0:
        if wrist_height > 0.7:  # High position
            return 'clear'
        else:
            return 'drive'
    elif wrist_velocity > 1.5:
        if court_zone in [1, 2, 3]:  # Front court
            return 'net'
        else:
            return 'drop'
    else:
        return 'lift'

# --- Final JSON Generation Functions ---
def aggregate_stats(data_list_vals):
    if not data_list_vals:
        return {}
    data_list_valid = [x for x in data_list_vals if x is not None and not math.isnan(x) and not math.isinf(x)]
    if not data_list_valid:
        return {}
    return {
        "mean": float(np.mean(data_list_valid)),
        "std": float(np.std(data_list_valid)),
        "min": float(np.min(data_list_valid)),
        "max": float(np.max(data_list_valid))
    }

def aggregate_std_only(data_list_vals):
    if not data_list_vals:
        return {}
    data_list_valid = [x for x in data_list_vals if x is not None and not math.isnan(x) and not math.isinf(x)]
    if not data_list_valid:
        return {}
    return {
        "mean": float(np.mean(data_list_valid)),
        "std": float(np.std(data_list_valid))
    }