# src/mock_data_utils.py
"""
Utility functions for generating mock badminton player feature data.
Used by generate_mock_data.py and generate_prediction_input.py.

Provides functions to create feature dictionaries structured similarly to the
expected output from the primary feature extraction module, based on the
'Achievable Feature List from CCTV Footage' document.

Author: Sujit
"""

import numpy as np
from typing import Dict, Any

# --- Helper Functions (Shared) ---

def _generate_stats_dict(rng: np.random.Generator, base_mean: float, base_std: float,
                         non_negative: bool = True, noise_factor: float = 0.2,
                         min_factor: float = 0.7, max_factor: float = 1.3
                         ) -> Dict[str, float]:
    """Generates a dictionary with mean, std, min, max based on base values.

    Introduces noise to the base mean and standard deviation to simulate variability.
    Ensures min <= mean <= max.

    Args:
        rng: NumPy random number generator instance.
        base_mean: The base mean value for the statistic.
        base_std: The base standard deviation for the statistic.
        non_negative: If True, ensures all generated values are >= 0.
        noise_factor: Controls the amount of noise added to std deviation.
        min_factor: Multiplier for mean to generate a lower bound for min value.
        max_factor: Multiplier for mean to generate an upper bound for max value.

    Returns:
        A dictionary containing 'mean', 'std', 'min', 'max' keys with float values.
    """
    std_dev = max(0.01, base_std + rng.normal(0, base_std * noise_factor))
    mean_val = base_mean + rng.normal(0, std_dev * 0.5)
    min_val = mean_val * rng.uniform(min_factor, 0.98)
    max_val = mean_val * rng.uniform(1.02, max_factor)
    if non_negative:
        mean_val = max(0, mean_val)
        min_val = max(0, min_val)
        max_val = max(0, max_val)
        min_val = min(min_val, mean_val)
        max_val = max(max_val, mean_val)
    return {'mean': mean_val, 'std': std_dev, 'min': min_val, 'max': max_val}

def _generate_single_value(rng: np.random.Generator, base_value: float,
                          noise_factor: float = 0.15, non_negative: bool = True) -> float:
     """Generates a single noisy value around a base.

     Args:
         rng: NumPy random number generator instance.
         base_value: The central value around which to generate the noisy value.
         noise_factor: Controls the standard deviation of the noise relative to the base value.
         non_negative: If True, ensures the returned value is >= 0.

     Returns:
         A single float value with added noise.
     """
     value = base_value + rng.normal(0, abs(base_value * noise_factor) + 0.01)
     return max(0, value) if non_negative else value

# --- Feature Dictionary Generation Helpers ---

def _generate_spatial_features(rng: np.random.Generator, grade_level: str, base_values: Dict) -> Dict:
    """Generates the spatial_features sub-dictionary based on grade-specific base values.

    Args:
        rng: NumPy random number generator instance.
        grade_level: The target grade ('A', 'B', 'C', 'D') influencing the stats.
        base_values: A dictionary containing pre-calculated base means/stds for the grade,
                     expected to have keys like 'elbow', 'knee', 'norm_ext', 'pos'.

    Returns:
        A dictionary representing the spatial_features part of the main feature dict.
    """
    elbow_mean, elbow_std = base_values['elbow']
    knee_mean, knee_std = base_values['knee']
    norm_ext_mean, norm_ext_std = base_values['norm_ext']
    pos_y_mean, pos_std = base_values['pos']

    return {
        'joint_angles': {
            'elbow_angle_L': _generate_stats_dict(rng, elbow_mean, elbow_std),
            'elbow_angle_R': _generate_stats_dict(rng, elbow_mean, elbow_std),
            'knee_angle_L': _generate_stats_dict(rng, knee_mean, knee_std),
            'knee_angle_R': _generate_stats_dict(rng, knee_mean, knee_std),
            'shoulder_angle_L': _generate_stats_dict(rng, 90, 15 if grade_level in ['A','B'] else 25),
            'shoulder_angle_R': _generate_stats_dict(rng, 90, 15 if grade_level in ['A','B'] else 25),
            'hip_angle_L': _generate_stats_dict(rng, 170, 10 if grade_level in ['A','B'] else 20),
            'hip_angle_R': _generate_stats_dict(rng, 170, 10 if grade_level in ['A','B'] else 20),
        },
        'relative_joint_distances': {
            'norm_elbow_to_wrist_L': {'mean': _generate_single_value(rng, norm_ext_mean, 0.1), 'std': _generate_single_value(rng, norm_ext_std, 0.2)},
            'norm_elbow_to_wrist_R': {'mean': _generate_single_value(rng, norm_ext_mean, 0.1), 'std': _generate_single_value(rng, norm_ext_std, 0.2)},
            'norm_shoulder_to_hip_L': {'mean': _generate_single_value(rng, 0.5, 0.05), 'std': _generate_single_value(rng, 0.05, 0.2)},
            'norm_shoulder_to_hip_R': {'mean': _generate_single_value(rng, 0.5, 0.05), 'std': _generate_single_value(rng, 0.05, 0.2)},
            'norm_knee_to_ankle_L': {'mean': _generate_single_value(rng, 0.4, 0.1), 'std': _generate_single_value(rng, 0.05, 0.2)},
            'norm_knee_to_ankle_R': {'mean': _generate_single_value(rng, 0.4, 0.1), 'std': _generate_single_value(rng, 0.05, 0.2)},
        },
        'court_positioning': {
            'avg_position_x': _generate_single_value(rng, 0, 0.1, non_negative=False),
            'avg_position_y': _generate_single_value(rng, pos_y_mean, 0.1),
            'position_std_x': _generate_single_value(rng, pos_std, 0.15),
            'position_std_y': _generate_single_value(rng, pos_std, 0.15),
            'zone_occupancy_freq': { i: r for i, r in enumerate(rng.dirichlet(np.ones(9)*3), 1) }
        }
    }

def _generate_temporal_features(rng: np.random.Generator, grade_level: str, base_values: Dict) -> Dict:
    """Generates the temporal_features sub-dictionary based on grade-specific base values.

    Args:
        rng: NumPy random number generator instance.
        grade_level: The target grade ('A', 'B', 'C', 'D') influencing the stats.
        base_values: A dictionary containing pre-calculated base values for the grade,
                     expected to have keys like 'wrist_max', 'ankle_max', 'wrist_accel_max',
                     'speed_rally', 'dist_rally', 'rally_dur', 'play_rest_ratio'.

    Returns:
        A dictionary representing the temporal_features part of the main feature dict.
    """
    wrist_max = base_values['wrist_max']
    ankle_max = base_values['ankle_max']
    wrist_accel_max = base_values['wrist_accel_max']
    speed_rally = base_values['speed_rally']
    dist_rally = base_values['dist_rally']
    rally_dur = base_values['rally_dur']
    play_rest_ratio = base_values['play_rest_ratio']

    return {
        'joint_velocities': {
            'wrist_velocity_R': _generate_stats_dict(rng, wrist_max*0.7, wrist_max*0.2, max_factor=1.1),
            'wrist_velocity_L': _generate_stats_dict(rng, wrist_max*0.6, wrist_max*0.25, max_factor=1.1),
            'elbow_velocity_L': _generate_stats_dict(rng, wrist_max*0.5, wrist_max*0.2, max_factor=1.1),
            'elbow_velocity_R': _generate_stats_dict(rng, wrist_max*0.5, wrist_max*0.2, max_factor=1.1),
            'ankle_velocity_R': _generate_stats_dict(rng, ankle_max*0.6, ankle_max*0.2, max_factor=1.1),
            'ankle_velocity_L': _generate_stats_dict(rng, ankle_max*0.6, ankle_max*0.2, max_factor=1.1),
            'knee_velocity_R': _generate_stats_dict(rng, ankle_max*0.8, ankle_max*0.2, max_factor=1.1),
            'knee_velocity_L': _generate_stats_dict(rng, ankle_max*0.8, ankle_max*0.2, max_factor=1.1),
        },
        'joint_accelerations': {
            'wrist_acceleration_R': {'max': _generate_single_value(rng, wrist_accel_max, 0.15), 'mean_positive': _generate_single_value(rng, wrist_accel_max*0.4, 0.2)},
            'wrist_acceleration_L': {'max': _generate_single_value(rng, wrist_accel_max*0.8, 0.2), 'mean_positive': _generate_single_value(rng, wrist_accel_max*0.3, 0.25)},
            'ankle_acceleration_R': {'max': _generate_single_value(rng, wrist_accel_max*0.5, 0.2), 'mean_positive': _generate_single_value(rng, wrist_accel_max*0.2, 0.25)},
            'ankle_acceleration_L': {'max': _generate_single_value(rng, wrist_accel_max*0.5, 0.2), 'mean_positive': _generate_single_value(rng, wrist_accel_max*0.2, 0.25)},
        },
        'body_movement': {
            'avg_body_speed_rally': _generate_single_value(rng, speed_rally, 0.1),
            'max_body_speed_rally': _generate_single_value(rng, speed_rally * 1.6, 0.15),
            'avg_body_acceleration_rally': _generate_single_value(rng, 1.0 + speed_rally*0.2, 0.2),
            'movement_distance_total': _generate_single_value(rng, dist_rally * 50, 0.2),
            'movement_distance_per_rally_avg': _generate_single_value(rng, dist_rally, 0.15),
        },
        'timing_and_intensity': {
            'avg_rally_duration': _generate_single_value(rng, rally_dur, 0.15),
            'avg_shots_per_rally': _generate_single_value(rng, max(1.0, rally_dur * 0.7), 0.2),
            'avg_time_between_shots': _generate_single_value(rng, 2.0 - speed_rally*0.3, 0.15),
            'avg_rest_time_between_rallies': _generate_single_value(rng, 10.0, 0.2),
            'play_rest_speed_ratio': _generate_single_value(rng, play_rest_ratio, 0.15),
        }
    }

def _generate_performance_metrics(rng: np.random.Generator, grade_level: str, base_values: Dict) -> Dict:
    """Generates the performance_metrics sub-dictionary based on grade-specific base values.

    Note:
        Stroke profile proportions are generated here but normalized later.

    Args:
        rng: NumPy random number generator instance.
        grade_level: The target grade ('A', 'B', 'C', 'D') influencing the stats.
        base_values: A dictionary containing pre-calculated base values for the grade,
                     expected to have keys like 'smash_prop', 'clear_prop', etc.,
                     'elbow_var_proxy', 'offensive_ratio', 'net_freq', 'baseline_freq',
                     'hull_area', 'reposition_speed'.

    Returns:
        A dictionary representing the performance_metrics part of the main feature dict.
    """
    # Normalize stroke profile later in the main function
    stroke_profile = {
        'stroke_dist_smash': base_values['smash_prop'], 'stroke_dist_clear': base_values['clear_prop'],
        'stroke_dist_drop': base_values['drop_prop'], 'stroke_dist_net': base_values['net_prop'],
        'stroke_dist_drive': base_values['drive_prop'], 'stroke_dist_lift': base_values['lift_prop'],
        'unknown_stroke_proportion': base_values['unknown_prop'],
    }
    return {
         'stroke_profile': stroke_profile,
         'technical_consistency_proxy': {
             'elbow_angle_variability': _generate_single_value(rng, base_values['elbow_var_proxy'], 0.2),
             'knee_angle_variability': _generate_single_value(rng, base_values['elbow_var_proxy'] * 1.1, 0.25),
             'wrist_speed_variability': _generate_single_value(rng, 15.0 / (base_values['speed_rally'] + 1.0), 0.3),
         },
         'tactical_proxies': {
             'shot_placement_zone_entropy': _generate_single_value(rng, 1.5 + base_values['offensive_ratio'] * 0.5, 0.1),
             'net_play_frequency': _generate_single_value(rng, base_values['net_freq'], 0.15),
             'baseline_play_frequency': _generate_single_value(rng, base_values['baseline_freq'], 0.15),
             'offensive_stroke_ratio': _generate_single_value(rng, base_values['offensive_ratio'], 0.1),
         },
         'court_coverage_metrics': {
             'coverage_area_hull': _generate_single_value(rng, base_values['hull_area'], 0.1),
             'coverage_efficiency_ratio': _generate_single_value(rng, 0.03 + base_values['hull_area'] * 0.001, 0.2),
             'avg_repositioning_speed': _generate_single_value(rng, base_values['reposition_speed'], 0.15),
         }
    }

# --- End Feature Dictionary Generation Helpers ---

# --- Main Data Generation Function (Shared) ---

def generate_sample_feature_dict(rng: np.random.Generator, grade_level: str, player_id_suffix: Any,
                                 id_prefix: str = "Player") -> Dict[str, Any]:
    """Generates a sample feature dictionary mimicking the achievable list structure for a given grade.

    This function defines base statistical values (means, standard deviations, proportions)
    that differ based on the target `grade_level`. It then uses helper functions
    (`_generate_spatial_features`, `_generate_temporal_features`, `_generate_performance_metrics`)
    to construct the nested feature dictionary, introducing random noise based on these
    base values.

    Finally, it normalizes stroke proportions and zone occupancy frequencies to sum to 1.

    Args:
        rng: NumPy random number generator instance.
        grade_level: The target grade ('A', 'B', 'C', 'D') to simulate.
        player_id_suffix: A unique suffix to append to the player ID.
        id_prefix: The prefix for the generated player ID.

    Returns:
        A nested dictionary representing a single player's features, structured
        according to the expected input format for the GradingAPI.
    """
    player_id = f"{id_prefix}_{grade_level}_{player_id_suffix}"

    # --- Define Base Values & Noise based on Grade --- #
    base_values = {
        'elbow': (125, 8) if grade_level == 'A' else (120, 12) if grade_level == 'B' else (110, 20) if grade_level == 'C' else (100, 30),
        'knee': (120, 10) if grade_level == 'A' else (115, 15) if grade_level == 'B' else (110, 25) if grade_level == 'C' else (105, 35),
        'norm_ext': (0.42, 0.04) if grade_level == 'A' else (0.40, 0.06) if grade_level == 'B' else (0.35, 0.08) if grade_level == 'C' else (0.30, 0.12),
        'pos': (0.65, 0.12) if grade_level == 'A' else (0.60, 0.16) if grade_level == 'B' else (0.50, 0.22) if grade_level == 'C' else (0.40, 0.30),
        'wrist_max': 4.5 if grade_level == 'A' else (3.8 if grade_level == 'B' else (3.0 if grade_level == 'C' else 2.2)),
        'ankle_max': 3.5 if grade_level == 'A' else (3.0 if grade_level == 'B' else (2.2 if grade_level == 'C' else 1.5)),
        'wrist_accel_max': 18.0 if grade_level == 'A' else (14.0 if grade_level == 'B' else (9.0 if grade_level == 'C' else 5.0)),
        'speed_rally': 2.8 if grade_level == 'A' else (2.2 if grade_level == 'B' else (1.6 if grade_level == 'C' else 1.1)),
        'dist_rally': 9.0 if grade_level == 'A' else (7.5 if grade_level == 'B' else (6.0 if grade_level == 'C' else 4.0)),
        'rally_dur': 10.0 if grade_level == 'A' else (8.0 if grade_level == 'B' else (6.0 if grade_level == 'C' else 4.5)),
        'play_rest_ratio': 3.5 if grade_level == 'A' else (2.8 if grade_level == 'B' else (2.0 if grade_level == 'C' else 1.4)),
        'smash_prop': 0.35 if grade_level == 'A' else (0.25 if grade_level == 'B' else (0.15 if grade_level == 'C' else 0.05)),
        'clear_prop': 0.15 if grade_level == 'A' else (0.20 if grade_level == 'B' else (0.25 if grade_level == 'C' else 0.35)),
        'drop_prop': 0.20 if grade_level in ['A', 'B'] else 0.10,
        'net_prop': 0.15 if grade_level in ['A', 'B'] else 0.10,
        'drive_prop': 0.10 if grade_level in ['A', 'B'] else 0.05,
        'lift_prop': 0.05 if grade_level in ['A', 'B'] else (0.15 if grade_level == 'C' else 0.25),
        'unknown_prop': 0.02 if grade_level == 'A' else (0.05 if grade_level == 'B' else (0.10 if grade_level == 'C' else 0.20)),
        'elbow_var_proxy': 8.0 if grade_level == 'A' else (12.0 if grade_level == 'B' else (20.0 if grade_level == 'C' else 30.0)),
        'offensive_ratio': 0.65 if grade_level == 'A' else (0.50 if grade_level == 'B' else (0.35 if grade_level == 'C' else 0.20)),
        'net_freq': 0.3 if grade_level in ['A', 'B'] else 0.15,
        'baseline_freq': 0.5 if grade_level in ['A', 'B'] else 0.7,
        'hull_area': 28.0 if grade_level == 'A' else (20.0 if grade_level == 'B' else (14.0 if grade_level == 'C' else 8.0)),
        'reposition_speed': 2.2 if grade_level == 'A' else (1.8 if grade_level == 'B' else (1.4 if grade_level == 'C' else 1.0))
    }

    # --- Construct the dictionary using helpers --- #
    feature_dict = {
        'player_id': player_id,
        'spatial_features': _generate_spatial_features(rng, grade_level, base_values),
        'temporal_features': _generate_temporal_features(rng, grade_level, base_values),
        'performance_metrics': _generate_performance_metrics(rng, grade_level, base_values)
    }

    # Normalize stroke profile proportions
    sp = feature_dict['performance_metrics']['stroke_profile']
    total_prop = sum(v for v in sp.values() if isinstance(v, (int, float)))
    if total_prop > 0:
        for k in sp:
            if isinstance(sp[k], (int, float)): sp[k] /= total_prop
    # Normalize zone occupancy
    zo = feature_dict['spatial_features']['court_positioning']['zone_occupancy_freq']
    total_zone = sum(zo.values())
    if total_zone > 0:
         for k in zo: zo[k] /= total_zone

    return feature_dict 