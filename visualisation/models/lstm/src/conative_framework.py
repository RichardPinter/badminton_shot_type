# src/conative_framework.py
"""
Conative Framework for Badminton Player Grading System.

Implements the five-stage conative framework (Dieu et al., 2020) for classifying
badminton players based on skill level and approach. Maps the 5 stages to the
Hunter Badminton Club's 4 grades (A-D). Loads thresholds from config.json and
uses feature names aligned with the 'Achievable Feature List'.

Author: Sujit
"""

import numpy as np
import pandas as pd
import json
import sys
import os
import logging
from typing import Dict, Tuple, List, Optional

# Import custom exceptions
try:
    from .exceptions import ConfigurationError
except ImportError: # Fallback for direct execution or potential structure issues
    from exceptions import ConfigurationError

# Configure logging
# Consider using a shared logger instance if part of a larger application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Feature Name Constants (Aligned with Achievable Feature List) ---
# These constants represent the *flattened* keys expected after API preprocessing.
# Define the core metrics needed for the Conative stages based on Dieu et al. concepts adapted for CCTV.
METRIC_COURT_COVERAGE = 'court_coverage_metrics_coverage_area_hull' # Example flattened name
METRIC_MOVEMENT_SPEED = 'body_movement_avg_body_speed_rally'     # Example flattened name
METRIC_TECH_CONSISTENCY = 'technical_consistency_proxy_overall_consistency_score' # Example: Using a proxy score
METRIC_PLAY_REST_RATIO = 'timing_and_intensity_play_rest_speed_ratio' # Example flattened name
# Tactical Awareness might be a composite or a specific feature if available
METRIC_TACTICAL_AWARENESS = 'tactical_proxies_composite_awareness_score' # Hypothetical composite score key

# --- TODO: Finalize the exact flattened names above based on the output of ---
# --- GradingAPI._preprocess_input_features after coordinating with Utsab/Kabir ---


class ConativeFramework:
    """
    Implements Dieu et al.'s (2020) five-stage conative framework using achievable features.

    Maps player skill stages (1-5) to grades (A-D) based on objective features
    and configured thresholds. Adapts metrics for CCTV-based analysis.

    Attributes:
        thresholds (Optional[Dict]): Dictionary containing the loaded thresholds
            from the config file. None if loading failed.
    """

    def __init__(self, config_path: str = 'config.json'):
        """Initializes the ConativeFramework by loading thresholds.

        Args:
            config_path: Path to the JSON configuration file containing
                'conative_thresholds'.

        Raises:
            ConfigurationError: If the config file cannot be found, loaded, parsed,
                or if it's missing the 'conative_thresholds' key.
        """
        try:
            self.thresholds: Optional[Dict] = self._load_config(config_path)
        except ConfigurationError as e:
            logger.error(f"Failed to initialize Conative Framework due to configuration error: {e}")
            # Set thresholds to None to indicate failure, matching previous behavior
            # Alternatively, could re-raise the exception here if initialization MUST succeed.
            self.thresholds = None

        if self.thresholds is None:
            logger.error("Conative Framework initialized WITHOUT thresholds. Stage determination will fail.")
        else:
            # Validate that thresholds exist for the core metrics used
            self._validate_thresholds()
            logger.info("Conative Framework initialized successfully with thresholds.")

    def _load_config(self, config_path: str) -> Optional[Dict]:
        """Loads thresholds from the specified JSON configuration file.

        Attempts to load from the provided path first, then falls back to
        a path relative to this script's directory.

        Args:
            config_path: The initial path to the configuration file.

        Returns:
            The dictionary containing the 'conative_thresholds'.

        Raises:
            ConfigurationError: If the file cannot be found, read, or parsed,
                or if the 'conative_thresholds' key is missing/invalid.
        """
        # Prioritize the path as provided (relative to execution dir)
        final_path = config_path

        if not os.path.exists(final_path):
            logger.warning(f"Config file not found at provided path: {final_path}. Attempting relative to script...")
            # Fallback: Try path relative to this script's directory
            script_dir = os.path.dirname(__file__)
            relative_path_attempt = os.path.join(script_dir, config_path)
            if os.path.exists(relative_path_attempt):
                final_path = relative_path_attempt
                logger.info(f"Found config file relative to script: {final_path}")
            else:
                # Critical error if not found in either location
                msg = f"Config file not found at '{config_path}' or '{relative_path_attempt}'."
                logger.error(f"CRITICAL: {msg} Cannot load thresholds.")
                raise ConfigurationError(msg)

        # Proceed only if a valid path was determined
        try:
            logger.info(f"Loading conative thresholds from: {final_path}")
            with open(final_path, 'r') as f:
                config_data = json.load(f)
            thresholds = config_data.get('conative_thresholds')
            if not thresholds or not isinstance(thresholds, dict):
                raise ValueError("'conative_thresholds' key missing or invalid.")

            # --- Ensure top-level keys are standard strings --- #
            stringified_thresholds = {str(k): v for k, v in thresholds.items()}
            # ------------------------------------------------- #

            return stringified_thresholds # Return the version with explicitly stringified keys
        except FileNotFoundError:
             # This might catch permissions errors or if the file vanishes after the exists() check
             msg = f"Error opening config file '{final_path}' (File not found or permissions issue)."
             logger.error(msg, exc_info=True)
             raise ConfigurationError(msg) from e
        except Exception as e:
            msg = f"Failed to load or parse config file '{final_path}': {e}"
            logger.error(msg, exc_info=True)
            raise ConfigurationError(msg) from e

    def _validate_thresholds(self):
        """Checks if necessary threshold keys exist after loading.

        Logs errors if required metrics ('court_coverage', 'movement_speed',
        'technical_consistency', 'tactical_awareness') or stages ('2'-'5')
        are missing from the loaded thresholds dictionary.
        Does not raise an error, allowing potentially partial functionality.
        """
        if self.thresholds is None: return # Skip if loading failed

        # Use the base metric names (without perf_, etc.) as keys in config.json
        # These should match the keys in config.json
        required_metrics = ['court_coverage', 'movement_speed', 'technical_consistency', 'tactical_awareness']
        required_stages = ['2', '3', '4', '5'] # Thresholds defined up to stage 5

        missing = False
        for metric in required_metrics:
            if metric not in self.thresholds or not isinstance(self.thresholds[metric], dict):
                logger.error(f"Config Error: Missing or invalid threshold structure for metric: '{metric}'")
                missing = True
                continue
            for stage in required_stages:
                if stage not in self.thresholds[metric]:
                    logger.error(f"Config Error: Missing threshold for stage '{stage}' in metric: '{metric}'")
                    missing = True
        if missing:
             logger.error("Threshold configuration is incomplete. Stage determination may be inaccurate.")
             # Optionally raise an error: raise ValueError("Threshold configuration incomplete.")

    def _extract_metric(self, data_dict: Optional[Dict], metric_key: str, default_value: float = 0.0) -> float:
        """Safely extracts a numeric metric value from a dictionary.

        Args:
            data_dict: The dictionary to extract from (can be nested).
            metric_key: The key for the desired metric.
            default_value: Value to return if the key is not found, the value is None,
                NaN, or cannot be converted to float.

        Returns:
            The extracted metric as a float, or the default_value on failure.
        """
        if data_dict is None or not isinstance(data_dict, dict):
            return default_value
        try:
            value = data_dict[metric_key]

            # Convert to float and handle potential NaN from JSON (represented as float np.nan)
            # Check for None explicitly as well
            if value is None or pd.isna(value):
                return default_value
            else:
                 extracted_value = float(value)
                 return extracted_value
        except KeyError:
            # Key not found at either level
            return default_value
        except (TypeError, ValueError) as e:
             # Error during float conversion
             return default_value

    def _calculate_technical_consistency_score(self, features_dict: Optional[Dict]) -> float:
        """Calculates a technical consistency score from raw features.

        Uses 'elbow_angle_variability' as a proxy. Maps the expected variability
        range (defined internally, e.g., 8.0-30.0) inversely to a target score
        range (e.g., 1.0-5.0) suitable for comparison with config thresholds.
        Lower variability results in a higher score.

        Args:
            features_dict: The raw feature dictionary containing
                performance_metrics['technical_consistency_proxy']['elbow_angle_variability'].

        Returns:
            The calculated technical consistency score (e.g., between 1.0 and 5.0).
            Returns 1.0 if input is None or data is missing.
        """
        if features_dict is None: return 1.0 # Default to lowest score if no data

        # Navigate to the correct sub-dictionary
        perf_metrics = features_dict.get('performance_metrics', {})
        tech_consistency_proxy = perf_metrics.get('technical_consistency_proxy', {})

        # Use elbow angle variability as the primary proxy for now
        # Lower variability -> higher score
        variability = self._extract_metric(tech_consistency_proxy, 'elbow_angle_variability', default_value=30.0) # Default to high variability

        # Define the expected range of variability (based on mock data generation)
        min_expected_variability = 8.0  # Corresponds to best consistency (Grade A)
        max_expected_variability = 30.0 # Corresponds to worst consistency (Grade D)

        # Define the target score range (based on config thresholds)
        min_score = 1.0 # Score corresponding to max variability (Stage 2 threshold)
        max_score = 5.0 # Score corresponding to min variability (Stage 5 threshold)

        # Clamp variability to the expected range to avoid extrapolation issues
        clamped_variability = max(min_expected_variability, min(max_expected_variability, variability))

        # Perform linear interpolation (inverse relationship: lower variability -> higher score)
        # Avoid division by zero if min and max variability are the same
        if max_expected_variability <= min_expected_variability:
            score = min_score if variability >= max_expected_variability else max_score
        else:
            score = max_score - (
                (clamped_variability - min_expected_variability) * (max_score - min_score) /
                (max_expected_variability - min_expected_variability)
            )

        # Ensure score is within the target bounds
        final_score = max(min_score, min(max_score, score))

        return final_score

    def _calculate_tactical_awareness(self, features_dict: Optional[Dict]) -> float:
        """
        Calculates the tactical awareness score from raw features.

        Placeholder Implementation:
            Currently uses a weighted combination of normalized court coverage,
            movement speed, and technical consistency score relative to Stage 5
            thresholds. This is a temporary proxy.
            **TODO**: Replace with actual calculation using tactical features
            (e.g., shot placement entropy, offensive ratio) when available.

        Args:
            features_dict: The raw feature dictionary.

        Returns:
            The calculated tactical awareness score (currently 0.0-1.0 placeholder).
            Returns 0.2 if input is None or thresholds/basic metrics are missing.
        """
        if self.thresholds is None: return 0.2 # Default if no thresholds
        if features_dict is None: return 0.2 # Default if no features

        # Extract relevant sub-dictionaries safely
        perf_metrics = features_dict.get('performance_metrics', {})
        court_coverage_metrics = perf_metrics.get('court_coverage_metrics', {})
        temporal_features = features_dict.get('temporal_features', {})
        body_movement = temporal_features.get('body_movement', {})
        tactical_proxies = perf_metrics.get('tactical_proxies', {})

        # --- TODO: Replace placeholder with actual calculation using available tactical proxies---
        # Examples:
        # entropy = self._extract_metric(tactical_proxies, 'shot_placement_zone_entropy', 0.0)
        # offensive_ratio = self._extract_metric(tactical_proxies, 'offensive_stroke_ratio', 0.0)
        # if entropy > 0 and offensive_ratio > 0:
        #     # Combine them using some logic...
        #     return calculated_score

        # --- Placeholder using basic metrics (less accurate) ---
        court_coverage = self._extract_metric(court_coverage_metrics, 'coverage_area_hull')
        movement_speed = self._extract_metric(body_movement, 'avg_body_speed_rally')
        tech_consistency = self._calculate_technical_consistency_score(features_dict) # Use calculated score

        if not all(v > 0 for v in [court_coverage, movement_speed, tech_consistency]):
            return 0.2 # Default if basic metrics are missing/zero

        try:
            # Use base metric names for threshold lookup from config
            thresh5 = self.thresholds
            # Note: thresholds in config are for the *conceptual* metrics, not necessarily direct feature values.
            # The calculation here normalizes based on the *feature values* relative to *stage 5 thresholds*.
            # This might need refinement depending on how thresholds relate to calculated scores.
            norm_coverage = court_coverage / thresh5['court_coverage']['5']
            norm_speed = movement_speed / thresh5['movement_speed']['5']
            # Use the calculated tech_consistency score (0-1 range) directly or normalize?
            # Let's assume the tech_consistency threshold in config refers to this 0-1 score.
            norm_consistency = tech_consistency / thresh5['technical_consistency']['5']
            # Similarly, assume tactical_awareness threshold refers to this placeholder calculation's output (0-1)
            tactical_awareness = 0.4 * norm_coverage + 0.3 * norm_speed + 0.3 * norm_consistency
            return max(0.0, min(1.0, tactical_awareness)) # Clamp placeholder score
        except (KeyError, ZeroDivisionError, TypeError) as e:
            logger.error(f"Error in placeholder tactical awareness calculation: {e}")
            return 0.2

    def determine_stage(self, features_dict: Dict) -> Tuple[int, Dict]:
        """Determines the conative stage (1-5) and calculates related metrics.

        Extracts or calculates key metrics (court coverage, movement speed,
        technical consistency, tactical awareness) from the raw features.
        Then, determines the highest stage (5 down to 2) for which the player meets
        all configured thresholds for these metrics using `_meets_thresholds`.
        If no thresholds for stages 2-5 are met, returns Stage 1.

        Args:
            features_dict: The raw feature dictionary for the player.

        Returns:
            A tuple containing:
                - int: The determined conative stage (1-5).
                - dict: The dictionary of calculated metrics used for stage determination
                  (keys: 'court_coverage', 'movement_speed', 'technical_consistency',
                  'tactical_awareness').

        Raises:
            ConfigurationError: If thresholds are not loaded or are invalid during checking.
        """
        # logger.debug(f"Received features_dict in determine_stage: {features_dict}")
        # --- END DEBUG --- #

        if self.thresholds is None:
            logger.error("Cannot determine conative stage: thresholds not loaded.")
            raise ConfigurationError("Cannot determine stage: Conative Framework thresholds not loaded.")

        # Extract relevant sub-dictionaries safely
        perf_metrics = features_dict.get('performance_metrics', {})
        court_coverage_metrics = perf_metrics.get('court_coverage_metrics', {})
        temporal_features = features_dict.get('temporal_features', {})
        body_movement = temporal_features.get('body_movement', {})

        # Extract / Calculate core metrics needed for staging directly from the correct sub-dictionaries
        metrics = {
            'court_coverage': self._extract_metric(court_coverage_metrics, 'coverage_area_hull'),
            'movement_speed': self._extract_metric(body_movement, 'avg_body_speed_rally'),
            'technical_consistency': self._calculate_technical_consistency_score(features_dict),
            'tactical_awareness': self._calculate_tactical_awareness(features_dict)
        }
        # logger.debug(f"Metrics for Stage Calc: { {k: f'{v:.2f}' for k, v in metrics.items()} }")

        # Determine stage by checking thresholds downwards from 5
        determined_stage = 1 # Default to stage 1
        if self._meets_thresholds(5, metrics):
            determined_stage = 5
        elif self._meets_thresholds(4, metrics):
            determined_stage = 4
        elif self._meets_thresholds(3, metrics):
            determined_stage = 3
        elif self._meets_thresholds(2, metrics):
            determined_stage = 2
        # If none of the above are met, it remains Stage 1

        logger.info(f"Player classified as Stage {determined_stage} based on meeting thresholds.")

        # Return the determined stage and the calculated metrics
        return determined_stage, metrics

    def _meets_thresholds(self, stage: int, metrics: Dict[str, float]) -> bool:
        """Checks if all calculated metrics meet the thresholds for the given stage.

        Args:
            stage: The conative stage (2-5) to check thresholds for.
            metrics: A dictionary of the player's calculated metric values.

        Returns:
            True if all required metrics meet or exceed the threshold for the stage,
            False otherwise.

        Raises:
            ConfigurationError: If thresholds are missing or invalid for the requested
                stage/metric during lookup.
        """
        if self.thresholds is None or not 2 <= stage <= 5:
            return False

        required_metrics = ['court_coverage', 'movement_speed', 'technical_consistency', 'tactical_awareness']
        try:
            for metric_key in required_metrics:
                threshold = float(self.thresholds[metric_key][str(stage)])
                metric_value = metrics.get(metric_key, -1.0) # Default to -1 if metric missing
                if metric_value < threshold:
                    return False # Did not meet this threshold
            # If loop completes, all thresholds for this stage were met
            return True
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error checking thresholds for stage {stage}, metric '{metric_key}': {e}")
            # Propagate as a configuration error, as thresholds are likely missing/invalid
            raise ConfigurationError(f"Error checking threshold for stage {stage}, metric '{metric_key}': {e}") from e

    def map_to_grade(self, stage: int) -> str:
        """Maps a conative stage (1-5) to a player grade ('A'-'D').

        Mapping:
            Stage 5 -> Grade A
            Stage 4 -> Grade B
            Stage 3 -> Grade C
            Stage 1, 2 -> Grade D

        Args:
            stage: The determined conative stage (1-5).

        Returns:
            The corresponding grade string ('A', 'B', 'C', 'D') or 'N/A' if stage is invalid.
        """
        grade_map = {5: 'A', 4: 'B', 3: 'C', 2: 'D', 1: 'D'}
        grade = grade_map.get(stage, 'N/A')
        return grade

    def get_stage_description(self, stage: int) -> str:
        """Provides a brief textual description for a given conative stage.

        Args:
            stage: The conative stage (1-5).

        Returns:
            A short string describing the characteristics of the stage.
        """
        descriptions = {
            5: "Expertise - Player imposes their game, adapts strategically...",
            4: "Contextual - Player effectively employs tactical sequences...",
            3: "Technical - Player focuses on executing specific winning strokes...",
            2: "Functional - Player focuses on directing the shuttlecock with some variation...",
            1: "Structural - Player primarily focuses on returning the shuttlecock..."
        } # Keep descriptions concise or load from config?
        description = descriptions.get(stage, "Unknown or invalid stage")
        return description

    def get_key_features_for_stage(self, stage: int) -> List[str]:
        """Identifies key features characteristic of each stage (using flattened names).

        Note:
            These are example features based on the framework concepts.
            **TODO**: Update with the FINAL flattened feature names most relevant
            to each stage, determined through analysis or expert input.

        Args:
            stage: The conative stage (1-5).

        Returns:
            A list of string feature names considered key for that stage.
        """
        # --- TODO: Update with the FINAL flattened feature names most relevant to each stage ---
        key_features_map = {
            5: [METRIC_COURT_COVERAGE, METRIC_TACTICAL_AWARENESS, METRIC_TECH_CONSISTENCY, METRIC_MOVEMENT_SPEED],
            4: ['tactical_proxies_avg_repositioning_speed', METRIC_TACTICAL_AWARENESS, 'tactical_proxies_shot_placement_zone_entropy'], # Example names
            3: [METRIC_TECH_CONSISTENCY, 'stroke_profile_stroke_dist_smash', 'joint_velocities_wrist_velocity_R_max'], # Example names
            2: ['tactical_proxies_shot_direction_control_score', METRIC_COURT_COVERAGE, METRIC_MOVEMENT_SPEED], # Example names
            1: ['performance_metrics_shuttle_return_rate', 'body_movement_avg_body_speed_rally', METRIC_TECH_CONSISTENCY] # Example names
        }
        features = key_features_map.get(stage, [])
        return features

