# src/preprocessing.py
"""
Feature Preprocessing Module for Badminton Player Grading System.

Handles preprocessing of feature vectors based on Yuan et al.'s (2024)
three-stage classification method:
1. Missing Value Imputation (Mean)
2. Feature Scaling (StandardScaler)
3. Feature Selection (Optional, using SelectKBest with f_classif)

Optimized for robustness and clarity.

Author: Sujit
Role: Senior Machine Learning Engineer
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.exceptions import NotFittedError
import logging
from typing import List, Optional, Tuple, Union

# Import custom exceptions
try:
    from .exceptions import PreprocessingError, DataError
except ImportError:
    from exceptions import PreprocessingError, DataError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeaturePreprocessor:
    """
    Preprocesses feature vectors for the grading model.

    Applies NaN imputation, standardization, and optional feature selection.
    Ensures consistent handling of features based on fitting stage.

    Attributes:
        scaler (StandardScaler): Scikit-learn scaler instance.
        feature_selector (Optional[SelectKBest]): Scikit-learn selector instance,
            or None if selection is not used.
        is_fitted (bool): Flag indicating if the preprocessor has been fitted.
        original_feature_names (Optional[List[str]]): Feature names before selection.
        selected_feature_names (Optional[List[str]]): Feature names after selection.
        original_feature_count (int): Number of features before selection.
        selected_feature_indices_ (Optional[np.ndarray]): Indices of selected features.
    """

    def __init__(self):
        """Initializes the FeaturePreprocessor with default components."""
        self.scaler = StandardScaler()
        self.feature_selector: Optional[SelectKBest] = None
        self.is_fitted: bool = False
        self.original_feature_names: Optional[List[str]] = None # Names before selection
        self.selected_feature_names: Optional[List[str]] = None # Names after selection
        self.original_feature_count: int = 0
        self.selected_feature_indices_: Optional[np.ndarray] = None # Indices of selected features

        logger.info("FeaturePreprocessor initialized.")

    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handles missing values (NaN) using column mean imputation.

        Calculates the mean of each column, ignoring existing NaNs. Replaces
        NaNs with their respective column's mean. If a column consists entirely
        of NaNs, its NaNs are replaced with 0.0.

        Args:
            features: The 2D NumPy array containing potential NaN values.

        Returns:
            A 2D NumPy array with NaNs imputed.

        Raises:
            DataError: If input is not a 2D NumPy array.
        """
        if not isinstance(features, np.ndarray) or features.ndim != 2:
            raise DataError("Input 'features' for NaN handling must be a 2D NumPy array.")

        nan_mask = np.isnan(features)
        if nan_mask.any():
            # logger.debug(f"Handling {nan_mask.sum()} missing values (NaNs) found in features.")
            features_copy = features.copy()
            # Calculate means ignoring NaNs, suppress warnings for all-NaN columns
            with np.errstate(invalid='ignore'):
                col_means = np.nanmean(features_copy, axis=0)

            # Replace NaNs in col_means (from all-NaN columns) with 0.0
            # Also handle potential inf/-inf from calculations
            col_means = np.nan_to_num(col_means, nan=0.0,
                                      posinf=np.finfo(features.dtype).max,
                                      neginf=np.finfo(features.dtype).min)
            # logger.debug(f"Calculated column means for imputation (NaNs/Infs replaced): {col_means}")

            # Replace NaNs using the calculated means for their respective columns
            nan_indices_rows, nan_indices_cols = np.where(nan_mask)
            features_copy[nan_indices_rows, nan_indices_cols] = np.take(col_means, nan_indices_cols)

            # Final check for safety
            if np.isnan(features_copy).any():
                 logger.warning("NaNs still present after imputation. Replacing remaining NaNs with 0.")
                 features_copy = np.nan_to_num(features_copy, nan=0.0)

            return features_copy
        else:
            # logger.debug("No missing values (NaNs) found in features.")
            return features # Return original if no NaNs

    # --- Feature Selection Helper ---
    def _fit_feature_selector(self, features_scaled: np.ndarray, labels: np.ndarray, k: Union[int, str]) -> Tuple[Optional[SelectKBest], np.ndarray, List[str]]:
        """Fits the SelectKBest feature selector if k is valid and labels are provided.

        Uses the f_classif scoring function. Validates k and label inputs.

        Args:
            features_scaled: The input features *after* scaling, used to fit the selector.
            labels: The target labels corresponding to the features.
            k: The number of top features to select (int) or 'all' to skip selection.

        Returns:
            A tuple containing:
                - Optional[SelectKBest]: The fitted selector instance, or None if skipped/failed.
                - np.ndarray: The indices of the selected features.
                - List[str]: The names of the selected features.

        Raises:
            ValueError: If labels are not a 1D NumPy array.
            PreprocessingError: If the SelectKBest fitting process fails.
        """
        feature_selector = None
        selected_indices = np.arange(self.original_feature_count) # Default to all
        selected_names = list(self.original_feature_names) # Default to all

        perform_selection = False
        if k != 'all':
            # Validate k
            if not isinstance(k, int) or k <= 0:
                logger.warning(f"Invalid value for k ('{k}'). Skipping feature selection.")
            elif k >= self.original_feature_count:
                logger.info(f"k ({k}) >= number of features ({self.original_feature_count}). Skipping feature selection.")
            # Validate labels (already checked for None in the main fit method before calling this)
            elif not isinstance(labels, np.ndarray) or labels.ndim != 1:
                 # This check might be redundant if already done in fit, but safer here too.
                 raise ValueError("Labels must be a 1D NumPy array for feature selection.")
            else:
                 # Check if labels have variance (SelectKBest needs variance)
                 if len(np.unique(labels)) < 2:
                      logger.warning("Labels have only one unique value. Skipping feature selection as it relies on variance.")
                 else:
                      perform_selection = True

        if perform_selection:
            logger.info(f"Fitting SelectKBest with k={k} using f_classif score function...")
            feature_selector = SelectKBest(f_classif, k=k)
            try:
                # Ensure no NaNs/Infs in scaled data before fitting selector
                if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                     logger.warning("NaNs or Infs detected in scaled data before feature selection. Attempting imputation with 0.")
                     features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

                feature_selector.fit(features_scaled, labels)
                selected_indices = feature_selector.get_support(indices=True)
                selected_names = [self.original_feature_names[i] for i in selected_indices]
                logger.info(f"SelectKBest fitted. Selected {len(selected_names)} features.")
            except Exception as e:
                logger.error(f"Error fitting SelectKBest: {e}. Skipping feature selection.", exc_info=True)
                # Revert to using all features if selection fails
                feature_selector = None
                selected_indices = np.arange(self.original_feature_count)
                selected_names = list(self.original_feature_names)
                # Raise PreprocessingError here? Or just log and continue with all features?
                # Let's raise, as selection was requested but failed.
                raise PreprocessingError(f"Feature selector (SelectKBest) failed during fit: {e}") from e

        return feature_selector, selected_indices, selected_names
    # --- End Feature Selection Helper ---

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None, k: Union[int, str] = 'all') -> 'FeaturePreprocessor':
        """Fits the preprocessor (scaler and optional selector) to the training data.

        Executes the three preprocessing stages:
        1. Imputes missing values using `_handle_missing_values`.
        2. Fits the `StandardScaler`.
        3. Optionally fits the `SelectKBest` feature selector using `_fit_feature_selector`
           if `k` is a valid integer and `labels` are provided.

        Args:
            features: The 2D NumPy array of training features.
            labels: The 1D NumPy array of training labels (required if k != 'all').
            feature_names: Optional list of names for the original features.
            k: Number of top features to select, or 'all' to skip selection.

        Returns:
            The fitted FeaturePreprocessor instance (self).

        Raises:
            DataError: If features or feature_names have invalid shape/type/length.
            PreprocessingError: If imputation, scaling, or selection (if attempted)
                fails during the fitting process.
        """
        logger.info(f"Fitting FeaturePreprocessor with data shape {features.shape}, k={k}...")
        # --- Input Validation ---
        if not isinstance(features, np.ndarray) or features.ndim != 2:
            raise DataError("Input 'features' for fitting must be a 2D NumPy array.")
        n_samples, self.original_feature_count = features.shape
        if n_samples == 0:
            raise DataError("Cannot fit preprocessor on empty dataset (0 samples).")
        # logger.debug(f"Original feature count: {self.original_feature_count}")

        # Validate and store feature names
        if feature_names is not None:
            if not isinstance(feature_names, list) or len(feature_names) != self.original_feature_count:
                raise DataError(f"feature_names must be a list of length {self.original_feature_count}, but got {len(feature_names)}.")
            self.original_feature_names = list(feature_names)
        else:
            self.original_feature_names = [f'feature_{i}' for i in range(self.original_feature_count)]
            # logger.debug("Generated generic feature names.")

        # --- Step 1: Handle Missing Values ---
        try:
            features_filled = self._handle_missing_values(features)
            # Check for remaining infinities after NaN handling (can happen if original data had inf)
            if np.isinf(features_filled).any():
                 inf_cols = np.where(np.isinf(features_filled).any(axis=0))[0]
                 logger.warning(f"Infinite values detected after NaN handling in columns: {inf_cols}. Consider clipping or investigation.")
                 # Replace inf with large finite numbers or handle based on domain knowledge
                 features_filled = np.nan_to_num(features_filled,
                                                 posinf=np.finfo(features.dtype).max,
                                                 neginf=np.finfo(features.dtype).min)
        except Exception as e:
            logger.error(f"Error during missing value handling in fit: {e}", exc_info=True)
            raise PreprocessingError("Failed during NaN handling during fit") from e

        # --- Step 2: Fit Scaler ---
        # logger.debug("Fitting StandardScaler...")
        try:
            self.scaler.fit(features_filled)
            features_scaled = self.scaler.transform(features_filled) # Scale data for selector
            # logger.debug("StandardScaler fitted successfully.")
        except Exception as e:
            logger.error(f"Error fitting/transforming with StandardScaler: {e}", exc_info=True)
            raise PreprocessingError("Failed to fit StandardScaler") from e

        # --- Step 3: Fit Feature Selector (Optional, using helper) ---
        self.feature_selector = None
        self.selected_feature_indices_ = np.arange(self.original_feature_count)
        self.selected_feature_names = list(self.original_feature_names)

        # Check if labels are provided if selection is requested
        if k != 'all' and labels is None:
            logger.warning("Labels (y) are required for feature selection (when k != 'all'). Skipping selection.")
        elif k != 'all': # Labels are provided (or will raise error in helper)
            try:
                self.feature_selector, self.selected_feature_indices_, self.selected_feature_names = \
                    self._fit_feature_selector(features_scaled, labels, k)
            except (ValueError, PreprocessingError) as e:
                # Catch errors from helper and log, but continue fitting with all features
                logger.error(f"Feature selection failed ({type(e).__name__}: {e}). Proceeding without selection.", exc_info=False)
                # Ensure state reflects no selection
                self.feature_selector = None
                self.selected_feature_indices_ = np.arange(self.original_feature_count)
                self.selected_feature_names = list(self.original_feature_names)
        else:
             logger.info("k='all', skipping feature selection.")

        self.is_fitted = True
        logger.info("FeaturePreprocessor fitting complete.")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transforms features using the fitted preprocessor.

        Applies the following steps in order:
        1. Imputes missing values using mean imputation based on the *training* data.
        2. Scales features using the fitted `StandardScaler`.
        3. Selects features using the fitted `SelectKBest` selector (if applicable).

        Args:
            features: The 2D NumPy array of features to transform.
                      Can also be a 1D array for a single sample.

        Returns:
            The transformed 2D NumPy array of features.

        Raises:
            NotFittedError: If the preprocessor has not been fitted.
            DataError: If the input features have the wrong shape or type.
            PreprocessingError: If any transformation step (imputation, scaling,
                selection) fails.
        """
        # logger.debug(f"Transforming features with shape {features.shape}...")
        if not self.is_fitted:
            raise NotFittedError("FeaturePreprocessor must be fitted before transforming data.")

        # --- Input Validation and Reshaping ---
        if not isinstance(features, np.ndarray):
            try: features = np.array(features, dtype=float)
            except ValueError as e: raise DataError(f"Input features cannot be converted to a numeric NumPy array: {e}")
        if features.ndim == 1: features = features.reshape(1, -1)
        elif features.ndim != 2: raise DataError(f"Input features must be a 1D or 2D NumPy array, but got {features.ndim} dimensions.")
        if features.shape[1] != self.original_feature_count:
            raise DataError(f"Input features have {features.shape[1]} columns, but preprocessor was fitted on {self.original_feature_count}.")

        # --- Step 1: Handle Missing Values ---
        try: features_filled = self._handle_missing_values(features)
        except Exception as e: raise PreprocessingError(f"Failed during NaN handling in transform: {e}") from e

        # --- Step 2: Scale Features ---
        try: features_scaled = self.scaler.transform(features_filled)
        except Exception as e:
            # Catch specific NotFittedError if possible, though scaler should be fitted if self.is_fitted is True
            if isinstance(e, NotFittedError):
                raise PreprocessingError(f"Scaler transform failed: Scaler is not fitted. {e}") from e
            else:
                raise PreprocessingError(f"Failed to transform with StandardScaler: {e}") from e

        # --- Step 3: Select Features ---
        if self.feature_selector is not None:
            try:
                features_selected = self.feature_selector.transform(features_scaled)
                # logger.debug(f"SelectKBest transformation applied. Output shape: {features_selected.shape}")
                return features_selected
            except Exception as e:
                # Catch specific NotFittedError
                if isinstance(e, NotFittedError):
                    raise PreprocessingError(f"Feature selector transform failed: Selector is not fitted. {e}") from e
                else:
                    raise PreprocessingError(f"Failed to transform with SelectKBest: {e}") from e
        else:
            # logger.debug("No feature selection applied. Returning scaled features.")
            return features_scaled

    def fit_transform(self, features: np.ndarray, labels: Optional[np.ndarray] = None,
                      feature_names: Optional[List[str]] = None, k: Union[int, str] = 'all') -> np.ndarray:
        """Fits the preprocessor and transforms the features in one step.

        Combines the `fit` and `transform` methods.

        Args:
            features: The 2D NumPy array of training features.
            labels: The 1D NumPy array of training labels (required if k != 'all').
            feature_names: Optional list of names for the original features.
            k: Number of top features to select, or 'all' to skip selection.

        Returns:
            The transformed 2D NumPy array of features.

        Raises:
            (Exceptions raised by `fit` or `transform`)
        """
        logger.info("Performing fit_transform...")
        self.fit(features, labels, feature_names, k)
        return self.transform(features)

    # --- Getter methods ---
    def get_original_feature_names(self) -> Optional[List[str]]:
        """Returns the list of original feature names provided during fitting."""
        return self.original_feature_names

    def get_selected_feature_names(self) -> Optional[List[str]]:
        """Returns the list of selected feature names after fitting."""
        if not self.is_fitted: logger.warning("Preprocessor not fitted."); return None
        return self.selected_feature_names

    def get_selected_indices(self) -> Optional[np.ndarray]:
        """Returns the indices of the selected features after fitting."""
        if not self.is_fitted: logger.warning("Preprocessor not fitted."); return None
        return self.selected_feature_indices_

