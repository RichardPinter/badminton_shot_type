# src/model.py
"""
Grading Model and Result Structure for Badminton Player Grading System.

Contains:
- GradingModel: Wrapper for the Random Forest classifier (Dieu et al., 2020).
- GradeClassification: Dataclass to hold the results of a grading prediction.

Optimized for clarity and robustness.

Author: Sujit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.exceptions import NotFittedError
import joblib
import logging
import os # For save/load directory creation
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Import custom exceptions
try:
    from .exceptions import ModelError, DataError
except ImportError:
    from exceptions import ModelError, DataError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Grading Model ---

class GradingModel:
    """
    Random Forest classifier wrapper for badminton player grading.

    Uses hyperparameters and approach inspired by Dieu et al. (2020).
    Handles training, prediction, probability estimation, feature importance,
    and model persistence.

    Attributes:
        DEFAULT_PARAMS (Dict): Default hyperparameters for the RandomForestClassifier.
        model (RandomForestClassifier): The underlying scikit-learn model instance.
        is_trained (bool): Flag indicating if the model has been trained.
        feature_names_when_trained (Optional[List[str]]): Names of features used during training.
        n_features_when_trained (Optional[int]): Number of features used during training.
        classes_when_trained (Optional[np.ndarray]): Unique numeric classes seen during training.
    """
    DEFAULT_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'bootstrap': True,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1 # Use all available CPU cores
    }

    def __init__(self, params: Optional[Dict] = None):
        """Initializes the GradingModel.

        Args:
            params: Optional dictionary of parameters to override the
                DEFAULT_PARAMS for the RandomForestClassifier.

        Raises:
            ValueError: If invalid hyperparameters are provided in `params`.
        """
        model_params = self.DEFAULT_PARAMS.copy()
        if params:
            model_params.update(params)
            logger.info(f"Initializing GradingModel with custom parameters: {params}")
        else:
             logger.info(f"Initializing GradingModel with default parameters (Dieu et al.): {model_params}")

        try:
            self.model = RandomForestClassifier(**model_params)
        except TypeError as e:
             logger.error(f"Invalid parameter provided for RandomForestClassifier: {e}", exc_info=True)
             raise ValueError("Invalid hyperparameters provided for GradingModel.") from e

        self.is_trained: bool = False
        self.feature_names_when_trained: Optional[List[str]] = None
        self.n_features_when_trained: Optional[int] = None
        self.classes_when_trained: Optional[np.ndarray] = None # Stores numeric classes [0, 1, 2, 3]

        logger.info("GradingModel initialized.")

    # --- Training Helpers ---
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, cv_folds: int) -> np.ndarray:
        """Performs Stratified K-Fold cross-validation.

        Validates the number of folds based on the minimum samples per class.
        Uses the same hyperparameters as the main model.

        Args:
            X: Training features.
            y: Training labels.
            cv_folds: The desired number of cross-validation folds.

        Returns:
            A NumPy array of accuracy scores for each fold, or an empty array
            if CV is skipped or fails.
        """
        cv_scores = np.array([])
        unique_classes_cv, counts_cv = np.unique(y, return_counts=True)
        min_samples_per_class = np.min(counts_cv) if len(counts_cv) > 0 else 0

        actual_cv_folds = cv_folds
        if not isinstance(cv_folds, int) or cv_folds < 2:
             logger.warning(f"Invalid cv_folds ({cv_folds}). Setting to default 5.")
             actual_cv_folds = 5
        if min_samples_per_class < actual_cv_folds:
            logger.warning(f"Min samples per class ({min_samples_per_class}) < cv_folds ({actual_cv_folds}). Reducing folds to {min_samples_per_class} or skipping.")
            actual_cv_folds = max(2, min_samples_per_class)
            if actual_cv_folds < 2: logger.warning("Skipping CV: less than 2 samples in each class."); actual_cv_folds = 0

        if actual_cv_folds >= 2:
            logger.info(f"Performing Stratified {actual_cv_folds}-Fold Cross-Validation...")
            try:
                cv_strategy = StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=self.model.random_state)
                cv_model = RandomForestClassifier(**self.model.get_params()) # Use fresh instance
                cv_scores = cross_val_score(cv_model, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=self.model.n_jobs, error_score='raise')
                logger.info(f"CV Accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
            except Exception as e:
                logger.error(f"Error during cross-validation: {e}. Skipping CV.", exc_info=True)
                cv_scores = np.array([]) # Ensure empty array on CV error
        else:
            logger.info("Skipping cross-validation.")

        return cv_scores
    # --- End Training Helpers ---

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None, cv_folds: int = 5) -> np.ndarray:
        """Trains the Random Forest model using preprocessed data.

        Performs input validation, stores feature names and class information,
        optionally runs cross-validation using `_perform_cross_validation`,
        and finally trains the main model on the entire dataset.

        Args:
            X: The 2D NumPy array of preprocessed training features.
            y: The 1D NumPy array of numeric training labels.
            feature_names: Optional list of names for the features in X.
            cv_folds: Number of folds for cross-validation.

        Returns:
            A NumPy array of cross-validation accuracy scores (empty if CV skipped).

        Raises:
            DataError: If input features (X), labels (y), or feature_names have
                invalid shape, type, or contain NaNs.
            ModelError: If the final model training process fails.
        """
        logger.info(f"Starting model training with data shape {X.shape}, cv_folds={cv_folds}...")
        # --- Input Validation ---
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise DataError("Input features X must be a 2D NumPy array.")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise DataError("Input labels y must be a 1D NumPy array.")
        n_samples, n_features = X.shape
        if n_samples != y.shape[0]:
            raise DataError(f"Sample count mismatch: X has {n_samples} samples, but y has {y.shape[0]} labels.")
        if n_samples == 0:
            raise DataError("Cannot train model on empty dataset (0 samples).")
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("NaNs or Infs detected in training data X. Model fitting may fail or be unreliable.")
        if pd.isna(y).any():
            raise DataError("Target labels y contain NaN values. Please impute or remove them before training.")

        # Store feature names and count used for training
        self.n_features_when_trained = n_features
        if feature_names is not None:
            if not isinstance(feature_names, list) or len(feature_names) != n_features:
                raise DataError(f"feature_names must be a list matching X columns ({n_features}), but got {len(feature_names)} names.")
            self.feature_names_when_trained = list(feature_names)
        else:
            self.feature_names_when_trained = [f'feature_{i}' for i in range(n_features)]

        # Store classes seen during training
        self.classes_when_trained = np.unique(y)
        logger.info(f"Training data contains classes: {self.classes_when_trained}")
        if len(self.classes_when_trained) < 2:
             logger.warning("Training data contains only one class. Model performance will be limited.")

        # --- Cross-Validation (using helper) ---
        cv_scores = self._perform_cross_validation(X, y, cv_folds)

        # --- Final Model Training ---
        logger.info("Training final model on the full dataset...")
        try:
            self.model.fit(X, y)
            self.is_trained = True
            # Verify classes match after fitting
            if hasattr(self.model, 'classes_') and not np.array_equal(self.model.classes_, self.classes_when_trained):
                 logger.warning(f"Model classes after fit ({self.model.classes_}) differ from unique labels in y ({self.classes_when_trained}). Updating stored classes.")
                 self.classes_when_trained = self.model.classes_
            logger.info("Final model training complete.")
        except Exception as e:
            self.is_trained = False
            error_msg = f"Error training final model: {e}"
            logger.error(error_msg, exc_info=True)
            raise ModelError(f"Final model training failed ({type(e).__name__}): {e}") from e

        return cv_scores

    def _check_is_fitted(self):
        """Raise NotFittedError if the model is not trained."""
        if not self.is_trained or not hasattr(self.model, 'classes_'):
            raise NotFittedError("This GradingModel instance is not fitted yet. Call 'train' before using this estimator.")

    def _validate_predict_input(self, X: np.ndarray) -> np.ndarray:
        """Validate input for prediction/probability methods.

        Checks if the model is fitted, validates the input shape and type,
        and ensures the number of features matches the number used during training.

        Args:
            X: Input features (1D or 2D NumPy array or convertible).

        Returns:
            The validated features as a 2D NumPy array.

        Raises:
            NotFittedError: If the model is not fitted.
            DataError: If input X cannot be converted, has wrong dimensions,
                or has a mismatching number of features.
        """
        self._check_is_fitted()

        if not isinstance(X, np.ndarray):
            try: X = np.array(X, dtype=float)
            except ValueError as e:
                raise DataError(f"Input X cannot be converted to a numeric NumPy array: {e}") from e
        if X.ndim == 1: X = X.reshape(1, -1)
        elif X.ndim != 2: raise DataError(f"Input X must be a 1D or 2D NumPy array, but got {X.ndim} dimensions.")
        if X.shape[1] != self.n_features_when_trained:
            raise DataError(f"Input features have {X.shape[1]} columns, but model was trained on {self.n_features_when_trained}.")
        if np.isnan(X).any() or np.isinf(X).any():
             logger.warning("NaNs or Infs detected in prediction input X. Model predictions may be unreliable.")
             # Optionally, impute here based on training data means/medians if scaler wasn't applied,
             # but ideally imputation happens in the preprocessor.
             # X = np.nan_to_num(X, nan=0.0, posinf=..., neginf=...)

        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts grades (numeric labels) using the trained model.

        Args:
            X: Input features (preprocessed) as a 1D or 2D NumPy array.

        Returns:
            A 1D NumPy array of predicted numeric labels.

        Raises:
            NotFittedError: If the model is not fitted.
            DataError: If input validation fails.
            ModelError: If prediction fails for other reasons.
        """
        X_validated = self._validate_predict_input(X)
        if self.model is None:
            logger.warning("Predict called before training. Raising ModelError.")
            raise ModelError("Model object is None. Cannot predict.")
        try:
            predictions = self.model.predict(X_validated)
            return predictions
        except Exception as e:
            error_msg = f"Error during prediction: {e}"
            logger.error(error_msg, exc_info=True)
            raise ModelError(f"Prediction failed ({type(e).__name__}): {e}") from e

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts grade probabilities using the trained model.

        Args:
            X: Input features (preprocessed) as a 1D or 2D NumPy array.

        Returns:
            A 2D NumPy array where rows correspond to samples and columns
            correspond to class probabilities (in the order of `classes_when_trained`).

        Raises:
            NotFittedError: If the model is not fitted.
            DataError: If input validation fails.
            ModelError: If probability prediction fails.
        """
        X_validated = self._validate_predict_input(X)
        if self.model is None:
            logger.warning("Predict_proba called before training. Raising ModelError.")
            raise ModelError("Model object is None. Cannot predict probabilities.")
        try:
            probabilities = self.model.predict_proba(X_validated)
            # Ensure probability columns align with classes_when_trained
            if probabilities.shape[1] != len(self.classes_when_trained):
                 logger.error(f"Probability output columns ({probabilities.shape[1]}) mismatch expected classes ({len(self.classes_when_trained)}).")
                 # Attempt to re-align or raise error? For now, just warn.
            return probabilities
        except Exception as e:
            error_msg = f"Error during probability prediction: {e}"
            logger.error(error_msg, exc_info=True)
            raise ModelError(f"Probability prediction failed ({type(e).__name__}): {e}") from e

    def get_feature_importance(self) -> Dict[str, float]:
        """Gets feature importance scores from the trained model.

        Returns feature importances (typically Gini importance for RandomForest)
        as a dictionary mapping feature names to importance scores.

        Returns:
            A dictionary where keys are feature names and values are their
            importance scores (float).

        Raises:
            NotFittedError: If the model is not fitted.
            ModelError: If the model does not support feature importances or an
                unexpected error occurs.
        """
        self._check_is_fitted()
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            logger.warning("Feature importances requested before model is trained or model lacks importance attribute.")
            raise ModelError("Model not trained or does not support feature importances.")
        try:
            importances = self.model.feature_importances_
            if self.feature_names_when_trained and len(self.feature_names_when_trained) == len(importances):
                importance_dict = dict(zip(self.feature_names_when_trained, importances))
                return importance_dict
            else:
                logger.warning("Feature names mismatch or unavailable. Returning importance by index.")
                return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)} # Ensure float
        except AttributeError:
            err_msg = "Model lacks 'feature_importances_' attribute."
            logger.error(err_msg)
            raise ModelError(err_msg)
        except Exception as e:
            err_msg = f"Error getting feature importance: {e}"
            logger.error(err_msg, exc_info=True)
            raise ModelError(f"Unexpected error getting feature importance ({type(e).__name__}): {e}") from e

    def visualize_feature_importance(self, top_n: int = 10) -> Optional[plt.Figure]:
        """Generates a bar chart of the top N feature importances.

        Requires matplotlib to be installed.

        Args:
            top_n: The number of top features to display.

        Returns:
            A matplotlib Figure object containing the plot, or None if visualization
            fails or the model is not fitted.
        """
        logger.info(f"Generating feature importance visualization for top {top_n} features...")
        self._check_is_fitted()
        if self.model is None:
            logger.warning("Cannot visualize: model is not trained.")
            return None

        feature_importance_dict = self.get_feature_importance()
        if not feature_importance_dict: logger.error("No feature importance data available."); return None

        try:
            imp_df = pd.DataFrame(feature_importance_dict.items(), columns=['feature', 'importance'])
            top_features_df = imp_df.sort_values('importance', ascending=False).head(top_n)
            if top_features_df.empty: logger.warning("No features to visualize."); return None

            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.5)))
            ax.barh(top_features_df['feature'], top_features_df['importance'], height=0.7)
            ax.set_xlabel('Importance Score (Gini Importance)')
            ax.set_ylabel('Feature')
            ax.set_title(f'Top {len(top_features_df)} Feature Importances')
            ax.invert_yaxis()
            fig.tight_layout()
            logger.info("Feature importance visualization generated.")
            return fig
        except Exception as e:
            logger.error(f"Error during visualization: {e}", exc_info=True)
            if 'fig' in locals() and isinstance(locals()['fig'], plt.Figure): plt.close(locals()['fig'])
            return None

    def save(self, filepath: str):
        """Saves the trained model state to a file using joblib.

        The saved state includes the fitted model, parameters, feature names,
        classes, and training status.

        Args:
            filepath: The path (including filename) where the model state will be saved.

        Raises:
            NotFittedError: If the model is not fitted.
            ModelError: If saving fails (e.g., directory creation error, joblib error).
        """
        self._check_is_fitted()
        logger.info(f"Saving trained model state to {filepath}...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure dir exists
        model_state = {
            'model_params': self.model.get_params(),
            'model_fitted': self.model,
            'feature_names_when_trained': self.feature_names_when_trained,
            'n_features_when_trained': self.n_features_when_trained,
            'classes_when_trained': self.classes_when_trained,
            'is_trained': self.is_trained
        }
        try:
            joblib.dump(model_state, filepath, compress=3)
            logger.info(f"Model state saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model state: {e}", exc_info=True)
            raise ModelError(f"Failed to save model state to {filepath} ({type(e).__name__}): {e}") from e

    def load(self, filepath: str) -> 'GradingModel':
        """Loads a trained model state from a file.

        Overwrites the current model's state with the loaded state.

        Args:
            filepath: The path to the saved model state file.

        Returns:
            The GradingModel instance (self) with the loaded state.

        Raises:
            FileNotFoundError: If the specified filepath does not exist.
            ModelError: If the file format is invalid, keys are missing,
                types are incorrect, feature count mismatches, or another
                loading error occurs.
        """
        logger.info(f"Loading model state from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model state file not found at path: {filepath}")
        try:
            model_state = joblib.load(filepath)
            required_keys = ['model_fitted', 'feature_names_when_trained', 'n_features_when_trained', 'classes_when_trained', 'is_trained']
            if not all(key in model_state for key in required_keys):
                missing = set(required_keys) - set(model_state.keys())
                raise ModelError(f"Invalid saved model format in '{filepath}'. Missing keys: {missing}")
            if not isinstance(model_state['model_fitted'], RandomForestClassifier):
                raise ModelError(f"Loaded 'model_fitted' from '{filepath}' is not a RandomForestClassifier (got {type(model_state['model_fitted'])}).")

            self.model = model_state['model_fitted']
            self.feature_names_when_trained = model_state['feature_names_when_trained']
            self.n_features_when_trained = model_state['n_features_when_trained']
            self.classes_when_trained = model_state['classes_when_trained']
            self.is_trained = model_state['is_trained']

            if not self.is_trained: logger.warning("Loading a model state marked as not trained.")
            if hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ != self.n_features_when_trained:
                 raise ModelError(f"Inconsistency: Loaded model expects {self.model.n_features_in_} features, saved state indicates {self.n_features_when_trained}.")

            logger.info(f"Model state loaded successfully from {filepath} (Trained: {self.is_trained})")
            return self
        except Exception as e:
            logger.error(f"Error loading model state from {filepath}: {e}", exc_info=True)
            self.is_trained = False # Ensure clean state on error
            raise ModelError(f"Unexpected error loading model state from {filepath} ({type(e).__name__}): {e}") from e


# --- Grade Classification Result Structure ---
@dataclass
class GradeClassification:
    """Structure to hold the results of a single player grading prediction."""
    grade: str                      # Predicted grade ('A', 'B', 'C', 'D', or 'N/A')
    confidence: float               # Confidence score (0-1) for the predicted grade
    conative_stage: int             # Determined conative stage (1-5, or 0 if error)
    stage_scores: Optional[Dict[int, float]] # DEPRECATED: Now holds calculated metrics from conative stage.
    explanation: str                # Human-readable explanation report (or error message)
    feature_importance: Optional[Dict[str, float]] = None # Feature importance from ML model
    processed_features: Optional[Dict[str, float]] = None # Processed features used by ML model (name: value)
    player_id: Optional[str] = None # Optional identifier for the player

    def __post_init__(self):
        # Clamp confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.grade not in ['A', 'B', 'C', 'D', 'N/A']:
             logger.warning(f"GradeClassification created with invalid grade '{self.grade}'.")


# --- Grade Classification Result Structure ---
@dataclass
class ShotClassification:
    """Structure to hold the results of a single shot classification prediction."""
    shot: str                       # Predicted shot ('Lob', etc)
    confidence: float               # Confidence score (0-1) for the predicted grade
    other_possibles: Optional[Dict[str, float]] = None # Other possible shots and their confidence scores

    def __post_init__(self):
        # Clamp confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))
        #if self.grade not in ['A', 'B', 'C', 'D', 'N/A']:
        #     logger.warning(f"GradeClassification created with invalid grade '{self.grade}'.")








