# src/evaluation.py
"""
Evaluation Module for Badminton Player Grading System.

Provides functions to evaluate the performance of the trained GradingModel,
including standard classification metrics and analysis of boundary cases.

Author: Sujit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.exceptions import NotFittedError
import logging
from typing import List, Dict, Optional, Tuple, Any

# Assuming model.py is in the same directory or src path
# Use relative import for intra-package imports
from .model import GradingModel # To type hint the model object

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define standard grade labels
GRADE_LABELS = ['D', 'C', 'B', 'A'] # Corresponds to numeric 0, 1, 2, 3
NUMERIC_LABELS = [0, 1, 2, 3]

def evaluate_model(model: GradingModel,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   grade_labels: List[str] = GRADE_LABELS
                   ) -> Tuple[Optional[Dict[str, Any]], Optional[plt.Figure]]:
    """
    Evaluates the trained GradingModel's performance on a test set.

    Calculates accuracy, generates a classification report, computes a
    confusion matrix, and creates a confusion matrix plot.

    Args:
        model: The trained GradingModel instance.
        X_test: Preprocessed test feature matrix (n_samples, n_selected_features).
        y_test: True numeric grade labels for the test set (n_samples,).
        grade_labels: List of string labels corresponding to numeric grades (e.g., ['D', 'C', 'B', 'A']).

    Returns:
        Tuple containing:
        - dict: Dictionary containing evaluation metrics ('accuracy', 'classification_report', 'confusion_matrix_raw').
                Returns None if evaluation fails.
        - matplotlib.figure.Figure: Figure object for the confusion matrix plot.
                                    Returns None if plotting fails or model not fitted.
                                    Caller is responsible for showing/saving the figure.
    """
    logger.info(f"Evaluating model performance on test set with shape {X_test.shape}...")

    try:
        # Ensure model is trained (predict methods will raise NotFittedError)
        model._check_is_fitted()

        # --- Make Predictions ---
        y_pred = model.predict(X_test)

        # Ensure labels are valid
        numeric_target_labels = list(range(len(grade_labels))) # e.g., [0, 1, 2, 3]
        if not np.all(np.isin(y_test, numeric_target_labels)):
             logger.warning(f"y_test contains labels outside the expected range {numeric_target_labels}. Metrics might be affected.")
        if not np.all(np.isin(y_pred, numeric_target_labels)):
             logger.warning(f"y_pred contains labels outside the expected range {numeric_target_labels}. Metrics might be affected.")


        # --- Calculate Metrics ---
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, target_names=grade_labels,
                                            labels=numeric_target_labels, output_dict=True, zero_division=0)
        cm_raw = confusion_matrix(y_test, y_pred, labels=numeric_target_labels)

        evaluation_results = {
            'accuracy': accuracy,
            'classification_report': report_dict, # Report as dict
            'confusion_matrix_raw': cm_raw.tolist() # Raw matrix as list of lists
        }
        logger.info(f"Evaluation Metrics Calculated - Accuracy: {accuracy:.4f}")
        # logger.debug(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=grade_labels, labels=numeric_target_labels, zero_division=0)}")
        # logger.debug(f"Confusion Matrix (raw):\n{cm_raw}")


        # --- Plot Confusion Matrix ---
        try:
            fig, ax = plt.subplots(figsize=(8, 6.5)) # Adjusted size slightly
            sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=grade_labels, yticklabels=grade_labels,
                        annot_kws={"size": 12}) # Adjust font size
            ax.set_xlabel('Predicted Grade', fontsize=12)
            ax.set_ylabel('True Grade', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14)
            plt.xticks(rotation=0, ha='center') # Ensure labels are horizontal
            plt.yticks(rotation=0)
            fig.tight_layout()
            logger.info("Confusion matrix plot generated.")
        except Exception as e:
             logger.error(f"Error generating confusion matrix plot: {e}", exc_info=True)
             if 'fig' in locals() and isinstance(locals()['fig'], plt.Figure):
                  plt.close(locals()['fig']) # Close plot if partially created
             fig = None # Indicate plotting failed


        return evaluation_results, fig

    except NotFittedError as e:
         logger.error(f"Cannot evaluate model: {e}")
         return None, None
    except ValueError as e: # Catch potential errors from metrics functions
         logger.error(f"Error during evaluation (check shapes and labels): {e}", exc_info=True)
         return None, None
    except Exception as e:
         logger.error(f"Unexpected error during model evaluation: {e}", exc_info=True)
         return None, None


def analyze_boundary_cases(model: GradingModel,
                           X: np.ndarray,
                           y: Optional[np.ndarray] = None,
                           confidence_threshold: float = 0.6,
                           grade_labels: List[str] = GRADE_LABELS
                           ) -> Optional[pd.DataFrame]:
    """
    Identifies and analyzes predictions with low confidence, potentially near grade boundaries.

    Args:
        model: The trained GradingModel instance.
        X: Preprocessed feature matrix (n_samples, n_selected_features).
        y: Optional true numeric grade labels (n_samples,). If provided, included in output.
        confidence_threshold: Confidence level below which a prediction is considered a boundary case.
        grade_labels: List of string labels corresponding to numeric grades.

    Returns:
        Pandas DataFrame containing information about boundary cases (predicted grade,
        confidence, true grade if provided, probabilities for all classes).
        Returns None if analysis fails or no boundary cases are found.
    """
    logger.info(f"Analyzing boundary cases with confidence threshold < {confidence_threshold}...")

    try:
        # Ensure model is trained
        model._check_is_fitted()

        # --- Get Predictions and Probabilities ---
        y_pred_numeric = model.predict(X)
        probabilities = model.predict_proba(X)

        # Calculate confidence for the predicted class
        confidences = np.max(probabilities, axis=1)

        # --- Create Results DataFrame ---
        results_data = {
            'predicted_numeric': y_pred_numeric,
            'predicted_grade': [grade_labels[i] for i in y_pred_numeric],
            'confidence': confidences
        }
        # Add probabilities for each class
        for i, label in enumerate(grade_labels):
             # Ensure column index exists in probabilities array
             if i < probabilities.shape[1]:
                  results_data[f'prob_{label}'] = probabilities[:, i]
             else:
                  logger.warning(f"Probability column index {i} out of bounds for shape {probabilities.shape}. Skipping prob_{label}.")


        # Add true grade if available
        if y is not None:
            if len(y) != len(y_pred_numeric):
                 logger.warning("Length mismatch between y_true and predictions. Cannot include true grade.")
            else:
                 results_data['true_numeric'] = y
                 results_data['true_grade'] = [grade_labels[i] if 0 <= i < len(grade_labels) else 'Invalid' for i in y]

        results_df = pd.DataFrame(results_data)
        # logger.debug(f"Results DataFrame created with {len(results_df)} samples.")

        # --- Filter for Boundary Cases ---
        boundary_cases_df = results_df[results_df['confidence'] < confidence_threshold].copy()
        boundary_cases_df.sort_values('confidence', ascending=True, inplace=True) # Sort by lowest confidence

        if boundary_cases_df.empty:
            logger.info("No boundary cases found below the specified confidence threshold.")
            return None
        else:
            logger.info(f"Found {len(boundary_cases_df)} boundary cases below confidence threshold {confidence_threshold}.")
            # logger.debug(f"Boundary Cases Head:\n{boundary_cases_df.head()}")
            return boundary_cases_df

    except NotFittedError as e:
        logger.error(f"Cannot analyze boundary cases: {e}")
        return None
    except ValueError as e: # Catch potential errors from predictions/probabilities
         logger.error(f"Error during boundary case analysis (check input shapes): {e}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"Unexpected error during boundary case analysis: {e}", exc_info=True)
        return None

