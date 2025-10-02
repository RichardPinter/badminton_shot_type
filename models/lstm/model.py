# ABOUTME: Simplified model structures for LSTM shot classification
# ABOUTME: Contains only ShotClassification dataclass for shot prediction results

"""
Shot Classification Result Structure for Badminton Shot Classifier.

Contains:
- ShotClassification: Dataclass to hold the results of a shot classification prediction.

Author: Drew (simplified for LSTM standalone use)
"""

from dataclasses import dataclass
from typing import Optional, Dict

# --- Shot Classification Result Structure ---
@dataclass
class ShotClassification:
    """Structure to hold the results of a single shot classification prediction."""
    shot: str                       # Predicted shot ('clear', 'drive', 'drop', 'lob', 'net', 'smash')
    confidence: float               # Confidence score (0-1) for the predicted shot
    other_possibles: Optional[Dict[str, float]] = None # Other possible shots and their confidence scores

    def __post_init__(self):
        # Clamp confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))
