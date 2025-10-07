"""
Custom exception types for the Badminton Player Grading System.
"""

class GradingSystemError(Exception):
    """Base class for exceptions in this grading system application."""
    pass

class ConfigurationError(GradingSystemError):
    """Exception raised for errors in configuration loading or validation."""
    pass

class DataError(GradingSystemError):
    """Exception raised for errors related to input data format or values."""
    pass

class ModelError(GradingSystemError):
    """Exception raised for errors during model training, loading, or prediction."""
    pass

class PreprocessingError(GradingSystemError):
    """Exception raised for errors during feature preprocessing (fit or transform)."""
    pass

class PredictionError(GradingSystemError):
    """Exception raised for general errors during the prediction workflow in the API."""
    pass 