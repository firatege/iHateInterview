import joblib
from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    """Abstract base class for all classifiers."""
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)

    @abstractmethod
    def predict(self, X):
        """Predict method to be implemented by subclasses."""
        pass

    def load_model(self, path):
        """Load model from file."""
        try:
            self.model = joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def save_model(self, path):
        """Save model to file."""
        try:
            joblib.dump(self.model, path)
        except Exception as e:
            raise RuntimeError(f"Model saving failed: {e}")

class ModelLoader:
    """Utility class for loading models."""
    @staticmethod
    def load(path):
        try:
            return joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

class PredictionValidator:
    """Utility class for validating predictions."""
    @staticmethod
    def validate(prediction, valid_labels):
        if prediction not in valid_labels:
            raise ValueError(f"Invalid prediction: {prediction}")
        return True
