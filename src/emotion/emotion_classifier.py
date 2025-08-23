
"""
EmotionClassifier: Class for emotion prediction using a trained model.
Inherits from BaseClassifier for unified interface and error handling.
"""

from src.core.base_classifier import BaseClassifier, PredictionValidator

class EmotionClassifier(BaseClassifier):
    def __init__(self, model_path, valid_labels=None):
        super().__init__(model_path)
        self.valid_labels = valid_labels

    def predict(self, features):
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before prediction.")
        prediction = self.model.predict(features)
        if self.valid_labels is not None:
            # Check all predictions are valid
            for p in prediction:
                PredictionValidator.validate(p, self.valid_labels)
        return prediction

    def predict_proba(self, features):
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before prediction.")
        return self.model.predict_proba(features)

