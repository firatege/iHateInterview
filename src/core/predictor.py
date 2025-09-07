"""Predictor wrapper that loads a trained sklearn model, scaler and label encoder
and provides simple prediction helpers for blendshape vectors/dicts.
"""
from pathlib import Path
import joblib
import numpy as np
from typing import Dict, Any, Optional


class Predictor:
    def __init__(self, model_path: str = "models/emotion_rf_model.joblib", scaler_path: str = "models/scaler.joblib", le_path: str = "models/label_encoder.joblib"):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.le_path = Path(le_path)
        self.model = None
        self.scaler = None
        self.le = None
        self._load()

    def _load(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
        else:
            # scaler optional (but recommended)
            self.scaler = None
        if self.le_path.exists():
            self.le = joblib.load(self.le_path)
        else:
            self.le = None

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def predict(self, X: np.ndarray):
        Xs = self._prepare(X)
        preds = self.model.predict(Xs)
        if self.le is not None:
            try:
                return self.le.inverse_transform(preds)
            except Exception:
                return preds
        return preds

    def predict_proba(self, X: np.ndarray):
        Xs = self._prepare(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(Xs)
        raise AttributeError('Model has no predict_proba')

    def predict_from_blendshape_dict(self, d: Dict[str, Any], feature_order: Optional[list] = None):
        """Accepts a blendshape dict {name:score,...}. If feature_order is provided,
        it will be used to order values; otherwise, dict.values() order is used.
        """
        if feature_order:
            vec = [d.get(k, 0.0) for k in feature_order]
        else:
            vec = list(d.values())
        return self.predict(np.array(vec, dtype=np.float32))
