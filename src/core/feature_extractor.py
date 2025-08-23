import numpy as np
import pandas as pd

class FeatureExtractor:
    """
    Main class for extracting features from raw face data.
    """
    def extract_blendshape_vector(self, blendshapes):
        """
        Convert blendshape dict/list to a numeric feature vector.
        """
        # Example: blendshapes = {'Smile': 0.8, 'Frown': 0.1, ...}
        return np.array(list(blendshapes.values()), dtype=np.float32)

    def extract_geometric_features(self, landmarks):
        """
        Extract geometric features from landmark coordinates.
        """
        # Example: landmarks = [(x1, y1), (x2, y2), ...]
        # Implement geometric calculations here
        return np.array(landmarks).flatten()

class GeometricFeatures:
    """
    Utility class for geometric feature calculations.
    """
    @staticmethod
    def euclidean_distance(p1, p2):
        """
        Calculate Euclidean distance between two points.
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Add more geometric feature methods as needed

class FeatureNormalizer:
    """
    Utility class for normalizing and scaling features.
    """
    @staticmethod
    def normalize(features):
        """
        Normalize feature vector (e.g., min-max scaling).
        """
        features = np.array(features)
        return (features - features.min()) / (features.max() - features.min() + 1e-8)

    @staticmethod
    def standardize(features):
        """
        Standardize feature vector (zero mean, unit variance).
        """
        features = np.array(features)
        return (features - features.mean()) / (features.std() + 1e-8)