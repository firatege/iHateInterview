
__all__ = ["GeometricFeatures"]
import numpy as np


class GeometricFeatures:
    """
    Class for extracting geometric features from facial landmarks.
    """
    def __init__(self, landmarks):
        self.landmarks = np.array(landmarks)

    def compute_distances(self):
        """
        Compute pairwise distances between key landmarks.
        """
        if len(self.landmarks) < 2:
            return np.array([])
        dists = np.linalg.norm(self.landmarks[:, np.newaxis] - self.landmarks[np.newaxis, :], axis=-1)
        return dists[np.triu_indices(len(self.landmarks), k=1)]

    def compute_angles(self):
        """
        Compute angles formed by triplets of key landmarks.
        """
        if len(self.landmarks) < 3:
            return np.array([])
        angles = []
        for i in range(len(self.landmarks)):
            for j in range(len(self.landmarks)):
                for k in range(len(self.landmarks)):
                    if i != j and j != k and i != k:
                        v1 = self.landmarks[i] - self.landmarks[j]
                        v2 = self.landmarks[k] - self.landmarks[j]
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                        angles.append(angle)
        return np.array(angles)

    def get_features(self):
        """
        Combine distances and angles into a single feature vector.
        """
        distances = self.compute_distances()
        angles = self.compute_angles()
        return np.concatenate([distances, angles])