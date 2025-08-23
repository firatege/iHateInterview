import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Constants
EMOTION_LABELS = ["Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise", "Neutral"]
EMOTION_COLORS = {
    "Happy": (255, 215, 0),
    "Sad": (70, 130, 180),
    "Angry": (220, 20, 60),
    "Fear": (138, 43, 226),
    "Disgust": (34, 139, 34),
    "Surprise": (255, 140, 0),
    "Neutral": (128, 128, 128)
}

# Utility functions
def label_to_index(label):
    """Convert emotion label to index."""
    return EMOTION_LABELS.index(label)

def index_to_label(index):
    """Convert index to emotion label."""
    return EMOTION_LABELS[index]

def preprocess_features(features):
    """Example preprocessing: convert to numpy array and normalize."""
    features = np.array(features)
    return (features - features.min()) / (features.max() - features.min() + 1e-8)

# Metrics
def compute_metrics(y_true, y_pred):
    """Compute accuracy and classification report."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS)
    return acc, report