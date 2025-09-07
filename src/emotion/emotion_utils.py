import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Constants (reduced set used in this project)
EMOTION_LABELS = ["Engaged", "Disengaged", "Neutral"]
EMOTION_COLORS = {
    "Engaged": (0, 255, 0),      # green
    "Disengaged": (0, 0, 255),   # red
    "Neutral": (255, 165, 0)     # orange
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