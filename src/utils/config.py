# MediaPipe FaceLandmarker ve kamera için modüler sınıflar
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

class FaceLandmarkerLoader:
    @staticmethod
    def load(model_path, num_faces=1):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=num_faces
        )
        return vision.FaceLandmarker.create_from_options(options)

class CameraLoader:
    @staticmethod
    def open(index=0):
        return cv2.VideoCapture(index)
import os

class Config:
    # Model Paths
    FACE_LANDMARKER_PATH = os.getenv('FACE_LANDMARKER_PATH', 'models/face_landmarker.task')
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/emotion_rf_model.joblib')
    BLENDSHAPE_SAVE_PATH = "data/blendshapes_only.csv"
    GEOM_SAVE_PATH = "data/distances_angles_only.csv"
    #Other Paths
    TRAIN_DATA_SAVE_PATH = "data/blendshapes_dataset.csv"
    # Camera Settings
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', 1280))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', 720))
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 0))
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', 30))

    # Visualization
    #SHOW_LANDMARKS = bool(int(os.getenv('SHOW_LANDMARKS', 1)))
    #WINDOW_NAME = os.getenv('WINDOW_NAME', 'FaceLandmarker')

    # Feature Extraction
    #BLENDSHAPE_THRESHOLD = float(os.getenv('BLENDSHAPE_THRESHOLD', 0.1))

    @classmethod
    def validate(cls):
        assert os.path.exists(cls.FACE_LANDMARKER_PATH), f"Model path not found: {cls.FACE_LANDMARKER_PATH}"
        assert cls.CAMERA_WIDTH > 0 and cls.CAMERA_HEIGHT > 0, "Camera dimensions must be positive"
        assert 0 <= cls.CAMERA_INDEX < 10, "Camera index out of range"
        assert cls.CAMERA_FPS > 0, "FPS must be positive"

# Kullanım örneği:
# Config.validate()
# print(Config.CAMERA_WIDTH)