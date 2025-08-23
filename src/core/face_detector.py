
import cv2
import mediapipe as mp
from src.utils import config
from src.utils import video_processor


class FaceDetector:
    """Main class for face detection using MediaPipe."""
    def __init__(self, camera_index):
        self.cap = video_processor.CameraManager(camera_index)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

    def get_frame(self):
        return self.cap.read_frame()

    def release(self):
        self.cap.release()

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame)


class LandmarkProcessor:
    """Processes face landmarks from MediaPipe results."""
    @staticmethod
    def has_landmarks(results):
        return bool(getattr(results, 'multi_face_landmarks', None))

    @staticmethod
    def draw_landmarks(frame, results):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )


class BlendshapeExtractor:
    """Placeholder for blendshape extraction logic."""
    @staticmethod
    def extract_blendshapes(results):
        # MediaPipe FaceMesh does not provide blendshapes by default.
        # If using FaceLandmarker, implement extraction here.
        return None


def main():
    settings = config.Config()
    detector = FaceDetector(settings.CAMERA_INDEX)

    while True:
        frame = detector.get_frame()
        if frame is None:
            break

        results = detector.process(frame)

        if LandmarkProcessor.has_landmarks(results):
            print("Face detected!")
            LandmarkProcessor.draw_landmarks(frame, results)
        else:
            print("No face detected.")

        # Blendshape extraction (placeholder)
        blendshapes = BlendshapeExtractor.extract_blendshapes(results)
        # print(blendshapes)  # If implemented

        cv2.imshow('Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.release()
    cv2.destroyAllWindows()
    print("Test completed!")


if __name__ == "__main__":
    main()
