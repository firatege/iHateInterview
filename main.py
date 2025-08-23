
"""
Mini Real-Time Emotion Recognition App (FaceLandmarker version)

- Kamera ile yüz tespiti (FaceLandmarker)
- Blendshape ile özellik çıkarımı
- Model ile duygu tahmini
- Sonucu ekranda gösterir
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.utils import config
from src.core.feature_extractor import FeatureExtractor
from src.emotion.emotion_classifier import EmotionClassifier
from src.emotion.emotion_utils import EMOTION_LABELS, EMOTION_COLORS
import warnings
warnings.filterwarnings("ignore")
# Config ve model yükle
settings = config.Config()
feature_extractor = FeatureExtractor()
classifier = EmotionClassifier(model_path=settings.MODEL_PATH, valid_labels=EMOTION_LABELS)


# FaceLandmarker ve kamera modüler config içinden yükleniyor
detector = config.FaceLandmarkerLoader.load(settings.FACE_LANDMARKER_PATH, num_faces=1)
cap = config.CameraLoader.open(settings.CAMERA_INDEX)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR -> RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    if result.face_blendshapes:
        blendshapes = result.face_blendshapes[0]
        blend_dict = {b.category_name: b.score for b in blendshapes}
        features = feature_extractor.extract_blendshape_vector(blend_dict).reshape(1, -1)
        emotion = classifier.predict(features)[0]
        color = EMOTION_COLORS.get(emotion, (255,255,255))
        cv2.putText(frame, f"Emotion: {emotion}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # En yüksek 5 blendshape'i ekrana yazdır
        top_blendshapes = sorted(blend_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        for idx, (name, score) in enumerate(top_blendshapes):
            text = f"{name}: {score:.2f}"
            cv2.putText(frame, text, (30, 100 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Landmarkları çiz (isteğe bağlı)
        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                for lm in landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    else:
        cv2.putText(frame, "No blendshapes detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    cv2.imshow('Real-Time Emotion App', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
