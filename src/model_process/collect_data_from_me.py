import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os
import time
import numpy as np
from src.utils import config
from src.micro_expressions import geometric_features
# =======================
# Settings
# =======================
settings = config.Config()


MODEL_PATH = settings.FACE_LANDMARKER_PATH
BLENDSHAPE_SAVE_PATH = settings.BLENDSHAPE_SAVE_PATH
GEOM_SAVE_PATH = settings.GEOM_SAVE_PATH

EMOTIONS = ["Engaged", "Disengaged", "Neutral"]
SECONDS_PER_EMOTION = 600  # 10 dakika

detector = config.FaceLandmarkerLoader.load(MODEL_PATH, num_faces=1)

# =======================
# Convert blendshapes to DataFrame
# =======================

# =======================
# Convert blendshapes and geometric features to DataFrame
# =======================

# =======================
# Convert blendshapes to DataFrame
# =======================
def blendshapes_to_dataframe(blendshapes, emotion):
    blendshape_data = {b.category_name: b.score for b in blendshapes}
    blendshape_data["emotion"] = emotion
    df = pd.DataFrame([blendshape_data])
    return df

# =======================
# Convert geometric features to DataFrame
# =======================
def geom_to_dataframe(geom_features, emotion):
    geom_data = {f"geom_{i}": v for i, v in enumerate(geom_features)}
    geom_data["emotion"] = emotion
    df = pd.DataFrame([geom_data])
    return df


# =======================
# Webcam loop
# =======================
cap = config.CameraLoader.open(settings.CAMERA_INDEX)

for emotion in EMOTIONS:
    print(f"Şimdi {emotion} duygusunu 10 dakika boyunca yapın!")
    start_time = time.time()
    while cap.isOpened() and (time.time() - start_time < SECONDS_PER_EMOTION):
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        result = detector.detect(mp_image)


        if result.face_blendshapes and result.face_landmarks:
            blendshapes = result.face_blendshapes[0]
            landmarks = result.face_landmarks[0]
            # Convert landmarks to (x, y, z) tuples
            landmark_coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
            # Compute geometric features
            geom = geometric_features.GeometricFeatures(landmark_coords)
            geom_features = geom.get_features()

            # Save blendshapes only
            df_blend = blendshapes_to_dataframe(blendshapes, emotion)
            if not os.path.exists(BLENDSHAPE_SAVE_PATH):
                df_blend.to_csv(BLENDSHAPE_SAVE_PATH, mode="w", header=True, index=False)
            else:
                df_blend.to_csv(BLENDSHAPE_SAVE_PATH, mode="a", header=False, index=False)

            # Save geometric features only
            df_geom = geom_to_dataframe(geom_features, emotion)
            if not os.path.exists(GEOM_SAVE_PATH):
                df_geom.to_csv(GEOM_SAVE_PATH, mode="w", header=True, index=False)
            else:
                df_geom.to_csv(GEOM_SAVE_PATH, mode="a", header=False, index=False)

            # Show strongest blendshape
            top_shape = max(blendshapes, key=lambda b: b.score)
            cv2.putText(frame, f"{emotion} / {top_shape.category_name} ({top_shape.score:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Ekranda hangi duyguda olduğunu göster
        cv2.putText(frame, f"Current: {emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        elapsed = int(time.time() - start_time)
        remaining = SECONDS_PER_EMOTION - elapsed
        cv2.putText(frame, f"Kalan: {remaining}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

        cv2.imshow("Data Collection (Engaged/Disengaged/Neutral)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Kullanıcı tarafından çıkıldı.")
            exit()

print("Tüm duygular için veri toplama tamamlandı!")
cap.release()
cv2.destroyAllWindows()
