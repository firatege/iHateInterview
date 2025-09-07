
"""
Mini Real-Time Engagement Detection App (Rule-Based)

- Kamera ile yüz tespiti (FaceLandmarker)
- Blendshape ile engagement durumu tespiti
- Rule-based sistem ile engagement analizi
- Sonucu ekranda gösterir
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.utils import config
from src.micro_expressions.movement_patterns_clean import FrameBuffer, MovementPatternDetector
import warnings
warnings.filterwarnings("ignore")

# Config yükle
settings = config.Config()

# Hareket pattern tanıma için buffer ve detector
frame_buffer = FrameBuffer(max_frames=60)  # ~2 saniye (30fps)
pattern_detector = MovementPatternDetector()


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
        
        # En yüksek 3 blendshape'i ekrana yazdır (daha az yer kaplasın)
        top_blendshapes = sorted(blend_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        for idx, (name, score) in enumerate(top_blendshapes):
            text = f"{name}: {score:.2f}"
            cv2.putText(frame, text, (30, 30 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Landmarkları çiz (isteğe bağlı)
        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                for lm in landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Hareket pattern tanıma - yüz landmarkları ile buffer'a ekle
            frame_buffer.add_frame(frame.copy(), result.face_landmarks[0], blendshapes)
            detected_patterns = pattern_detector.detect_patterns(frame_buffer)
            
            # Tespit edilen tüm patterns'ı ekranda göster
            if detected_patterns:
                y_offset = 130  # Blendshape listesinin altından başla
                
                for idx, pattern in enumerate(detected_patterns):
                    # Pattern türüne göre renk seç
                    if pattern == 'Engaged':
                        color = (0, 255, 0)  # Yeşil
                    elif pattern == 'Disengaged':
                        color = (0, 0, 255)  # Kırmızı
                    elif pattern == 'Neutral':
                        color = (255, 165, 0)  # Turuncu
                    elif 'Thinking' in pattern or 'Contemplat' in pattern:
                        color = (255, 255, 0)  # Sarı - Düşünme
                    elif 'Looking' in pattern:
                        color = (255, 0, 255)  # Magenta - Bakış yönü
                    elif 'Eye' in pattern:
                        color = (0, 255, 255)  # Cyan - Göz hareketleri
                    elif 'Surprise' in pattern or 'Interest' in pattern:
                        color = (0, 255, 128)  # Açık yeşil - Pozitif
                    elif 'Stress' in pattern or 'Discomfort' in pattern:
                        color = (0, 0, 255)  # Kırmızı - Negatif
                    elif 'Boredom' in pattern or 'Fatigue' in pattern:
                        color = (128, 128, 128)  # Gri - Yorgunluk
                    else:
                        color = (255, 255, 255)  # Beyaz - Diğer
                    
                    # Türkçe çeviri (sadece İngilizce karakterler)
                    turkish_patterns = {
                        'Engaged': 'Ilgili',
                        'Disengaged': 'Ilgisiz', 
                        'Neutral': 'Notr',
                        'LookingUp_Thinking': 'Yukari Bakiyor/Dusunuyor',
                        'LookingDown_Avoidance': 'Asagi Bakiyor/Kaciyor',
                        'LookingLeft_Distracted': 'Sola Bakiyor/Dikkati Dagik',
                        'LookingRight_Distracted': 'Saga Bakiyor/Dikkati Dagik',
                        'EyeRoll_Annoyed': 'Goz Deviriyor/Sinirli',
                        'EyeSquint_Suspicious': 'Goz Kisiyor/Supheli',
                        'EyeWiden_Surprised': 'Goz Genisletiyor/Saskin',
                        'Thinking_Contemplating': 'Dusunce Halinde',
                        'Confusion_Puzzled': 'Kafasi Karisik',
                        'Concentration_Focused': 'Konsantre/Odaklanmis',
                        'Surprise_Shocked': 'Sasirmis/Sok',
                        'Skepticism_Doubtful': 'Supheci/Kuskulu',
                        'Boredom_Tired': 'Sikilmis/Yorgun',
                        'Stress_Tense': 'Stresli/Gergin',
                        'Contemplation_DeepThought': 'Derin Dusunce',
                        'Doubt_Uncertain': 'Suphe/Belirsizlik',
                        'Interest_Curious': 'Ilgili/Merakli',
                        'Fatigue_Exhausted': 'Yorgun/Bitkin',
                        'Alertness_Focused': 'Tetikte/Odaklanmis',
                        'Discomfort_Uneasy': 'Rahatsiz/Huzursuz'
                    }
                    
                    display_text = turkish_patterns.get(pattern, pattern)
                    cv2.putText(frame, display_text, 
                              (30, y_offset + idx * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Ana engagement durumunu büyük yazıyla göster
                main_engagement = detected_patterns[0] if detected_patterns else 'Neutral'
                if main_engagement in ['Engaged', 'Disengaged', 'Neutral']:
                    main_color = (0, 255, 0) if main_engagement == 'Engaged' else (0, 0, 255) if main_engagement == 'Disengaged' else (255, 165, 0)
                    main_text = turkish_patterns.get(main_engagement, main_engagement)
                    cv2.putText(frame, f"Durum: {main_text}", 
                              (frame.shape[1] - 250, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, 2)
    else:
        cv2.putText(frame, "No blendshapes detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    cv2.imshow('Mikro Ifade Tanima Sistemi - 20+ Davranis Tespiti', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()