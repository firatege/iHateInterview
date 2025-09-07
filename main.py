
"""
Mini Real-Time Engagement Detection App (Rule-Based)

- Face detection with camera (FaceLandmarker)
- Engagement state detection with blendshapes
- Engagement analysis with rule-based system
- Shows results on screen
"""

import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.utils import config
from src.micro_expressions.movement_patterns_clean import FrameBuffer, MovementPatternDetector
from src.visualization.advanced_visualizer import AdvancedVisualizer
import warnings
warnings.filterwarnings("ignore")

# Load config
settings = config.Config()

# Buffer and detector for movement pattern recognition
frame_buffer = FrameBuffer(max_frames=60)  # ~2 seconds (30fps)
pattern_detector = MovementPatternDetector()

# Persistent list storing detected micro expressions
persistent_patterns = []
pattern_timestamps = {}  # Timestamp for each pattern
pattern_display_time = 5.0  # How many seconds to display

# Visualizer for advanced visualization
USE_ADVANCED_VISUALIZATION = True  # Toggle advanced visualization

# Loading FaceLandmarker and camera from modular config
detector = config.FaceLandmarkerLoader.load(settings.FACE_LANDMARKER_PATH, num_faces=1)
cap = config.CameraLoader.open(settings.CAMERA_INDEX)

# Create visualizer for advanced visualization
if USE_ADVANCED_VISUALIZATION:
    # Get camera dimensions
    _, frame = cap.read()
    if frame is not None:
        frame_height, frame_width = frame.shape[:2]
        visualizer = AdvancedVisualizer(frame_width, frame_height)
    else:
        USE_ADVANCED_VISUALIZATION = False
        print("Camera image cannot be retrieved, advanced visualization disabled.")

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
        
        # Print top 3 blendshapes on screen (to take up less space)
        top_blendshapes = sorted(blend_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        for idx, (name, score) in enumerate(top_blendshapes):
            text = f"{name}: {score:.2f}"
            cv2.putText(frame, text, (30, 30 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Draw landmarks (optional)
        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                for lm in landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Movement pattern recognition - add to buffer with face landmarks
            frame_buffer.add_frame(frame.copy(), result.face_landmarks[0], blendshapes)
            detected_patterns = pattern_detector.detect_patterns(frame_buffer)
            
            # Current time
            current_time = time.time()
            
            # Delete old patterns (if certain time has passed)
            patterns_to_remove = []
            for pattern in persistent_patterns:
                if pattern in pattern_timestamps:
                    if current_time - pattern_timestamps[pattern] > pattern_display_time:
                        patterns_to_remove.append(pattern)
            
            for pattern in patterns_to_remove:
                if pattern in persistent_patterns:
                    persistent_patterns.remove(pattern)
                    if pattern in pattern_timestamps:
                        del pattern_timestamps[pattern]
            
            # Always update the main engagement state
            if detected_patterns:
                main_engagement = detected_patterns[0]
                # Update the list, put engagement state at the beginning
                if 'Engaged' in persistent_patterns or 'Disengaged' in persistent_patterns or 'Neutral' in persistent_patterns:
                    # Remove previous state from the list
                    persistent_patterns = [p for p in persistent_patterns 
                                        if p != 'Engaged' and p != 'Disengaged' and p != 'Neutral']
                
                # Add main engagement state to the beginning and update timestamp
                persistent_patterns.insert(0, main_engagement)
                pattern_timestamps[main_engagement] = current_time
                
                # Add other micro expressions (only new ones)
                for pattern in detected_patterns[1:]:
                    if pattern not in persistent_patterns:
                        persistent_patterns.append(pattern)
                        pattern_timestamps[pattern] = current_time  # Add timestamp
            
            # Show all detected persistent patterns on screen
            if persistent_patterns:
                y_offset = 130  # Start below the blendshape list
                
                # English translation dictionary
                turkish_patterns = {
                    'Engaged': 'Engaged',
                    'Disengaged': 'Disengaged', 
                    'Neutral': 'Neutral',
                    'LookingUp_Thinking': 'Looking Up/Thinking',
                    'LookingDown_Avoidance': 'Looking Down/Avoiding',
                    'LookingLeft_Distracted': 'Looking Left/Distracted',
                    'LookingRight_Distracted': 'Looking Right/Distracted',
                    'EyeRoll_Annoyed': 'Eye Rolling/Annoyed',
                    'EyeSquint_Suspicious': 'Eye Squinting/Suspicious',
                    'EyeWiden_Surprised': 'Eyes Widening/Surprised',
                    'Thinking_Contemplating': 'Thinking',
                    'Confusion_Puzzled': 'Confused/Puzzled',
                    'Concentration_Focused': 'Concentrated/Focused',
                    'Surprise_Shocked': 'Surprised/Shocked',
                    'Skepticism_Doubtful': 'Skeptical/Doubtful',
                    'Boredom_Tired': 'Bored/Tired',
                    'Stress_Tense': 'Stressed/Tense',
                    'Contemplation_DeepThought': 'Deep Contemplation',
                    'Doubt_Uncertain': 'Doubt/Uncertainty',
                    'Interest_Curious': 'Interested/Curious',
                    'Fatigue_Exhausted': 'Fatigued/Exhausted',
                    'Alertness_Focused': 'Alert/Focused',
                    'Discomfort_Uneasy': 'Uncomfortable/Uneasy',
                    # Newly added engagement indicators
                    'Smile_Happy': 'Smiling/Happy',
                    'Excitement_Enthusiastic': 'Excited/Enthusiastic',
                    'Agreement_Approving': 'Agreement/Approving',
                    'Appreciation_Impressed': 'Appreciation/Impressed',
                    'Curiosity_Interested': 'Curious/Interested'
                }
                
                for idx, pattern in enumerate(persistent_patterns):
                    # Select color based on pattern type
                    if pattern == 'Engaged':
                        color = (0, 255, 0)  # Green
                    elif pattern == 'Disengaged':
                        color = (0, 0, 255)  # Red
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
                    # Yeni eklenen ilgi ifadeleri için renkler
                    elif 'Smile' in pattern:
                        color = (0, 255, 100)  # Açık yeşil - Gülümseme
                    elif 'Excitement' in pattern:
                        color = (100, 255, 0)  # Yeşil sarı - Heyecan
                    elif 'Agreement' in pattern:
                        color = (0, 200, 100)  # Yeşil turkuaz - Onaylama
                    elif 'Appreciation' in pattern:
                        color = (50, 200, 50)  # Orta yeşil - Takdir
                    elif 'Curiosity' in pattern:
                        color = (0, 200, 200)  # Turkuaz - Merak
                    else:
                        color = (255, 255, 255)  # Beyaz - Diğer
                    
                    display_text = turkish_patterns.get(pattern, pattern)
                    cv2.putText(frame, display_text, 
                              (30, y_offset + idx * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Ana engagement durumunu büyük yazıyla göster
                main_engagement = persistent_patterns[0] if persistent_patterns else 'Neutral'
                if main_engagement in ['Engaged', 'Disengaged', 'Neutral']:
                    main_color = (0, 255, 0) if main_engagement == 'Engaged' else (0, 0, 255) if main_engagement == 'Disengaged' else (255, 165, 0)
                    main_text = turkish_patterns.get(main_engagement, main_engagement)
                    cv2.putText(frame, f"Durum: {main_text}", 
                              (frame.shape[1] - 250, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, 2)
    # Use advanced visualization mode
    if USE_ADVANCED_VISUALIZATION and result.face_landmarks:
        landmarks = result.face_landmarks[0]
        blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
        
        # Gelişmiş görselleştirme işlemlerini uygula
        frame = visualizer.process_frame(frame, landmarks, blendshapes, persistent_patterns)
        
    # Standart (eski) görselleştirme
    else:
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
            
            # Tespit edilen tüm kalıcı patterns'ı ekranda göster
            if persistent_patterns:
                y_offset = 130  # Blendshape listesinin altından başla
                
                for idx, pattern in enumerate(persistent_patterns):
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
                    # Yeni eklenen ilgi ifadeleri için renkler
                    elif 'Smile' in pattern:
                        color = (0, 255, 100)  # Açık yeşil - Gülümseme
                    elif 'Excitement' in pattern:
                        color = (100, 255, 0)  # Yeşil sarı - Heyecan
                    elif 'Agreement' in pattern:
                        color = (0, 200, 100)  # Yeşil turkuaz - Onaylama
                    elif 'Appreciation' in pattern:
                        color = (50, 200, 50)  # Orta yeşil - Takdir
                    elif 'Curiosity' in pattern:
                        color = (0, 200, 200)  # Turkuaz - Merak
                    else:
                        color = (255, 255, 255)  # Beyaz - Diğer
                    
                    display_text = turkish_patterns.get(pattern, pattern)
                    cv2.putText(frame, display_text, 
                              (30, y_offset + idx * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Ana engagement durumunu büyük yazıyla göster
                main_engagement = persistent_patterns[0] if persistent_patterns else 'Neutral'
                if main_engagement in ['Engaged', 'Disengaged', 'Neutral']:
                    main_color = (0, 255, 0) if main_engagement == 'Engaged' else (0, 0, 255) if main_engagement == 'Disengaged' else (255, 165, 0)
                    main_text = turkish_patterns.get(main_engagement, main_engagement)
                    cv2.putText(frame, f"Durum: {main_text}", 
                              (frame.shape[1] - 250, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, 2)
        else:
            cv2.putText(frame, "No blendshapes detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    # Title change - add advanced mode info
    title = 'Micro Expression Detection System - Advanced View' if USE_ADVANCED_VISUALIZATION else 'Micro Expression Detection System - 20+ Behaviors'
    cv2.imshow(title, frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Key controls
    if key == ord('q'):
        break
    elif key == ord('c'):  # Clear with 'c' key
        persistent_patterns = []
        pattern_timestamps = {}
        print("All text on screen has been cleared")
    elif key == ord('v'):  # Toggle visualization mode with 'v' key
        USE_ADVANCED_VISUALIZATION = not USE_ADVANCED_VISUALIZATION
        print(f"Advanced visualization: {'Enabled' if USE_ADVANCED_VISUALIZATION else 'Disabled'}")

cap.release()
cv2.destroyAllWindows()