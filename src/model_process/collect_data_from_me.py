import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os
import time
import numpy as np
import argparse
import threading
import queue
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

# Global detector - load once
detector = None

# =======================
# Helper Functions
# =======================
def blendshapes_to_dataframe(blendshapes, emotion):
    blendshape_data = {b.category_name: b.score for b in blendshapes}
    blendshape_data["emotion"] = emotion
    df = pd.DataFrame([blendshape_data])
    return df

def geom_to_dataframe(geom_features, emotion):
    geom_data = {f"geom_{i}": v for i, v in enumerate(geom_features)}
    geom_data["emotion"] = emotion
    df = pd.DataFrame([geom_data])
    return df

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=5.0, help="Minutes per emotion (default 5)")
    return p.parse_args()

class CameraManager:
    def __init__(self):
        self.cap = None
        self.running = False
        
    def initialize(self):
        """Agresif kamera başlatma"""
        print("Kamera başlatılıyor...")
        
        # Önce tüm kameraları serbest bırak
        for i in range(10):
            try:
                temp_cap = cv2.VideoCapture(i)
                temp_cap.release()
            except:
                pass
        
        time.sleep(0.5)  # Kısa bekleme
        
        # Kamerayı bul ve aç
        for idx in range(10):
            try:
                print(f"Kamera {idx} deneniyor...")
                cap = cv2.VideoCapture(idx)
                
                if cap.isOpened():
                    # Test frame oku
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Kamera {idx} başarılı!")
                        
                        # Agresif ayarlar
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FPS, 15)  # Daha düşük FPS
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        
                        # Warm up
                        for _ in range(3):
                            cap.read()
                        
                        self.cap = cap
                        self.running = True
                        return True
                
                cap.release()
            except Exception as e:
                print(f"Kamera {idx} hatası: {e}")
        
        raise RuntimeError("Hiçbir kamera bulunamadı!")
    
    def read_frame_safe(self):
        """Güvenli frame okuma"""
        if not self.running or not self.cap:
            return False, None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                # 3 kez dene
                for _ in range(3):
                    ret, frame = self.cap.read()
                    if ret:
                        break
                    time.sleep(0.01)
            return ret, frame
        except Exception as e:
            print(f"Frame okuma hatası: {e}")
            return False, None
    
    def cleanup(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except:
                pass

class MediaPipeProcessor:
    def __init__(self):
        self.detector = None
        self.processing_queue = queue.Queue(maxsize=1)  # Tek frame
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.last_result = None
        
    def initialize(self):
        """MediaPipe başlat"""
        try:
            print("MediaPipe yükleniyor...")
            self.detector = config.FaceLandmarkerLoader.load(MODEL_PATH, num_faces=1)
            print("✅ MediaPipe yüklendi!")
            
            # Worker thread başlat
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            return True
        except Exception as e:
            print(f"❌ MediaPipe yükleme hatası: {e}")
            return False
    
    def _worker(self):
        """Arka plan MediaPipe işlemci"""
        frame_skip_counter = 0
        
        while not self.stop_event.is_set():
            try:
                # Frame al (timeout ile)
                frame = self.processing_queue.get(timeout=0.1)
                
                # Her 3. frame'i işle (performans için)
                frame_skip_counter += 1
                if frame_skip_counter % 3 != 0:
                    continue
                
                # MediaPipe işle
                try:
                    # Frame'i küçült (performans için)
                    small_frame = cv2.resize(frame, (320, 240))
                    
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    )
                    
                    result = self.detector.detect(mp_image) # type: ignore
                    
                    # Sonucu koy (non-blocking)
                    try:
                        self.result_queue.put_nowait(result)
                        self.last_result = result
                    except queue.Full:
                        pass  # Queue doluysa geç
                        
                except Exception as e:
                    print(f"MediaPipe işlem hatası: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker hatası: {e}")
                break
    
    def process_async(self, frame):
        """Frame'i async işleme gönder"""
        try:
            self.processing_queue.put_nowait(frame)
        except queue.Full:
            pass  # Queue doluysa skip
    
    def get_latest_result(self):
        """En son sonucu al"""
        try:
            result = self.result_queue.get_nowait()
            self.last_result = result
            return result
        except queue.Empty:
            return self.last_result  # En son sonucu döndür
    
    def cleanup(self):
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

def main():
    args = parse_args()
    SECONDS_PER_EMOTION = int(args.minutes * 60)
    
    # Kamera başlat
    camera = CameraManager()
    if not camera.initialize():
        return
    
    # MediaPipe başlat
    processor = MediaPipeProcessor()
    if not processor.initialize():
        camera.cleanup()
        return
    
    # OpenCV window
    WINDOW_NAME = "Data Collection - Press Q to quit"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    
    try:
        frame_counter = 0
        
        for emotion_idx, emotion in enumerate(EMOTIONS):
            print(f"\n🎬 {emotion_idx+1}/{len(EMOTIONS)}: {emotion} duygusunu {int(SECONDS_PER_EMOTION/60)} dakika yapın!")
            print("Hazır olduğunuzda herhangi bir tuşa basın...")
            
            # Kullanıcı hazır olana kadar preview göster
            while True:
                ret, frame = camera.read_frame_safe()
                if ret:
                    cv2.putText(frame, f"HAZIRLIK: {emotion}", (10, 30),  # type: ignore
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # type: ignore
                    cv2.putText(frame, "Herhangi bir tusa basin", (10, 70),   # type: ignore
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # type: ignore
                    cv2.imshow(WINDOW_NAME, frame)  # type: ignore

                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Herhangi bir tuş
                    break
            
            # Video writer hazırla: günlük klasör (YYYYMMDD) içinde sakla
            date_str = time.strftime("%Y%m%d")
            date_dir = os.path.join("data", "self_data", date_str)
            ensure_dir(date_dir)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_filename = f"{emotion.lower()}_{timestamp}.avi"
            out_path = os.path.join(date_dir, out_filename)
            
            fourcc = cv2.VideoWriter_fourcc(*"MJPG") # type: ignore
            video_writer = cv2.VideoWriter(out_path, fourcc, 15.0, (640, 480) )
            
            start_time = time.time()
            emotion_frames = 0
            save_counter = 0
            
            print(f"🔴 KAYIT BAŞLADI - {emotion}")
            
            # Ana döngü - sadece kamera ve display
            while camera.running and (time.time() - start_time < SECONDS_PER_EMOTION):
                loop_start = time.time()
                frame_counter += 1
                
                # Frame oku
                ret, frame = camera.read_frame_safe()
                if not ret:
                    continue
                
                emotion_frames += 1
                
                # Video yaz
                try: 
                    video_writer.write(frame) # type: ignore
                except:
                    pass
                
                # MediaPipe'a gönder (her 5. frame)
                if frame_counter % 5 == 0:
                    processor.process_async(frame.copy()) # type: ignore
                
                # Sonuçları al
                result = processor.get_latest_result()
                
                # Display frame hazırla
                display_frame = frame.copy() # type: ignore
                
                if result and result.face_blendshapes:
                    blendshapes = result.face_blendshapes[0]
                    
                    # Veri kaydet (her 30 frame'de bir)
                    save_counter += 1
                    if save_counter % 30 == 0:
                        try:
                            # Sadece blendshapes kaydet (geometric features çok yavaş)
                            df_blend = blendshapes_to_dataframe(blendshapes, emotion)
                            if not os.path.exists(BLENDSHAPE_SAVE_PATH):
                                df_blend.to_csv(BLENDSHAPE_SAVE_PATH, mode="w", header=True, index=False)
                            else:
                                df_blend.to_csv(BLENDSHAPE_SAVE_PATH, mode="a", header=False, index=False)
                        except Exception as e:
                            print(f"Kayit hatası: {e}")
                    
                    # En güçlü blendshape göster
                    top = max(blendshapes, key=lambda b: b.score)
                    status = f"{top.category_name}: {top.score:.2f}"
                else:
                    status = "Yuz bulunamadi"
                
                # UI çiz
                elapsed = int(time.time() - start_time)
                remaining = max(0, SECONDS_PER_EMOTION - elapsed)
                
                cv2.putText(display_frame, f"KAYIT: {emotion}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Kalan: {remaining}s", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(display_frame, status, (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Progress bar
                progress = min(1.0, elapsed / SECONDS_PER_EMOTION)
                bar_width = 400
                cv2.rectangle(display_frame, (10, 110), (10 + bar_width, 130), (100, 100, 100), -1)
                cv2.rectangle(display_frame, (10, 110), (10 + int(bar_width * progress), 130), (0, 255, 0), -1)
                
                # Göster
                cv2.imshow(WINDOW_NAME, display_frame)
                
                # FPS kontrol
                loop_time = time.time() - loop_start
                target_time = 1.0 / 15.0  # 15 FPS
                if loop_time < target_time:
                    time.sleep(target_time - loop_time)
                
                # Çıkış kontrolü
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n⏹️ Kullanıcı tarafından durduruldu")
                    video_writer.release()
                    return
            
            # Video kapat
            video_writer.release()
            print(f"✅ {emotion} tamamlandı! ({emotion_frames} frame)")
            
            # Metadata kaydet
            try:
                metadata_path = os.path.join(date_dir, "metadata.csv")
                meta_entry = {
                    "filepath": out_path,
                    "label": emotion, 
                    "start_time": timestamp,
                    "duration_s": SECONDS_PER_EMOTION,
                    "frames": emotion_frames,
                }
                
                import csv
                write_header = not os.path.exists(metadata_path)
                with open(metadata_path, "a", newline="", encoding="utf-8") as mf:
                    writer = csv.DictWriter(mf, fieldnames=list(meta_entry.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(meta_entry)
            except Exception as e:
                print(f"Metadata kayıt hatası: {e}")
        
        print("\n🎉 Tüm duygular tamamlandı!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Klavyeden durduruldu")
    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        processor.cleanup()
        camera.cleanup()
        cv2.destroyAllWindows()
        print("🧹 Temizlik tamamlandı")

if __name__ == "__main__":
    main()