import numpy as np
import time

class FrameBuffer:
    """
    Son N frame'in landmark ve blendshape verisini saklar.
    """
    def __init__(self, max_frames=60):
        self.buffer = []
        self.max_frames = max_frames

    def add_frame(self, frame, landmarks, blendshapes):
        self.buffer.append({
            'timestamp': time.time(),
            'frame': frame,
            'landmarks': landmarks,
            'blendshapes': blendshapes
        })
        if len(self.buffer) > self.max_frames:
            self.buffer.pop(0)

    def get_landmark_series(self, landmark_ids):
        # landmark_ids: tek id veya liste
        if isinstance(landmark_ids, int):
            landmark_ids = [landmark_ids]
        series = []
        for entry in self.buffer:
            coords = []
            for lid in landmark_ids:
                lm = entry['landmarks'][lid]
                coords.append((lm.x, lm.y, lm.z))
            series.append(coords)
        return np.array(series)  # shape: (frames, len(landmark_ids), 3)
        
    def get_blendshape_series(self, blendshape_names):
        """
        Belirtilen blendshape değerlerinin zaman serisini döndürür.
        """
        # blendshape_names: tek isim veya liste
        if isinstance(blendshape_names, str):
            blendshape_names = [blendshape_names]
        series = []
        for entry in self.buffer:
            values = []
            for name in blendshape_names:
                # Her bir blendshape için score değerini al
                for bs in entry['blendshapes']:
                    if bs.category_name == name:
                        values.append(bs.score)
                        break
                else:
                    values.append(0.0)  # Blendshape bulunamadı ise 0 değeri ekle
            series.append(values)
        return np.array(series)  # shape: (frames, len(blendshape_names))

    def __len__(self):
        return len(self.buffer)

class MovementPatternDetector:
    """
    Mikro ifadeler ve vücut dili analizi ile etkileşim seviyesini tespit eden sınıf.
    İlgili (Engaged), İlgisiz (Disengaged) ve Nötr (Neutral) durumları tespit edilir.
    """
    def __init__(self):
        # Eşik değerleri
        self.thresholds = {
            # Genel eşikler
            'blendshape_activation': 0.25,  # Blendshape aktivasyon eşiği
            'time_window': 0.5,  # Zaman penceresi (saniye)
            
            # İlgili (Engaged) durumu blendshape eşikleri
            'engaged_smile': 0.3,  # Gülümseme eşiği
            'engaged_eye_wide': 0.25,  # Göz genişleme eşiği
            'engaged_brow_raise': 0.25,  # Kaş kaldırma eşiği
            
            # İlgisiz (Disengaged) durumu blendshape eşikleri
            'disengaged_eye_look': 0.3,  # Göz kaçırma eşiği
            'disengaged_brow_down': 0.25,  # Kaş çatma eşiği
            'disengaged_mouth_press': 0.25,  # Dudak sıkma eşiği
        }
        
        # Durum takip değişkenleri
        self.last_engaged_time = 0
        self.last_disengaged_time = 0
        self.engagement_state = 'Neutral'  # Başlangıç durumu
        self.state_start_time = time.time()
        
        # Detection frekans kontrolü
        self.last_detection_time = 0
        self.detection_interval = 0.1  # 100ms aralıklarla tespit yap
        
        # Tepki izleme için blendshape grup tanımları
        self.engaged_blendshapes = [
            'mouthSmileLeft', 'mouthSmileRight',  # Gülümseme
            'eyeWideLeft', 'eyeWideRight',  # Göz genişleme
            'browOuterUpLeft', 'browOuterUpRight', 'browInnerUp'  # Kaş kaldırma
        ]
        
        self.disengaged_blendshapes = [
            'eyeLookOutLeft', 'eyeLookOutRight',  # Göz kaçırma dışa
            'eyeLookDownLeft', 'eyeLookDownRight',  # Göz kaçırma aşağı
            'browDownLeft', 'browDownRight',  # Kaş çatma
            'mouthPressLeft', 'mouthPressRight',  # Dudak sıkma
            'mouthFrownLeft', 'mouthFrownRight'  # Ağız büzme
        ]
        
        self.all_tracked_blendshapes = list(set(self.engaged_blendshapes + self.disengaged_blendshapes))

    def detect_engagement_state(self, buffer):
        """
        Konuşmaya verilen tepkilere göre ilgili/ilgisiz durumunu tespit eder.
        Blendshape tabanlı mikro ifadeleri analiz ederek etkileşim seviyesini belirler.
        """
        try:
            # Yeterli veri yoksa nötr kabul et
            if len(buffer) < 5:
                return 'Neutral'
                
            # İlgili ve ilgisiz blendshape gruplarını al
            engaged_values = buffer.get_blendshape_series(self.engaged_blendshapes)
            disengaged_values = buffer.get_blendshape_series(self.disengaged_blendshapes)
            
            if len(engaged_values) < 3 or len(disengaged_values) < 3:
                return 'Neutral'
                
            # Son 15 frame'i incele (yaklaşık 0.5 saniye)
            frames_to_check = min(15, len(engaged_values))
            recent_engaged = engaged_values[-frames_to_check:]
            recent_disengaged = disengaged_values[-frames_to_check:]
            
            # İlgili blendshape'lerin ortalama aktivasyon değerlerini hesapla
            engaged_activations = []
            for i in range(len(self.engaged_blendshapes)):
                if i < recent_engaged.shape[1]:
                    engaged_activations.append(np.mean(recent_engaged[:, i]))
                else:
                    engaged_activations.append(0)
            
            # İlgisiz blendshape'lerin ortalama aktivasyon değerlerini hesapla
            disengaged_activations = []
            for i in range(len(self.disengaged_blendshapes)):
                if i < recent_disengaged.shape[1]:
                    disengaged_activations.append(np.mean(recent_disengaged[:, i]))
                else:
                    disengaged_activations.append(0)
            
            # En yüksek ilgili ve ilgisiz aktivasyon değerlerini bul
            max_engaged = max(engaged_activations) if engaged_activations else 0
            max_disengaged = max(disengaged_activations) if disengaged_activations else 0
            
            # Smile özel kontrolü (gülümseme en güçlü ilgili göstergesi)
            smile_left_idx = self.engaged_blendshapes.index('mouthSmileLeft') if 'mouthSmileLeft' in self.engaged_blendshapes else -1
            smile_right_idx = self.engaged_blendshapes.index('mouthSmileRight') if 'mouthSmileRight' in self.engaged_blendshapes else -1
            
            smile_avg = 0
            if smile_left_idx >= 0 and smile_right_idx >= 0:
                if smile_left_idx < len(engaged_activations) and smile_right_idx < len(engaged_activations):
                    smile_avg = (engaged_activations[smile_left_idx] + engaged_activations[smile_right_idx]) / 2
            
            # İlgili tepkilerin ağırlığı (gülümseme için bonus)
            engaged_score = max_engaged
            if smile_avg > self.thresholds['engaged_smile']:
                engaged_score = max(engaged_score, smile_avg * 1.2)  # Gülümseme bonusu
            
            # İlgisiz tepkilerin ağırlığı
            disengaged_score = max_disengaged
            
            # Tepki eşiklerinin kontrolü
            current_time = time.time()
            
            # İlgili durum tespiti
            if engaged_score > self.thresholds['blendshape_activation'] and engaged_score > disengaged_score * 1.2:
                if self.last_engaged_time == 0:
                    self.last_engaged_time = current_time
                elif current_time - self.last_engaged_time > self.thresholds['time_window']:
                    # Süre geçtiyse ilgili olarak işaretle
                    if self.engagement_state != 'Engaged':
                        print(f"Engagement state changed: {self.engagement_state} -> Engaged (score: {engaged_score:.3f})")
                        self.engagement_state = 'Engaged'
                        self.state_start_time = current_time
                    # İlgili durumunu tazele
                    self.last_engaged_time = current_time
                    self.last_disengaged_time = 0
                    return self.engagement_state
            else:
                self.last_engaged_time = 0
            
            # İlgisiz durum tespiti
            if disengaged_score > self.thresholds['blendshape_activation'] and disengaged_score > engaged_score * 1.2:
                if self.last_disengaged_time == 0:
                    self.last_disengaged_time = current_time
                elif current_time - self.last_disengaged_time > self.thresholds['time_window']:
                    # Süre geçtiyse ilgisiz olarak işaretle
                    if self.engagement_state != 'Disengaged':
                        print(f"Engagement state changed: {self.engagement_state} -> Disengaged (score: {disengaged_score:.3f})")
                        self.engagement_state = 'Disengaged'
                        self.state_start_time = current_time
                    # İlgisiz durumunu tazele
                    self.last_disengaged_time = current_time
                    self.last_engaged_time = 0
                    return self.engagement_state
            else:
                self.last_disengaged_time = 0
            
            # Nötr durum - her iki durum da tespit edilmezse veya dengeliyse
            if (max_engaged < self.thresholds['blendshape_activation'] and 
                max_disengaged < self.thresholds['blendshape_activation']):
                # Yeterince uzun süredir nötr durumdaysa durumu güncelle
                if self.engagement_state != 'Neutral' and current_time - self.state_start_time > 1.5:
                    print(f"Engagement state changed: {self.engagement_state} -> Neutral (low activity)")
                    self.engagement_state = 'Neutral'
                    self.state_start_time = current_time
            
            # Mevcut durumu döndür
            return self.engagement_state
                
        except Exception as e:
            print(f"Engagement detection error: {e}")
            return 'Neutral'  # Hata durumunda nötr döndür

    def get_top_active_blendshapes(self, buffer, top_n=5):
        """
        En aktif blendshape'leri tespit eder.
        """
        try:
            if len(buffer) < 3:
                return []
                
            # Tüm takip edilen blendshape'leri al
            values = buffer.get_blendshape_series(list(self.all_tracked_blendshapes))
            
            if len(values) < 3:
                return []
                
            # Son 10 frame'i incele
            frames_to_check = min(10, len(values))
            recent_values = values[-frames_to_check:]
            
            # Her blendshape için ortalama değerleri hesapla
            avg_values = {}
            for i, name in enumerate(self.all_tracked_blendshapes):
                if i < recent_values.shape[1]:
                    avg = np.mean(recent_values[:, i])
                    if avg > 0.05:  # Çok düşük değerleri filtrele
                        avg_values[name] = avg
            
            # En yüksek değerli blendshape'leri sırala
            top_blendshapes = sorted(avg_values.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return top_blendshapes
                
        except Exception as e:
            print(f"Top blendshapes detection error: {e}")
            return []

    def detect_patterns(self, buffer):
        """
        Mikro ifade analizi ile durumu tespit eder ve sonuçları döndürür.
        """
        detected_patterns = []
        current_time = time.time()
        
        # İlgili/ilgisiz durumunu tespit et
        engagement_state = self.detect_engagement_state(buffer)
        detected_patterns.append(engagement_state)
        
        # Frekans kontrolü - sadece belirli aralıklarla diğer tespitleri yap
        if current_time - self.last_detection_time < self.detection_interval:
            return detected_patterns
        
        self.last_detection_time = current_time
        
        # Yeni mikro ifade tespitleri (daha az sıklıkta)
        patterns_to_check = [
            ('LookingUp', self.detect_looking_up),
            ('LookingDown', self.detect_looking_down),
            ('LookingLeft', self.detect_looking_left),
            ('LookingRight', self.detect_looking_right),
            ('EyeRoll', self.detect_eye_roll),
            ('EyeSquint', self.detect_eye_squint),
            ('EyeWiden', self.detect_eye_widen),
            ('ThinkingExpression', self.detect_thinking_expression),
            ('Confusion', self.detect_confusion),
            ('Concentration', self.detect_concentration),
            ('Surprise', self.detect_surprise),
            ('Skepticism', self.detect_skepticism),
            ('Boredom', self.detect_boredom),
            ('Stress', self.detect_stress),
            ('Contemplation', self.detect_contemplation),
            ('Doubt', self.detect_doubt),
            ('Interest', self.detect_interest),
            ('Fatigue', self.detect_fatigue),
            ('Alertness', self.detect_alertness),
            ('Discomfort', self.detect_discomfort),
            # Yeni eklenen ilgi göstergeleri
            ('Smile', self.detect_smile),
            ('Excitement', self.detect_excitement),
            ('Agreement', self.detect_agreement),
            ('Appreciation', self.detect_appreciation),
            ('Curiosity', self.detect_curiosity)
        ]
        
        for pattern_name, detection_func in patterns_to_check:
            result = detection_func(buffer)
            if result:
                detected_patterns.append(result)
                print(f"Tespit edilen: {result}")
        
        # En aktif blendshape'leri tespit et (debug için)
        top_blendshapes = self.get_top_active_blendshapes(buffer)
        if top_blendshapes:
            blendshape_str = ", ".join([f"{name}: {value:.2f}" for name, value in top_blendshapes[:3]])
            #print(f"Top active blendshapes: {blendshape_str}")
        
        # Tespit edilen durumları döndür
        return detected_patterns

    # ============= YENİ MİKRO İFADE TESPİT METHODLARİ =============
    
    def detect_looking_up(self, buffer):
        """Yukarı bakma - Düşünme, hatırlama veya sıkılma göstergesi"""
        try:
            if len(buffer) < 5:
                return None
            
            # Yukarı bakma blendshape'leri
            look_up_blendshapes = ['eyeLookUpLeft', 'eyeLookUpRight']
            values = buffer.get_blendshape_series(look_up_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-10:]  # Son 10 frame
            avg_values = np.mean(recent_values, axis=0)
            max_look_up = np.max(avg_values)
            
            if max_look_up > 0.3:
                return 'LookingUp_Thinking'
                
        except Exception as e:
            print(f"Looking up detection error: {e}")
        return None
    
    def detect_looking_down(self, buffer):
        """Aşağı bakma - Utanma, üzüntü veya kaçınma göstergesi"""
        try:
            if len(buffer) < 5:
                return None
            
            look_down_blendshapes = ['eyeLookDownLeft', 'eyeLookDownRight']
            values = buffer.get_blendshape_series(look_down_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-10:]
            avg_values = np.mean(recent_values, axis=0)
            max_look_down = np.max(avg_values)
            
            if max_look_down > 0.35:
                return 'LookingDown_Avoidance'
                
        except Exception as e:
            print(f"Looking down detection error: {e}")
        return None
    
    def detect_looking_left(self, buffer):
        """Sola bakma - Kaçınma veya dikkat dağınıklığı"""
        try:
            if len(buffer) < 5:
                return None
            
            look_left_blendshapes = ['eyeLookOutLeft', 'eyeLookInRight']
            values = buffer.get_blendshape_series(look_left_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-8:]
            avg_values = np.mean(recent_values, axis=0)
            max_look_left = np.max(avg_values)
            
            if max_look_left > 0.4:
                return 'LookingLeft_Distracted'
                
        except Exception as e:
            print(f"Looking left detection error: {e}")
        return None
    
    def detect_looking_right(self, buffer):
        """Sağa bakma - Kaçınma veya dikkat dağınıklığı"""
        try:
            if len(buffer) < 5:
                return None
            
            look_right_blendshapes = ['eyeLookOutRight', 'eyeLookInLeft']
            values = buffer.get_blendshape_series(look_right_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-8:]
            avg_values = np.mean(recent_values, axis=0)
            max_look_right = np.max(avg_values)
            
            if max_look_right > 0.4:
                return 'LookingRight_Distracted'
                
        except Exception as e:
            print(f"Looking right detection error: {e}")
        return None
    
    def detect_eye_roll(self, buffer):
        """Göz devirme - Sıkılma, sinirlenme veya eleştiri"""
        try:
            if len(buffer) < 8:
                return None
            
            # Kombinasyon: önce yukarı sonra yana hareket
            up_blendshapes = ['eyeLookUpLeft', 'eyeLookUpRight']
            side_blendshapes = ['eyeLookOutLeft', 'eyeLookOutRight']
            
            up_values = buffer.get_blendshape_series(up_blendshapes)
            side_values = buffer.get_blendshape_series(side_blendshapes)
            
            if len(up_values) < 8 or len(side_values) < 8:
                return None
            
            # Son 8 frame'de pattern ara
            recent_up = up_values[-8:]
            recent_side = side_values[-8:]
            
            # Önce yukarı bakma var mı?
            up_peak = np.max(recent_up[:4])  # İlk yarı
            side_peak = np.max(recent_side[4:])  # İkinci yarı
            
            if up_peak > 0.3 and side_peak > 0.25:
                return 'EyeRoll_Annoyed'
                
        except Exception as e:
            print(f"Eye roll detection error: {e}")
        return None
    
    def detect_eye_squint(self, buffer):
        """Göz kısma - Şüphe, konsantrasyon veya rahatsızlık"""
        try:
            if len(buffer) < 8:  # Daha fazla frame gerekli
                return None
            
            squint_blendshapes = ['eyeSquintLeft', 'eyeSquintRight']
            values = buffer.get_blendshape_series(squint_blendshapes)
            
            if len(values) < 5:
                return None
            
            recent_values = values[-15:]  # Daha uzun zaman dilimi
            avg_squint = np.mean(recent_values, axis=0)
            max_squint = np.max(avg_squint)
            
            # Eşik değerini yükselttik (0.3 -> 0.6)
            if max_squint > 0.6:
                # Ek kontrol: süreklilik kontrolü
                sustained_squint = np.mean(recent_values[-8:], axis=0)
                if np.max(sustained_squint) > 0.45:
                    return 'EyeSquint_Suspicious'
                
        except Exception as e:
            print(f"Eye squint detection error: {e}")
        return None
    
    def detect_eye_widen(self, buffer):
        """Göz genişletme - Şaşırma, ilgi veya korku"""
        try:
            if len(buffer) < 5:
                return None
            
            wide_blendshapes = ['eyeWideLeft', 'eyeWideRight']
            values = buffer.get_blendshape_series(wide_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-8:]
            avg_wide = np.mean(recent_values, axis=0)
            max_wide = np.max(avg_wide)
            
            if max_wide > 0.4:
                return 'EyeWiden_Surprised'
                
        except Exception as e:
            print(f"Eye widen detection error: {e}")
        return None
    
    def detect_thinking_expression(self, buffer):
        """Düşünme ifadesi - Yukarı bakma + kaş çatma kombinasyonu"""
        try:
            if len(buffer) < 8:
                return None
            
            # Düşünme pattern: yukarı bakma + hafif kaş çatma
            thinking_blendshapes = ['eyeLookUpLeft', 'eyeLookUpRight', 'browDownLeft', 'browDownRight']
            values = buffer.get_blendshape_series(thinking_blendshapes)
            
            if len(values) < 5:
                return None
            
            recent_values = values[-10:]
            avg_values = np.mean(recent_values, axis=0)
            
            look_up_score = np.mean(avg_values[:2])  # İlk 2: yukarı bakma
            brow_down_score = np.mean(avg_values[2:])  # Son 2: kaş çatma
            
            if look_up_score > 0.25 and brow_down_score > 0.15:
                return 'Thinking_Contemplating'
                
        except Exception as e:
            print(f"Thinking expression detection error: {e}")
        return None
    
    def detect_confusion(self, buffer):
        """Kafa karışıklığı - Asimetrik kaş hareketleri"""
        try:
            if len(buffer) < 6:
                return None
            
            # Tek kaş kaldırma veya asimetrik ifadeler
            asymmetric_blendshapes = ['browOuterUpLeft', 'browOuterUpRight', 'browInnerUp']
            values = buffer.get_blendshape_series(asymmetric_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-8:]
            avg_values = np.mean(recent_values, axis=0)
            
            # Asimetri kontrolü
            left_up = avg_values[0]
            right_up = avg_values[1]
            inner_up = avg_values[2]
            
            asymmetry = abs(left_up - right_up)
            if asymmetry > 0.2 and (left_up > 0.2 or right_up > 0.2):
                return 'Confusion_Puzzled'
                
        except Exception as e:
            print(f"Confusion detection error: {e}")
        return None
    
    def detect_concentration(self, buffer):
        """Konsantrasyon - Kaş çatma + göz kısma"""
        try:
            if len(buffer) < 8:  # Daha fazla frame
                return None
            
            concentration_blendshapes = ['browDownLeft', 'browDownRight', 'eyeSquintLeft', 'eyeSquintRight']
            values = buffer.get_blendshape_series(concentration_blendshapes)
            
            if len(values) < 5:
                return None
            
            recent_values = values[-12:]  # Daha uzun zaman
            avg_values = np.mean(recent_values, axis=0)
            
            brow_down = np.mean(avg_values[:2])
            eye_squint = np.mean(avg_values[2:])
            
            # Göz kısma eşiğini yükselttik (0.2 -> 0.4)
            if brow_down > 0.3 and eye_squint > 0.4:
                return 'Concentration_Focused'
                
        except Exception as e:
            print(f"Concentration detection error: {e}")
        return None
    
    def detect_surprise(self, buffer):
        """Şaşırma - Göz genişleme + kaş kaldırma"""
        try:
            if len(buffer) < 5:
                return None
            
            surprise_blendshapes = ['eyeWideLeft', 'eyeWideRight', 'browOuterUpLeft', 'browOuterUpRight']
            values = buffer.get_blendshape_series(surprise_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-6:]
            avg_values = np.mean(recent_values, axis=0)
            
            eye_wide = np.mean(avg_values[:2])
            brow_up = np.mean(avg_values[2:])
            
            if eye_wide > 0.35 and brow_up > 0.3:
                return 'Surprise_Shocked'
                
        except Exception as e:
            print(f"Surprise detection error: {e}")
        return None
    
    def detect_skepticism(self, buffer):
        """Şüphecilik - Tek kaş kaldırma + dudak büzme"""
        try:
            if len(buffer) < 6:
                return None
            
            skeptical_blendshapes = ['browOuterUpLeft', 'browOuterUpRight', 'mouthPucker']
            values = buffer.get_blendshape_series(skeptical_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-8:]
            avg_values = np.mean(recent_values, axis=0)
            
            brow_asymmetry = abs(avg_values[0] - avg_values[1])
            mouth_pucker = avg_values[2] if len(avg_values) > 2 else 0
            
            if brow_asymmetry > 0.15 and mouth_pucker > 0.2:
                return 'Skepticism_Doubtful'
                
        except Exception as e:
            print(f"Skepticism detection error: {e}")
        return None
    
    def detect_boredom(self, buffer):
        """Sıkılma - Aşağı bakma + ağız açma"""
        try:
            if len(buffer) < 8:
                return None
            
            boredom_blendshapes = ['eyeLookDownLeft', 'eyeLookDownRight', 'jawOpen']
            values = buffer.get_blendshape_series(boredom_blendshapes)
            
            if len(values) < 5:
                return None
            
            recent_values = values[-12:]
            avg_values = np.mean(recent_values, axis=0)
            
            look_down = np.mean(avg_values[:2])
            jaw_open = avg_values[2] if len(avg_values) > 2 else 0
            
            if look_down > 0.3 and jaw_open > 0.1:
                return 'Boredom_Tired'
                
        except Exception as e:
            print(f"Boredom detection error: {e}")
        return None
    
    def detect_stress(self, buffer):
        """Stres - Kaş çatma + dudak sıkma"""
        try:
            if len(buffer) < 6:
                return None
            
            stress_blendshapes = ['browDownLeft', 'browDownRight', 'mouthPressLeft', 'mouthPressRight']
            values = buffer.get_blendshape_series(stress_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-10:]
            avg_values = np.mean(recent_values, axis=0)
            
            brow_tension = np.mean(avg_values[:2])
            mouth_press = np.mean(avg_values[2:])
            
            if brow_tension > 0.3 and mouth_press > 0.25:
                return 'Stress_Tense'
                
        except Exception as e:
            print(f"Stress detection error: {e}")
        return None
    
    def detect_contemplation(self, buffer):
        """Derin düşünce - Uzun süreli yukarı bakma"""
        try:
            if len(buffer) < 15:
                return None
            
            contemplate_blendshapes = ['eyeLookUpLeft', 'eyeLookUpRight']
            values = buffer.get_blendshape_series(contemplate_blendshapes)
            
            if len(values) < 10:
                return None
            
            # Uzun süre yukarı bakma kontrolü
            extended_values = values[-15:]
            sustained_look_up = np.mean(extended_values, axis=0)
            max_sustained = np.max(sustained_look_up)
            
            if max_sustained > 0.25:
                return 'Contemplation_DeepThought'
                
        except Exception as e:
            print(f"Contemplation detection error: {e}")
        return None
    
    def detect_doubt(self, buffer):
        """Şüphe - Göz kısma + kaş çatma kombinasyonu"""
        try:
            if len(buffer) < 10:  # Daha fazla frame
                return None
            
            doubt_blendshapes = ['eyeSquintLeft', 'eyeSquintRight', 'browDownLeft', 'browDownRight']
            values = buffer.get_blendshape_series(doubt_blendshapes)
            
            if len(values) < 5:
                return None
            
            recent_values = values[-12:]  # Daha uzun zaman
            avg_values = np.mean(recent_values, axis=0)
            
            eye_squint = np.mean(avg_values[:2])
            brow_down = np.mean(avg_values[2:])
            
            # Her iki eşiği de yükselttik (0.2 -> 0.4)
            if eye_squint > 0.4 and brow_down > 0.4:
                return 'Doubt_Uncertain'
                
        except Exception as e:
            print(f"Doubt detection error: {e}")
        return None
    
    def detect_interest(self, buffer):
        """İlgi - Göz genişleme + hafif kaş kaldırma"""
        try:
            if len(buffer) < 5:
                return None
            
            interest_blendshapes = ['eyeWideLeft', 'eyeWideRight', 'browInnerUp']
            values = buffer.get_blendshape_series(interest_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-8:]
            avg_values = np.mean(recent_values, axis=0)
            
            eye_wide = np.mean(avg_values[:2])
            brow_up = avg_values[2] if len(avg_values) > 2 else 0
            
            if eye_wide > 0.25 and brow_up > 0.15:
                return 'Interest_Curious'
                
        except Exception as e:
            print(f"Interest detection error: {e}")
        return None
    
    def detect_fatigue(self, buffer):
        """Yorgunluk - Göz kapama + ağız açma"""
        try:
            if len(buffer) < 8:
                return None
            
            fatigue_blendshapes = ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen']
            values = buffer.get_blendshape_series(fatigue_blendshapes)
            
            if len(values) < 5:
                return None
            
            recent_values = values[-12:]
            
            # Uzun göz kapama kontrolü
            blink_freq = np.mean(recent_values[:, :2], axis=0)
            jaw_open = np.mean(recent_values[:, 2]) if recent_values.shape[1] > 2 else 0
            
            if np.max(blink_freq) > 0.4 and jaw_open > 0.15:
                return 'Fatigue_Exhausted'
                
        except Exception as e:
            print(f"Fatigue detection error: {e}")
        return None
    
    def detect_alertness(self, buffer):
        """Tetikte olma - Göz genişleme + düz bakış"""
        try:
            if len(buffer) < 5:
                return None
            
            alert_blendshapes = ['eyeWideLeft', 'eyeWideRight']
            values = buffer.get_blendshape_series(alert_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-8:]
            avg_alert = np.mean(recent_values, axis=0)
            max_alert = np.max(avg_alert)
            
            # Yan bakış yok mu kontrol et
            side_look_blendshapes = ['eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookInLeft', 'eyeLookInRight']
            side_values = buffer.get_blendshape_series(side_look_blendshapes)
            side_activity = np.mean(side_values[-5:]) if len(side_values) > 2 else 0
            
            if max_alert > 0.3 and side_activity < 0.1:
                return 'Alertness_Focused'
                
        except Exception as e:
            print(f"Alertness detection error: {e}")
        return None
    
    def detect_discomfort(self, buffer):
        """Rahatsızlık - Göz kısma + kaçınma"""
        try:
            if len(buffer) < 10:  # Daha fazla frame
                return None
            
            discomfort_blendshapes = ['eyeSquintLeft', 'eyeSquintRight', 'eyeLookDownLeft', 'eyeLookDownRight']
            values = buffer.get_blendshape_series(discomfort_blendshapes)
            
            if len(values) < 5:
                return None
            
            recent_values = values[-15:]  # Daha uzun zaman
            avg_values = np.mean(recent_values, axis=0)
            
            eye_squint = np.mean(avg_values[:2])
            look_away = np.mean(avg_values[2:])
            
            # Göz kısma eşiğini yükselttik (0.2 -> 0.5)
            if eye_squint > 0.5 and look_away > 0.35:
                return 'Discomfort_Uneasy'
                
        except Exception as e:
            print(f"Discomfort detection error: {e}")
        return None

    def detect_smile(self, buffer):
        """Gülümseme - İlgi, memnuniyet, mutluluk"""
        try:
            if len(buffer) < 5:
                return None
            
            smile_blendshapes = ['mouthSmileLeft', 'mouthSmileRight']
            values = buffer.get_blendshape_series(smile_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-8:]
            avg_values = np.mean(recent_values, axis=0)
            max_smile = np.max(avg_values)
            
            if max_smile > 0.4:  # Belirgin gülümseme
                return 'Smile_Happy'
                
        except Exception as e:
            print(f"Smile detection error: {e}")
        return None
    
    def detect_excitement(self, buffer):
        """Heyecan - Genişleyen gözler + hafif ağız açma kombinasyonu"""
        try:
            if len(buffer) < 8:
                return None
            
            excitement_blendshapes = ['eyeWideLeft', 'eyeWideRight', 'jawOpen', 'browInnerUp']
            values = buffer.get_blendshape_series(excitement_blendshapes)
            
            if len(values) < 4:
                return None
            
            recent_values = values[-10:]
            avg_values = np.mean(recent_values, axis=0)
            
            eye_wide = np.mean(avg_values[:2])
            jaw_open = avg_values[2] if len(avg_values) > 2 else 0
            brow_up = avg_values[3] if len(avg_values) > 3 else 0
            
            # Heyecan genellikle hafif ağız açma ve göz genişletme ile gösterilir
            if eye_wide > 0.3 and jaw_open > 0.15 and brow_up > 0.2:
                return 'Excitement_Enthusiastic'
                
        except Exception as e:
            print(f"Excitement detection error: {e}")
        return None
    
    def detect_agreement(self, buffer):
        """Onaylama/Katılma - Hafif kafa sallama, kaş kaldırma"""
        try:
            if len(buffer) < 10:
                return None
            
            # Onay genellikle kaş kaldırma ile gösterilir
            agreement_blendshapes = ['browOuterUpLeft', 'browOuterUpRight', 'browInnerUp']
            values = buffer.get_blendshape_series(agreement_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-12:]
            avg_values = np.mean(recent_values, axis=0)
            
            brow_outer = np.mean(avg_values[:2])
            brow_inner = avg_values[2] if len(avg_values) > 2 else 0
            
            # Kaş kaldırma kombinasyonu
            if brow_outer > 0.3 and brow_inner > 0.25:
                return 'Agreement_Approving'
                
        except Exception as e:
            print(f"Agreement detection error: {e}")
        return None
    
    def detect_appreciation(self, buffer):
        """Takdir/Beğenme - Hafif gülümseme ve baş eğme"""
        try:
            if len(buffer) < 8:
                return None
            
            appreciation_blendshapes = ['mouthSmileLeft', 'mouthSmileRight', 'browInnerUp']
            values = buffer.get_blendshape_series(appreciation_blendshapes)
            
            if len(values) < 3:
                return None
            
            recent_values = values[-10:]
            avg_values = np.mean(recent_values, axis=0)
            
            smile = np.mean(avg_values[:2])
            brow_inner = avg_values[2] if len(avg_values) > 2 else 0
            
            # Takdir genellikle hafif gülümseme ve iç kaş kaldırma ile gösterilir
            if smile > 0.3 and brow_inner > 0.2:
                return 'Appreciation_Impressed'
                
        except Exception as e:
            print(f"Appreciation detection error: {e}")
        return None
    
    def detect_curiosity(self, buffer):
        """Merak - Tek kaş kaldırma, hafif öne eğilme"""
        try:
            if len(buffer) < 6:
                return None
            
            curiosity_blendshapes = ['browOuterUpLeft', 'browOuterUpRight', 'eyeWideLeft', 'eyeWideRight']
            values = buffer.get_blendshape_series(curiosity_blendshapes)
            
            if len(values) < 4:
                return None
            
            recent_values = values[-8:]
            avg_values = np.mean(recent_values, axis=0)
            
            # Asimetrik kaş kaldırma (merak işareti)
            left_brow = avg_values[0]
            right_brow = avg_values[1]
            eye_wide = np.mean(avg_values[2:4])
            
            # Asimetri kontrolü
            brow_asymmetry = abs(left_brow - right_brow)
            brow_max = max(left_brow, right_brow)
            
            # Merak genellikle asimetrik kaş kaldırma ve göz genişletme ile gösterilir
            if (brow_asymmetry > 0.15 and brow_max > 0.25) or (eye_wide > 0.3 and brow_max > 0.2):
                return 'Curiosity_Interested'
                
        except Exception as e:
            print(f"Curiosity detection error: {e}")
        return None

