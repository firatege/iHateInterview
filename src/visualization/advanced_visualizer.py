import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class AdvancedVisualizer:
    """
    Class providing advanced visualization features.
    Contains various methods for visualizing engagement state, micro-expressions, 
    and facial landmarks with detailed highlighting of facial features.
    """
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height
        
        # Gösterge paneli alanları - increase dashboard height for better visibility
        self.dashboard_height = 180
        self.sidebar_width = 200
        
        # Zaman serisi verileri için depolama - increase history length for better time series visualization
        self.max_history = 150  # ~5 seconds at 30fps
        self.engagement_history = []
        self.timestamp_history = []
        self.emotion_history = {}
        
        # Renk paleti
        self.colors = {
            'background': (40, 44, 52),
            'panel': (30, 34, 42),
            'grid': (60, 64, 72),
            'text': (220, 220, 220),
            'engaged': (0, 255, 0),
            'disengaged': (0, 0, 255),
            'neutral': (255, 165, 0),
            'thinking': (255, 255, 0),
            'looking': (255, 0, 255),
            'eye': (0, 255, 255),
            'positive': (0, 255, 128),
            'negative': (0, 0, 255),
            'tired': (128, 128, 128),
            'highlight': (255, 255, 255),
            'warning': (0, 165, 255)  # Orange color for fake smile/warning
        }
        
        # Başlangıç zamanı
        self.start_time = time.time()
        
        # Önemli anlar listesi
        self.key_moments = []
        self.last_pattern_detected = None
        
    def update_history(self, engagement_state, detected_patterns, current_time=None):
        """
        Updates time series data for visualization and tracks important moments.
        Also includes fake smile detection logic (currently disabled).
        """
        if current_time is None:
            current_time = time.time()
            
        # Record engagement state
        engagement_score = 1.0 if engagement_state == 'Engaged' else 0.0 if engagement_state == 'Disengaged' else 0.5
        self.engagement_history.append(engagement_score)
        self.timestamp_history.append(current_time - self.start_time)
        
        # Limit history length
        if len(self.engagement_history) > self.max_history:
            self.engagement_history.pop(0)
            self.timestamp_history.pop(0)
        
        # Track patterns for emotion history
        for pattern in detected_patterns:
            if pattern not in self.emotion_history:
                self.emotion_history[pattern] = 1
            else:
                self.emotion_history[pattern] += 1
        
        # Fake smile detection logic (currently disabled)
        # The code is kept for future development but is currently commented out
        """
        has_smile = any('Smile' in pattern for pattern in detected_patterns)
        contradictory_signals = any(signal in ''.join(detected_patterns) for signal in 
                                  ['Discomfort', 'Stress', 'Disengaged', 'EyeSquint', 'EyeRoll'])
        
        # Detect fake smile when smile is present but with contradictory signals
        if has_smile and contradictory_signals:
            self.key_moments.append({
                'time': current_time - self.start_time,
                'pattern': 'Fake Smile Detected',
                'type': 'warning'  # Special type for fake smile
            })
        """
                
        # Check for any new important micro-expressions
        if detected_patterns:
            new_pattern = detected_patterns[-1] if len(detected_patterns) > 1 else detected_patterns[0]
            pattern_changed = new_pattern != self.last_pattern_detected
            
            # If we have a new pattern that's different from the last one
            if pattern_changed:
                self.last_pattern_detected = new_pattern
                
                # Expanded list of positive patterns to track
                positive_indicators = [
                    'Smile', 'Excitement', 'Interest', 'Surprise', 'Agreement', 
                    'Appreciation', 'Curiosity', 'Engaged', 'Concentration'
                ]
                
                # Expanded list of negative patterns to track
                negative_indicators = [
                    'Stress', 'Discomfort', 'Disengaged', 'Boredom', 'Fatigue',
                    'Confusion', 'Skepticism', 'Doubt', 'EyeRoll', 'Frown'
                ]
                
                # Check for positive micro-expressions
                if any(keyword in new_pattern for keyword in positive_indicators):
                    # Only add as key moment if it's a significant positive indicator
                    self.key_moments.append({
                        'time': current_time - self.start_time,
                        'pattern': new_pattern,
                        'type': 'positive'
                    })
                    
                # Check for negative micro-expressions
                elif any(keyword in new_pattern for keyword in negative_indicators):
                    self.key_moments.append({
                        'time': current_time - self.start_time,
                        'pattern': new_pattern,
                        'type': 'negative'
                    })
                    
                # If we detect a sudden change in engagement status, mark it as a key moment
                elif engagement_state == 'Engaged' and len(self.engagement_history) > 5:
                    # Check if there was a recent transition from disengaged to engaged
                    if np.mean(self.engagement_history[-5:-1]) < 0.3 and engagement_score > 0.7:
                        self.key_moments.append({
                            'time': current_time - self.start_time,
                            'pattern': 'Sudden Engagement',
                            'type': 'positive'
                        })
                elif engagement_state == 'Disengaged' and len(self.engagement_history) > 5:
                    # Check if there was a recent transition from engaged to disengaged
                    if np.mean(self.engagement_history[-5:-1]) > 0.7 and engagement_score < 0.3:
                        self.key_moments.append({
                            'time': current_time - self.start_time,
                            'pattern': 'Lost Engagement',
                            'type': 'negative'
                        })
                
    def draw_dashboard(self, frame, detected_patterns, blendshapes=None):
        """
        Draws a simplified dashboard at the bottom of the screen with enhanced time series visualization.
        """
        # Dashboard arka planı
        dashboard = np.zeros((self.dashboard_height, frame.shape[1], 3), dtype=np.uint8)
        dashboard[:, :] = self.colors['panel']
        
        # Draw light grid lines for better time series readability
        grid_color = (50, 50, 50)  # Dark gray grid
        for i in range(0, dashboard.shape[1], 60):  # Vertical lines every 60 pixels (time markers)
            cv2.line(dashboard, (i, 0), (i, dashboard.shape[0]), grid_color, 1)
        
        # Add horizontal grid lines at key levels
        cv2.line(dashboard, (0, int(dashboard.shape[0] * 0.2)), (dashboard.shape[1], int(dashboard.shape[0] * 0.2)), grid_color, 1)  # 80% engagement
        cv2.line(dashboard, (0, int(dashboard.shape[0] * 0.5)), (dashboard.shape[1], int(dashboard.shape[0] * 0.5)), grid_color, 1)  # 50% engagement
        cv2.line(dashboard, (0, int(dashboard.shape[0] * 0.8)), (dashboard.shape[1], int(dashboard.shape[0] * 0.8)), grid_color, 1)  # 20% engagement
        
        # Add level labels
        cv2.putText(dashboard, "High", (5, int(dashboard.shape[0] * 0.2) + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(dashboard, "Neutral", (5, int(dashboard.shape[0] * 0.5) + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(dashboard, "Low", (5, int(dashboard.shape[0] * 0.8) + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Draw time series with thicker lines and better visibility
        if len(self.engagement_history) > 1:
            # Scale values to dashboard height
            values = np.array(self.engagement_history) * (self.dashboard_height * 0.8)
            values = self.dashboard_height - values  # Invert Y-axis
            
            # Calculate x values for line graph across most of the width
            x_values = np.linspace(60, dashboard.shape[1] - 10, len(values))
            
            # Draw engagement trend line with thicker lines and fill
            pts = []
            for i in range(len(values)):
                pts.append([int(x_values[i]), int(values[i])])
            
            pts = np.array(pts, np.int32)
            
            # Create fill points by adding bottom corners
            fill_pts = pts.copy()
            fill_pts = np.vstack([fill_pts, [[int(x_values[-1]), dashboard.shape[0]], [int(x_values[0]), dashboard.shape[0]]]])
            
            # Draw filled area under the curve with transparency
            overlay = dashboard.copy()
            cv2.fillPoly(overlay, [fill_pts], (0, 100, 0))  # Dark green fill
            cv2.addWeighted(overlay, 0.3, dashboard, 0.7, 0, dashboard)
            
            # Draw the main line with gradient color based on engagement
            for i in range(1, len(values)):
                color = self.get_color_by_value(self.engagement_history[i])
                cv2.line(dashboard, 
                         (int(x_values[i-1]), int(values[i-1])), 
                         (int(x_values[i]), int(values[i])), 
                         color, 3)  # Thicker line (3px)
                
                # Add dots at data points for better visibility
                cv2.circle(dashboard, (int(x_values[i]), int(values[i])), 2, color, -1)
                
                    # Mark key moments on the timeline
            current_time = time.time() - self.start_time
            for moment in self.key_moments:
                # Only show moments that would be visible on the current time scale
                if current_time - moment['time'] < self.max_history / 30:  # Assuming 30fps
                    # Calculate position on timeline
                    ratio = (current_time - moment['time']) / (self.max_history / 30)
                    moment_x = int(dashboard.shape[1] - 10 - ratio * (dashboard.shape[1] - 70))
                    
                    if 60 <= moment_x < dashboard.shape[1] - 10:
                        # Draw marker based on moment type (simplified - fake smile detection disabled)
                        marker_color = self.colors['positive'] if moment['type'] == 'positive' else self.colors['negative']
                        cv2.line(dashboard, (moment_x, 0), (moment_x, dashboard.shape[0] - 40), marker_color, 2)
                        
                        # Add small label
                        cv2.circle(dashboard, (moment_x, dashboard.shape[0] - 20), 5, marker_color, -1)
                        
                        # Code for fake smile detection marker (kept but disabled)
                        """
                        # Add special indicator for fake smile
                        if moment['pattern'] == 'Fake Smile Detected':
                            # Draw attention-grabbing marker
                            cv2.drawMarker(dashboard, (moment_x, 30), 
                                         self.colors['warning'], cv2.MARKER_DIAMOND, 10, 2)
                        """        # Information panel on the right side
        info_x = dashboard.shape[1] - 220
        
        # Draw semi-transparent background for info panel
        cv2.rectangle(dashboard, (info_x, 10), (dashboard.shape[1] - 10, dashboard.shape[0] - 10), (50, 50, 50), -1)
        
        # Show elapsed time
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins:02d}:{secs:02d}"
        cv2.putText(dashboard, f"Time: {time_str}", (info_x + 10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Show key moments count
        cv2.putText(dashboard, f"Key Moments: {len(self.key_moments)}", (info_x + 10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        # Show last detected pattern with appropriate color
        if self.last_pattern_detected:
            pattern_color = self.colors['positive']
            
            # Fake smile detection logic kept but disabled
            """
            # Check for fake smile in recent key moments
            fake_smile_detected = False
            recent_moments = [m for m in self.key_moments if (current_time - m['time']) < 3.0]
            for moment in recent_moments:
                if moment['pattern'] == 'Fake Smile Detected':
                    fake_smile_detected = True
                    break
            
            if fake_smile_detected:
                pattern_text = "Fake Smile Detected"
                pattern_color = self.colors['warning']
            """
            
            # Simple pattern display (fake smile detection disabled)
            if any(key in self.last_pattern_detected for key in ['Stress', 'Discomfort', 'Disengaged']):
                pattern_color = self.colors['negative']
                pattern_text = self.last_pattern_detected
            elif self.last_pattern_detected == 'Neutral':
                pattern_color = self.colors['neutral']
                pattern_text = self.last_pattern_detected
            else:
                pattern_text = self.last_pattern_detected
                
            cv2.putText(dashboard, "Last Pattern:", (info_x + 10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            cv2.putText(dashboard, f"{pattern_text}", (info_x + 10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, pattern_color, 2)
            
            # Fake smile indicator code kept but disabled
            """
            # Add additional indicator for fake smile
            if fake_smile_detected:
                cv2.putText(dashboard, "INSINCERE EXPRESSION", (info_x + 10, 155), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['warning'], 1)
            """
        
        # Add title for the time series
        cv2.putText(dashboard, "Engagement Timeline", (70, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Dashboard'ı ana frame'in altına ekle
        result = np.vstack([frame, dashboard])
        return result
    
    def draw_face_landmarks(self, frame, landmarks, blendshapes=None):
        """
        Enhanced face landmark visualization with facial feature highlighting.
        Draws and highlights specific facial regions (eyes, eyebrows, nose, mouth).
        """
        if not landmarks:
            return frame
            
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Define facial feature indices for MediaPipe Face Landmarker
        # These are based on MediaPipe's landmark indices
        facial_features = {
            'silhouette': list(range(0, 10)),
            'left_eyebrow': list(range(336, 342)),
            'right_eyebrow': list(range(296, 302)),
            'left_eye': list(range(362, 374)),
            'right_eye': list(range(382, 394)),
            'outer_lips': list(range(0, 12)),
            'inner_lips': list(range(60, 72)),
            'nose_bridge': list(range(168, 174)),
            'nose_tip': list(range(128, 134))
        }
        
        # Define colors for each feature
        feature_colors = {
            'silhouette': (200, 200, 200),  # Light gray
            'left_eyebrow': (0, 165, 255),  # Orange
            'right_eyebrow': (0, 165, 255),  # Orange
            'left_eye': (0, 255, 255),      # Yellow
            'right_eye': (0, 255, 255),     # Yellow
            'outer_lips': (0, 0, 255),      # Red
            'inner_lips': (255, 0, 0),      # Blue
            'nose_bridge': (255, 0, 255),   # Purple
            'nose_tip': (255, 0, 255)       # Purple
        }
        
        # Get coordinates for all landmarks
        landmark_coords = []
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_coords.append((x, y))
        
        # Adaptive processing - use smaller set of landmarks if not enough available
        if len(landmark_coords) < 400:  # MediaPipe full face mesh has ~468 points
            # Simplified feature set for fewer landmarks
            simplified_features = {
                'eyes': list(range(33, 40)),  # Approximate eye region
                'mouth': list(range(48, 68)), # Approximate mouth region
                'nose': list(range(27, 36))   # Approximate nose region
            }
            
            simplified_colors = {
                'eyes': (0, 255, 255),      # Yellow
                'mouth': (0, 0, 255),       # Red
                'nose': (255, 0, 255)       # Purple
            }
            
            # Draw simplified features
            for feature, indices in simplified_features.items():
                color = simplified_colors[feature]
                pts = []
                valid_indices = [i for i in indices if i < len(landmark_coords)]
                if valid_indices:
                    for i in valid_indices:
                        if i < len(landmark_coords):
                            pts.append(landmark_coords[i])
                    
                    if len(pts) > 2:
                        pts = np.array(pts, np.int32)
                        cv2.polylines(frame, [pts], True, color, 2)
                        # Add semi-transparent fill
                        cv2.fillPoly(overlay, [pts], color)
            
            # Draw all landmarks with small circles
            for x, y in landmark_coords:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        else:
            # Draw full facial features with connecting lines
            for feature, indices in facial_features.items():
                color = feature_colors[feature]
                pts = []
                valid_indices = [i for i in indices if i < len(landmark_coords)]
                if valid_indices:
                    for i in valid_indices:
                        if i < len(landmark_coords):
                            pts.append(landmark_coords[i])
                    
                    if len(pts) > 2:
                        pts = np.array(pts, np.int32)
                        cv2.polylines(frame, [pts], True, color, 2)
                        # Add semi-transparent fill for eyes and mouth
                        if feature in ['left_eye', 'right_eye', 'outer_lips', 'inner_lips']:
                            cv2.fillPoly(overlay, [pts], color)
            
            # Draw all landmarks with small dots
            for x, y in landmark_coords:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Blend the overlay with semi-transparency
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # If we have blendshapes, show activation near relevant facial regions
        if blendshapes:
            # Map blendshapes to face regions and draw activation indicators
            for bs in blendshapes:
                activation = bs.score
                
                # Only show significant activations
                if activation > 0.4:
                    # Map blendshape to position based on name
                    if 'eye' in bs.category_name.lower():
                        pos_y = h // 3
                        color = (0, 255, 255)
                    elif 'brow' in bs.category_name.lower():
                        pos_y = h // 4
                        color = (0, 165, 255)
                    elif 'mouth' in bs.category_name.lower() or 'lip' in bs.category_name.lower():
                        pos_y = 2 * h // 3
                        color = (0, 0, 255)
                    elif 'cheek' in bs.category_name.lower():
                        pos_y = h // 2
                        color = (255, 0, 255)
                    else:
                        continue
                        
                    # Calculate activation radius based on score
                    radius = int(20 * activation)
                    if 'left' in bs.category_name.lower():
                        pos_x = w // 3
                    elif 'right' in bs.category_name.lower():
                        pos_x = 2 * w // 3
                    else:
                        pos_x = w // 2
                    
                    # Draw activation circle with transparency
                    overlay_bs = frame.copy()
                    cv2.circle(overlay_bs, (pos_x, pos_y), radius, color, -1)
                    alpha_bs = 0.4 * activation  # More activation = more opacity
                    cv2.addWeighted(overlay_bs, alpha_bs, frame, 1 - alpha_bs, 0, frame)
        
        return frame
    
    def draw_emotion_heatmap(self, frame, blendshapes):
        """
        Creates a heatmap visualization showing emotional activity in different facial regions.
        Uses blendshapes to detect which areas of the face are most active during expression.
        """
        if not blendshapes:
            return frame
        
        # Define facial regions and associated blendshapes
        regions = {
            'forehead': ['browInnerUp', 'browDownLeft', 'browDownRight', 'browOuterUpLeft', 'browOuterUpRight'],
            'eyes': ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight'],
            'cheeks': ['cheekPuff', 'cheekSquintLeft', 'cheekSquintRight'],
            'nose': ['noseSneerLeft', 'noseSneerRight'],
            'mouth': ['mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthPucker', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthClose', 'mouthOpen'],
            'jaw': ['jawOpen', 'jawForward', 'jawLeft', 'jawRight']
        }
        
        # Map blendshapes to emotional meanings
        emotion_mappings = {
            'happiness': ['mouthSmileLeft', 'mouthSmileRight', 'cheekSquintLeft', 'cheekSquintRight'],
            'surprise': ['eyeWideLeft', 'eyeWideRight', 'browInnerUp', 'mouthOpen', 'jawOpen'],
            'anger': ['browDownLeft', 'browDownRight', 'noseSneerLeft', 'noseSneerRight', 'mouthFrownLeft', 'mouthFrownRight'],
            'concentration': ['eyeSquintLeft', 'eyeSquintRight', 'browDownLeft', 'browDownRight'],
            'confusion': ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'jawForward'],
            'interest': ['eyeWideLeft', 'eyeWideRight', 'eyeLookUpLeft', 'eyeLookUpRight']
        }
        
        # Calculate activation scores for each region
        region_scores = {}
        for region, blendshape_names in regions.items():
            scores = []
            for bs in blendshapes:
                if bs.category_name in blendshape_names:
                    scores.append(bs.score)
            
            region_scores[region] = max(scores) if scores else 0.0
        
        # Calculate emotion scores
        emotion_scores = {}
        for emotion, blendshape_names in emotion_mappings.items():
            # Get average of all related blendshapes
            emotion_values = []
            for bs in blendshapes:
                if bs.category_name in blendshape_names:
                    emotion_values.append(bs.score)
            
            # Calculate weighted score
            if emotion_values:
                emotion_scores[emotion] = sum(emotion_values) / len(emotion_values)
            else:
                emotion_scores[emotion] = 0.0
        
        # Create heat map overlay
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw region heatmaps based on activation scores
        # Forehead region
        if region_scores['forehead'] > 0.2:
            color = self.get_heatmap_color(region_scores['forehead'])
            cv2.ellipse(overlay, (w//2, h//5), (w//3, h//10), 0, 0, 360, color, -1)
        
        # Eyes region - draw two ellipses for eyes
        if region_scores['eyes'] > 0.2:
            color = self.get_heatmap_color(region_scores['eyes'])
            # Left eye
            cv2.ellipse(overlay, (w//3, h//3), (w//8, h//16), 0, 0, 360, color, -1)
            # Right eye
            cv2.ellipse(overlay, (2*w//3, h//3), (w//8, h//16), 0, 0, 360, color, -1)
        
        # Cheeks region
        if region_scores['cheeks'] > 0.2:
            color = self.get_heatmap_color(region_scores['cheeks'])
            # Left cheek
            cv2.ellipse(overlay, (w//4, h//2), (w//10, h//8), 0, 0, 360, color, -1)
            # Right cheek
            cv2.ellipse(overlay, (3*w//4, h//2), (w//10, h//8), 0, 0, 360, color, -1)
        
        # Nose region
        if region_scores['nose'] > 0.2:
            color = self.get_heatmap_color(region_scores['nose'])
            cv2.ellipse(overlay, (w//2, h//2), (w//10, h//8), 0, 0, 360, color, -1)
        
        # Mouth region
        if region_scores['mouth'] > 0.2:
            color = self.get_heatmap_color(region_scores['mouth'])
            cv2.ellipse(overlay, (w//2, 2*h//3), (w//4, h//8), 0, 0, 360, color, -1)
        
        # Jaw region
        if region_scores['jaw'] > 0.2:
            color = self.get_heatmap_color(region_scores['jaw'])
            cv2.ellipse(overlay, (w//2, 3*h//4), (w//3, h//10), 0, 0, 360, color, -1)
        
        # Blend overlay with original frame
        alpha = 0.3  # Transparency level
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add emotion labels if any emotion is detected strongly
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1]) if emotion_scores else (None, 0)
        if dominant_emotion[0] is not None and dominant_emotion[1] > 0.4:  # Only show if confidence is high enough
            emotion_name = dominant_emotion[0].capitalize()
            emotion_score = dominant_emotion[1]
            
            # Choose text color based on emotion
            if emotion_name == 'Happiness':
                text_color = (0, 255, 255)  # Yellow
            elif emotion_name == 'Surprise':
                text_color = (0, 165, 255)  # Orange
            elif emotion_name == 'Anger':
                text_color = (0, 0, 255)    # Red
            elif emotion_name == 'Concentration':
                text_color = (255, 0, 0)    # Blue
            elif emotion_name == 'Confusion':
                text_color = (255, 0, 255)  # Purple
            elif emotion_name == 'Interest':
                text_color = (0, 255, 0)    # Green
            else:
                text_color = (255, 255, 255)  # White
            
            # Draw text with background for better visibility
            text = f"{emotion_name}: {emotion_score:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_w, text_h = text_size
            
            # Position the text near the top of the frame
            text_x = w - text_w - 10
            text_y = 30
            
            # Draw background rectangle
            cv2.rectangle(frame, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (40, 40, 40), -1)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        return frame
    
    def get_color_by_value(self, value):
        """
        Değere göre renk döndürür (engagement için).
        """
        if value > 0.7:  # İlgili
            return self.colors['engaged']
        elif value < 0.3:  # İlgisiz
            return self.colors['disengaged']
        else:  # Nötr
            return self.colors['neutral']
    
    def get_heatmap_color(self, value):
        """
        Returns a color for the heatmap based on activation value.
        Creates a smooth color transition from blue (cold) to red (hot).
        
        Args:
            value: Activation value between 0.0 and 1.0
            
        Returns:
            BGR color tuple
        """
        # Clamp value between 0 and 1
        value = max(0.0, min(1.0, value))
        
        # Blue (cold) -> Cyan -> Green -> Yellow -> Red (hot)
        if value < 0.25:
            # Blue to Cyan (0.0 - 0.25)
            ratio = value / 0.25
            b = 255
            g = int(255 * ratio)
            r = 0
        elif value < 0.5:
            # Cyan to Green (0.25 - 0.5)
            ratio = (value - 0.25) / 0.25
            b = int(255 * (1 - ratio))
            g = 255
            r = 0
        elif value < 0.75:
            # Green to Yellow (0.5 - 0.75)
            ratio = (value - 0.5) / 0.25
            b = 0
            g = 255
            r = int(255 * ratio)
        else:
            # Yellow to Red (0.75 - 1.0)
            ratio = (value - 0.75) / 0.25
            b = 0
            g = int(255 * (1 - ratio))
            r = 255
            
        return (b, g, r)
    
    def highlight_key_moments(self, frame, current_time=None):
        """
        Records important moments but doesn't display the banner at the top of the screen.
        Only marks key moments on the timeline in the dashboard.
        """
        if not self.key_moments:
            return frame
            
        if current_time is None:
            current_time = time.time() - self.start_time
            
        # Check for important moments in the last 5 seconds
        recent_moments = [m for m in self.key_moments if current_time - m['time'] < 5.0]
        
        if not recent_moments:
            return frame
            
        # Get the most recent important moment (used for dashboard timeline markers)
        last_moment = recent_moments[-1]
        
        # No banner displayed at the top - removed as requested
        
        # The timeline markers in the dashboard will still show key moments
        return frame
    
    def process_frame(self, frame, landmarks, blendshapes, detected_patterns):
        """
        Apply all visualization effects to a frame.
        """
        # Skip drawing face landmarks and heatmap as requested
        
        # Zaman serisi verilerini güncelle
        if detected_patterns:
            self.update_history(detected_patterns[0], detected_patterns)
        
        # Önemli anları vurgula
        frame = self.highlight_key_moments(frame)
        
        # Gösterge panelini ekle
        frame = self.draw_dashboard(frame, detected_patterns, blendshapes)
        
        return frame
