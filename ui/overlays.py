"""
UI Overlays Module

This module contains all the visualization and UI overlay logic for the computer vision middleware.
It provides functions to draw various overlays on frames, including:
- Performance metrics
- Face landmarks
- Eye state indicators
- Posture indicators
- Emotion indicators
- Visualization graphs
"""

import cv2
import numpy as np
import time
import psutil
import mediapipe as mp
from collections import deque

class PerformanceOverlay:
    """Handles drawing of performance metrics on frames."""
    
    def __init__(self, history_size=100):
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.fps_history = deque(maxlen=history_size)
        self.last_measure_time = time.time()
        self.measure_interval = 0.5
        self.cpu_alert_threshold = 20.0
        
        self.current_cpu = 0.0
        self.current_memory = 0.0
        self.current_fps = 0.0
        self.average_cpu = 0.0
        self.average_memory = 0.0
        self.average_fps = 0.0
        
        self.process = psutil.Process()
        self._measure()
    
    def _measure(self):
        """Take performance measurements."""
        try:
            self.current_cpu = psutil.cpu_percent(interval=0)
            self.cpu_history.append(self.current_cpu)
            self.average_cpu = sum(self.cpu_history) / len(self.cpu_history)
        except Exception as e:
            print(f"Error measuring CPU: {e}")
        
        try:
            self.current_memory = psutil.virtual_memory().percent
            self.memory_history.append(self.current_memory)
            self.average_memory = sum(self.memory_history) / len(self.memory_history)
        except Exception as e:
            print(f"Error measuring memory: {e}")
        
        self.last_measure_time = time.time()
    
    def update(self, current_fps):
        """Update performance metrics."""
        self.current_fps = current_fps
        self.fps_history.append(current_fps)
        if len(self.fps_history) > 0:
            self.average_fps = sum(self.fps_history) / len(self.fps_history)
        
        current_time = time.time()
        if current_time - self.last_measure_time >= self.measure_interval:
            self._measure()
    
    def is_cpu_high(self):
        """Check if CPU usage is above the alert threshold."""
        return self.current_cpu > self.cpu_alert_threshold
    
    def draw(self, frame):
        """Draw performance metrics on the frame."""
        h, w, _ = frame.shape
        
        metrics_x = 10
        metrics_y = 70
        
        # CPU usage
        cpu_color = (0, 255, 0) if self.current_cpu <= self.cpu_alert_threshold else (0, 0, 255)
        cv2.putText(frame, f"CPU: {self.current_cpu:.1f}%", 
                   (metrics_x, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, cpu_color, 1)
        
        # Memory usage
        memory_color = (0, 255, 0)
        if self.current_memory > 80:
            memory_color = (0, 0, 255)
        elif self.current_memory > 60:
            memory_color = (0, 165, 255)
            
        cv2.putText(frame, f"MEM: {self.current_memory:.1f}%", 
                   (metrics_x, metrics_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, memory_color, 1)
        
        # FPS
        fps_color = (0, 255, 0)
        if self.current_fps < 15:
            fps_color = (0, 0, 255)
        elif self.current_fps < 25:
            fps_color = (0, 165, 255)
            
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (metrics_x, metrics_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, fps_color, 1)
        
        # CPU alert
        if self.is_cpu_high() and int(time.time() * 2) % 2 == 0:
            alert_text = "HIGH CPU USAGE!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            alert_x = w // 2 - text_size[0] // 2
            alert_y = 30
            
            cv2.rectangle(frame,
                         (alert_x - 10, alert_y - 20),
                         (alert_x + text_size[0] + 10, alert_y + 5),
                         (0, 0, 255), -1)
            
            cv2.putText(frame, alert_text,
                       (alert_x, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

class FaceOverlay:
    """Handles drawing of face-related overlays."""
    
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize history for graphs
        self.left_eye_history = deque(maxlen=100)
        self.right_eye_history = deque(maxlen=100)
        self.blink_history = deque(maxlen=100)
        self.posture_history = deque(maxlen=100)
        self.eyebrow_history = deque(maxlen=100)
    
    def draw_landmarks(self, frame, face_landmarks, w, h):
        """Draw facial landmarks on the frame."""
        for idx, landmark in enumerate(face_landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            if idx in range(0, 68):  # Jaw region
                color = (0, 255, 0)  # Green
            elif idx in range(68, 151):  # Eye region
                color = (255, 0, 0)  # Blue
            elif idx in range(151, 200):  # Eyebrow region
                color = (0, 0, 255)  # Red
            elif idx in range(200, 300):  # Nose region
                color = (255, 255, 0)  # Yellow
            else:  # Mouth region
                color = (255, 0, 255)  # Magenta
                
            cv2.circle(frame, (x, y), 1, color, -1)
    
    def draw_eye_state(self, frame, eye_state, w):
        """Draw eye state indicator."""
        if isinstance(eye_state, dict):
            # Handle the case where the full eye state dict is passed
            state = eye_state.get('eye_state', 'unknown')
            blink_count = eye_state.get('blink_counter', 0)
            expressions = eye_state.get('expressions', {})
            
            # Update graph history
            self.left_eye_history.append(eye_state.get('left_eye_distance', 0))
            self.right_eye_history.append(eye_state.get('right_eye_distance', 0))
            if eye_state.get('blink_detected', False):
                self.blink_history.append(1)
            else:
                self.blink_history.append(0)
            
            # Draw eye state
            if state == "wide_open":
                color = (0, 255, 255)  # Yellow
            elif state == "squinting":
                color = (255, 255, 0)  # Cyan
            elif state == "blinking":
                color = (255, 0, 255)  # Magenta
            elif state == "closed":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            cv2.rectangle(frame, (w - 150, 10), (w - 10, 40), color, -1)
            cv2.putText(frame, state, (w - 140, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw blink counter
            cv2.putText(frame, f"Blinks: {blink_count}", (w - 140, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw facial expressions if available
            if expressions:
                # Draw frowning status
                if expressions.get('is_frowning', False):
                    cv2.putText(frame, "Frowning", (w - 140, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw smiling status
                if expressions.get('smiling', False):
                    cv2.putText(frame, "Smiling", (w - 140, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Handle the case where just the eye state string is passed
            if eye_state == "wide_open":
                color = (0, 255, 255)  # Yellow
            elif eye_state == "squinting":
                color = (255, 255, 0)  # Cyan
            elif eye_state == "blinking":
                color = (255, 0, 255)  # Magenta
            elif eye_state == "closed":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            cv2.rectangle(frame, (w - 150, 10), (w - 10, 40), color, -1)
            cv2.putText(frame, eye_state, (w - 140, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def draw_emotion(self, frame, emotion_name, y_offset):
        """Draw emotion indicator."""
        if isinstance(emotion_name, dict):
            # Handle the case where a full emotion dict is passed
            color = emotion_name.get('color', (0, 255, 0))
            emotion_text = emotion_name.get('emotion', "Unknown")
            
            # Add valence and arousal info for debugging if needed
            valence = emotion_name.get('valence', 0)
            arousal = emotion_name.get('arousal', 0)
            
            # Draw detailed emotion with valence/arousal values
            cv2.putText(frame, f"Emotion: {emotion_text}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Optionally display valence/arousal values (comment out if not needed)
            # cv2.putText(frame, f"Valence: {valence:.2f} Arousal: {arousal:.2f}", 
            #            (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            # Handle the case where just the emotion name is passed
            color = (0, 255, 0) if emotion_name == "Relaxed" else (0, 0, 255)
            emotion_text = emotion_name
            
            cv2.putText(frame, f"Emotion: {emotion_text}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_posture(self, frame, posture_info, y_offset):
        """Draw posture indicator."""
        # Get hunch state from the detector
        if "hunch_state" in posture_info:
            hunch_state = posture_info["hunch_state"]
            
            if "Good" in hunch_state:
                color = (0, 255, 0)  # Green
            elif "Slight" in hunch_state:
                color = (0, 255, 255)  # Yellow
            elif "Medium" in hunch_state:
                color = (0, 165, 255)  # Orange
            else:  # Severe Hunch
                color = (0, 0, 255)  # Red
        else:
            # Fallback if hunch_state isn't available
            if posture_info["hunch_detected"]:
                hunch_state = "Poor Posture"
                color = (0, 0, 255)  # Red
            else:
                hunch_state = "Good Posture"
                color = (0, 255, 0)  # Green
        
        # Update posture history for graphs
        # Store full information for shoulder-based detection
        if "relative_drop" in posture_info:
            # Store dict with relative drop for shoulder-based detection
            self.posture_history.append({
                "relative_drop": posture_info["relative_drop"],
                "is_hunched": posture_info.get("is_hunched", False)
            })
        else:
            # For face-based detection, just store the forward tilt
            self.posture_history.append(posture_info.get("forward_tilt", 0))
        
        cv2.putText(frame, f"Posture: {hunch_state}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show hunch counter if any hunches detected
        if posture_info["hunch_counter"] > 0:
            y_offset += 30
            cv2.putText(frame, f"Hunches: {posture_info['hunch_counter']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show hunch duration if hunched
        if posture_info.get("is_hunched", False) and posture_info.get("hunch_duration", 0) > 0:
            y_offset += 30
            duration = posture_info["hunch_duration"]
            cv2.putText(frame, f"Hunched for: {duration:.1f}s", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show posture statistics if available
        if "posture_stats" in posture_info and posture_info["posture_stats"]["total_time"] > 0:
            y_offset += 30
            upright_pct = posture_info.get("upright_percentage", 0)
            cv2.putText(frame, f"Upright: {upright_pct:.1f}%", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def plot_eye_graph(self, width=400, height=200):
        """Create a visualization of eye metrics over time.
        
        Args:
            width: Width of the plot
            height: Height of the plot
            
        Returns:
            numpy.ndarray: Plot image
        """
        # Create plot image (dark background)
        plot_img = np.zeros((height, width, 3), dtype=np.uint8)
        plot_img[:] = (30, 30, 30)  # Dark gray background
        
        # Draw grid lines
        for i in range(0, height, height//4):
            cv2.line(plot_img, (0, i), (width, i), (50, 50, 50), 1)
            
        for i in range(0, width, width//4):
            cv2.line(plot_img, (i, 0), (i, height), (50, 50, 50), 1)
        
        # Draw title
        cv2.putText(plot_img, "Eye Metrics", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw left eye history (green)
        if len(self.left_eye_history) > 1:
            points = []
            for i, dist in enumerate(self.left_eye_history):
                x = int(i * width / max(1, len(self.left_eye_history) - 1))
                y = int(height - dist * height * 5)  # Scale for visibility
                y = max(0, min(height-1, y))  # Clamp to valid range
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(plot_img, points[i], points[i+1], (0, 255, 0), 2)
        
        # Draw right eye history (blue)
        if len(self.right_eye_history) > 1:
            points = []
            for i, dist in enumerate(self.right_eye_history):
                x = int(i * width / max(1, len(self.right_eye_history) - 1))
                y = int(height - dist * height * 5)  # Scale for visibility
                y = max(0, min(height-1, y))  # Clamp to valid range
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(plot_img, points[i], points[i+1], (255, 0, 0), 2)
        
        # Draw blink markers (magenta)
        if len(self.blink_history) > 1:
            for i, blinked in enumerate(self.blink_history):
                if blinked:
                    x = int(i * width / max(1, len(self.blink_history) - 1))
                    cv2.circle(plot_img, (x, height // 2), 5, (255, 0, 255), -1)
        
        # Draw legend
        cv2.putText(plot_img, "Left Eye", (width - 100, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(plot_img, "Right Eye", (width - 100, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(plot_img, "Blinks", (width - 100, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        return plot_img
    
    def plot_posture_graph(self, width=400, height=150):
        """Create a visualization of posture metrics over time.
        
        Args:
            width: Width of the plot
            height: Height of the plot
            
        Returns:
            numpy.ndarray: Plot image
        """
        # Create plot image (dark background)
        plot_img = np.zeros((height, width, 3), dtype=np.uint8)
        plot_img[:] = (30, 30, 30)  # Dark gray background
        
        # Draw grid lines
        for i in range(0, height, height//4):
            cv2.line(plot_img, (0, i), (width, i), (50, 50, 50), 1)
            
        for i in range(0, width, width//4):
            cv2.line(plot_img, (i, 0), (i, height), (50, 50, 50), 1)
        
        # Check if we're using shoulder-based detection (preferred) or face-based
        using_shoulder_detection = len(self.posture_history) > 0 and not isinstance(self.posture_history[0], float)
        
        if using_shoulder_detection:
            # Draw title for shoulder-based detection
            cv2.putText(plot_img, "Shoulder Drop", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Draw zero line at middle
            zero_y = height // 2
            cv2.line(plot_img, (0, zero_y), (width, zero_y), (100, 100, 100), 1)
            
            # Draw relative drop with color coding
            if len(self.posture_history) > 1:
                points = []
                for i, drop in enumerate(self.posture_history):
                    x = int(i * width / max(1, len(self.posture_history) - 1))
                    
                    # For shoulder data, we display the relative drop
                    if isinstance(drop, dict) and 'relative_drop' in drop:
                        # Scale the drop value for visibility
                        # Note: 1.0 is the baseline, higher values indicate hunching
                        relative_drop = drop['relative_drop']
                        y = int(zero_y - (relative_drop - 1.0) * height)
                    else:
                        # Default to middle if no valid data
                        y = zero_y
                        
                    y = max(0, min(height-1, y))  # Clamp to valid range
                    points.append((x, y))
                
                for i in range(len(points) - 1):
                    # Get the relative drop for current point
                    if isinstance(self.posture_history[i], dict) and 'relative_drop' in self.posture_history[i]:
                        relative_drop = self.posture_history[i]['relative_drop']
                        
                        # Determine color based on drop
                        if relative_drop >= 1.25:  # Severe
                            color = (0, 0, 255)  # Red
                        elif relative_drop >= 1.15:  # Medium
                            color = (0, 165, 255)  # Orange
                        elif relative_drop >= 1.08:  # Slight
                            color = (0, 255, 255)  # Yellow
                        else:  # Good
                            color = (0, 255, 0)  # Green
                    else:
                        color = (100, 100, 100)  # Default gray
                    
                    cv2.line(plot_img, points[i], points[i+1], color, 2)
            
            # Draw threshold lines (using typical values from HunchDetector)
            slight_y = int(zero_y - 0.08 * height)  # 1.08 threshold
            medium_y = int(zero_y - 0.15 * height)  # 1.15 threshold  
            severe_y = int(zero_y - 0.25 * height)  # 1.25 threshold
            
            cv2.line(plot_img, (0, slight_y), (width, slight_y), (0, 255, 255), 1)
            cv2.line(plot_img, (0, medium_y), (width, medium_y), (0, 165, 255), 1)
            cv2.line(plot_img, (0, severe_y), (width, severe_y), (0, 0, 255), 1)
        else:
            # Original code for face-based forward tilt
            cv2.putText(plot_img, "Posture Forward Tilt", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Draw zero line
            zero_y = height // 2
            cv2.line(plot_img, (0, zero_y), (width, zero_y), (100, 100, 100), 1)
            
            # Draw posture history with color coding
            if len(self.posture_history) > 1:
                points = []
                for i, tilt in enumerate(self.posture_history):
                    x = int(i * width / max(1, len(self.posture_history) - 1))
                    # Scale and center the tilt values
                    y = int(zero_y - tilt * height * 0.3)
                    y = max(0, min(height-1, y))  # Clamp to valid range
                    points.append((x, y))
                
                for i in range(len(points) - 1):
                    # Determine color based on tilt value (red for poor posture, green for good)
                    tilt = abs(self.posture_history[i])
                    if tilt > 1.1:  # Severe
                        color = (0, 0, 255)  # Red
                    elif tilt > 0.9:  # Medium
                        color = (0, 165, 255)  # Orange
                    elif tilt > 0.7:  # Slight
                        color = (0, 255, 255)  # Yellow
                    else:  # Good
                        color = (0, 255, 0)  # Green
                    
                    cv2.line(plot_img, points[i], points[i+1], color, 2)
            
            # Draw threshold lines
            slight_y = int(zero_y - 0.7 * height * 0.3)
            medium_y = int(zero_y - 0.9 * height * 0.3)
            severe_y = int(zero_y - 1.1 * height * 0.3)
            
            cv2.line(plot_img, (0, slight_y), (width, slight_y), (0, 255, 255), 1)
            cv2.line(plot_img, (0, medium_y), (width, medium_y), (0, 165, 255), 1)
            cv2.line(plot_img, (0, severe_y), (width, severe_y), (0, 0, 255), 1)
            
            # Draw mirrored thresholds
            slight_y_mirror = int(zero_y + 0.7 * height * 0.3)
            medium_y_mirror = int(zero_y + 0.9 * height * 0.3)
            severe_y_mirror = int(zero_y + 1.1 * height * 0.3)
            
            cv2.line(plot_img, (0, slight_y_mirror), (width, slight_y_mirror), (0, 255, 255), 1)
            cv2.line(plot_img, (0, medium_y_mirror), (width, medium_y_mirror), (0, 165, 255), 1)
            cv2.line(plot_img, (0, severe_y_mirror), (width, severe_y_mirror), (0, 0, 255), 1)
        
        return plot_img
    
    def draw_graphs(self, frame):
        """Add visualization graphs to the frame.
        
        Args:
            frame: The frame to draw on
            
        Returns:
            frame: Frame with graphs
        """
        h, w, _ = frame.shape
        
        # Create eye metrics graph
        eye_graph = self.plot_eye_graph(width=w//3, height=150)
        
        # Create posture graph
        posture_graph = self.plot_posture_graph(width=w//3, height=150)
        
        # Position graphs at bottom of frame
        y_offset = h - eye_graph.shape[0] - posture_graph.shape[0] - 10
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_offset-10), (w//3+10, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Insert eye graph
        eye_region = frame[y_offset:y_offset+eye_graph.shape[0], 5:5+eye_graph.shape[1]]
        cv2.addWeighted(eye_graph, 1.0, eye_region, 0.0, 0, eye_region)
        
        # Insert posture graph below eye graph
        posture_y = y_offset + eye_graph.shape[0] + 5
        posture_region = frame[posture_y:posture_y+posture_graph.shape[0], 5:5+posture_graph.shape[1]]
        cv2.addWeighted(posture_graph, 1.0, posture_region, 0.0, 0, posture_region)
        
        return frame

class CalibrationOverlay:
    """Handles drawing of calibration-related overlays."""
    
    def draw_calibration(self, frame, eye_calibration_in_progress, hunch_calibration_in_progress,
                        eye_calibration_start_time, eye_calibration_duration,
                        hunch_calibration_samples):
        """Draw calibration overlay."""
        if eye_calibration_in_progress or hunch_calibration_in_progress:
            if eye_calibration_in_progress:
                elapsed_time = time.time() - eye_calibration_start_time
                remaining_time = max(0, eye_calibration_duration - elapsed_time)
                calibration_text = f"EYE CALIBRATION: {remaining_time:.1f}s remaining"
            else:
                remaining_frames = max(0, 30 - len(hunch_calibration_samples))
                calibration_text = f"POSTURE CALIBRATION: {remaining_frames} frames remaining"
            
            h, w, _ = frame.shape
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            cv2.putText(frame, "CALIBRATION IN PROGRESS", (w//2 - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, calibration_text, (w//2 - 200, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Please sit with UPRIGHT posture and look at the camera", 
                       (w//2 - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Keep your shoulders level and back straight", 
                       (w//2 - 250, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 