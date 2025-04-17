"""
Core monitor class that coordinates CV processing and UI rendering.
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from ..utils.hunch import HunchDetector
from ..utils.eye import EyeStateDetector
from ..utils.emotion import EmotionDetector
from ui.overlays import PerformanceOverlay, FaceOverlay, CalibrationOverlay

class PostureMonitor:
    """Main application class that coordinates CV processing and UI rendering."""
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Pose for shoulder detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize detectors
        self.hunch_detector = HunchDetector()
        self.eye_detector = EyeStateDetector()
        self.emotion_detector = EmotionDetector()
        
        # Initialize UI overlays
        self.performance_overlay = PerformanceOverlay()
        self.face_overlay = FaceOverlay()
        self.calibration_overlay = CalibrationOverlay()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Calibration state
        self.eye_calibration_in_progress = True
        self.eye_calibration_start_time = time.time()
        self.eye_calibration_duration = 5.0  # seconds
        
        # Hunch calibration state
        self.hunch_calibration_in_progress = True
        self.start_hunch_calibration()
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
    
    def start_hunch_calibration(self):
        """Start the hunch detector calibration process"""
        self.hunch_detector.start_calibration()
        self.hunch_calibration_in_progress = True
    
    def process_frame(self, frame):
        """Process a single frame and return the annotated frame."""
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Process frame with MediaPipe Face Mesh
        face_results = self.face_mesh.process(rgb_frame)
        
        # Process frame with MediaPipe Pose for shoulder tracking
        pose_results = self.pose.process(rgb_frame)
        
        face_landmarks = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # Draw facial landmarks
            self.face_overlay.draw_landmarks(frame, face_landmarks, w, h)
            
            # Process eye state first to get expressions
            eye_state = self.eye_detector.process_eyes(face_landmarks)
            
            # Draw eye state 
            self.face_overlay.draw_eye_state(frame, eye_state, w)
            
            # Extract expressions from eye state
            expressions = eye_state.get('expressions', {})
                
            # Process emotion using expressions
            emotion = self.emotion_detector.process_emotion(face_landmarks, expressions)
            
            # Draw emotion with proper y_offset so it doesn't overlap with other UI elements
            self.face_overlay.draw_emotion(frame, emotion, 100)
            
            # Handle eye calibration
            if self.eye_calibration_in_progress:
                elapsed_time = time.time() - self.eye_calibration_start_time
                if elapsed_time >= self.eye_calibration_duration:
                    self.eye_calibration_in_progress = False
        
        # Process posture - prefer pose landmarks, fall back to face landmarks
        if pose_results.pose_landmarks:
            # Convert to grayscale for optical flow
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process posture with pose landmarks
            posture_info = self.hunch_detector.process_posture(
                pose_results.pose_landmarks.landmark,
                frame=gray_frame
            )
            
            # Draw shoulder tracking indicators
            if self.hunch_detector.calibrated:
                frame = self.hunch_detector.draw_posture_indicators(frame)
            
            # Draw posture state
            self.face_overlay.draw_posture(frame, posture_info, 130)
            
            # Update hunch calibration state
            if self.hunch_calibration_in_progress and self.hunch_detector.calibrated:
                self.hunch_calibration_in_progress = False
        elif face_landmarks:
            # Fall back to face landmarks if pose not detected
            posture_info = self.hunch_detector.process_posture(face_landmarks)
            self.face_overlay.draw_posture(frame, posture_info, 130)
        
        # Draw calibration overlay if needed
        self.calibration_overlay.draw_calibration(
            frame,
            self.eye_calibration_in_progress,
            self.hunch_calibration_in_progress,
            self.eye_calibration_start_time,
            self.eye_calibration_duration,
            self.hunch_detector.calibration_samples if hasattr(self.hunch_detector, 'calibration_samples') else []
        )
        
        # Draw visualization graphs
        self.face_overlay.draw_graphs(frame)
        
        # Update and draw performance metrics
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_frame_time = current_time
        
        self.performance_overlay.update(self.fps)
        frame = self.performance_overlay.draw(frame)
        
        return frame

    def run(self):
        """Main application loop."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Posture Monitor', annotated_frame)
                
                # Break loop on 'q' press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Start recalibration on 'c' key
                    self.eye_detector.start_calibration()
                    self.eye_calibration_in_progress = True
                    self.eye_calibration_start_time = time.time()
                    self.start_hunch_calibration()
                    print("Recalibration started...")
                elif key == ord('r'):
                    # Reset counters on 'r' key
                    self.eye_detector.reset_blink_counter()
                    self.hunch_detector.reset_hunch_counter()
                    print("Counters reset")
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            self.pose.close() 