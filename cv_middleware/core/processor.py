"""
Core processor class that provides a simple interface for computer vision functionality.
"""

import cv2
import numpy as np
import mediapipe as mp
from ..utils.emotion import EmotionDetector
from ..utils.eye import EyeStateDetector
from ..utils.hunch import HunchDetector

class CVProcessor:
    """Main interface for computer vision functionality."""
    
    def __init__(self, history_size=30):
        """Initialize the computer vision processor.
        
        Args:
            history_size (int): Number of frames to keep in history for smoothing
        """
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
        self.emotion_detector = EmotionDetector(history_size)
        self.eye_detector = EyeStateDetector(history_size)
        self.hunch_detector = HunchDetector(history_size)
        
        # Initialize state
        self.current_emotion = None
        self.current_eye_state = None
        self.current_posture = None
        self.hunch_calibration_started = False
        
    def process_frame(self, frame):
        """Process a single frame and return results.
        
        Args:
            frame (numpy.ndarray): BGR image frame
            
        Returns:
            dict: Dictionary containing processing results
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        face_results = self.face_mesh.process(rgb_frame)
        
        # Process with MediaPipe Pose for shoulder landmarks
        pose_results = self.pose.process(rgb_frame)
        
        # Default return values
        result = {
            'emotion': None,
            'eye_state': None,
            'posture': None,
            'landmarks': None,
            'pose_landmarks': None
        }
            
        # If face landmarks detected
        if face_results.multi_face_landmarks:
            # Get face landmarks
            face_landmarks = face_results.multi_face_landmarks[0]
            result['landmarks'] = face_landmarks
            
            # Process eye state first to get facial expressions
            eye_result = self.eye_detector.process_eyes(face_landmarks)
            self.current_eye_state = eye_result
            result['eye_state'] = eye_result
            
            # Extract expressions from eye result
            expressions = eye_result.get('expressions', {})
            
            # Process emotion with expressions
            emotion_result = self.emotion_detector.process_emotion(face_landmarks, expressions)
            self.current_emotion = emotion_result
            result['emotion'] = emotion_result
        
        # Process posture using pose landmarks if available (more accurate),
        # otherwise fall back to face landmarks
        if pose_results.pose_landmarks:
            # Convert frame to grayscale for optical flow
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process posture with pose landmarks and optical flow
            posture_result = self.hunch_detector.process_posture(
                pose_results.pose_landmarks.landmark, 
                frame=gray_frame
            )
            result['pose_landmarks'] = pose_results.pose_landmarks
        elif face_results.multi_face_landmarks:
            # Fall back to face landmarks if pose not detected
            posture_result = self.hunch_detector.process_posture(face_landmarks)
        else:
            # No landmarks detected
            posture_result = {
                "calibrated": False,
                "hunch_state": "Unknown",
                "is_hunched": False,
                "hunch_counter": 0
            }
        
        self.current_posture = posture_result
        result['posture'] = posture_result
        
        # Auto-start calibration when not calibrated and we have face/pose landmarks
        if (not self.hunch_calibration_started and 
            (face_results.multi_face_landmarks or pose_results.pose_landmarks)):
            self.hunch_detector.start_calibration()
            self.hunch_calibration_started = True
        
        return result
        
    def get_current_emotion(self):
        """Get the current emotion state."""
        return self.current_emotion
        
    def get_eye_state(self):
        """Get the current eye state."""
        return self.current_eye_state
        
    def get_posture_state(self):
        """Get the current posture state."""
        return self.current_posture
        
    def release(self):
        """Release resources."""
        self.face_mesh.close()
        self.pose.close()
        
    def start_calibration(self):
        """Start calibration for all detectors."""
        self.eye_detector.start_calibration()
        self.hunch_detector.start_calibration()
        self.hunch_calibration_started = True 