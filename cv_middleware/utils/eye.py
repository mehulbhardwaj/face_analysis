"""
Eye state detection utility class.
"""

import numpy as np
import cv2
import time
import pygame
from collections import deque

class EyeStateDetector:
    """Class for detecting eye states and blinks from facial landmarks."""
    
    def __init__(self, history_size=30):
        """Initialize the eye state detector.
        
        Args:
            history_size: Number of frames to keep in the eye history (default: 30)
        """
        # Initialize blink detection variables
        self.blink_counter = 0
        self.blink_threshold = 0.2  # Threshold for eye closure to be considered a blink
        self.min_blink_frames = 2  # Minimum frames eye must be closed to count as blink
        self.closed_frame_counter = 0
        
        # Initialize eye lid distance history for plotting
        self.left_eye_history = deque(maxlen=history_size)
        self.right_eye_history = deque(maxlen=history_size)
        self.start_time = 0
        
        # Blink detection parameters for graph-based approach
        self.blink_detected = False
        self.blink_window_size = 5 # Window size for detecting dips
        self.blink_dip_threshold = 0.1  # Minimum relative dip to consider as blink
        self.blink_min_duration = 1  # Minimum duration of dip in frames
        self.blink_cooldown = 5  # Frames to wait before detecting another blink
        self.blink_cooldown_counter = 0
        
        # Calibration variables
        self.calibrated = False
        self.baseline_eye_openness = 0.0
        self.baseline_std = 0.0
        self.calibration_samples = []
        self.calibration_duration = 5  # seconds
        self.calibration_start_time = 0
        self.calibration_in_progress = False
        
        # Additional calibration metrics
        self.baseline_eyebrow_distance = 0.0
        self.baseline_eyebrow_height = 0.0
        self.baseline_mouth_height = 0.0
        self.baseline_mouth_width = 0.0
        
        # Eye frowness detection
        self.baseline_left_eyebrow_eye_distance = 0.0
        self.baseline_right_eyebrow_eye_distance = 0.0
        self.eyebrow_eye_distance_history = deque(maxlen=history_size)
        self.eyebrow_raise_threshold = 1.1 # Eyebrows are considered raised when 10% above baseline
        self.eyebrow_frown_threshold = 0.95  # Eyes are considered frowning when 5% below baseline
        
        # Eye state detection thresholds
        self.wide_open_threshold = 1.2  # Eyes are considered wide open when 20% above baseline
        self.squint_threshold = 0.9  # Eyes are considered squinting when 20% below baseline
        self.blink_threshold_factor = 0.6  # Eyes are considered blinking when 40% below baseline
        
        # Facial expression detection thresholds
        self.smile_threshold = 1.2  # Mouth is considered smiling when 20% above baseline width
        self.frown_threshold = 0.8  # Mouth is considered frowning when 20% below baseline width
        
        # Eye landmark indices
        self.left_eye_upper = 159
        self.left_eye_lower = 145
        self.right_eye_upper = 386
        self.right_eye_lower = 374
        
        # Additional landmarks for expressions
        self.left_eyebrow = 105
        self.right_eyebrow = 334
        self.left_eye_corner = 33
        self.right_eye_corner = 263
        self.mouth_left = 78
        self.mouth_right = 308
        self.mouth_top = 13
        self.mouth_bottom = 14
        
        # Add temporal smoothing for frowning detection
        self.frowning_history = deque(maxlen=3)  # Store last 3 frames of frowning state
        
        # Initialize audio for cue
        try:
            pygame.mixer.init()
            self.audio_initialized = True
            try:
                self.cue_sound = pygame.mixer.Sound("cue.mp3")
                print("Successfully loaded audio cue")
            except Exception as e:
                print(f"Error loading audio cue: {e}")
                self.cue_sound = None
        except (ImportError, pygame.error):
            print("Pygame not available, audio cues disabled")
            self.audio_initialized = False
            self.cue_sound = None
            
    def play_audio_cue(self):
        """Play audio cue when a blink is detected."""
        if self.cue_sound is None or not self.audio_initialized:
            return
        
        try:
            import threading
            # Play in a separate thread to avoid blocking
            def play_sound():
                self.cue_sound.play()
                
            threading.Thread(target=play_sound).start()
        except Exception as e:
            print(f"Error playing audio cue: {e}")
        
    def start_calibration(self):
        """Start the calibration process to establish a baseline for eye measurements."""
        self.calibration_in_progress = True
        self.calibration_samples = []
        self.calibration_start_time = time.time()
        self.calibrated = False
        print("Calibration started. Please gaze normally at the camera for 5 seconds...")
    
    def update_calibration(self, landmarks, face_size):
        """Update calibration with new measurements.
        
        Args:
            landmarks: MediaPipe NormalizedLandmarkList object
            face_size: Normalization factor based on face size
            
        Returns:
            bool: True if calibration is complete, False otherwise
        """
        if not self.calibration_in_progress:
            return False
            
        # Calculate eye openness
        left_eye_distance, right_eye_distance = self.calculate_eye_distances(landmarks, face_size)
        eye_openness = (left_eye_distance + right_eye_distance) / 2
        self.calibration_samples.append(eye_openness)
        
        # Calculate eyebrow distance (between inner corners of eyebrows)
        eyebrow_distance = self.calculate_distance(landmarks, self.left_eyebrow, self.right_eyebrow) / face_size
        
        # Calculate eyebrow to eye distance (frowning)
        left_eyebrow_eye_distance = self.calculate_distance(landmarks, self.left_eyebrow, self.left_eye_upper) / face_size
        right_eyebrow_eye_distance = self.calculate_distance(landmarks, self.right_eyebrow, self.right_eye_upper) / face_size
        
        # Calculate mouth height and width
        mouth_height = self.calculate_distance(landmarks, self.mouth_top, self.mouth_bottom) / face_size
        mouth_width = self.calculate_distance(landmarks, self.mouth_left, self.mouth_right) / face_size
        
        # Check if calibration duration has elapsed
        elapsed_time = time.time() - self.calibration_start_time
        if elapsed_time >= self.calibration_duration:
            # Calculate baseline and standard deviation
            if len(self.calibration_samples) > 0:
                self.baseline_eye_openness = np.mean(self.calibration_samples)
                self.baseline_std = np.std(self.calibration_samples)
                
                # Set baseline for additional metrics
                self.baseline_eyebrow_distance = eyebrow_distance
                self.baseline_left_eyebrow_eye_distance = left_eyebrow_eye_distance
                self.baseline_right_eyebrow_eye_distance = right_eyebrow_eye_distance
                self.baseline_mouth_height = mouth_height
                self.baseline_mouth_width = mouth_width
                
                self.calibrated = True
                self.start_time = time.time()
                self.calibration_in_progress = False
                print(f"Calibration complete. Baseline: {self.baseline_eye_openness:.4f}, Std: {self.baseline_std:.4f}")
                return True
            else:
                print("Calibration failed: No samples collected")
                self.calibration_in_progress = False
                return False
        
        return False
    
    def detect_eye_state(self, eye_openness):
        """Detect the current state of the eyes based on calibration baseline.
        
        Args:
            eye_openness: Current eye openness measurement
            
        Returns:
            str: Current eye state ("wide_open", "normal", "squinting", "blinking", "closed")
        """
        if not self.calibrated:
            return "unknown"
            
        # Calculate relative eye openness compared to baseline
        relative_openness = eye_openness / self.baseline_eye_openness
        
        # Determine eye state based on thresholds
        if relative_openness >= self.wide_open_threshold:
            return "wide_open"
        elif relative_openness <= self.blink_threshold_factor:
            return "blinking"
        elif relative_openness <= self.squint_threshold:
            return "squinting"
        elif relative_openness <= 0.1:  # Almost completely closed
            return "closed"
        else:
            return "normal"
    
    def detect_facial_expressions(self, landmarks, face_size):
        """Detect facial expressions from landmarks.
        
        Args:
            landmarks: MediaPipe NormalizedLandmarkList object
            face_size: Normalization factor based on face size
            
        Returns:
            dict: Dictionary containing expression information
        """
        if not self.calibrated:
            return {
                "is_frowning": False,
                "smiling": False
            }
        
        # Calculate eyebrow to eye distances
        left_eyebrow_eye_distance = self.calculate_distance(landmarks, self.left_eyebrow, self.left_eye_upper) / face_size
        right_eyebrow_eye_distance = self.calculate_distance(landmarks, self.right_eyebrow, self.right_eye_upper) / face_size
        
        # Calculate mouth width
        mouth_width = self.calculate_distance(landmarks, self.mouth_left, self.mouth_right) / face_size
        
        # Calculate relative measurements
        relative_left_eyebrow_eye_distance = left_eyebrow_eye_distance / self.baseline_left_eyebrow_eye_distance
        relative_right_eyebrow_eye_distance = right_eyebrow_eye_distance / self.baseline_right_eyebrow_eye_distance
        relative_mouth_width = mouth_width / self.baseline_mouth_width
        
        # Determine frowning by eyebrow-eye distance
        left_margin = relative_left_eyebrow_eye_distance - self.eyebrow_frown_threshold
        right_margin = relative_right_eyebrow_eye_distance - self.eyebrow_frown_threshold
        
        is_frowning = (left_margin < 0 and right_margin < 0) or left_margin < 0 or right_margin < 0
        
        # Update frowning history for temporal smoothing
        self.frowning_history.append(is_frowning)
        
        # Only consider frowning if detected in majority of recent frames
        is_frowning_smoothed = sum(self.frowning_history) > len(self.frowning_history) / 2
        
        # Determine smiling by mouth width
        is_smiling = relative_mouth_width >= self.smile_threshold
        
        return {
            "is_frowning": is_frowning_smoothed,
            "smiling": is_smiling,
            "left_eyebrow_eye_distance": left_eyebrow_eye_distance,
            "right_eyebrow_eye_distance": right_eyebrow_eye_distance,
            "relative_left_eyebrow_eye_distance": relative_left_eyebrow_eye_distance,
            "relative_right_eyebrow_eye_distance": relative_right_eyebrow_eye_distance,
            "mouth_width": mouth_width,
            "relative_mouth_width": relative_mouth_width
        }
    
    def detect_blink_from_graph(self, eye_openness):
        """Detect blinks by analyzing dips in the eye openness graph.
        
        Args:
            eye_openness: The current eye openness value
            
        Returns:
            bool: True if a blink was detected, False otherwise
        """
        # Add current eye openness to history
        self.left_eye_history.append(eye_openness)
        
        # If we're in cooldown period, decrement counter and return
        if self.blink_cooldown_counter > 0:
            self.blink_cooldown_counter -= 1
            return False
        
        # Need enough history to detect a dip
        if len(self.left_eye_history) < self.blink_window_size:
            return False
        
        # Check if we're in a dip (blink)
        current_value = eye_openness
        window_values = list(self.left_eye_history)[-self.blink_window_size:]
        
        # Calculate average of window excluding current value
        window_avg = sum(window_values[:-1]) / (len(window_values) - 1)
        
        # Calculate relative dip
        relative_dip = (window_avg - current_value) / max(0.01, window_avg)
        
        # Check if this is a significant dip
        if relative_dip > self.blink_dip_threshold:
            # Check if we've been in a dip for minimum duration
            if self.closed_frame_counter >= self.blink_min_duration:
                # This is a valid blink
                self.blink_counter += 1
                self.blink_detected = True
                self.closed_frame_counter = 0
                self.blink_cooldown_counter = self.blink_cooldown
                # Play audio cue for blink
                # self.play_audio_cue()
                return True
            else:
                # We're in a dip but not long enough yet
                self.closed_frame_counter += 1
                return False
        else:
            # Not in a dip
            self.closed_frame_counter = 0
            return False

    def detect_blink(self, eye_openness):
        """Detect if a blink has occurred based on eye openness.
        
        Args:
            eye_openness: The normalized eye openness value
            
        Returns:
            bool: True if a blink was detected, False otherwise
        """
        # Use the graph-based blink detection
        return self.detect_blink_from_graph(eye_openness)
    
    def calculate_distance(self, landmarks, index1, index2):
        """Calculate distance between two landmarks.
        
        Args:
            landmarks: MediaPipe NormalizedLandmarkList object
            index1: Index of first landmark
            index2: Index of second landmark
            
        Returns:
            float: Distance between landmarks
        """
        point1 = landmarks.landmark[index1]
        point2 = landmarks.landmark[index2]
        
        return np.sqrt((point1.x - point2.x)**2 + 
                      (point1.y - point2.y)**2 + 
                      (point1.z - point2.z)**2)

    def calculate_eye_distances(self, landmarks, face_size):
        """Calculate the distance between upper and lower eyelids for both eyes.
        
        Args:
            landmarks: MediaPipe NormalizedLandmarkList object
            face_size: Normalization factor based on face size
            
        Returns:
            tuple: (left_eye_distance, right_eye_distance) normalized by face size
        """
        # Left eye landmarks (upper lid: 159, lower lid: 145)
        left_upper = landmarks.landmark[self.left_eye_upper]
        left_lower = landmarks.landmark[self.left_eye_lower]
        left_eye_distance = np.sqrt((left_upper.x - left_lower.x)**2 + 
                                  (left_upper.y - left_lower.y)**2 + 
                                  (left_upper.z - left_lower.z)**2) / face_size
        
        # Right eye landmarks (upper lid: 386, lower lid: 374)
        right_upper = landmarks.landmark[self.right_eye_upper]
        right_lower = landmarks.landmark[self.right_eye_lower]
        right_eye_distance = np.sqrt((right_upper.x - right_lower.x)**2 + 
                                   (right_upper.y - right_lower.y)**2 + 
                                   (right_upper.z - right_lower.z)**2) / face_size
        
        return left_eye_distance, right_eye_distance
    
    def process_eyes(self, landmarks, face_size=1.0):
        """Process eye measurements from facial landmarks.
        
        Args:
            landmarks: MediaPipe NormalizedLandmarkList object
            face_size: Normalization factor based on face size
            
        Returns:
            dict: Dictionary containing eye state information
        """
        # Calculate eye distances
        left_eye_distance, right_eye_distance = self.calculate_eye_distances(landmarks, face_size)
        
        # Add to history
        self.right_eye_history.append(right_eye_distance)
        
        # Calculate average eye openness for blink detection
        eye_openness = (left_eye_distance + right_eye_distance) / 2
        
        # Update calibration if in progress
        if self.calibration_in_progress:
            self.update_calibration(landmarks, face_size)
        
        # Detect blink using graph-based approach
        blink_detected = self.detect_blink(eye_openness)
        
        # Determine eye state based on calibration
        eye_state = self.detect_eye_state(eye_openness)
        
        # Detect facial expressions
        expressions = self.detect_facial_expressions(landmarks, face_size)
        
        return {
            "left_eye_distance": left_eye_distance,
            "right_eye_distance": right_eye_distance,
            "eye_openness": eye_openness,
            "blink_detected": blink_detected,
            "eye_state": eye_state,
            "blink_counter": self.blink_counter,
            "expressions": expressions
        }
    
    def reset_blink_counter(self):
        """Reset the blink counter to zero."""
        self.blink_counter = 0
        print("Blink counter reset") 