"""
Computer Vision Middleware Package

This package provides a simple interface for computer vision functionality including:
- Emotion detection
- Eye state tracking
- Posture analysis (hunch detection)

Usage:
    from cv_middleware import CVProcessor
    
    # Initialize the processor
    processor = CVProcessor()
    
    # Process a frame
    results = processor.process_frame(frame)
    
    # Get current emotion
    emotion = processor.get_current_emotion()
    
    # Get eye state
    eye_state = processor.get_eye_state()
    
    # Get posture state
    posture = processor.get_posture_state()
    
    # Clean up
    processor.release()
"""

from .core.processor import CVProcessor

__all__ = ['CVProcessor'] 