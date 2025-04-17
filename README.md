# Computer Vision Middleware

A comprehensive computer vision middleware package that provides emotion detection, eye state tracking, and posture analysis capabilities with rich visualization options. This package is designed to be easily integrated into UI applications.

## Features

- **Emotion Detection**: Identifies emotions (Happy, Sad, Relaxed, Stressed) with valence and arousal values
- **Eye State Tracking**: Monitors eye states (Open, Closed, Blinking) and counts blinks
- **Posture Analysis**: Detects hunching and poor posture with severity levels
- **Facial Expression Analysis**: Detects smiling, frowning, and other expressions
- **Rich Visualization Overlays**:
  - Performance metrics (CPU, Memory, FPS)
  - Face landmarks visualization
  - Eye state indicators
  - Posture indicators with hunch statistics
  - Emotion indicators
  - Historical data graphs (eye metrics, posture)
- **Calibration Tools**: For personalized eye and posture detection
- **Real-time Processing**: Optimized for performance with metrics monitoring

## Installation

1. Clone this repository
2. Create and activate the conda environment:
```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate <env name>
```

## Quick Start

Here's a simple example of how to use the middleware in your UI application:

```python
from cv_middleware import CVProcessor
import cv2

# Initialize the processor
processor = CVProcessor()

# Initialize your video source (webcam, video file, etc.)
cap = cv2.VideoCapture(0)

try:
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results = processor.process_frame(frame)
        
        # Get current states
        emotion = processor.get_current_emotion()
        eye_state = processor.get_eye_state()
        posture = processor.get_posture_state()
        
        # Use the results in your UI
        # ...
        
finally:
    # Clean up
    cap.release()
    processor.release()
```

## UI Integration

### Using Visualization Overlays

The middleware includes a powerful visualization module (`ui.overlays`) that you can use to enhance your UI:

```python
from cv_middleware import CVProcessor
from ui.overlays import PerformanceOverlay, FaceOverlay, CalibrationOverlay
import cv2

# Initialize components
processor = CVProcessor()
perf_overlay = PerformanceOverlay()
face_overlay = FaceOverlay()
calib_overlay = CalibrationOverlay()

cap = cv2.VideoCapture(0)
fps = 0
last_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate FPS
        current_time = time.time()
        if current_time - last_time > 0:
            fps = 1 / (current_time - last_time)
        last_time = current_time
        
        # Process frame
        results = processor.process_frame(frame)
        
        # Add performance overlay
        perf_overlay.update(fps)
        frame = perf_overlay.draw(frame)
        
        # Add face overlays
        h, w, _ = frame.shape
        if 'landmarks' in results:
            face_overlay.draw_landmarks(frame, results['landmarks'], w, h)
        
        face_overlay.draw_eye_state(frame, results['eye_state'], w)
        face_overlay.draw_emotion(frame, results['emotion'], 30)
        face_overlay.draw_posture(frame, results['posture'], 60)
        
        # Add visualization graphs
        frame = face_overlay.draw_graphs(frame)
        
        # Display the frame in your UI
        # ...
        
finally:
    cap.release()
    processor.release()
```

### Calibration Interface

The middleware supports calibration for eye and posture detection:

```python
# Start eye calibration
processor.start_eye_calibration()

# Start posture calibration
processor.start_posture_calibration()

# While calibration is in progress, display the calibration overlay
if processor.is_calibrating():
    calib_overlay.draw_calibration(
        frame,
        processor.is_eye_calibration_active(),
        processor.is_posture_calibration_active(),
        processor.get_eye_calibration_start_time(),
        processor.get_eye_calibration_duration(),
        processor.get_posture_calibration_samples()
    )
```

## API Reference

### CVProcessor

The main interface for computer vision functionality.

#### Methods

- `process_frame(frame)`: Process a single frame and return results
- `get_current_emotion()`: Get the current emotion state
- `get_eye_state()`: Get the current eye state
- `get_posture_state()`: Get the current posture state
- `start_eye_calibration(duration=5.0)`: Start eye calibration process
- `start_posture_calibration()`: Start posture calibration process
- `is_calibrating()`: Check if any calibration is in progress
- `is_eye_calibration_active()`: Check if eye calibration is active
- `is_posture_calibration_active()`: Check if posture calibration is active
- `get_eye_calibration_start_time()`: Get the start time of eye calibration
- `get_eye_calibration_duration()`: Get the duration of eye calibration
- `get_posture_calibration_samples()`: Get current posture calibration samples
- `release()`: Release resources

### UI Overlay Classes

#### PerformanceOverlay

Handles drawing of performance metrics on frames.

- `update(current_fps)`: Update performance metrics with current FPS
- `is_cpu_high()`: Check if CPU usage is above the alert threshold
- `draw(frame)`: Draw performance metrics on the frame

#### FaceOverlay

Handles drawing of face-related overlays.

- `draw_landmarks(frame, face_landmarks, w, h)`: Draw facial landmarks on the frame
- `draw_eye_state(frame, eye_state, w)`: Draw eye state indicator
- `draw_emotion(frame, emotion_name, y_offset)`: Draw emotion indicator
- `draw_posture(frame, posture_info, y_offset)`: Draw posture indicator
- `plot_eye_graph(width, height)`: Create a visualization of eye metrics over time
- `plot_posture_graph(width, height)`: Create a visualization of posture metrics over time
- `draw_graphs(frame)`: Add visualization graphs to the frame

#### CalibrationOverlay

Handles drawing of calibration-related overlays.

- `draw_calibration(frame, eye_calibration_in_progress, hunch_calibration_in_progress, eye_calibration_start_time, eye_calibration_duration, hunch_calibration_samples)`: Draw calibration overlay

### Results Format

The `process_frame()` method returns a dictionary with the following structure:

```python
{
    'emotion': {
        'emotion': str,  # Emotion name (Happy, Sad, Relaxed, Stressed)
        'valence': float,  # Valence value (0-1)
        'arousal': float,  # Arousal value (0-1)
        'color': tuple  # BGR color for visualization
    },
    'eye_state': {
        'eye_state': str,  # Eye state (wide_open, squinting, blinking, closed)
        'blink_counter': int,  # Number of blinks
        'blink_detected': bool,  # Whether a blink was detected in this frame
        'left_eye_distance': float,  # Left eye openness metric
        'right_eye_distance': float,  # Right eye openness metric
        'expressions': {
            'is_frowning': bool,  # Whether the user is frowning
            'smiling': bool  # Whether the user is smiling
        }
    },
    'posture': {
        'hunch_state': str,  # Posture state (Good, Slight Hunch, Medium Hunch, Severe Hunch)
        'hunch_detected': bool,  # Whether a hunch is detected
        'hunch_counter': int,  # Number of hunches detected
        'is_hunched': bool,  # Whether currently hunched
        'hunch_duration': float,  # Duration of current hunch in seconds
        'relative_drop': float,  # Shoulder drop metric (shoulder-based detection)
        'forward_tilt': float,  # Forward tilt metric (face-based detection)
        'upright_percentage': float,  # Percentage of time in upright posture
        'posture_stats': {
            'total_time': float,  # Total monitoring time
            'upright_time': float,  # Time spent upright
            'hunched_time': float  # Time spent hunched
        }
    },
    'landmarks': object  # MediaPipe face landmarks
}
```

## Customization

### Visualization Settings

You can customize the visualization overlays by modifying the parameters when initializing the overlay classes:

```python
# Customize performance overlay
perf_overlay = PerformanceOverlay(
    history_size=200,  # Increase history for graphs
    cpu_alert_threshold=30.0  # Adjust CPU alert threshold
)

# Customize graph sizes when drawing
eye_graph = face_overlay.plot_eye_graph(width=500, height=200)
```

### Integration with Different UI Frameworks

#### Pygame Integration

```python
import pygame

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((1280, 720))

# In your main loop
frame = processor.process_frame(frame)
# Add overlays
# ...

# Convert OpenCV frame to Pygame surface
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = np.rot90(frame)
frame = pygame.surfarray.make_surface(frame)
screen.blit(frame, (0, 0))
pygame.display.flip()
```

#### Qt Integration

```python
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap

# In your Qt application
label = QLabel()

# In your frame processing loop
frame = processor.process_frame(frame)
# Add overlays
# ...

# Convert to Qt format
h, w, ch = frame.shape
bytesPerLine = ch * w
converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
qImg = QImage(converted.data, w, h, bytesPerLine, QImage.Format_RGB888)
label.setPixmap(QPixmap.fromImage(qImg))
```

## Examples

See `example.py` for a complete working example of how to use the middleware.

Additional examples can be found in the `examples/` directory:
- `basic_integration.py`: Simple integration example
- `full_ui_integration.py`: Complete UI integration with all overlays
- `calibration_example.py`: Example showing calibration process
- `qt_integration.py`: Example of integration with Qt

## Requirements

- Python 3.9+
- OpenCV
- MediaPipe
- NumPy
- Pygame (for audio feedback)
- Matplotlib (for visualization)

See `environment.yml` for complete list of dependencies.

## Best Practices

1. **Performance Monitoring**: Use the PerformanceOverlay to monitor system load and adjust processing resolution if needed
2. **Calibration**: Always perform calibration before using posture detection for accurate results
3. **Error Handling**: Implement proper error handling for frames where face detection fails
4. **UI Responsiveness**: Process frames in a separate thread to keep your UI responsive
5. **Customization**: Adjust visualization colors and styles to match your application theme

## License

This project is licensed under the MIT License - see the LICENSE file for details.
