"""
Main Application Entry Point

This module serves as the entry point for the computer vision application.
It initializes and runs the PostureMonitor application.
"""

from cv_middleware.core.monitor import PostureMonitor

def main():
    """Initialize and run the posture monitoring application."""
    monitor = PostureMonitor()
    monitor.run()

if __name__ == "__main__":
    main() 