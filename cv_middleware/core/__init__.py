"""
Core module for computer vision middleware.

This module provides the main interfaces and classes for computer vision functionality.
"""

from .processor import CVProcessor
from .monitor import PostureMonitor

__all__ = ['CVProcessor', 'PostureMonitor'] 