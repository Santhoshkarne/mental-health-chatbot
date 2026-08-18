"""
emotion package — exposes EmotionDetector and EmotionService for clean imports.

Usage:
    from emotion import EmotionDetector, EmotionService
"""

from emotion.models.emotion_detector import EmotionDetector
from emotion.service import EmotionService

__all__ = ["EmotionDetector", "EmotionService"]
