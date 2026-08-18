"""
EmotionService — the single entry point the chatbot (or orchestrator) calls.

Wraps EmotionDetector so the rest of the app never touches the model directly.
This makes it easy to add caching, logging, or swap the model later without
changing any calling code.

Usage:
    from emotion import EmotionService

    service = EmotionService()
    result = service.analyze("I feel really anxious about my exam.")
    print(result.primary_emotion)   # "nervousness"
    print(result.emotions)          # top-3 EmotionScore objects
"""

from schemas.emotion_schema import EmotionResult
from metrics.emotion.models.emotion_detector import EmotionDetector


class EmotionService:
    """
    High-level service that the chatbot orchestrator calls.

    Why not use EmotionDetector directly?
    - The service handles initialisation, so the orchestrator doesn't
      need to know about model names, top_k, or HuggingFace pipelines.
    - If you later want to add caching (so repeated identical messages
      don't re-run the model), you add it here — one place.
    - If you swap models, only this file changes.
    """

    def __init__(self, top_k: int = 3):
        self._detector = EmotionDetector(top_k=top_k)

    def analyze(self, text: str) -> EmotionResult:
        """
        Analyze the emotional content of user text.

        Args:
            text: Raw user message string.

        Returns:
            EmotionResult with primary_emotion, top-k emotions, and raw_scores.
        """
        return self._detector.detect(text)

    def get_primary_emotion(self, text: str) -> str:
        """Shortcut — returns just the primary emotion label as a string."""
        return self._detector.detect(text).primary_emotion

    def get_emotion_summary(self, text: str) -> dict[str, float]:
        """
        Returns a simple {label: score} dict of just the top-k emotions.
        Useful for quick lookups without the full EmotionResult object.
        """
        result = self._detector.detect(text)
        return {e.label: e.score for e in result.emotions}
