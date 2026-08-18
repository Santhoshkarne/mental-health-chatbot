"""
Unit tests for EmotionDetector and EmotionService.

These tests load the real HuggingFace model, so they take ~30-60 seconds
on the first run (model download). Subsequent runs are faster (cached).

Run with:
    python -m pytest emotion/tests/test_emotion_detector.py -v
"""

import pytest
from schemas.emotion_schema import EmotionScore, EmotionResult
from emotion.models.emotion_detector import EmotionDetector
from emotion.service import EmotionService


# ──────────────────────────────────────────────────────────────
# Fixtures — shared setup used by multiple tests
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector():
    """
    Create one EmotionDetector for the entire test module.

    scope="module" means this detector is created ONCE and reused
    across all tests in this file. Without it, every test would
    reload the 500MB model — wasting time.
    """
    return EmotionDetector(top_k=3)


@pytest.fixture(scope="module")
def service():
    """Create one EmotionService for the entire test module."""
    return EmotionService(top_k=3)


# ──────────────────────────────────────────────────────────────
# Test 1: Basic output structure
# ──────────────────────────────────────────────────────────────

class TestEmotionDetectorStructure:
    """Verify the detector returns correctly shaped data."""

    def test_returns_emotion_result(self, detector: EmotionDetector):
        """detect() should return an EmotionResult Pydantic model."""
        result = detector.detect("I feel happy today")
        assert isinstance(result, EmotionResult)

    def test_has_primary_emotion(self, detector: EmotionDetector):
        """The result must always have a primary_emotion string."""
        result = detector.detect("I feel happy today")
        assert isinstance(result.primary_emotion, str)
        assert len(result.primary_emotion) > 0

    def test_returns_top_3_emotions(self, detector: EmotionDetector):
        """The emotions list should contain exactly top_k (3) items."""
        result = detector.detect("I feel happy today")
        assert len(result.emotions) == 3

    def test_emotions_are_emotion_scores(self, detector: EmotionDetector):
        """Each item in the emotions list should be an EmotionScore."""
        result = detector.detect("I feel happy today")
        for emotion in result.emotions:
            assert isinstance(emotion, EmotionScore)
            assert isinstance(emotion.label, str)
            assert isinstance(emotion.score, float)

    def test_raw_scores_has_28_labels(self, detector: EmotionDetector):
        """The GoEmotions model has 28 labels — all should appear in raw_scores."""
        result = detector.detect("I feel happy today")
        assert len(result.raw_scores) == 28

    def test_scores_between_0_and_1(self, detector: EmotionDetector):
        """All scores should be valid probabilities (0 to 1)."""
        result = detector.detect("I feel happy today")
        for emotion in result.emotions:
            assert 0.0 <= emotion.score <= 1.0
        for score in result.raw_scores.values():
            assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────
# Test 2: Sorting & ranking
# ──────────────────────────────────────────────────────────────

class TestEmotionDetectorRanking:
    """Verify that emotions are sorted correctly."""

    def test_emotions_sorted_descending(self, detector: EmotionDetector):
        """The top-k emotions should be sorted highest score first."""
        result = detector.detect("I am so angry right now")
        scores = [e.score for e in result.emotions]
        assert scores == sorted(scores, reverse=True)

    def test_primary_emotion_is_top_scored(self, detector: EmotionDetector):
        """primary_emotion should match the first (highest) emotion label."""
        result = detector.detect("I am so angry right now")
        assert result.primary_emotion == result.emotions[0].label


# ──────────────────────────────────────────────────────────────
# Test 3: Emotion detection accuracy (sanity checks)
# ──────────────────────────────────────────────────────────────

class TestEmotionDetectorAccuracy:
    """
    Sanity checks — the model should detect obvious emotions.
    These are NOT strict assertions (ML models can be unpredictable),
    so we check if the expected emotion appears in the top 3.
    """

    def test_detects_anger(self, detector: EmotionDetector):
        result = detector.detect("I am so angry right now")
        top_labels = [e.label for e in result.emotions]
        assert "anger" in top_labels, f"Expected 'anger' in top 3, got {top_labels}"

    def test_detects_joy(self, detector: EmotionDetector):
        result = detector.detect("This is the best day of my life, I'm so happy!")
        top_labels = [e.label for e in result.emotions]
        assert "joy" in top_labels, f"Expected 'joy' in top 3, got {top_labels}"

    def test_detects_sadness(self, detector: EmotionDetector):
        result = detector.detect("I lost my best friend and I can't stop crying")
        top_labels = [e.label for e in result.emotions]
        assert "sadness" in top_labels, f"Expected 'sadness' in top 3, got {top_labels}"

    def test_detects_fear(self, detector: EmotionDetector):
        result = detector.detect("I'm scared but relieved it's over")
        top_labels = [e.label for e in result.emotions]
        assert "fear" in top_labels or "relief" in top_labels, (
            f"Expected 'fear' or 'relief' in top 3, got {top_labels}"
        )


# ──────────────────────────────────────────────────────────────
# Test 4: Custom top_k
# ──────────────────────────────────────────────────────────────

class TestCustomTopK:
    """Verify that top_k parameter works correctly."""

    def test_top_k_1(self):
        det = EmotionDetector(top_k=1)
        result = det.detect("I feel happy")
        assert len(result.emotions) == 1

    def test_top_k_5(self):
        det = EmotionDetector(top_k=5)
        result = det.detect("I feel happy")
        assert len(result.emotions) == 5


# ──────────────────────────────────────────────────────────────
# Test 5: EmotionService
# ──────────────────────────────────────────────────────────────

class TestEmotionService:
    """Test the service layer wrapper."""

    def test_analyze_returns_emotion_result(self, service: EmotionService):
        result = service.analyze("I feel anxious")
        assert isinstance(result, EmotionResult)

    def test_get_primary_emotion_returns_string(self, service: EmotionService):
        label = service.get_primary_emotion("I feel anxious")
        assert isinstance(label, str)
        assert len(label) > 0

    def test_get_emotion_summary_returns_dict(self, service: EmotionService):
        summary = service.get_emotion_summary("I feel anxious")
        assert isinstance(summary, dict)
        assert len(summary) == 3  # top_k = 3
        for label, score in summary.items():
            assert isinstance(label, str)
            assert isinstance(score, float)


# ──────────────────────────────────────────────────────────────
# Test 6: Edge cases
# ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Handle unusual inputs gracefully."""

    def test_whitespace_input(self, detector: EmotionDetector):
        """Input with leading/trailing whitespace should be preprocessed."""
        result = detector.detect("   I feel happy   ")
        assert result.text == "I feel happy"  # preprocess() strips whitespace

    def test_text_preserved_in_result(self, detector: EmotionDetector):
        """The original (preprocessed) text should appear in the result."""
        query = "I feel nothing, just numb"
        result = detector.detect(query)
        assert result.text == query
