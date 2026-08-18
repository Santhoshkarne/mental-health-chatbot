from typing import Any

from transformers import pipeline
from core.base_detector import BaseDetector
from schemas.emotion_schema import EmotionScore, EmotionResult


class EmotionDetector(BaseDetector):
    """
    Wraps SamLowe/roberta-base-go_emotions.
    Takes a user query and returns a structured EmotionResult.
    """

    def __init__(self, top_k: int = 3):
        self.model_name = "SamLowe/roberta-base-go_emotions"
        self.top_k = top_k
        self.pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None  # return all 28 labels, not just the top one
        )

    def detect(self, text: str) -> EmotionResult:
        text = self.preprocess(text)
        raw_scores: list[dict[str, Any]] = self.pipeline(text)[0]  # type: ignore[assignment]

        # Sort all 28 emotions by score (highest first) and pick top_k
        sorted_scores = sorted(raw_scores, key=lambda r: r["score"], reverse=True)
        top_emotions = sorted_scores[:self.top_k]

        return EmotionResult(
            text=text,
            primary_emotion=top_emotions[0]["label"] if top_emotions else "neutral",
            emotions=[EmotionScore(label=r["label"], score=r["score"]) for r in top_emotions],
            raw_scores={r["label"]: r["score"] for r in raw_scores}
        )


if __name__ == "__main__":
    detector = EmotionDetector()
    query = "I feel really anxious about my exam tomorrow but also kind of hopeful."
    result = detector.detect(query)
    print(result.model_dump_json(indent=2))