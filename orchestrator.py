"""
Unified Orchestrator
====================

The master pipeline that runs BOTH branches in parallel for a user query:

  Branch A — Dialog Analysis (Emotion, Cause/Effect, Intent, Severity)
  Branch B — Knowledge Graph Retrieval (Top 5 paragraphs via MMR)

After both branches complete, they are merged into a single combined
context dictionary that is ready to be sent to the next phase
(e.g., LLM response generation).

Usage:
    python orchestrator.py
"""

import warnings
import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Global Model Initialization (Load once to avoid slow reloads)
# ──────────────────────────────────────────────────────────────
try:
    from emotion.service import EmotionService
    emotion_service = EmotionService()
except Exception as e:
    print(f"Warning: Could not load EmotionService: {e}")
    emotion_service = None

try:
    from transformers import pipeline as hf_pipeline
    cause_nlp = hf_pipeline(
        "token-classification",
        model="tanfiona/unicausal-tok-baseline",
        aggregation_strategy="simple",
    )
except Exception as e:
    print(f"Warning: Could not load cause extraction model: {e}")
    cause_nlp = None

try:
    from transformers import pipeline as hf_pipeline
    intent_classifier = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )
except Exception as e:
    print(f"Warning: Could not load intent extraction model: {e}")
    intent_classifier = None

try:
    from knowledge.retrieve import HybridRetriever
    knowledge_retriever = HybridRetriever()
except Exception as e:
    print(f"Warning: Could not load Knowledge Retriever: {e}")
    knowledge_retriever = None



# ──────────────────────────────────────────────────────────────
# Branch A: Dialog Analysis
# ──────────────────────────────────────────────────────────────

def run_dialog_analysis(query: str) -> dict:
    """
    Extracts all dialog-level features from the user query:
        - Emotion (primary + top-3)
        - Cause / Effect / Signal
        - Intent (action + object)
        - Severity (Low / Medium / High)

    Returns a flat dictionary with all extracted parts.
    """
    result = {
        "emotion": "",
        "emotion_scores": {},
        "cause": [],
        "effect": [],
        "signal": [],
        "intent": {"action": "", "object": ""},
        "severity": "",
        "severity_score": 0.0,
    }

    # ── 1. Emotion Detection ──
    try:
        if emotion_service:
            emotion_result = emotion_service.analyze(query)
            result["emotion"] = emotion_result.primary_emotion
            result["emotion_scores"] = {e.label: e.score for e in emotion_result.emotions}
    except Exception as e:
        print(f"  [Dialog] Emotion detection failed: {e}")

    # ── 2. Cause / Effect / Signal (using unicausal model) ──
    try:
        if cause_nlp:
            cause_tokens = cause_nlp(query)
            causes = []
            effects = []
            signals = []
            for token in cause_tokens:
                label = token["entity_group"].upper()
                word = token["word"].strip()
                if not word:
                    continue
                if "CAUSE" in label:
                    causes.append(word)
                elif "EFFECT" in label:
                    effects.append(word)
                elif "SIGNAL" in label:
                    signals.append(word)

            result["cause"] = causes
            result["effect"] = effects
            result["signal"] = signals
    except Exception as e:
        print(f"  [Dialog] Cause extraction failed: {e}")

    # ── 3. Intent Extraction (zero-shot classification) ──
    try:
        if intent_classifier:
            intent_labels = [
                "seeking information",
                "expressing distress",
                "asking for help",
                "sharing experience",
                "seeking diagnosis",
                "looking for treatment options",
            ]
            intent_result = intent_classifier(query, intent_labels, multi_label=False)
            result["intent"] = {
                "action": intent_result["labels"][0],
                "confidence": round(intent_result["scores"][0], 4),
            }
    except Exception as e:
        print(f"  [Dialog] Intent extraction failed: {e}")

    # ── 4. Severity Detection ──
    try:
        from metrics.severity import detect_severity
        severity_label, severity_score = detect_severity(query)
        result["severity"] = severity_label
        result["severity_score"] = severity_score
    except Exception as e:
        print(f"  [Dialog] Severity detection failed: {e}")

    return result


# ──────────────────────────────────────────────────────────────
# Branch B: Knowledge Graph Retrieval
# ──────────────────────────────────────────────────────────────

def run_knowledge_retrieval(query: str) -> list[dict]:
    """
    Retrieves the Top 5 most relevant paragraphs from the
    book knowledge graph using SBERT + Graph Expansion + MMR.

    Returns a list of dicts, each with: text, source, score, relevance.
    """
    try:
        if knowledge_retriever:
            results = knowledge_retriever.retrieve_top_k(query, k=5, similarity_pool=10)
            # Simplify the output for the combined context
            return [
                {
                    "text": r["text"],
                    "source": r["source"],
                    "relevance": round(r["relevance"], 4),
                    "mmr_score": round(r["score"], 4),
                }
                for r in results
            ]
        return []
    except Exception as e:
        print(f"  [Knowledge] Retrieval failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────
# Merge: Combine Both Branches into a Single Context
# ──────────────────────────────────────────────────────────────

def merge_contexts(
    query: str,
    dialog: dict,
    knowledge: list[dict],
) -> dict:
    """
    Combines Dialog Analysis and Knowledge Retrieval into a single
    unified context dictionary ready for the next phase (e.g., LLM).

    Structure:
    {
        "query": "...",
        "dialog_analysis": { emotion, cause, effect, severity, ... },
        "knowledge_contexts": [ { text, source, relevance }, ... ],
        "combined_prompt": "A formatted text block ready for an LLM"
    }
    """
    # Build a human-readable combined prompt that an LLM can consume
    prompt_parts = []

    prompt_parts.append(f"User Query: {query}")
    prompt_parts.append("")

    # Dialog Analysis Section
    prompt_parts.append("── Dialog Analysis ──")
    if dialog.get("emotion"):
        prompt_parts.append(f"Detected Emotion: {dialog['emotion']}")
    if dialog.get("emotion_scores"):
        scores_str = ", ".join(f"{k}: {v:.3f}" for k, v in dialog["emotion_scores"].items())
        prompt_parts.append(f"Emotion Scores: {scores_str}")
    if dialog.get("cause"):
        prompt_parts.append(f"Cause(s): {', '.join(dialog['cause'])}")
    if dialog.get("effect"):
        prompt_parts.append(f"Effect(s): {', '.join(dialog['effect'])}")
    if dialog.get("signal"):
        prompt_parts.append(f"Signal(s): {', '.join(dialog['signal'])}")
    if dialog.get("intent", {}).get("action"):
        prompt_parts.append(f"Intent: Action='{dialog['intent']['action']}', Object='{dialog['intent']['object']}'")
    if dialog.get("severity"):
        prompt_parts.append(f"Severity: {dialog['severity']} (confidence: {dialog.get('severity_score', 0):.3f})")

    prompt_parts.append("")

    # Knowledge Contexts Section
    prompt_parts.append("── Retrieved Knowledge (Top 5 Book Paragraphs) ──")
    if knowledge:
        for i, ctx in enumerate(knowledge, 1):
            prompt_parts.append(f"[{i}] Source: {ctx['source']} | Relevance: {ctx['relevance']}")
            prompt_parts.append(f"    {ctx['text'][:500]}")
            prompt_parts.append("")
    else:
        prompt_parts.append("No relevant knowledge found.")

    combined_prompt = "\n".join(prompt_parts)

    return {
        "query": query,
        "dialog_analysis": dialog,
        "knowledge_contexts": knowledge,
        "combined_prompt": combined_prompt,
    }


# ──────────────────────────────────────────────────────────────
# Main Orchestrator
# ──────────────────────────────────────────────────────────────

def orchestrate(query: str) -> dict:
    """
    The master function. Runs Branch A and Branch B in PARALLEL,
    then merges their outputs into one combined context.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Processing: '{query}'")
    print(f"{'='*60}")

    start = time.time()

    # Run both branches in parallel using threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dialog = executor.submit(run_dialog_analysis, query)
        future_knowledge = executor.submit(run_knowledge_retrieval, query)

        dialog_result = future_dialog.result()
        knowledge_result = future_knowledge.result()

    elapsed = time.time() - start

    # Merge both branches
    combined = merge_contexts(query, dialog_result, knowledge_result)

    print(f"\n⏱️  Total processing time: {elapsed:.2f}s")

    return combined


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Unified Orchestrator Starting...")
    print("   Branch A: Dialog Analysis (Emotion, Cause, Intent, Severity)")
    print("   Branch B: Knowledge Graph Retrieval (Top 5 via MMR)")
    print("=" * 60)

    while True:
        try:
            query = input("\n👤 You: ")
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not query.strip():
                continue

            result = orchestrate(query)

            # Display the combined context
            print("\n" + "─" * 60)
            print("📋 COMBINED CONTEXT FOR NEXT PHASE:")
            print("─" * 60)
            print(result["combined_prompt"])
            print("─" * 60)

            # Save to file for next phase consumption
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "latest_context.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved combined context to: {output_path}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
