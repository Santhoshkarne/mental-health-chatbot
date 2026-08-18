import warnings
import os
import sys
from metrics.intent import extract_intent
from metrics.severity import detect_severity
# Suppress some noisy warnings from transformers if any
warnings.filterwarnings("ignore")

# Load models once so they can be reused
print("Loading models... This may take a moment.")

# Emotion
try:
    from emotion.service import EmotionService
    emotion_service = EmotionService()
except ImportError:
    print("Warning: Could not import EmotionService. Make sure you run this script from the 'mental-health-chatbot' directory.")
    emotion_service = None

# Cause
cause_bosch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cause_bosch")
if cause_bosch_path not in sys.path:
    sys.path.append(cause_bosch_path)

try:
    from cause_bosch.extract import CauseEffectExtractor
    cause_extractor = CauseEffectExtractor(
        model_dir=".",
        base_model_name="roberta-large",
    )
except ImportError as e:
    print(f"Warning: Could not import CauseEffectExtractor: {e}")
    cause_extractor = None

def analyze_query(query: str):
    print(f"\nAnalyzing query: '{query}'\n" + "-"*50)
    
    # 1. Emotion
    if emotion_service:
        primary_emotion = emotion_service.get_primary_emotion(query)
        print(f"Emotion : {primary_emotion}")
    else:
        primary_emotion = None
        print("Emotion : Not available")
    
    # 2. Cause
    causes = []
    effects = []
    signals = []
    if cause_extractor:
        try:
            cause_results = cause_extractor.predict([query])[0]
            causes = [str(rel['cause']) for rel in cause_results if rel.get('cause')]
            effects = [str(rel['effect']) for rel in cause_results if rel.get('effect')]
            signals = [str(rel['signal']) for rel in cause_results if rel.get('signal')]
        except Exception as e:
            print(f"Error extracting cause/effect/signal: {e}")
            
    print(f"Cause   : {', '.join(causes) if causes else 'None detected'}")
    print(f"Effect  : {', '.join(effects) if effects else 'None detected'}")
    print(f"Signal  : {', '.join(signals) if signals else 'None detected'}")
    
    # 3. Intent
    action, obj = extract_intent(query)
    if action and obj:
        print(f"Intent  : Action='{action}', Object='{obj}'")
    else:
        print("Intent  : None detected")
    
    # 4. Severity
    predicted_severity, severity_score = detect_severity(query)
    print(f"Severity: {predicted_severity} (Confidence: {severity_score:.4f})")
    print("-" * 50)
    
    return {
        "emotion": primary_emotion,
        "cause": causes,
        "effect": effects,
        "signal": signals,
        "intent": {"action": action, "object": obj},
        "severity": predicted_severity
    }

if __name__ == "__main__":
    print("\nPipeline Ready!")
    while True:
        try:
            user_input = input("\nEnter a query (or type 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input.strip():
                analyze_query(user_input)
        except (KeyboardInterrupt, EOFError):
            break