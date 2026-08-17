from transformers import pipeline

# Load directly — no training needed
nlp = pipeline(
    "token-classification",
    model="tanfiona/unicausal-tok-baseline",
    aggregation_strategy="simple"  # merges B/I tokens into full spans
)
sentence = "I'm not lazy, plus, were do you come from, reading tests differ from countries? Also, I know myself, that I am a good reader. And, all and all, are saying that I'm not dyslexic, just too lazy? "
result = nlp(sentence)
print(result)
