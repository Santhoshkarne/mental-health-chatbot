import spacy
nlp = spacy.load("en_core_web_trf")  

def extract_intent(text):
    doc = nlp(text)
    for token in doc:
        if token.dep_ == "dobj":
            action = token.head.lemma_        # verb governing the object
            obj = token.text                  # or expand to full noun phrase via token.subtree
            return action, obj
    return None, None

# print(extract_intent("i want to book a flight for hyderabad to chandigarh"))