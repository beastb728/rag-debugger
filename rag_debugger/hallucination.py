import re
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)
    return set([ent.text.lower() for ent in doc.ents])


def extract_numbers(text):
    return set(re.findall(r"\b\d+\.?\d*\b", text))


def detect_unsupported_entities(answer, retrieved_chunks):
    retrieved_text = " ".join(retrieved_chunks).lower()
    answer_entities = extract_entities(answer)

    unsupported = []
    for ent in answer_entities:
        if ent not in retrieved_text:
            unsupported.append(ent)

    return unsupported


def detect_unsupported_numbers(answer, retrieved_chunks):
    retrieved_text = " ".join(retrieved_chunks)
    answer_numbers = extract_numbers(answer)

    unsupported = []
    for num in answer_numbers:
        if num not in retrieved_text:
            unsupported.append(num)

    return unsupported