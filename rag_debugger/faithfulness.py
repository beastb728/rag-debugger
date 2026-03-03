import re


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.strip()) > 0]


def compute_faithfulness(answer, retrieved_chunks):
    retrieved_text = " ".join(retrieved_chunks).lower()
    sentences = split_into_sentences(answer)

    grounded = 0
    total = len(sentences)

    for sentence in sentences:
        sentence_clean = sentence.lower()
        if any(word in retrieved_text for word in sentence_clean.split()):
            grounded += 1

    score = grounded / total if total > 0 else 0.0

    return {
        "grounded_sentences": grounded,
        "total_sentences": total,
        "faithfulness_score": round(score, 3),
    }