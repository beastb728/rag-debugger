import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import embed_texts


def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.strip()) > 0]


def compute_faithfulness(answer: str, retrieved_chunks: list[str], threshold: float = 0.7):
    """
    Semantic faithfulness scoring using embedding similarity.
    A sentence is grounded if its embedding is sufficiently
    similar to at least one retrieved chunk.
    """

    sentences = split_into_sentences(answer)

    if not sentences or not retrieved_chunks:
        return {
            "grounded_sentences": 0,
            "total_sentences": 0,
            "faithfulness_score": 0.0,
        }

    # Embed sentences and chunks
    sentence_embeddings = embed_texts(sentences)
    chunk_embeddings = embed_texts(retrieved_chunks)

    grounded = 0

    for sentence_embedding in sentence_embeddings:
        sims = cosine_similarity(
            sentence_embedding.reshape(1, -1),
            chunk_embeddings
        )
        max_sim = np.max(sims)

        if max_sim >= threshold:
            grounded += 1

    total = len(sentences)
    score = grounded / total if total > 0 else 0.0

    return {
        "grounded_sentences": grounded,
        "total_sentences": total,
        "faithfulness_score": round(score, 3),
    }