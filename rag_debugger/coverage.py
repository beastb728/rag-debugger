import re
from collections import Counter


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
    return tokens


def extract_query_terms(query, min_length=4):
    tokens = tokenize(query)
    return set([t for t in tokens if len(t) >= min_length])


def compute_coverage(query_terms, retrieved_chunks, answer):
    retrieved_text = " ".join(retrieved_chunks).lower()
    answer_text = answer.lower()

    covered_in_retrieval = {t for t in query_terms if t in retrieved_text}
    covered_in_answer = {t for t in query_terms if t in answer_text}

    missing_in_retrieval = query_terms - covered_in_retrieval
    missing_in_answer = query_terms - covered_in_answer

    return {
        "query_terms": query_terms,
        "covered_in_retrieval": covered_in_retrieval,
        "covered_in_answer": covered_in_answer,
        "missing_in_retrieval": missing_in_retrieval,
        "missing_in_answer": missing_in_answer,
    }