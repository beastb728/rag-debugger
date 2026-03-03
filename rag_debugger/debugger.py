from .embeddings import embed_texts
from .scoring import compute_overall_score
from .similarity import compute_query_chunk_similarity, compute_chunk_redundancy
from .coverage import extract_query_terms, compute_coverage
from .hallucination import (
    detect_unsupported_entities,
    detect_unsupported_numbers,
)
from .faithfulness import compute_faithfulness
from .report import generate_report


class RagDebugger:
    """
    Plug-and-play evaluation layer for RAG systems.
    """

    def evaluate(self, query: str, retrieved_chunks: list[str], answer: str) -> dict:

        # Embeddings
        query_embedding = embed_texts([query])[0]
        chunk_embeddings = embed_texts(retrieved_chunks)

        similarity_scores = compute_query_chunk_similarity(
            query_embedding, chunk_embeddings
        )

        redundancy = compute_chunk_redundancy(chunk_embeddings)

        # Coverage
        query_terms = extract_query_terms(query)
        coverage_data = compute_coverage(query_terms, retrieved_chunks, answer)

        # Hallucination
        unsupported_entities = detect_unsupported_entities(answer, retrieved_chunks)
        unsupported_numbers = detect_unsupported_numbers(answer, retrieved_chunks)

        hallucination_data = {
            "unsupported_entities": unsupported_entities,
            "unsupported_numbers": unsupported_numbers,
        }

        # Faithfulness
        faithfulness_data = compute_faithfulness(answer, retrieved_chunks)

        # 🆕 Overall Score (v1.1.0)
        overall_score, quality_label = compute_overall_score(
            similarity_scores.tolist(),
            coverage_data,
            unsupported_entities,
            unsupported_numbers,
            faithfulness_data,
        )

        report = {
            "retrieval_similarity": similarity_scores.tolist(),
            "redundant_chunks": redundancy,
            "coverage": coverage_data,
            "hallucination": hallucination_data,
            "faithfulness": faithfulness_data,
            "overall_score": overall_score,
            "quality_label": quality_label,
        }

        return report

    def pretty_print(self, report):
        print(generate_report(
            report["retrieval_similarity"],
            report["redundant_chunks"],
            report["coverage"],
            report["hallucination"]["unsupported_entities"],
            report["hallucination"]["unsupported_numbers"],
            report["faithfulness"],
            report["overall_score"],
            report["quality_label"],
        ))