import json

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

        # Overall Score
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

    def evaluate_dataset(
        self,
        dataset: list[dict],
        export_path: str | None = None,
    ) -> dict:
        """
        Evaluate a dataset of RAG examples.

        Each item must contain:
        {
            "query": str,
            "retrieved_chunks": list[str],
            "answer": str
        }

        If export_path is provided, results will be written to JSON.
        """

        results = []
        overall_scores = []
        faithfulness_scores = []
        coverage_scores = []
        hallucination_counts = []

        for item in dataset:
            report = self.evaluate(
                item["query"],
                item["retrieved_chunks"],
                item["answer"],
            )

            results.append(report)

            overall_scores.append(report["overall_score"])
            faithfulness_scores.append(
                report["faithfulness"]["faithfulness_score"]
            )

            # Coverage rate
            total_terms = len(report["coverage"]["query_terms"])
            missing = len(report["coverage"]["missing_in_retrieval"])
            coverage_rate = (
                1 - (missing / total_terms) if total_terms > 0 else 0
            )
            coverage_scores.append(coverage_rate)

            # Hallucination count
            hallucination_count = (
                len(report["hallucination"]["unsupported_entities"])
                + len(report["hallucination"]["unsupported_numbers"])
            )
            hallucination_counts.append(hallucination_count)

        n = len(dataset)

        summary = {
            "num_samples": n,
            "average_overall_score": round(sum(overall_scores) / n, 3),
            "average_faithfulness": round(sum(faithfulness_scores) / n, 3),
            "average_coverage": round(sum(coverage_scores) / n, 3),
            "average_hallucinations_per_sample": round(
                sum(hallucination_counts) / n, 3
            ),
        }

        final_output = {
            "summary": summary,
            "individual_results": results,
        }

        # 🔥 JSON Export (v1.3.0 feature)
        if export_path:
            with open(export_path, "w") as f:
                json.dump(final_output, f, indent=4)

        return final_output

    def pretty_print(self, report):
        print(
            generate_report(
                report["retrieval_similarity"],
                report["redundant_chunks"],
                report["coverage"],
                report["hallucination"]["unsupported_entities"],
                report["hallucination"]["unsupported_numbers"],
                report["faithfulness"],
                report["overall_score"],
                report["quality_label"],
            )
        )