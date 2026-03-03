def compute_overall_score(
    similarity_scores,
    coverage_data,
    unsupported_entities,
    unsupported_numbers,
    faithfulness_data,
):
    # Retrieval score
    retrieval_score = sum(similarity_scores) / len(similarity_scores)

    # Coverage score
    total_terms = len(coverage_data["query_terms"])
    missing = len(coverage_data["missing_in_retrieval"])
    coverage_score = 1 - (missing / total_terms) if total_terms > 0 else 0

    # Hallucination penalty
    hallucination_count = len(unsupported_entities) + len(unsupported_numbers)
    hallucination_penalty = 1 - min(1, 0.1 * hallucination_count)

    # Faithfulness score
    faithfulness_score = faithfulness_data["faithfulness_score"]

    overall = (
        0.35 * retrieval_score
        + 0.25 * coverage_score
        + 0.25 * faithfulness_score
        + 0.15 * hallucination_penalty
    )

    overall = round(overall, 3)

    # Quality label
    if overall >= 0.8:
        label = "Excellent"
    elif overall >= 0.65:
        label = "Good"
    elif overall >= 0.5:
        label = "Moderate"
    else:
        label = "Poor"

    return overall, label