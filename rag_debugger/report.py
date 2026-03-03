def generate_report(similarities,
                    redundancy,
                    coverage_data,
                    unsupported_entities,
                    unsupported_numbers,
                    faithfulness_data,
                    overall_score,
                    quality_label):

    report = []
    report.append("=== RAG DEBUG REPORT ===\n")

    # Retrieval
    report.append("Retrieval Similarity:")
    for idx, score in enumerate(similarities):
        report.append(f"  Chunk {idx+1}: {round(score,3)}")

    if redundancy:
        report.append("Redundant Chunks Detected:")
        for i, j, score in redundancy:
            report.append(f"  Chunk {i+1} and Chunk {j+1} (sim={round(score,3)})")
    report.append("")

    # Coverage
    report.append("Coverage Analysis:")
    report.append(f"  Query Terms: {coverage_data['query_terms']}")
    report.append(f"  Missing in Retrieval: {coverage_data['missing_in_retrieval']}")
    report.append(f"  Missing in Answer: {coverage_data['missing_in_answer']}")
    report.append("")

    # Hallucination
    report.append("Unsupported Entities:")
    report.append(f"  {unsupported_entities}")
    report.append("Unsupported Numbers:")
    report.append(f"  {unsupported_numbers}")
    report.append("")

    # Faithfulness
    report.append("Faithfulness:")
    report.append(f"  Grounded Sentences: {faithfulness_data['grounded_sentences']}")
    report.append(f"  Total Sentences: {faithfulness_data['total_sentences']}")
    report.append(f"  Score: {faithfulness_data['faithfulness_score']}")
    report.append("")

    # 🆕 Overall Health (v1.1.0)
    report.append("Overall RAG Health:")
    report.append(f"  Score: {round(overall_score,3)}")
    report.append(f"  Quality: {quality_label}")
    report.append("")

    return "\n".join(report)