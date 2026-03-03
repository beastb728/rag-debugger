from embeddings import EmbeddingModel
from similarity import compute_query_chunk_similarity, compute_chunk_redundancy
from coverage import extract_query_terms, compute_coverage
from hallucination import detect_unsupported_entities, detect_unsupported_numbers
from faithfulness import compute_faithfulness
from report import generate_report


def run_rag_debugger(query, retrieved_chunks, answer):
    embedder = EmbeddingModel()

    query_embedding = embedder.encode(query)[0]
    chunk_embeddings = embedder.encode(retrieved_chunks)

    similarities = compute_query_chunk_similarity(query_embedding, chunk_embeddings)
    redundancy = compute_chunk_redundancy(chunk_embeddings)

    query_terms = extract_query_terms(query)
    coverage_data = compute_coverage(query_terms, retrieved_chunks, answer)

    unsupported_entities = detect_unsupported_entities(answer, retrieved_chunks)
    unsupported_numbers = detect_unsupported_numbers(answer, retrieved_chunks)

    faithfulness_data = compute_faithfulness(answer, retrieved_chunks)

    report = generate_report(
        similarities,
        redundancy,
        coverage_data,
        unsupported_entities,
        unsupported_numbers,
        faithfulness_data
    )

    return report


if __name__ == "__main__":
    query = "What is the normal LDL cholesterol range?"
    retrieved_chunks = [
        "LDL cholesterol is considered optimal below 100 mg/dL.",
        "High LDL levels increase cardiovascular risk."
    ]
    answer = "The normal LDL cholesterol range is 100 mg/dL. It should not exceed 130."

    print(run_rag_debugger(query, retrieved_chunks, answer))