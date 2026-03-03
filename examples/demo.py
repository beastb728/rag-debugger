from rag_debugger import RagDebugger, __version__


if __name__ == "__main__":
    print(f"RAG Debugger Version: {__version__}\n")

    debugger = RagDebugger()

    query = "What is the normal LDL cholesterol range?"

    retrieved_chunks = [
        "LDL cholesterol is considered optimal below 100 mg/dL.",
        "High LDL levels increase cardiovascular risk."
    ]

    answer = "The normal LDL cholesterol range is 100 mg/dL. It should not exceed 130."

    # ---- Single Example Evaluation ----
    print("=== Single Example Evaluation ===\n")
    report = debugger.evaluate(query, retrieved_chunks, answer)
    debugger.pretty_print(report)

    # ---- Dataset Evaluation (v1.2.0) ----
    print("\n=== Dataset Evaluation ===\n")

    dataset = [
        {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
        },
        {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
        }
    ]

    dataset_report = debugger.evaluate_dataset(dataset)

    print("Dataset Summary:")
    print(dataset_report["summary"])