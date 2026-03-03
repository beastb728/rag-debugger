from rag_debugger import RagDebugger


if __name__ == "__main__":
    debugger = RagDebugger()

    query = "What is the normal LDL cholesterol range?"

    retrieved_chunks = [
        "LDL cholesterol is considered optimal below 100 mg/dL.",
        "High LDL levels increase cardiovascular risk."
    ]

    answer = "The normal LDL cholesterol range is 100 mg/dL. It should not exceed 130."

    report = debugger.evaluate(query, retrieved_chunks, answer)

    debugger.pretty_print(report)