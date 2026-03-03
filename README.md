# RAG Debugger 🔍

A modular evaluation framework for **Retrieval-Augmented Generation (RAG)** systems. 

RAG Debugger provides a plug-and-play diagnostic layer to quantitatively evaluate retrieval quality, context coverage, hallucination risk, and grounding—at both the individual query and full-dataset benchmarking levels.

---

## 💡 Why RAG Debugger?

Most RAG pipelines optimize for generation "vibes" but lack rigorous metrics. This tool exists to objectively measure:

* **Retrieval Accuracy:** Are the retrieved documents actually relevant?
* **Grounding:** Is the answer strictly derived from the retrieved context?
* **Hallucination Risk:** Are there entities or numbers in the answer not found in the source?
* **Intent Coverage:** Does the response fully address the user's original query?

---

## ✨ Core Features

* **Retrieval Evaluation:** Embedding-based similarity scoring and redundant chunk detection.
* **Faithfulness Analysis:** Entity and numeric hallucination detection + semantic faithfulness scoring.
* **System-Level Metrics:** Generates a composite **RAG Health Score** for your entire pipeline.
* **Developer First:** Clean CLI interface, installable Python package, and JSON export support.

---

## ⚙️ Installation

We recommend using a virtual environment to ensure isolation and reproducibility:

```bash
# Clone the repository
git clone [https://github.com/beastb728/rag-debugger.git](https://github.com/beastb728/rag-debugger.git)
cd rag-debugger

# Setup environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .

```

## 🛠 CLI Usage
RAG Debugger is designed to be immediately testable via the command line.

**Evaluate a Dataset**
```bash
rag-debugger evaluate dataset.json
```

**Save Results to File**
```bash
rag-debugger evaluate dataset.json --output results.json
```

## 📄 Dataset Format
To run an evaluation, your input must be a JSON list of evaluation samples following this structure:
```bash
[
  {
    "query": "What is the capital of France?",
    "retrieved_chunks": [
      "Paris is the capital and largest city of France.",
      "France is located in Western Europe."
    ],
    "answer": "The capital of France is Paris."
  }
]
```