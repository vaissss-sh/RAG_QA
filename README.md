
# Hybrid RAG Document Q&A System

A production-grade, local-first **Retrieval-Augmented Generation (RAG)** Document Q&A system built from scratch with Python, LangChain, FAISS, and BM25, utilizing Groq (Llama 3.1 8B) for fast inference. 

This project is built to demonstrate modern retrieval architectures that solve key corporate search problems, including hallucination, domain-specific search mismatches, and poor search precision.

---

## 🏗️ System Architecture

The project features a **two-phase architecture**: an offline **Ingestion Pipeline** and an online **Hybrid Retrieval & Generation Pipeline**.

```mermaid
flowchart TD
    subgraph Ingestion Pipeline (Offline)
        A[Documents: PDF, DOCX, TXT] --> B[Section-Aware Parser]
        B --> C[Paragraph Reconstruction]
        C --> D[Sliding Window Chunking with Overlap]
        D --> E[Embeddings Generator: BGE-small-en-v1.5]
        E --> F[(FAISS Dense Index)]
        D --> G[(BM25 Sparse Database)]
    end

    subgraph Q&A Pipeline (Online)
        H[User Query] --> I[Dense Query Vector]
        H --> J[Sparse Text Tokenizer]
        I --> K[FAISS Dense Search]
        J --> L[BM25 Sparse Search]
        K --> M[RRF Fusion Ranking]
        L --> M
        M --> N[Cross-Encoder Reranking]
        N --> O[Top K Relevant Context Chunks]
        O --> P[Grounded LLM Prompt Template]
        P --> Q[Groq API: Llama 3.1 8B]
        Q --> R[Streaming Response with Inline Citations]
    end
```

### Core Architecture Components
1. **Section-Aware Chunking**: Custom parser detects markdown headings, numbered headings, and all-caps titles. Paragraphs are grouped under active headers. Oversized paragraphs are split at sentence boundaries, and chunks are constructed using a character-bound backtracking window to ensure semantic context remains contiguous.
2. **Hybrid Retrieval**: Combines semantic embeddings (`BAAI/bge-small-en-v1.5` on CPU) with sparse token keyword matching (`rank_bm25` using `BM25Okapi`).
3. **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse retrievals using rank reciprocals ($k=60$). This matches exact terminology (numbers, IDs) while retaining semantic queries.
4. **Cross-Encoder Reranker**: Scores query-chunk pairs directly using `ms-marco-MiniLM-L-6-v2` on CPU to compute deep semantic intersections, prioritizing the most critical chunks for the LLM.
5. **Grounded Generation**: Feeds relevant contexts to Groq Llama 3.1, enforcing strict formatting, inline citations (`[filename.pdf, Page X]`), and returning `"not found in the provided documents"` if the answer is not supported.

---

## ⚡ Setup & Installation

### 1. Prerequisites
- Python 3.11
- Groq API Key (Free at [console.groq.com](https://console.groq.com/))

### 2. Environment Setup
Clone this repository and create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Unix/macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create Sample Documents
Generate the benchmark DOCX and PDF files used for evaluation:
```bash
python data/create_sample_docs.py
```

### 5. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📊 Evaluation & Benchmarking (RAGAS)

Evaluation is critical to demonstrating RAG reliability. This system integrates the **RAGAS** framework to evaluate retrieval strategies across **12 structured Q&A benchmarks**:

1. **Faithfulness** (Groundedness): Is the answer strictly derived from the context?
2. **Answer Relevancy**: Does the answer directly address the user's question?
3. **Context Precision**: Are the relevant documents placed first in retrieved context?
4. **Context Recall**: Does the retrieved context contain all the necessary ground-truth facts?

To run the evaluation across all retrieval configurations (Dense-only vs. Hybrid RRF vs. Hybrid + Reranker), click the **Run RAGAS Evaluation** button inside the Streamlit UI, or execute:
```bash
python evaluation.py
```

### Benchmark Results
Results are saved to `eval_results.md` and `indices/eval_results.csv` after evaluation.
Below is the typical comparison of retrieval modes:

| Retrieval Mode | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Only (FAISS)** | *TBD* | *TBD* | *TBD* | *TBD* |
| **Hybrid RRF** | *TBD* | *TBD* | *TBD* | *TBD* |
| **Hybrid + Reranker** | *TBD* | *TBD* | *TBD* | *TBD* |

*(Note: Click "Run RAGAS Evaluation" in the Streamlit Sidebar to generate your local benchmark results).*
