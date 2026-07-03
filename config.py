import os
from pathlib import Path

# Project Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indices"

# Create directories if they do not exist
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Ingestion Configuration
CHUNK_SIZE = 500  # Target characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks
MIN_CHUNK_LEN = 50  # Ignore extremely small chunks (e.g. noise)

# Model Configurations
# BGE-small-en-v1.5 is a highly efficient, CPU-friendly local dense embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Reranker model
# Lightweight Cross-Encoder to re-score matches on CPU
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM for QA and evaluation
# Llama 3.1 8B is hosted on Groq, which is fast and has a generous free tier
LLM_MODEL_NAME = "llama-3.1-8b-instant"

# Retrieval Hyperparameters
TOP_K_DENSE = 10      # Candidates to fetch from FAISS
TOP_K_SPARSE = 10     # Candidates to fetch from BM25
RRF_K = 60            # Constant for Reciprocal Rank Fusion
RERANK_TOP_K = 5      # Final number of context chunks sent to LLM

# Saved Indices Names
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index"
BM25_INDEX_PATH = INDEX_DIR / "bm25_corpus.pkl"
PROCESSED_FILES_REGISTRY = INDEX_DIR / "ingested_files.json"
EVAL_RESULTS_PATH = BASE_DIR / "eval_results.md"
