import re
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import config

def get_embeddings():
    """Load local CPU-friendly HuggingFace embeddings."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Use cosine similarity metrics
    )

def load_cross_encoder():
    """Load CPU-friendly cross-encoder model for reranking."""
    return CrossEncoder(config.RERANKER_MODEL_NAME, device='cpu')

def build_indices(chunks: list[dict]):
    """Build and save FAISS dense index and BM25 sparse index."""
    # 1. Build FAISS index
    embeddings = get_embeddings()
    documents = []
    for idx, c in enumerate(chunks):
        # Inject chunk_id into metadata to uniquely track chunks during fusion
        metadata = c["metadata"].copy()
        metadata["chunk_id"] = idx
        doc = Document(page_content=c["text"], metadata=metadata)
        documents.append(doc)
        
    faiss_index = FAISS.from_documents(documents, embeddings)
    faiss_index.save_local(str(config.FAISS_INDEX_PATH))
    
    # 2. Build BM25 index
    tokenized_corpus = []
    for c in chunks:
        # Lowercase alphanumeric tokenization
        tokens = re.findall(r'\w+', c["text"].lower())
        tokenized_corpus.append(tokens)
        
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save BM25 index alongside chunk text data
    with open(config.BM25_INDEX_PATH, 'wb') as f:
        pickle.dump({"chunks": chunks, "bm25": bm25}, f)
        
    print(f"Successfully generated indices for {len(chunks)} chunks.")
    return faiss_index, bm25

def load_indices():
    """Load existing dense and sparse indexes from disk."""
    if not config.FAISS_INDEX_PATH.exists() or not config.BM25_INDEX_PATH.exists():
        raise FileNotFoundError("Indices do not exist. Please run ingestion first.")
        
    embeddings = get_embeddings()
    faiss_index = FAISS.load_local(
        str(config.FAISS_INDEX_PATH), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    with open(config.BM25_INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
        chunks = data["chunks"]
        bm25 = data["bm25"]
        
    return faiss_index, bm25, chunks

def hybrid_retrieve(query: str, all_chunks: list[dict], faiss_index, bm25, 
                    k_dense: int, k_sparse: int, rrf_k: int, top_n: int) -> list[tuple[dict, float]]:
    """Combine FAISS and BM25 results using Reciprocal Rank Fusion (RRF)."""
    # 1. Dense search
    dense_hits = faiss_index.similarity_search_with_score(query, k=k_dense)
    
    # 2. Sparse search
    query_tokens = re.findall(r'\w+', query.lower())
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Sort sparse document scores in descending order and select top candidates
    sparse_indices = np.argsort(bm25_scores)[::-1][:k_sparse]
    
    # 3. Apply Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    
    # Process dense ranks (1-indexed)
    for rank, (doc, _) in enumerate(dense_hits):
        chunk_id = doc.metadata["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + (rank + 1))
        
    # Process sparse ranks (1-indexed)
    for rank, idx in enumerate(sparse_indices):
        # Ignore items with 0 keyword matches
        if bm25_scores[idx] <= 0:
            continue
        chunk_id = idx  # The index in the BM25 corpus corresponds directly to all_chunks index
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + (rank + 1))
        
    # Sort candidates by combined RRF score descending
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top_n candidates
    retrieved_chunks = []
    for chunk_id, score in sorted_chunks[:top_n]:
        # Return full chunk dictionary and its fusion score
        retrieved_chunks.append((all_chunks[chunk_id], score))
        
    return retrieved_chunks

def retrieve(query: str, all_chunks: list[dict], faiss_index, bm25, cross_encoder=None, 
             mode: str = "hybrid_rerank") -> list[dict]:
    """Unified routing function for dense, hybrid, or hybrid+rerank retrieval."""
    if mode == "dense":
        # Dense search only
        hits = faiss_index.similarity_search_with_score(query, k=config.RERANK_TOP_K)
        results = []
        for doc, score in hits:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)  # FAISS L2 distance
            })
        return results
        
    elif mode == "hybrid":
        # Hybrid RRF only
        hits = hybrid_retrieve(query, all_chunks, faiss_index, bm25, 
                              k_dense=config.TOP_K_DENSE, 
                              k_sparse=config.TOP_K_SPARSE, 
                              rrf_k=config.RRF_K, 
                              top_n=config.RERANK_TOP_K)
        results = []
        for chunk, score in hits:
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(score)  # RRF score
            })
        return results
        
    elif mode == "hybrid_rerank":
        # Hybrid RRF + Cross-Encoder Rerank
        if cross_encoder is None:
            cross_encoder = load_cross_encoder()
            
        # Retrieve more candidates (e.g. top 10) to feed into the reranker
        rrf_hits = hybrid_retrieve(query, all_chunks, faiss_index, bm25, 
                                  k_dense=config.TOP_K_DENSE, 
                                  k_sparse=config.TOP_K_SPARSE, 
                                  rrf_k=config.RRF_K, 
                                  top_n=config.TOP_K_DENSE)
        
        candidates = [chunk for chunk, _ in rrf_hits]
        if not candidates:
            return []
            
        # Compute relevance scores using Cross-Encoder model
        pairs = [(query, c["text"]) for c in candidates]
        scores = cross_encoder.predict(pairs)
        
        # Sort candidates by relevance score descending
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk, score in ranked[:config.RERANK_TOP_K]:
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(score)  # Cross-Encoder score
            })
        return results
    else:
        raise ValueError(f"Invalid retrieval mode: {mode}")
