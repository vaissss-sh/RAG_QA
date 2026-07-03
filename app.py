import sys
from unittest.mock import MagicMock

# Mock deprecated/removed VertexAI packages to satisfy Ragas imports
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
sys.modules["langchain_community.embeddings.vertexai"] = MagicMock()

import streamlit as st
import os
import json
import pandas as pd
import shutil
from pathlib import Path
from dotenv import load_dotenv

import config
import ingestion
import retrieval
import generation
import evaluation

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Ask Your Docs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling (Indigo-accented dark theme)
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Pill badge styling for metadata and metrics */
    .badge {
        background-color: #1e1b4b;
        color: #c7d2fe;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 8px;
        border: 1px solid #4f46e5;
        display: inline-block;
        margin-bottom: 6px;
    }
    
    /* Citation block card styling */
    .source-box {
        border-left: 4px solid #6366f1;
        background-color: #111827;
        padding: 12px 16px;
        border-radius: 0px 8px 8px 0px;
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 12px;
        color: #e5e7eb;
        border-top: 1px solid #1f2937;
        border-right: 1px solid #1f2937;
        border-bottom: 1px solid #1f2937;
    }
    
    /* Highlight inline citations */
    .inline-citation {
        background-color: #2e1065;
        color: #d8b4fe;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 12px;
        font-weight: bold;
        border: 1px solid #6b21a8;
    }
    
    /* Styling headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Streamlit container border styling */
    div[data-testid="stForm"] {
        border: 1px solid #1f2937;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load cached models
@st.cache_resource
def get_embeddings():
    return retrieval.get_embeddings()

@st.cache_resource
def get_cross_encoder():
    return retrieval.load_cross_encoder()

def check_or_load_indices():
    """Attempts to load indexes from disk; returns None components if not found."""
    try:
        faiss_index, bm25, chunks = retrieval.load_indices()
        return faiss_index, bm25, chunks
    except FileNotFoundError:
        return None, None, []

# Initialize Session State values
if "messages" not in st.session_state:
    st.session_state.messages = []
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

# Check index status and load assets
faiss_index, bm25, chunks = check_or_load_indices()
index_ready = faiss_index is not None and bm25 is not None and len(chunks) > 0

# Retrieve Ingestion files registry
try:
    with open(config.PROCESSED_FILES_REGISTRY, 'r', encoding='utf-8') as f:
        registry = json.load(f)
except Exception:
    registry = {}

# Layout design using Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=64)
    st.title("RAG Controls")
    
    # 1. Settings Section
    with st.container(border=True):
        st.subheader("⚙️ Settings")
        
        # Groq API Key
        api_key = os.getenv("GROQ_API_KEY", "")
        user_api_key = st.text_input(
            "Groq API Key",
            value=api_key,
            type="password",
            placeholder="gsk_...",
            help="Create a free key at console.groq.com"
        )
        active_api_key = user_api_key if user_api_key else api_key
        
        # Retrieval Mode select box
        retrieval_mode = st.selectbox(
            "Retrieval Strategy",
            options=["dense", "hybrid", "hybrid_rerank"],
            format_func=lambda x: {
                "dense": "Dense Only (FAISS)",
                "hybrid": "Hybrid RRF (FAISS + BM25)",
                "hybrid_rerank": "Hybrid + Cross-Encoder Rerank"
            }[x],
            index=2,
            help="Dense matches meaning; Hybrid adds exact words; Reranking optimizes precision."
        )

    # 2. Documents Section
    with st.container(border=True):
        st.subheader("📄 Knowledge Documents")
        
        is_empty = len(registry) == 0
        
        with st.expander("📤 Upload Documents", expanded=is_empty):
            uploaded_files = st.file_uploader(
                "Upload files",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                save_dir = config.DATA_DIR
                save_dir.mkdir(exist_ok=True)
                files_written = False
                
                for f in uploaded_files:
                    save_path = save_dir / f.name
                    if not save_path.exists() or save_path.stat().st_size != len(f.getvalue()):
                        with open(save_path, "wb") as out_file:
                            out_file.write(f.getvalue())
                        files_written = True
                
                if files_written:
                    with st.spinner("Parsing & embedding documents..."):
                        chunks_list, changed = ingestion.ingest_all_documents()
                        if changed:
                            retrieval.build_indices(chunks_list)
                            st.success("Indices updated successfully!")
                            st.cache_resource.clear()
                            st.rerun()

        # Render loaded file list outside the expander
        if registry:
            st.markdown("**Ingested Files:**")
            for filename, info in registry.items():
                chunk_cnt = len(info.get("chunks", []))
                st.markdown(f"- 📄 `{filename}` *({chunk_cnt} chunks)*")
        else:
            st.info("No documents loaded.")

    # 3. System Status Section
    with st.container(border=True):
        st.subheader("📊 System Status")
        st.markdown(f"**Knowledge Base**: {'✅ Active' if index_ready else '❌ Inactive'}")
        st.markdown(f"**Total Indexed Chunks**: `{len(chunks) if index_ready else 0}`")
        st.markdown(f"**Embedding Model**: `{config.EMBEDDING_MODEL_NAME.split('/')[-1]}`")
        st.markdown(f"**Reranker Model**: `{config.RERANKER_MODEL_NAME.split('/')[-1]}`")

    # 4. Action Operations & Confirmations
    st.markdown("---")
    
    # Confirm Delete UI
    if st.session_state.confirm_delete:
        st.warning("⚠️ Delete all documents and indices? This action is permanent.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", type="primary", use_container_width=True):
                # Clean local directories
                for f in config.DATA_DIR.glob("*"):
                    if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".txt"]:
                        f.unlink()
                if config.FAISS_INDEX_PATH.exists():
                    shutil.rmtree(config.FAISS_INDEX_PATH)
                if config.BM25_INDEX_PATH.exists():
                    config.BM25_INDEX_PATH.unlink()
                if config.PROCESSED_FILES_REGISTRY.exists():
                    config.PROCESSED_FILES_REGISTRY.unlink()
                
                # Clear session state and caches
                st.cache_resource.clear()
                st.session_state.messages = []
                st.session_state.confirm_delete = False
                st.success("Cleaned knowledge store.")
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_delete = False
                st.rerun()
    else:
        # Clear Conversation
        if st.button("🧹 Clear Chat Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        # Delete All Documents
        if st.button("🗑️ Remove All Documents", use_container_width=True):
            st.session_state.confirm_delete = True
            st.rerun()

    # 5. RAGAS Benchmark Summary
    if config.EVAL_RESULTS_PATH.exists():
        with st.expander("📈 RAGAS Performance Report", expanded=False):
            with open(config.EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
                st.markdown(f.read())
                
    # Run evaluation trigger
    if st.button("⚡ Run RAGAS Benchmark", use_container_width=True):
        if not active_api_key:
            st.error("Groq API key required.")
        else:
            with st.spinner("Running RAGAS evaluation (takes ~2 mins)..."):
                try:
                    report = evaluation.run_evaluation(active_api_key)
                    st.success("Benchmarks complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation error: {e}")

# Main Chat Panel
# Header title
st.title("🧠 Ask Your Docs")
st.markdown("Query Llama 3.1 about your local documents with verifiable hybrid search and cross-encoder rerankers.")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # For Assistant messages, show metadata tags and collapsible sources
        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            ret_mode = msg.get("retrieval_mode", "")
            eval_scores = msg.get("eval_metrics", None)
            
            # Print tag bar
            tags_html = ""
            if ret_mode:
                mode_lbl = {
                    "dense": "🔍 Dense Only",
                    "hybrid": "🔍 Hybrid (RRF)",
                    "hybrid_rerank": "🔍 Hybrid + Reranker"
                }.get(ret_mode, ret_mode)
                tags_html += f"<span class='badge'>{mode_lbl}</span>"
            if eval_scores:
                tags_html += f"<span class='badge' style='border-color: #10b981; color: #a7f3d0;'>📊 Faithfulness: {eval_scores['faithfulness']:.3f}</span>"
                tags_html += f"<span class='badge' style='border-color: #10b981; color: #a7f3d0;'>📊 Relevancy: {eval_scores['answer_relevancy']:.3f}</span>"
            
            if tags_html:
                st.markdown(f"<div style='margin-bottom: 6px;'>{tags_html}</div>", unsafe_allow_html=True)
                
            if sources:
                with st.expander("📄 View Sources Used", expanded=False):
                    for idx, src in enumerate(sources):
                        meta = src.get("metadata", {})
                        source = meta.get("source", "Unknown")
                        page = meta.get("page", "Unknown")
                        section = meta.get("section", "Unknown")
                        score = src.get("score", 0.0)
                        
                        st.markdown(f"**[{idx+1}] {source}** | Page {page} | Section: *{section}* (Score: `{score:.4f}`)")
                        st.markdown(f"<div class='source-box'>{src['text']}</div>", unsafe_allow_html=True)

# EMPTY STATE & SUGGESTION CARDS
if len(st.session_state.messages) == 0:
    st.markdown("---")
    if not index_ready:
        st.markdown("### 👋 Welcome! Let's get started.")
        st.markdown("Upload documents in the sidebar to populate the knowledge base. Once loaded, you can ask questions like:")
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **What are the standard working hours at Acme Corp?**\n\n*Requires company_policy.docx*")
        with col2:
            st.info("💡 **Why does the Groq LPU use SRAM instead of DRAM?**\n\n*Requires ai_handbook.pdf*")
    else:
        st.markdown("### 💡 Suggestion Prompts")
        st.markdown("Click one of these suggestions to query the active indices instantly:")
        
        # Inspect files loaded to suggest smart prompts
        loaded_files = list(registry.keys())
        suggestions = []
        if any("company_policy" in f for f in loaded_files):
            suggestions.append("What are the standard working hours at Acme Corp?")
            suggestions.append("How many days of PTO do employees receive?")
        if any("ai_handbook" in f for f in loaded_files):
            suggestions.append("Why does the Groq LPU use SRAM instead of standard memory?")
            suggestions.append("What context window size is supported by Llama 3.1?")
        if any("rag_guide" in f for f in loaded_files):
            suggestions.append("What are the three core phases of a standard RAG pipeline?")
            suggestions.append("Why is RAG more cost-effective than fine-tuning?")
            
        if not suggestions:
            suggestions = [
                "Summarize the key takeaways from the documents.",
                "What is the main topic covered in the knowledge base?"
            ]
            
        # Draw columns for buttons
        cols = st.columns(min(len(suggestions), 3))
        for i, sug in enumerate(suggestions[:3]):
            with cols[i]:
                if st.button(sug, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": sug})
                    st.rerun()

# Answer Generation Pipeline trigger
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        # Multi-stage progress indicators using st.status
        with st.status("🔍 Processing query...", expanded=True) as status:
            if not index_ready:
                status.update(label="No documents indexed!", state="error", expanded=True)
                st.error("Please upload and index documents in the sidebar first.")
            elif not active_api_key:
                status.update(label="Groq API key missing!", state="error", expanded=True)
                st.error("Please enter a Groq API Key in the sidebar.")
            else:
                st.write("Searching dense and sparse indices...")
                cross_encoder = get_cross_encoder() if retrieval_mode == "hybrid_rerank" else None
                
                # Retrieve chunks
                retrieved_chunks = retrieval.retrieve(
                    query=user_query,
                    all_chunks=chunks,
                    faiss_index=faiss_index,
                    bm25=bm25,
                    cross_encoder=cross_encoder,
                    mode=retrieval_mode
                )
                
                st.write(f"✅ Found {len(retrieved_chunks)} relevant source text chunks.")
                st.write("Synthesizing grounded response using Llama 3.1...")
                status.update(label="🧠 Generating answer...", state="running")
                
                response_placeholder = st.empty()
                response_text = ""
                
                try:
                    # Stream tokens from Groq
                    stream = generation.generate_answer_stream(
                        query=user_query,
                        retrieved_chunks=retrieved_chunks,
                        api_key=active_api_key
                    )
                    
                    for token in stream:
                        response_text += token
                        response_placeholder.markdown(response_text + "▌")
                    response_placeholder.markdown(response_text)
                    
                    # Look up RAGAS scores if evaluation CSV is present
                    eval_metrics = None
                    try:
                        eval_csv = config.INDEX_DIR / "eval_results.csv"
                        if eval_csv.exists():
                            df_eval = pd.read_csv(eval_csv)
                            mode_mapping = {
                                "dense": "Dense Only (FAISS)",
                                "hybrid": "Hybrid Retrieval (FAISS + BM25)",
                                "hybrid_rerank": "Hybrid + Cross-Encoder Rerank"
                            }
                            row = df_eval[df_eval["Retrieval Mode"] == mode_mapping[retrieval_mode]]
                            if not row.empty:
                                eval_metrics = {
                                    "faithfulness": float(row.iloc[0]["Faithfulness"]),
                                    "answer_relevancy": float(row.iloc[0]["Answer Relevancy"])
                                }
                    except Exception:
                        pass
                    
                    # Update status step to complete
                    status.update(label="Complete!", state="complete", expanded=False)
                    
                    # Save reply to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": retrieved_chunks,
                        "eval_metrics": eval_metrics,
                        "retrieval_mode": retrieval_mode
                    })
                    st.rerun()
                    
                except Exception as e:
                    status.update(label="Generation failed!", state="error", expanded=True)
                    st.error(f"Inference error: {e}")

# Standard Chat Input Box
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()
