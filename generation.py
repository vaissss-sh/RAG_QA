import os
from groq import Groq
import config

def format_context(retrieved_chunks: list[dict]) -> str:
    """Format chunk text and metadata into a clean text block for LLM prompt context."""
    context_blocks = []
    for c in retrieved_chunks:
        meta = c["metadata"]
        source = meta.get("source", "Unknown")
        page = meta.get("page", "Unknown")
        section = meta.get("section", "Unknown")
        block = f"Source: {source} | Page: {page} | Section: {section}\nContent:\n{c['text']}"
        context_blocks.append(block)
        
    return "\n\n---\n\n".join(context_blocks)

def get_system_prompt() -> str:
    """Returns system guidelines enforcing grounding, citations, and 'not found' behavior."""
    return (
        "You are answering questions using ONLY the retrieved document contexts. Follow these rules strictly:\n"
        "1. Do NOT describe your reasoning process or mention ambiguity in the question itself.\n"
        "2. If multiple documents are present in the context, answer using all relevant ones and clearly label which document each part of your answer comes from (e.g. 'According to [filename]: ...' with inline citations like [filename.pdf, Page X] or [filename.docx, Section Y] where the information was found).\n"
        "3. If a document is not relevant to the question, simply omit it — do not say 'not found' for documents that weren't asked about.\n"
        "4. Only say 'not found in the provided documents' if NONE of the retrieved context answers the question at all.\n"
        "5. Give a direct, well-organized answer. No meta-commentary about the question being broad."
    )

def generate_answer(query: str, retrieved_chunks: list[dict], api_key: str) -> str:
    """Synchronous answer generation. Useful for evaluation scripts."""
    if not api_key:
        raise ValueError("Groq API key is missing. Please provide your key in .env or the Streamlit sidebar.")
        
    context_text = format_context(retrieved_chunks)
    client = Groq(api_key=api_key)
    
    completion = client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ],
        temperature=0.0  # Zero temperature for deterministic, grounded results
    )
    
    return completion.choices[0].message.content

def generate_answer_stream(query: str, retrieved_chunks: list[dict], api_key: str):
    """Generator yielding answer tokens in real-time. Useful for Streamlit chats."""
    if not api_key:
        raise ValueError("Groq API key is missing. Please provide your key in .env or the Streamlit sidebar.")
        
    context_text = format_context(retrieved_chunks)
    client = Groq(api_key=api_key)
    
    completion = client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ],
        temperature=0.0,
        stream=True
    )
    
    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
