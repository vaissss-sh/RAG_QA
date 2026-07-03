import os
import re
import json
import hashlib
from pathlib import Path
from pypdf import PdfReader
import docx

import config

# Regular expressions for section header detection
HEADING_PATTERNS = [
    r'^#+\s+(.+)$',                            # Markdown Headers: # Header
    r'^\d+(\.\d+)*\s+[A-Z].*$',                # Numbered headings: 1. Introduction, 1.1 Background
    r'^[A-Z][A-Z\s\-\,\.\&\(\)\/\:\;\@]{4,50}$', # Short all-caps titles
    r'^(Section|Chapter|Appendix)\s+\d+.*$',   # Specific Section markers
]

def is_heading(text: str) -> bool:
    """Helper to detect if a line or paragraph text is a section heading."""
    text = text.strip()
    if not text or len(text) > 100:  # Headings are generally short
        return False
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, text):
            return True
    return False

def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file to check for updates."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def parse_pdf(file_path: Path) -> list:
    """Extract paragraphs and headers from a PDF page-by-page."""
    reader = PdfReader(file_path)
    paragraphs = []
    current_section = "Introduction"
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text()
        if not text:
            continue
        
        # Split by newlines and group lines that form paragraphs
        lines = text.split('\n')
        curr_para = []
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                if curr_para:
                    para_text = " ".join(curr_para).strip()
                    if para_text:
                        if is_heading(para_text):
                            current_section = para_text
                        paragraphs.append({
                            "text": para_text,
                            "page": page_num,
                            "section": current_section,
                            "source": file_path.name
                        })
                    curr_para = []
            else:
                curr_para.append(line_strip)
                
        if curr_para:
            para_text = " ".join(curr_para).strip()
            if para_text:
                if is_heading(para_text):
                    current_section = para_text
                paragraphs.append({
                    "text": para_text,
                    "page": page_num,
                    "section": current_section,
                    "source": file_path.name
                })
    return paragraphs

def parse_docx(file_path: Path) -> list:
    """Extract paragraphs and headers from a DOCX file."""
    doc = docx.Document(file_path)
    paragraphs = []
    current_section = "Introduction"
    
    # Estimate page numbers since Word flow is dynamic (approx 1500 chars/page)
    char_count = 0
    chars_per_page = 1500
    
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
            
        is_heading_style = p.style and p.style.name and p.style.name.startswith('Heading')
        if is_heading_style or is_heading(text):
            current_section = text
            
        page_num = (char_count // chars_per_page) + 1
        char_count += len(text)
        
        paragraphs.append({
            "text": text,
            "page": page_num,
            "section": current_section,
            "source": file_path.name
        })
    return paragraphs

def parse_txt(file_path: Path) -> list:
    """Extract paragraphs and headers from a TXT file."""
    paragraphs = []
    current_section = "Introduction"
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    raw_paras = content.split('\n\n')
    char_count = 0
    chars_per_page = 1500
    
    for rp in raw_paras:
        text = rp.strip()
        if not text:
            continue
            
        if is_heading(text):
            current_section = text
            
        page_num = (char_count // chars_per_page) + 1
        char_count += len(text)
        
        paragraphs.append({
            "text": text,
            "page": page_num,
            "section": current_section,
            "source": file_path.name
        })
    return paragraphs

def split_large_paragraph(para: dict, chunk_size: int) -> list:
    """Split an oversized paragraph into sentence-level sub-paragraphs."""
    text = para["text"]
    # Simple regex to split on sentences (period/exclamation/question followed by space)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sub_paras = []
    current_sub = ""
    for s in sentences:
        if len(current_sub) + len(s) + 1 < chunk_size:
            current_sub += (" " + s if current_sub else s)
        else:
            if current_sub:
                sub_paras.append({
                    "text": current_sub,
                    "page": para["page"],
                    "section": para["section"],
                    "source": para["source"]
                })
            current_sub = s
    if current_sub:
        sub_paras.append({
            "text": current_sub,
            "page": para["page"],
            "section": para["section"],
            "source": para["source"]
        })
    return sub_paras

def get_chunks_from_file(file_path: Path, chunk_size: int, chunk_overlap: int) -> list:
    """Read a document, divide it into paragraphs, and apply sliding window chunking."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        raw_paragraphs = parse_pdf(file_path)
    elif ext == ".docx":
        raw_paragraphs = parse_docx(file_path)
    elif ext == ".txt":
        raw_paragraphs = parse_txt(file_path)
    else:
        print(f"Skipping unsupported file extension: {ext}")
        return []
        
    # Split any individual paragraph that is too large
    processed_paragraphs = []
    for rp in raw_paragraphs:
        if len(rp["text"]) > chunk_size * 1.5:
            processed_paragraphs.extend(split_large_paragraph(rp, chunk_size))
        else:
            processed_paragraphs.append(rp)
            
    # Combine paragraphs into chunks with overlap
    chunks = []
    i = 0
    while i < len(processed_paragraphs):
        p = processed_paragraphs[i]
        chunk_paras = [p]
        curr_len = len(p["text"])
        j = i + 1
        
        while j < len(processed_paragraphs) and curr_len + len(processed_paragraphs[j]["text"]) < chunk_size:
            chunk_paras.append(processed_paragraphs[j])
            curr_len += len(processed_paragraphs[j]["text"]) + 1
            j += 1
            
        chunk_text = "\n".join([cp["text"] for cp in chunk_paras])
        
        if len(chunk_text.strip()) >= config.MIN_CHUNK_LEN:
            first_para = chunk_paras[0]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": first_para["source"],
                    "page": first_para["page"],
                    "section": first_para["section"]
                }
            })
            
        if j >= len(processed_paragraphs):
            break
            
        # Overlap backtracking
        overlap_len = 0
        backtrack_idx = j - 1
        while backtrack_idx >= i and overlap_len + len(processed_paragraphs[backtrack_idx]["text"]) < chunk_overlap:
            overlap_len += len(processed_paragraphs[backtrack_idx]["text"]) + 1
            backtrack_idx -= 1
            
        i = max(backtrack_idx + 1, i + 1)
        
    return chunks

def ingest_all_documents() -> tuple[list[dict], bool]:
    """
    Scans the data directory, detects modifications, parses changed files,
    updates registry, and returns (all_chunks, has_changes).
    """
    registry_path = config.PROCESSED_FILES_REGISTRY
    if registry_path.exists():
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    else:
        registry = {}
        
    current_files = list(config.DATA_DIR.glob("*"))
    current_files = [f for f in current_files if f.suffix.lower() in [".pdf", ".docx", ".txt"]]
    
    has_changes = False
    new_registry = {}
    all_chunks = []
    
    # Check for additions and modifications
    for file_path in current_files:
        filename = file_path.name
        file_hash = get_file_hash(file_path)
        
        # If file is unchanged, load existing chunks from registry if present
        if filename in registry and registry[filename]["hash"] == file_hash:
            new_registry[filename] = registry[filename]
            all_chunks.extend(registry[filename]["chunks"])
        else:
            print(f"Ingesting and chunking {filename}...")
            has_changes = True
            chunks = get_chunks_from_file(file_path, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            new_registry[filename] = {
                "hash": file_hash,
                "chunks": chunks
            }
            all_chunks.extend(chunks)
            
    # Check for deletions
    for filename in registry:
        if filename not in new_registry:
            print(f"Detected deletion of {filename}")
            has_changes = True
            
    # Save the updated registry if changes were made
    if has_changes:
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(new_registry, f, indent=4, ensure_ascii=False)
            
    return all_chunks, has_changes

if __name__ == "__main__":
    chunks, changed = ingest_all_documents()
    print(f"Total chunks: {len(chunks)}, Ingestion updated: {changed}")
