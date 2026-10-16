import hashlib
from typing import List, Dict, Any


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 250
) -> List[str]:
    """
    Splits text into overlapping chunks, prioritizing line breaks and sentence
    boundaries so formulas, tables, and paragraphs stay coherent.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Look for natural split points in the second half of the chunk window
        if end < text_length:
            # 1. Newline (keeps table rows and formula lines together)
            last_newline = text.rfind("\n", start + (chunk_size // 2), end)
            # 2. Sentence boundary (period followed by space or newline)
            last_period = text.rfind(". ", start + (chunk_size // 2), end)
            # 3. Word boundary (space)
            last_space = text.rfind(" ", start + (chunk_size // 2), end)

            if last_newline != -1:
                end = last_newline + 1
            elif last_period != -1:
                end = last_period + 2
            elif last_space != -1:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Step forward with overlap
        start = max(start + 1, end - chunk_overlap)

    return chunks


def create_document_chunks(
    loaded_segments: List[Dict[str, Any]],
    chunk_size: int = 1200,
    chunk_overlap: int = 250
) -> List[Dict[str, Any]]:
    """
    Creates chunks enriched with contextual headers for each section.
    
    Prepending the Document, Level, Section, and Page directly into the text
    guarantees that:
    1. Dense embeddings encode the course/section topic.
    2. BM25 keyword matching matches course names (e.g. DBMS, CS2001) even on
       continuation pages.
    3. The LLM always knows the authoritative source and course context.
    """
    final_chunks = []

    for segment in loaded_segments:
        raw_text = segment["text"]
        base_meta = segment["metadata"]

        level_str = base_meta.get("level", "Document")
        section_str = base_meta.get("section", "General")
        page_num = base_meta["page"]
        source_name = base_meta["source"]

        context_header = (
            f"[Document: {source_name} | Level: {level_str} | "
            f"Section: {section_str} | Page {page_num}]\n"
        )

        sub_chunks = chunk_text(raw_text, chunk_size, chunk_overlap)

        for idx, chunk_str in enumerate(sub_chunks):
            enriched_text = context_header + chunk_str

            unique_seed = f"{source_name}_p{page_num}_s{section_str}_c{idx}_{chunk_str[:60]}"
            chunk_id = hashlib.md5(unique_seed.encode("utf-8")).hexdigest()

            chunk_meta = {
                "source": source_name,
                "page": page_num,
                "level": level_str,
                "section": section_str,
                "file_type": base_meta["file_type"],
                "chunk_index": idx
            }

            final_chunks.append({
                "id": chunk_id,
                "text": enriched_text,
                "metadata": chunk_meta
            })

    print(f"✂️  Created {len(final_chunks)} context-enriched chunks across all documents.")
    return final_chunks
