"""
src/chunking.py

Text chunking utilities for Task 2: splitting long complaint narratives
into smaller, embedding-friendly chunks.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter(chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter.

    Defaults (chunk_size=500, chunk_overlap=50) match the pre-built
    complaint_embeddings.parquet vector store spec used in Tasks 3-4,
    so chunk granularity stays consistent across the project even though
    Task 2 builds its own smaller-scale index.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_dataframe(df, text_column: str, splitter: RecursiveCharacterTextSplitter, id_column: str = "Complaint ID"):
    """
    Splits every narrative in `df[text_column]` into chunks and returns a
    flat list of dicts, one per chunk, with metadata for traceability.

    Each dict contains:
        complaint_id, product_category, chunk_index, total_chunks, chunk_text
    """
    records = []

    for _, row in df.iterrows():
        text = row[text_column]
        if not isinstance(text, str) or text.strip() == "":
            continue

        chunks = splitter.split_text(text)
        total_chunks = len(chunks)

        for idx, chunk_text in enumerate(chunks):
            records.append({
                "complaint_id": row[id_column],
                "product_category": row["product_category"],
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_text": chunk_text,
            })

    return records