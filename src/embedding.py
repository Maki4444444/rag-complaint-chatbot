"""
src/embedding.py

Embedding and vector store utilities for Task 2: turning text chunks into
vectors and persisting them in a ChromaDB collection with metadata.
"""
import chromadb


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Loads the sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_chunks(model, chunk_records: list, batch_size: int = 64):
    """
    Generates embeddings for a list of chunk dicts (from chunk_dataframe).
    Returns the same list with an added "embedding" key per record.
    """
    texts = [r["chunk_text"] for r in chunk_records]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    for record, embedding in zip(chunk_records, embeddings):
        record["embedding"] = embedding.tolist()

    return chunk_records


def build_chroma_store(chunk_records: list, persist_dir: str, collection_name: str = "complaints"):
    """
    Creates (or overwrites) a persisted ChromaDB collection from chunk
    records that already have an "embedding" key.

    Each chunk's metadata includes complaint_id, product_category,
    chunk_index, and total_chunks for traceability back to the source
    complaint.
    """
    client = chromadb.PersistentClient(path=persist_dir)

    # Drop existing collection of the same name so re-runs are idempotent
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    ids = [f"{r['complaint_id']}_{r['chunk_index']}" for r in chunk_records]
    embeddings = [r["embedding"] for r in chunk_records]
    documents = [r["chunk_text"] for r in chunk_records]
    metadatas = [
        {
            "complaint_id": str(r["complaint_id"]),
            "product_category": str(r["product_category"]),
            "chunk_index": int(r["chunk_index"]),
            "total_chunks": int(r["total_chunks"]),
        }
        for r in chunk_records
    ]

    # Chroma has a max batch add size; insert in batches to be safe.
    BATCH = 5000
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            documents=documents[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )

    return collection