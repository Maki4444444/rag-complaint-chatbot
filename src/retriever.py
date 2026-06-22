"""
src/retriever.py

Retriever module for Task 3: embeds a user question and performs
semantic similarity search against the ChromaDB vector store,
returning the top-k most relevant complaint chunks with metadata.

Supports optional product_category filtering so questions about a
specific product only search within that category's chunks.
"""
import chromadb
from src.embedding import load_embedding_model, EMBEDDING_MODEL_NAME

VALID_CATEGORIES = {
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
}


def load_vector_store(persist_dir: str, collection_name: str = "complaints"):
    """
    Load the persisted ChromaDB collection from disk.

    For Task 3, point this at the pre-built full-scale vector store from
    the challenge dataset resources (1.37M chunks across 464K complaints).
    For local testing, point at the Task 2 sample index in vector_store/
    (12,500 complaints, 23,114 chunks).
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)
    print(f"Loaded collection '{collection_name}' with {collection.count():,} chunks")
    return collection


def build_retriever(collection, model=None):
    """
    Returns a retriever function bound to the given ChromaDB collection
    and embedding model.

    The returned retrieve() function accepts:
        question         -- plain-English question string
        top_k            -- number of chunks to return (default 5)
        product_category -- optional category filter, one of:
                            "Credit Card", "Personal Loan",
                            "Savings Account", "Money Transfer"
                            If None, searches across all categories.
    """
    if model is None:
        model = load_embedding_model(EMBEDDING_MODEL_NAME)

    def retrieve(question: str, top_k: int = 5, product_category: str = None) -> list:
        """
        Embed the question and return the top_k most semantically similar
        complaint chunks from the vector store.

        Args:
            question:         plain-English question from the user
            top_k:            number of chunks to retrieve (default 5)
            product_category: optional filter — restrict search to one of:
                              "Credit Card", "Personal Loan",
                              "Savings Account", "Money Transfer"
                              If None, searches across all categories.

        Returns:
            list of dicts, each containing:
                chunk_text, complaint_id, product_category,
                chunk_index, total_chunks, similarity_score
        """
        # Validate category filter if provided
        if product_category and product_category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid product_category '{product_category}'. "
                f"Must be one of: {sorted(VALID_CATEGORIES)}"
            )

        # Embed the question using the same model that built the index
        query_embedding = (
            model.encode(question, normalize_embeddings=True)
            .tolist()
        )

        # Build optional metadata filter for ChromaDB
        where_filter = None
        if product_category:
            where_filter = {"product_category": {"$eq": product_category}}

        # Similarity search against the vector store
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        # Flatten into a clean list of dicts
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "chunk_text": doc,
                "complaint_id": meta.get("complaint_id", ""),
                "product_category": meta.get("product_category", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                # ChromaDB returns L2 distance; convert to similarity
                "similarity_score": round(1 - dist, 4),
            })

        return chunks

    return retrieve