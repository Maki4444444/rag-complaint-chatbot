"""
src/generator.py

Generator module for Task 3: combines the prompt template, retrieved
chunks, and user question, then calls the HuggingFace Inference API
to produce a grounded, evidence-backed answer.

Uses Qwen/Qwen2.5-7B-Instruct via HuggingFace's free Inference API --
no model download, no GPU required. The model runs on HF servers.

Setup:
    1. Create a free account at huggingface.co
    2. Go to Settings -> Access Tokens -> New Token (Read access)
    3. Add HF_TOKEN=hf_... to your .env file in the repo root
"""
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ── Model configuration ────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL    = "Qwen/Qwen2.5-7B-Instruct"

# ── Prompt template ────────────────────────────────────────────────────────
# This template is a core rubric requirement:
#   - Sets the analyst role and domain (CrediTrust / financial complaints)
#   - Enforces groundedness: answer ONLY from the provided context
#   - Includes a fallback: admit when context is insufficient
RAG_PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. \
Your task is to answer questions about customer complaints.
Use ONLY the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain enough information to answer the question, \
state clearly: "I don't have enough information to answer that."
Do not add any information not present in the excerpts below.

Retrieved complaint excerpts:
{context}

Question: {question}

Answer:"""


def build_context_block(retrieved_chunks: list) -> str:
    """
    Format the list of retrieved chunks into a numbered context block
    for injection into the prompt, including key metadata for traceability.
    """
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(
            f"[Excerpt {i} | Product: {chunk['product_category']} | "
            f"Complaint ID: {chunk['complaint_id']}]\n{chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


def build_llm_client(token: str = HF_TOKEN, model: str = MODEL) -> InferenceClient:
    """
    Connect to the HuggingFace Inference API.
    Raises a clear error if the token is missing.
    """
    if not token:
        raise ValueError(
            "HF_TOKEN is not set. Add HF_TOKEN=hf_... to your .env file.\n"
            "Get a free token at: https://huggingface.co/settings/tokens"
        )
    return InferenceClient(model=model, token=token)


def generate_answer(
    client: InferenceClient,
    question: str,
    retrieved_chunks: list,
    max_new_tokens: int = 300,
) -> dict:
    """
    Full RAG generator step:
      1. Build a numbered context block from the retrieved chunks
      2. Inject context + question into the prompt template
      3. Call the LLM via HuggingFace Inference API
      4. Return the answer along with the prompt and sources for traceability

    Args:
        client:           HuggingFace InferenceClient
        question:         the user's plain-English question
        retrieved_chunks: list of dicts from the retriever
        max_new_tokens:   max length of the generated answer

    Returns:
        dict with keys: question, answer, context_block, sources, prompt
    """
    context_block = build_context_block(retrieved_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_block,
        question=question,
    )

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
    )
    answer = (response.choices[0].message.content or "").strip()

    return {
        "question":      question,
        "answer":        answer,
        "context_block": context_block,
        "sources":       retrieved_chunks,
        "prompt":        prompt,
    }