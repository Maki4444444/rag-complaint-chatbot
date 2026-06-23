"""
app.py — CrediTrust Complaint Intelligence Chatbot
====================================================
Task 4: Interactive Chat Interface

Run with:
    python app.py

Requires:
    - .env file with HF_TOKEN=hf_...
    - vector_store/ directory with persisted ChromaDB index
    - pip install gradio sentence-transformers chromadb huggingface_hub python-dotenv
"""

import os
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
import gradio as gr

from src.retriever import load_vector_store, build_retriever, VALID_CATEGORIES
from src.generator import build_llm_client, RAG_PROMPT_TEMPLATE
from src.embedding import load_embedding_model, EMBEDDING_MODEL_NAME

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────
VECTOR_STORE_DIR  = "./vector_store"
COLLECTION_NAME   = "complaints_sample"
TOP_K             = 5
CATEGORY_OPTIONS  = ["All Categories"] + sorted(VALID_CATEGORIES)

# ── Load models and vector store once at startup ───────────────────────────
print("Loading embedding model...")
embed_model  = load_embedding_model(EMBEDDING_MODEL_NAME)

print("Loading vector store...")
collection   = load_vector_store(VECTOR_STORE_DIR, COLLECTION_NAME)

print("Connecting to LLM...")
llm_client   = build_llm_client()

retrieve     = build_retriever(collection, model=embed_model)
print("Ready!\n")


# ── Streaming RAG function ─────────────────────────────────────────────────
def rag_stream(question: str, category: str, history: list):
    """
    Streaming RAG pipeline:
      1. Retrieve top-k chunks (with optional category filter)
      2. Build prompt from chunks + question
      3. Stream LLM tokens back to the UI one by one
      4. Yield (history, sources_markdown) at each token
    """
    if not question.strip():
        yield history, ""
        return

    # Resolve category filter
    product_category = None if category == "All Categories" else category

    # ── RETRIEVE ──────────────────────────────────────────────────────────
    chunks = retrieve(question, top_k=TOP_K, product_category=product_category)

    # ── BUILD CONTEXT ─────────────────────────────────────────────────────
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} | Product: {chunk['product_category']} | "
            f"Complaint ID: {chunk['complaint_id']}]\n{chunk['chunk_text']}"
        )
    context_block = "\n\n".join(context_parts)
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_block,
        question=question,
    )

    # ── BUILD SOURCES MARKDOWN ─────────────────────────────────────────────
    sources_md = "### 📎 Retrieved Sources\n\n"
    for i, chunk in enumerate(chunks, 1):
        score = chunk["similarity_score"]
        cat   = chunk["product_category"]
        cid   = chunk["complaint_id"]
        text  = chunk["chunk_text"][:300].replace("\n", " ")
        sources_md += (
            f"**Source {i}** — {cat} | Complaint #{cid} "
            f"| Similarity: `{score:.3f}`\n"
            f"> {text}...\n\n"
        )

    # ── STREAM GENERATION ─────────────────────────────────────────────────
    # Add user message and placeholder assistant message to history
    history = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": ""},
    ]

    streamed = ""
    for token in llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        stream=True,
    ):
        delta = token.choices[0].delta.content or ""
        streamed += delta
        history[-1]["content"] = streamed
        yield history, sources_md

    # Final yield with complete response
    yield history, sources_md


def clear_all():
    """Reset the chat history, sources panel, and input box."""
    return [], "", "", "All Categories"


# ── Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(
    title="CrediTrust Complaint Intelligence",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
    /* Page background */
    .gradio-container {
        max-width: 1100px !important;
        margin: auto;
    }

    /* Header */
    #header {
        text-align: center;
        padding: 24px 0 8px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    #header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 0;
    }
    #header p {
        color: #64748b;
        font-size: 0.95rem;
        margin: 6px 0 0 0;
    }

    /* Chatbot bubbles */
    .message.svelte-1axk9f7 {
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Sources panel */
    #sources-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        font-size: 0.88rem;
        line-height: 1.6;
        max-height: 420px;
        overflow-y: auto;
    }

    /* Input row */
    #input-row {
        align-items: flex-end;
    }

    /* Buttons */
    #ask-btn {
        min-width: 90px;
    }
    #clear-btn {
        min-width: 80px;
    }
    """
) as demo:

    # ── Header ─────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="header">
        <h1>🏦 CrediTrust Complaint Intelligence</h1>
        <p>Ask questions about customer complaints across Credit Cards,
           Savings Accounts, Money Transfers, and Personal Loans.</p>
    </div>
    """)

    # ── Main layout: chat left, sources right ──────────────────────────────
    with gr.Row():

        # Left column — chat
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                value=[],
                label="Conversation",
                type="messages",
                height=480,
                bubble_full_width=False,
                show_label=False,
                placeholder=(
                    "<div style='text-align:center; color:#94a3b8; padding:40px 0'>"
                    "<div style='font-size:2rem'>💬</div>"
                    "<div style='font-size:1rem; margin-top:8px'>"
                    "Ask a question about customer complaints</div>"
                    "</div>"
                ),
            )

            # Input row
            with gr.Row(elem_id="input-row"):
                question_box = gr.Textbox(
                    placeholder="e.g. Why are customers unhappy with their credit cards?",
                    label="",
                    lines=1,
                    max_lines=4,
                    scale=8,
                    show_label=False,
                    container=False,
                    autofocus=True,
                )
                ask_btn = gr.Button(
                    "Ask ➤",
                    variant="primary",
                    scale=1,
                    elem_id="ask-btn",
                )

            # Controls row
            with gr.Row():
                category_filter = gr.Dropdown(
                    choices=CATEGORY_OPTIONS,
                    value="All Categories",
                    label="🔍 Filter by Product",
                    scale=4,
                    interactive=True,
                )
                clear_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary",
                    scale=1,
                    elem_id="clear-btn",
                )

            # Example questions
            gr.Examples(
                examples=[
                    ["Why are customers unhappy with their credit cards?"],
                    ["What fraud issues are credit card customers reporting?"],
                    ["What problems are customers having with savings accounts?"],
                    ["What issues are customers facing with money transfers?"],
                    ["What are the main complaints about personal loans?"],
                    ["How are customers describing their customer service experience?"],
                ],
                inputs=question_box,
                label="💡 Example Questions",
            )

        # Right column — sources
        with gr.Column(scale=4):
            gr.Markdown("### 📎 Retrieved Sources")
            sources_panel = gr.Markdown(
                value="*Sources will appear here after you ask a question.*",
                elem_id="sources-panel",
            )

    # ── Event wiring ───────────────────────────────────────────────────────
    # Submit on button click
    ask_btn.click(
        fn=rag_stream,
        inputs=[question_box, category_filter, chatbot],
        outputs=[chatbot, sources_panel],
    ).then(
        fn=lambda: "",
        outputs=question_box,
    )

    # Submit on Enter key
    question_box.submit(
        fn=rag_stream,
        inputs=[question_box, category_filter, chatbot],
        outputs=[chatbot, sources_panel],
    ).then(
        fn=lambda: "",
        outputs=question_box,
    )

    # Clear button resets everything
    clear_btn.click(
        fn=clear_all,
        outputs=[chatbot, sources_panel, question_box, category_filter],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )