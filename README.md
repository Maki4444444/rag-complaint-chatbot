# Intelligent Complaint Analysis for Financial Services

**RAG-Powered Chatbot for CrediTrust Financial: turning raw customer complaints into actionable insights.**

> 10 Academy AI Mastery Week 7 Challenge (17–23 Jun 2026)

## Business Context

CrediTrust Financial is a fast-growing digital finance company operating across three East
African markets, serving 500,000+ users across four product lines: **Credit Cards, Personal
Loans, Savings Accounts, and Money Transfers**. The company receives thousands of customer
complaints per month and currently has no efficient way to surface trends from this
unstructured feedback.

This project builds an internal **Retrieval-Augmented Generation (RAG)** chatbot that lets
Product, Support, and Compliance teams ask plain-English questions (e.g. *"Why are people
unhappy with Credit Cards?"*) and receive synthesized, evidence-backed answers in seconds.

## Project Goals

- Reduce the time to identify a major complaint trend from days to minutes
- Let non-technical teams self-serve insights without a data analyst
- Shift the organization from reactive to proactive issue identification

## Project Structure
rag-complaint-chatbot/

├── .vscode/                  # Editor settings

├── .github/workflows/        # CI: automated unit tests

├── data/

│   ├── raw/                  # Original CFPB dataset (not tracked in git)

│   └── processed/            # Cleaned/filtered data (not tracked in git)

├── vector_store/             # Persisted FAISS/ChromaDB index (not tracked in git)

├── notebooks/                # EDA, chunking, and RAG prototyping notebooks

├── src/                      # Production pipeline code (preprocessing, chunking,

│                              #   embedding, retriever, generator)

├── tests/                    # Unit tests

├── app.py                    # Gradio/Streamlit chat interface

├── requirements.txt

└── README.md

## Tasks

| Task | Description | Status |
|---|---|---|
| 1 | EDA & Data Preprocessing | ✅ |
| 2 | Text Chunking, Embedding & Vector Store Indexing | ✅ |
| 3 | RAG Core Logic & Evaluation | ✅ |
| 4 | Interactive Chat Interface | ✅ |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Running the App

```bash
python app.py
```

## Data Source

[Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/data-research/consumer-complaints/)
complaint dataset, filtered to four product categories: Credit Card, Personal Loan,
Savings Account, Money Transfer.

## Task 1: EDA & Preprocessing Summary

**Status:**  Complete

### Dataset Overview
- **Source:** Full CFPB Consumer Complaint Database export, ~9.6M rows (~5GB CSV)
- **Narrative coverage:** Only 31.0% of complaints (2,980,756 / 9,609,797) include a free-text consumer narrative
- **Narrative length:** Median 114 words, mean 176 words, right-skewed with a max of 6,469 words

### Filtering
Filtered to four target product categories and removed records with empty narratives:

| Product Category | Complaints |
|---|---|
| Credit Card | 189,334 |
| Savings Account | 155,204 |
| Money Transfer | 98,685 |
| Personal Loan | 37,341 |
| **Total** | **480,564** |

Category distribution is imbalanced (~5:1 between largest and smallest category) this is carried forward into stratified sampling in Task 2.

### Text Cleaning Pipeline
Implemented in `src/preprocessing.py`, run via `notebooks/01_eda_preprocessing.ipynb`:

1. **Noise removal** lowercasing, stripping URLs, phone numbers, HTML tags, CFPB redaction placeholders (`XXXX`), and punctuation
2. **NLP normalization** tokenization, English stopword removal, and lemmatization (verbs + nouns) via NLTK

Cleaning ran in ~10.5 minutes across 480,564 rows and removed only 4 records that became empty post-cleaning.

### Outputs
- Cleaned dataset: `data/processed/filtered_complaints.csv`
- EDA visualizations: `data/processed/eda_product_distribution.png`, `data/processed/eda_narrative_length.png`
- Unit tests: `tests/test_preprocessing.py` (15 tests covering text cleaning and product mapping)

## Task 2: Text Chunking, Embedding & Vector Store Indexing Summary

**Status:**  Complete

### Data Sampling, Chunking, and Vector Store Construction

A stratified sample of **12,500 complaints** was drawn from the **480,564 cleaned complaints** produced in Task 1, preserving the original product-category distribution. The resulting sample contained:

- **Credit Card:** 4,925 complaints (39.4%)
- **Savings Account:** 4,037 complaints (32.3%)
- **Money Transfer:** 2,567 complaints (20.5%)
- **Personal Loan:** 971 complaints (7.8%)

These proportions exactly match those of the full dataset, ensuring that the sample remains representative while reducing computational requirements for embedding and retrieval.

Complaint narratives were segmented using LangChain's `RecursiveCharacterTextSplitter` with a **chunk size of 500 characters** and **50-character overlap**. This configuration was selected to balance context preservation with embedding specificity while maintaining consistency with the retrieval granularity used in the downstream RAG pipeline.

The chunking process generated **23,114 text chunks** from the 12,500 sampled complaints, corresponding to an average of **1.85 chunks per complaint**.

Chunks were embedded using the **`sentence-transformers/all-MiniLM-L6-v2`** model, a lightweight transformer model (~80 MB) that produces **384-dimensional dense vector embeddings** optimized for semantic similarity search. Embedding generation for all chunks required approximately **233 seconds**.

The resulting embeddings were indexed into a persisted **ChromaDB** collection. Each chunk was stored together with metadata fields including:

- `complaint_id`
- `product_category`
- `chunk_index`
- `total_chunks`

This metadata ensures complete traceability between retrieved chunks and their source complaints, supporting filtering, debugging, evaluation, and explainability throughout the RAG workflow.

### Outputs
- Vector store: `vector_store/` (persisted ChromaDB collection)
- Pipeline code: `src/chunking.py`, `src/embedding.py`
- Notebook: `notebooks/02_chunking_embedding.ipynb`

## Task 3: Retrieval-Augmented Generation (RAG) Core Logic Summary

**Status:**  Complete

### Semantic Retrieval and Response Generation

A Retrieval-Augmented Generation (RAG) pipeline was developed to enable natural-language question answering over customer complaint narratives. The system combines semantic search with a Large Language Model (LLM), allowing responses to be grounded in actual complaint data rather than relying solely on model knowledge.

User questions are embedded using the same **`sentence-transformers/all-MiniLM-L6-v2`** model employed during vector store construction. These query embeddings are then matched against the persisted ChromaDB collection using cosine similarity search.

The retriever supports both **cross-category retrieval** and **product-specific filtering**, enabling focused searches within:

* **Credit Card**
* **Savings Account**
* **Money Transfer**
* **Personal Loan**

For each question, the top **5 most relevant complaint chunks** are retrieved and combined into a structured prompt containing complaint excerpts, complaint identifiers, and product-category information.

Responses are generated using **Qwen2.5-7B-Instruct** via the Hugging Face Inference API. Retrieved complaint evidence is supplied as context to ensure answers remain grounded in customer narratives and accurately reflect the issues present in the dataset.

The complete RAG pipeline was evaluated using representative complaint-analysis questions spanning all product categories. Evaluation included both qualitative review and embedding-based metrics assessing answer relevance, context utilization, and response faithfulness to retrieved evidence.

### Outputs

* Retriever module: `src/retriever.py`
* Generator module: `src/generator.py`
* Evaluation notebook: `notebooks/03_rag_pipeline.ipynb`
* Prompt templates and retrieval logic
* End-to-end RAG question-answering workflow

## Task 4: Interactive Complaint Intelligence Chatbot Summary

**Status:**  Complete

### User Interface and Chat Application

An interactive chatbot application was developed to provide a user-friendly interface for the RAG pipeline created in Task 3. The application enables users to explore customer complaints through natural-language conversations while maintaining full transparency into the retrieved evidence used to generate responses.

The chatbot was implemented using **Gradio** and integrates the embedding model, ChromaDB vector store, retriever, and LLM components into a single end-to-end application.

Users can submit free-form questions about customer complaints and optionally filter retrieval results by product category:

* **Credit Card**
* **Savings Account**
* **Money Transfer**
* **Personal Loan**
* **All Categories**

For each query, the system retrieves the top relevant complaint chunks, constructs a contextual prompt, and streams the generated response back to the interface in real time.

To improve explainability and trustworthiness, a dedicated sources panel displays the retrieved complaint excerpts together with their associated complaint IDs, product categories, and similarity scores. This allows users to verify the evidence supporting each generated answer.

The interface also includes example questions, conversation history, category filtering controls, and session reset functionality to enhance usability and support exploratory analysis of complaint trends.

### Outputs

* Chat application: `app.py`
* Interactive Gradio interface
* Real-time response streaming
* Evidence and source display panel
* End-to-end complaint intelligence chatbot
