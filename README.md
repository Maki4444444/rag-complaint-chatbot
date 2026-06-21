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
| 2 | Text Chunking, Embedding & Vector Store Indexing | 🔲 |
| 3 | RAG Core Logic & Evaluation | 🔲 |
| 4 | Interactive Chat Interface | 🔲 |

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

**Status:** ✅ Complete

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