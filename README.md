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
| 1 | EDA & Data Preprocessing | 🔲 |
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
