# Chat With Your Documents (Advanced Grounded RAG Pipeline)

An advanced, production-grade Retrieval-Augmented Generation (RAG) pipeline in Python. Built specifically to handle large, complex structured documents (such as 79+ page academic grading documents, policy manuals, and technical specifications) with exact formula preservation, schedule parsing, and inline page citations.

## Key Features

- **Section & Course-Aware Ingestion:** Tracks multi-page course hierarchies (Foundation, Diploma, Degree levels) across continuation pages so chunks never lose their course context.
- **Hybrid Retrieval (Dense + BM25 with RRF):**
  - **Dense Semantic Embeddings:** Uses `all-mpnet-base-v2` (768-dimensional local embeddings via `sentence-transformers`).
  - **Sparse Keyword Search:** Uses `BM25Okapi` to pinpoint exact acronyms and course codes (`DBMS`, `CS2001`, `OPPE`, `GAA`).
  - **Reciprocal Rank Fusion (RRF):** Fuses dense and sparse rankings to retrieve the most relevant sections every time.
- **Exact Formula & Schedule Preservation:** Configured to accurately reproduce mathematical grading formulas (weights, max conditions, bonus rules) and exam dates.
- **Robust API Resilience:** Built-in retry logic with exponential backoff against transient Gemini API spikes.
- **Strict Anti-Hallucination Guardrail:** Returns `"I don't know based on the provided documents."` for ungrounded queries.

## Quickstart Guide

### 1. Set Up Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Your API Key
In `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Ingest Documents
Place your `.pdf`, `.md`, or `.txt` files in `./documents/` and index:
```bash
python main.py --ingest
```

### 4. Interactive Chat
```bash
python main.py
```
Or with retrieval inspection scores:
```bash
python main.py --debug
```

## Example Questions & Answers

```text
🧑 You: What is the grading formula for DBMS course?

🤖 Assistant:
For the Database Management System (DBMS) course (CS2001), the final course score (T) is calculated as:

T = 0.03GAA2 + 0.02GAA3 + 0.2 OPPE + 0.45F + max(0.2max(Qz1, Qz2), (0.10Qz1 + 0.20Qz2))

Where:
• GAA2 = Average score of week 2 and 3 SQL-based assignments (grading_document.pdf, Page 32)
• GAA3 = Average score of week 7 programming assignment (grading_document.pdf, Page 32)
• Qz1 = Score in Quiz I (0 if not attempted) (grading_document.pdf, Page 32)
• Qz2 = Score in Quiz II (0 if not attempted) (grading_document.pdf, Page 32)
• OPPE = Score in online remote proctored programming exam (grading_document.pdf, Page 32)
• F = Score in final exam (grading_document.pdf, Page 32)

📌 Verified Sources: grading_document.pdf (Page 31), grading_document.pdf (Page 32)
```
