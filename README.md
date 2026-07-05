# Conversational AI for Financial Q&A

A conversational AI system that answers natural-language questions over company financial statements, combining a **Retrieval-Augmented Generation (RAG)** pipeline with **LoRA fine-tuned language models**, built and evaluated on TCS annual reports (FY24, FY25).

## Overview

Financial statements are dense, tabular, and hard to query directly. This project turns them into a question-answering system that can respond to queries like *"What was the Revenue from operations in Q4 FY24?"* by:

1. Extracting structured data from raw PDF annual reports
2. Converting tabular KPIs into retrievable text chunks
3. Retrieving the most relevant chunks for a given question (RAG)
4. Generating a grounded natural-language answer using an LLM

Two complementary approaches are implemented and compared:

- **RAG pipeline** — retrieval over embedded financial statement chunks, answered with FLAN-T5
- **Fine-tuned models** — FLAN-T5-small and Phi-2, adapted with LoRA/PEFT directly on financial Q&A pairs

## Architecture

```
Raw PDFs (TCS_2023-24.pdf, TCS_2024-25.pdf)
        │  pymupdf / pdfplumber / camelot (table extraction)
        ▼
Structured JSON (financial_statement_fixed_2024.json, _2025.json)
        │  rows_to_sentences() — tabular KPI rows → natural-language sentences
        ▼
Token-aware chunking (tiktoken, cl100k_base, configurable chunk sizes)
        │
        ├──► Embeddings (sentence-transformers/all-MiniLM-L6-v2) → FAISS index
        │                                                              │
        │                                                     top-k similarity search
        │                                                              ▼
        │                                                 FLAN-T5 (text2text generation)
        │                                                              │
        └──► QA pair datasets (rag_qas.json, test_qas.json) ──► Answer
                        │
                        ▼
        LoRA/PEFT fine-tuning
        ├── FLAN-T5-small (seq2seq)
        └── Phi-2 (causal LM, target_modules=q_proj/v_proj)
```

## Repository Structure

```
conv_ai_financial_qa/
├── data/
│   ├── raw_pdfs/              # Source annual reports (TCS FY24, FY25)
│   ├── processed_files/       # Extracted, structured financial statement JSON
│   ├── rag_chunks/            # Chunked text, FAISS index, and metadata for retrieval
│   ├── qa_datasets/           # Generated QA pairs (RAG eval + T5 fine-tuning + test set)
│   └── fine_tune/             # Fine-tuning JSONL datasets
├── rag_system/
│   └── preprocessor.py        # PDF → sentences → token-aware chunks with metadata
├── fine_tuned_model/
│   ├── train.py                # LoRA fine-tuning of microsoft/phi-2 on QA pairs
│   └── inference.py            # Interactive CLI inference over the fine-tuned model
├── notebooks/
│   ├── data_extraction.ipynb   # PDF table/text extraction into structured JSON
│   ├── faiss_index.ipynb       # Embedding generation + FAISS index build
│   ├── rag_pipeline.ipynb      # End-to-end RAG retrieval + generation pipeline
│   ├── pre_fine_tune.ipynb     # Baseline (pre-fine-tuning) evaluation
│   ├── finetune_qa.ipynb       # QA dataset preparation for fine-tuning
│   ├── finetune_flant5_lora.ipynb  # LoRA fine-tuning of FLAN-T5-small
│   └── results/flan_t5_lora/checkpoint-2814/  # Training checkpoint
├── results/flan_t5_lora_adapter/    # Saved LoRA adapter (base: google/flan-t5-small)
└── requirements.txt
```

## Tech Stack

| Component | Tool/Library |
|---|---|
| PDF/table extraction | PyMuPDF, pdfplumber, camelot-py |
| Tokenization / chunking | tiktoken (cl100k_base) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS |
| Generation (RAG) | google/flan-t5-base |
| Fine-tuning | HuggingFace Transformers, PEFT (LoRA), Datasets, Accelerate |
| Fine-tuned models | google/flan-t5-small (seq2seq), microsoft/phi-2 (causal LM) |

## How It Works

**1. Data extraction** (`notebooks/data_extraction.ipynb`)
Parses TCS annual report PDFs into structured JSON of `[KPI, period, value]` rows using PyMuPDF/pdfplumber/camelot for table-aware extraction.

**2. Chunking** (`rag_system/preprocessor.py`)
Converts each KPI row into a plain-language sentence (e.g. *"Revenue from operations in Q4 FY24 was 61,237. (Source: financial_statement_fixed_2024)"*), then groups sentences into token-bounded chunks (configurable sizes, e.g. 100/400 tokens) using tiktoken.

**3. Retrieval index** (`notebooks/faiss_index.ipynb`)
Embeds each chunk with `all-MiniLM-L6-v2` and builds a FAISS similarity index (`data/rag_chunks/faiss_index.idx` + `metadata.json`).

**4. RAG inference** (`notebooks/rag_pipeline.ipynb`)
For a user question, retrieves the top-k most similar chunks from FAISS and passes them as context to `google/flan-t5-base` for answer generation.

**5. Fine-tuning** (`fine_tuned_model/train.py`, `notebooks/finetune_flant5_lora.ipynb`)
As an alternative to retrieval, both FLAN-T5-small and Phi-2 are fine-tuned directly on curated financial QA pairs using LoRA (rank 8, alpha 16) via PEFT, so the model can answer without an explicit retrieval step.

## Example

```
Q: What was the Revenue from operations in Q4 FY24?
A: The Revenue from operations in Q4 FY24 was 61,237.
```

QA pairs are auto-generated from every KPI × period combination in the structured financial statements, giving a consistent, verifiable ground truth for evaluation (see `data/qa_datasets/rag_qas.json`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
python -m ipykernel install --user --name=conv_ai_env
jupyter lab
```

Run notebooks in order: `data_extraction.ipynb` → `faiss_index.ipynb` → `rag_pipeline.ipynb` for the RAG path, or `finetune_qa.ipynb` → `finetune_flant5_lora.ipynb` for the fine-tuning path.
