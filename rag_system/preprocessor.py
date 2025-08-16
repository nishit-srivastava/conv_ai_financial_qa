import json
import os
import uuid
from pathlib import Path
from typing import List, Dict
import tiktoken  # OpenAI tokenizer

def load_financial_json(path: str) -> List[Dict]:
    """Load financial statement JSON and return rows."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["rows"]

def rows_to_sentences(rows: List[List[str]], source: str) -> List[str]:
    """Convert tabular rows into human-readable sentences."""
    sentences = []

    headers = rows[0]   # Already: ["KPI", "Q4 FY24", "Q3 FY23", ...]
    for row in rows[1:]:
        kpi = row[0].strip()
        for col_name, val in zip(headers[1:], row[1:]):
            if kpi and val.strip():
                sentences.append(f"{kpi} in {col_name} was {val}. (Source: {source})")

    return sentences


def chunk_text(sentences: List[str], chunk_size: int = 100) -> List[Dict]:
    """Split sentences into token chunks with metadata."""
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    buffer, buffer_tokens = [], 0

    for sent in sentences:
        tokens = len(enc.encode(sent))
        if buffer_tokens + tokens > chunk_size:
            chunks.append(" ".join(buffer))
            buffer, buffer_tokens = [], 0
        buffer.append(sent)
        buffer_tokens += tokens

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks

def process_files(data_dir: str, chunk_sizes=[100, 400]) -> List[Dict]:

    print("📂 Looking for files...")  
    abs_path = Path(data_dir).resolve()

    print("🔍 Searching in:", abs_path)

    for file in Path(data_dir).glob("financial_statement_fixed_*.json"):
        print("   👉", file.name)
    """Process all JSON files into chunks with metadata."""
    all_chunks = []
    for file in Path(data_dir).glob("financial_statement_fixed_*.json"):  
        if not file.exists():
            print(f"⚠️ File {file} not found, skipping.")
            continue
        rows = load_financial_json(file)
        if not rows:
            print(f"⚠️ No data found in {file.name}, skipping.")
            continue

        sentences = rows_to_sentences(rows, source=file.stem)
        print(f"📄 {file.name}: {len(sentences)} sentences extracted")

        for size in chunk_sizes:
            chunks = chunk_text(sentences, chunk_size=size)
            for i, ch in enumerate(chunks):
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "source": file.name,
                    "year": file.stem.split("_")[-1],
                    "chunk_size": size,
                    "text": ch
                })
    return all_chunks


if __name__ == "__main__":
    print(f" File processing...")
    chunks = process_files("./data/processed_files")
    os.makedirs("./data/rag_chunks", exist_ok=True)
    with open("./data/rag_chunks/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Processed {len(chunks)} chunks saved in ../data/rag_chunks/chunks.json")
