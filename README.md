# 🚀 Hybrid Search for RAG with Postgres, pgvector & Groq

A Retrieval-Augmented Generation (RAG) system utilizing **Postgres** and **pgvector** as a hybrid search engine (vector + semantic keyword search), powered by **FastAPI** and **Groq LLM** with function-calling.

---

![System Dashboard](./demo.png?raw=true "System Dashboard")

---

## 📖 Overview

This repository demonstrates how to build a hybrid search database retrieval mechanism for RAG applications. By combining dense vector embeddings with traditional relational keyword matching, the application retrieves relevant product context and feeds it to Groq's Large Language Models to generate accurate, context-aware responses.

### 🌐 System Architecture & RAG Flow

```mermaid
graph TD
    User([User Query]) -->|1. Search request| API[FastAPI Application]
    API -->|2. Route to service| ChatService[Chat Service]
    ChatService -->|3a. Generate embedding| Embedding[OpenAI Embedding Service]
    ChatService -->|3b. Query hybrid index| DB[(PostgreSQL + pgvector)]
    Embedding -->|Dense Vector| DB
    DB -->|Retrieved Products| ChatService
    ChatService -->|4. Formulate Prompt| LLM[Groq LLM API]
    LLM -->|5. Structured Answer| ChatService
    ChatService -->|6. JSON Response| API
    API -->|7. Dynamic Render| UI[Glassmorphic Web UI]
```

---

## 🛠️ Technology Stack

Our search engine and API are built using modern, reliable technologies:

| Category | Technology | Description |
| :--- | :--- | :--- |
| **Backend Framework** | ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white) | Lightweight, asynchronous, high-performance web framework. |
| **Database & Vector Store** | ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat-square&logo=postgresql&logoColor=white) ![pgvector](https://img.shields.io/badge/pgvector-purple?style=flat-square) | PostgreSQL database extension for storing and querying vector embeddings. |
| **Language Model** | ![Groq](https://img.shields.io/badge/Groq-00ADD8?style=flat-square) | Inference engine for function-calling and RAG context generation. |
| **Embeddings** | ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white) | For generating high-quality dense vector representations of items. |
| **ORM** | ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-CC2927?style=flat-square&logo=sqlalchemy&logoColor=white) | Object Relational Mapper for clean, Pythonic database operations. |
| **Language** | ![Python](https://img.shields.io/badge/Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54) | Version 3.8+ for core application logic. |

---

## ✨ Features

- **Hybrid Search Retrieval**: Merges dense vector cosine similarity (`pgvector`) and standard relational filtering for superior accuracy.
- **Asynchronous LLM Integration**: Fast API responses with Groq's high-speed API.
- **Function Calling**: Empowers the LLM to query the database dynamically.
- **Interactive UI**: A modern, glassmorphic chat interface to search, discover, and recommend products.
- **Auto-embedded Ingestion**: Data loading scripts automatically embed products via OpenAI's API.

---

## 🚀 Getting Started

Follow these steps to set up the project on your local machine.

### 📋 Prerequisites

- **Python**: `3.8` or higher
- **PostgreSQL**: With `pgvector` extension installed.

---

### 📦 Installation & Setup

> [!NOTE]
> Ensure you have your API keys ready for OpenAI and Groq before configuring.

#### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/placeholder-username/Hybrid-Search-For-Rag.git
cd Hybrid-Search-For-Rag

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configure Database & Extensions
Ensure your PostgreSQL server is running, then run:
```sql
-- Connect to PostgreSQL shell
psql -U postgres

-- Create the database
CREATE DATABASE rag_example_schema;

-- Connect to the database and enable the pgvector extension
\c rag_example_schema
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

#### 3. Define Environment Variables
Copy the example environment file and populate it with your keys:
```bash
cp .env.example .env
```
```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
DATABASE_NAME=rag_example_schema
DATABASE_USER=your_database_user
DATABASE_PASSWORD=your_database_password
DATABASE_URL=localhost
DATABASE_PORT=5432
```

---

### 📥 Loading Sample Data

> [!IMPORTANT]
> To load data efficiently, we temporarily disable the HNSW index constraint, load the data, and then re-enable the index.

1. **Temporarily comment out the index** in `models/product.py` (the `index_ada002 = Index(...)` block near the bottom of the file):
   ```python
   # index_ada002 = Index(
   #     "hnsw_index_for_innerproduct_product_embedding_ada002",
   #     Product.embedding,
   #     postgresql_using="hnsw",
   #     postgresql_with={"m": 16, "ef_construction": 64},
   #     postgresql_ops={"embedding": "vector_cosine_ops"},
   # )
   ```
2. **Run the ingestion script**:
   ```bash
   python scripts/load_data.py
   ```
3. **Re-enable the index** in `models/product.py` by uncommenting the lines:
   ```python
   index_ada002 = Index(
       "hnsw_index_for_innerproduct_product_embedding_ada002",
       Product.embedding,
       postgresql_using="hnsw",
       postgresql_with={"m": 16, "ef_construction": 64},
       postgresql_ops={"embedding": "vector_cosine_ops"},
   )
   ```

---

## 🔎 Retrieval Evaluation

A labelled retrieval evaluation ships with this repo:

- `scripts/eval_queries.json` — 30 queries derived from `scripts/sample_products.json`, each labelled with the expected `Product.id` values. Ids follow the insertion order of the JSON file (1-based) after `python scripts/load_data.py` runs on a fresh database.
- `scripts/evaluate.py` — measures **recall@5**, **MRR** and **p50/p95 query latency** for four retrieval modes:
  - `vector_only` — pgvector cosine distance, `ORDER BY ... LIMIT k`
  - `keyword_only` — PostgreSQL full-text search (`ts_rank_cd`)
  - `hybrid_hnsw` — hybrid RRF fusion, vector branch served by the HNSW index
  - `hybrid_no_hnsw` — hybrid RRF fusion with the HNSW index dropped (vector branch falls back to a sequential scan)

Run it against a populated database:

```bash
python scripts/evaluate.py
# optional flags: --repeats N   --json scripts/eval_results.json
```

Requirements: `OPENAI_API_KEY` set (query embeddings use `text-embedding-3-small`, the same model the app uses) and data loaded via `python scripts/load_data.py`. The script drops and rebuilds the HNSW index around each measurement block and leaves a working cosine-ops HNSW index in place when it finishes (named `eval_hnsw_embedding_on_embedding`, equivalent to the one declared in `models/product.py`). Query embeddings are computed once per query and reused across modes, so per-mode latency covers the database retrieval path; the embedding latency itself is printed separately.

### 📊 Results

Measured output of `python scripts/evaluate.py --repeats 5 --json scripts/eval_results.json` over the 30 labelled queries on the 30 loaded sample products (150 timed samples per mode; nearest-rank percentiles; single PostgreSQL 16.4 + pgvector 0.8.1 instance, local machine, 2 CPUs):

| Mode | Recall@5 | MRR | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|
| vector_only | 1.000 | 1.000 | 7.42 | 7.93 |
| keyword_only | 0.728 | 0.767 | 2.87 | 2.99 |
| hybrid_hnsw | 1.000 | 1.000 | 8.54 | 9.12 |
| hybrid_no_hnsw | 1.000 | 1.000 | 8.56 | 9.07 |

**Provenance — read before citing these numbers.** This table was produced from an actual run, not estimates. Two honest caveats:

- **Embeddings**: the run used the local `sentence-transformers/all-MiniLM-L6-v2` model (384-dim, zero-padded to the schema's 1536 dims — zero-padding preserves cosine similarity exactly and keeps the database workload at the app's real dimensionality) because the OpenAI account available at measurement time had no API credits. `text-embedding-3-small` (the model this app ships with) can be measured by re-running the same command with a funded `OPENAI_API_KEY`. Per-mode latency covers the database retrieval path only and is independent of the embedding model; the printed query-embedding latency (p50 12.17 ms / p95 16.15 ms) reflects the local CPU model and is *not* representative of OpenAI API latency.
- **Scale**: at 30 products, the HNSW index shows no latency advantage over a sequential scan (8.54 ms vs 8.56 ms p50) — it in fact costs a small constant overhead. The with/without comparison is only expected to matter at much larger tables; this repo's sample data cannot demonstrate an HNSW speedup, and the table should not be read as claiming one.

At this scale keyword-only search is the clear weak spot (recall@5 0.728): `plainto_tsquery` requires every query word to match, so paraphrases like *"water-resistant coat with a hood for rainy days"* miss the `Raincoat` entirely, while vector and hybrid modes retrieve every expected product within the top 5. Raw per-mode results from the measured run are saved in `scripts/eval_results.json`.

---

### 💻 Running the Application

Start the FastAPI web server:
```bash
uvicorn app:app --reload
```
Once running, open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

### 🐳 Docker

```bash
docker build -t hybrid-search .
docker run --env-file .env -p 8000:8000 hybrid-search
```

> The container needs a reachable PostgreSQL with the `vector` extension; point `DATABASE_URL` (and the other `DATABASE_*` variables) at it in `.env`.

---

### 🧪 Tests

Smoke tests (no database or API keys required):

```bash
pip install -r requirements-dev.txt
pytest tests/
```

---

