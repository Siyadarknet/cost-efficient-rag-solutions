# Cost-Efficient RAG Application

A production-ready Retrieval-Augmented Generation (RAG) application built using **FastAPI**, **ChromaDB**, **Sentence Transformers**, and **Llama 3.3 (Groq)**.

This project demonstrates that a low-cost vector database can provide competitive retrieval performance while significantly reducing infrastructure costs compared to managed vector databases.



# Features

PDF / HTML / Markdown ingestion
Configurable chunk size and overlap
Duplicate-safe ingestion (idempotent)
ChromaDB persistent vector database
Top-k semantic retrieval
Grounded LLM answers with citations
"No relevant context" handling
REST API with FastAPI
Retrieval evaluation
Cost comparison
Query logging
Environment-based configuration



# Technology Stack

| Component | Technology |
|------------|------------|
| API | FastAPI |
| Vector Store | ChromaDB |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Embeddings | 384 Dimensions |
| LLM | Llama 3.3 via Groq |
| Evaluation | Custom Retrieval Metrics |
| Logging | Loguru |



# Project Structure

```
cost-efficient-rag/

app/
api.py
config.py
embeddings.py
ingest.py
retriever.py
llm.py
rag.py
logger.py
metrics.py
evaluation.py

documents/

evaluation/
questions.json
results.csv
summary.json

logs/

main.py
requirements.txt
README.md
.env
```


# Installation

Clone repository

```bash
git clone <repository-url>

cd cost-efficient-rag
```

Install dependencies

```bash
pip install -r requirements.txt
```


# Environment Variables

```
GROQ_API_KEY=xxxxxxxxxxxxxxxx

EMBED_MODEL=BAAI/bge-small-en-v1.5

CHROMA_PATH=./chroma_db

COLLECTION_NAME=rag_documents

CHUNK_SIZE=800

CHUNK_OVERLAP=150

TOP_K=5

SIMILARITY_THRESHOLD=0.45

LLM_MODEL=llama-3.3-70b-versatile
```


# Running API

```bash
python -m uvicorn app.api:api --reload
```

Swagger

```
http://127.0.0.1:8000/docs
```


# Ingest Documents

POST

```
/ingest
```

Upload

PDF
HTML
Markdown


# Ask Questions

POST

```
/ask
```

Example

```json
{
    "question":"What is SIP?",
    "k":5,
    "category":"finance"
}
```


# Evaluation

Run

```bash
python -m app.evaluation
```

Generated files

```
evaluation/results.csv

evaluation/summary.json
```

Metrics

Recall@k
Hit Rate
MRR
nDCG@k
Context Precision


# Cost Comparison

|Vectors|ChromaDB|Managed DB|
|-------|---------|----------|
|100K|$0|≈$70/month|
|1M|≈$5–10/month (disk)|≈$300/month|
|10M|≈$20/month SSD|≈$1000+/month|

Assumptions

 Local SSD
 Single server
 No replication
 No autoscaling


# Latency

Typical

Embedding

40–60 ms

Retrieval

20–40 ms

LLM

1–2 sec

Total

≈1.5 sec


# Trade-offs

Advantages

 Zero infrastructure cost
 Simple deployment
 Persistent local storage
 Easy experimentation

Limitations

 Single-node deployment
 No built-in replication
 Manual backups
 Limited horizontal scalability


# When to Switch to Managed Vector DB

Use a managed vector database when:

 Dataset exceeds tens of millions of vectors
 High availability is required
 Multi-region deployment is needed
 Automatic scaling is important
 Enterprise-grade monitoring is required


# Future Improvements

 Hybrid Search (BM25 + Vector)
 Cross Encoder Re-ranking
 OCR for scanned PDFs
 Streaming Responses
 Redis Query Cache
 Async Embedding Pipeline
 Docker & Kubernetes Deployment
 Authentication & Rate Limiting


# License

MIT License