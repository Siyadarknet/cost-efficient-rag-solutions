# Cost-Efficient RAG (Retrieval-Augmented Generation)

A production-oriented Retrieval-Augmented Generation (RAG) system that ingests documents, stores embeddings in ChromaDB, retrieves the most relevant chunks using semantic search, and generates grounded answers using Llama 3.3 through the Groq API.

This project demonstrates how an open-source vector database (ChromaDB) can provide accurate retrieval with significantly lower infrastructure cost than managed vector databases.


# Features

PDF document ingestion
Automatic document chunking
Duplicate-safe ingestion
Persistent ChromaDB vector database
Semantic search using BAAI/bge-small-en-v1.5 embeddings
Grounded answer generation using Groq Llama 3.3
Retrieval evaluation
LLM-as-Judge evaluation
Latency benchmarking
Cost-efficient architecture
FastAPI REST API
Swagger documentation



# Project Structure

```
cost_efficient_rag/
│
├── app/
│   ├── api.py
│   ├── config.py
│   ├── embeddings.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── rag.py
│   ├── llm.py
│   ├── judge.py
│   ├── metrics.py
│   ├── evaluation.py
│   ├── logger.py
│   └── utils.py
│
├── chroma_db/
│
├── documents/
│
├── evaluation/
│   ├── questions.json
│   ├── results.csv
│   └── summary.json
│
├── logs/
│
├── requirements.txt
├── .env
├── main.py
└── README.md
```



# System Architecture

```
                Documents
                    │
                    ▼
            Document Loader
                    │
                    ▼
             Text Chunking
                    │
                    ▼
        BAAI/bge-small-en-v1.5
              Embeddings
                    │
                    ▼
               ChromaDB
                    │
                    ▼
          Semantic Retrieval
                    │
                    ▼
          Retrieved Context
                    │
                    ▼
       Llama 3.3 (Groq API)
                    │
                    ▼
          Grounded Response
```



# Technologies Used

| Component | Technology |
|------------|------------|
| Language | Python 3.13 |
| API | FastAPI |
| Vector Database | ChromaDB |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| LLM | Llama-3.3-70B-Versatile (Groq) |
| Evaluation | Custom Metrics + LLM Judge |
| Data Processing | Pandas, NumPy |
| Logging | Python Logging |



# Installation

Clone repository

```bash
git clone <repository-url>

cd cost_efficient_rag
```

Install dependencies

```bash
pip install -r requirements.txt
```



# Environment Variables

Create `.env`

```env
GROQ_API_KEY=your_api_key

LLM_MODEL=llama-3.3-70b-versatile

EMBED_MODEL=BAAI/bge-small-en-v1.5

CHROMA_DB=./chroma_db
```


# Running API

```bash
python -m uvicorn app.api:api --reload
```

Open Swagger

```
http://127.0.0.1:8000/docs
```


# API Endpoints

## Health Check

GET

```
/health
```


## Ingest Document

POST

```
/ingest
```

Upload PDF file.


## Ingest Folder

POST

```
/ingest-folder
```

Example

```
category=general
```


## Ask Question

POST

```
/ask
```

Example

```json
{
    "question":"What is the purpose of this document?",
    "k":5,
    "category":"general"
}
```


# Retrieval Pipeline

1. User submits a question.
2. Question embedding is generated.
3. ChromaDB retrieves Top-K relevant chunks.
4. Retrieved chunks are passed to the LLM.
5. LLM generates a grounded answer using only retrieved context.
6. Source citations are returned with the answer.


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

# Evaluation Metrics

## Retrieval Metrics

Hit Rate
Recall@K
Mean Reciprocal Rank (MRR)
nDCG@K
Context Precision


## LLM-as-Judge Metrics

Faithfulness
Answer Relevance
Overall Score


## Performance Metrics

Average Latency
P50 Latency
P95 Latency
Minimum Latency
Maximum Latency
Token Usage


# Cost Analysis

This project uses ChromaDB instead of a managed vector database.

Advantages

Zero licensing cost
Runs locally
Persistent storage
Easy deployment
Suitable for small and medium datasets

Compared with managed vector databases, this significantly reduces operational cost while maintaining competitive retrieval performance.


# Current Results

Example evaluation

| Metric | Value |
|----------|-------|
| Questions | 20 |
| Recall@K | 0.567 |
| MRR | 0.548 |
| nDCG | 0.624 |
| Context Precision | 0.400 |
| P50 Latency | 0.074 sec |
| P95 Latency | 0.112 sec |


# LLM-as-Judge Note

The project includes an LLM-based evaluation module (`judge.py`) to measure:

Faithfulness
Answer Relevance
Overall Answer Quality

During development, this module can be disabled to avoid exceeding the Groq free-tier token limits.

Enable it by setting:

```python
USE_LLM_JUDGE = True
```

# Limitations

Groq free-tier API has token rate limits.
Retrieval quality depends on chunk size and embedding quality.
ChromaDB is designed for single-node deployments.
Large datasets may require distributed vector databases.


# Future Improvements

Hybrid Search (BM25 + Vector Search)
Cross-Encoder Re-ranking
OCR support for scanned PDFs
Streaming responses
Docker deployment
Kubernetes deployment
Redis caching
User authentication
Multi-document collections


# Author

**Sahil Mehta**

B.Tech Computer Science Engineering


# License

This project is developed for educational and research purposes.

MIT License.