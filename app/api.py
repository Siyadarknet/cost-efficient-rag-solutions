from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
import shutil

from app.rag import rag_pipeline
from app.ingest import ingest_file, ingest_folder
from app.config import settings


api = FastAPI(title="Cost Efficient RAG API")


class AskRequest(BaseModel):
    question: str
    k: int | None = None
    category: str | None = None


@api.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store": "ChromaDB",
        "embedding_model": settings.EMBED_MODEL,
        "embedding_dim": settings.EMBED_DIM,
        "llm_model": settings.LLM_MODEL,
    }


@api.post("/ask")
def ask(request: AskRequest):
    return rag_pipeline.ask(
        question=request.question,
        k=request.k,
        category=request.category,
    )


@api.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    category: str = Form("general"),
):
    documents_dir = Path("documents")
    documents_dir.mkdir(exist_ok=True)

    file_path = documents_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = ingest_file(str(file_path), category=category)

    return result


@api.post("/ingest-folder")
def ingest_folder_api(category: str = "general"):
    return ingest_folder("documents", category=category)