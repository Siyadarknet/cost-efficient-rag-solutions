import hashlib
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader

from app.config import settings
from app.embeddings import embedding_model
from app.logger import logger


def create_chunk_id(text: str, source: str, page: int) -> str:
    """
    Create unique id for each chunk.
    This prevents duplicate vectors during re-ingestion.
    """
    raw = f"{source}-{page}-{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_document(file_path: str):
    """
    Load PDF, HTML, MD, or TXT file.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in [".html", ".htm"]:
        loader = BSHTMLLoader(str(path))
    elif suffix in [".md", ".txt"]:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return loader.load()


def split_documents(documents):
    """
    Split documents into chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents(documents)


def get_collection():
    """
    Get ChromaDB persistent collection.
    """
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def ingest_file(file_path: str, category: str = "general") -> Dict[str, Any]:
    """
    Ingest one file into ChromaDB.
    """

    logger.info(f"Starting ingestion: {file_path}")

    docs = load_document(file_path)
    chunks = split_documents(docs)
    collection = get_collection()

    inserted = 0
    skipped = 0

    for index, chunk in enumerate(chunks):
        text = chunk.page_content.strip()

        if not text:
            continue

        source = Path(file_path).name
        page = chunk.metadata.get("page", 0)

        chunk_id = create_chunk_id(text, source, page)

        existing = collection.get(ids=[chunk_id])

        if existing and existing.get("ids"):
            skipped += 1
            continue

        vector = embedding_model.embed_text(text)

        metadata = {
            "source": source,
            "page": page,
            "chunk_index": index,
            "category": category,
            "embedding_model": settings.EMBED_MODEL,
            "embedding_dim": settings.EMBED_DIM,
        }

        collection.add(
            ids=[chunk_id],
            documents=[text],
            embeddings=[vector],
            metadatas=[metadata],
        )

        inserted += 1

    logger.info(
        f"Ingestion completed: inserted={inserted}, skipped={skipped}, total_chunks={len(chunks)}"
    )

    return {
        "file": file_path,
        "total_chunks": len(chunks),
        "inserted": inserted,
        "skipped_duplicates": skipped,
        "collection_count": collection.count(),
    }


def ingest_folder(folder_path: str = "documents", category: str = "general"):
    """
    Ingest all supported files from a folder.
    """

    folder = Path(folder_path)

    supported = [".pdf", ".html", ".htm", ".md", ".txt"]

    results = []

    for file in folder.rglob("*"):
        if file.suffix.lower() in supported:
            result = ingest_file(str(file), category=category)
            results.append(result)

    return results