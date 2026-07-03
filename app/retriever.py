import chromadb
from app.config import settings
from app.embeddings import embedding_model


def get_collection():
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def retrieve_chunks(
    question: str,
    k: int | None = None,
    category: str | None = None,
):
    """
    Retrieve top-k relevant chunks from ChromaDB.
    """

    if k is None:
        k = settings.TOP_K

    collection = get_collection()

    query_embedding = embedding_model.embed_text(question)

    where_filter = None
    if category:
        where_filter = {"category": category}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        similarity = 1 - distance

        if similarity < settings.SIMILARITY_THRESHOLD:
            continue

        chunks.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": distance,
                "similarity": similarity,
                "citation": f"{metadata.get('source')} page {metadata.get('page')}",
            }
        )

    return chunks