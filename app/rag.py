import time
from typing import Optional

from app.retriever import retrieve_chunks
from app.llm import generate_answer
from app.logger import log_query


class RAGPipeline:
    """
    Production-ready Retrieval-Augmented Generation pipeline.
    """

    def __init__(self):
        pass

    def ask(
        self,
        question: str,
        k: Optional[int] = None,
        category: Optional[str] = None,
    ) -> dict:
        """
        End-to-end RAG workflow.

        1. Retrieve relevant chunks
        2. Generate grounded answer
        3. Log metrics
        4. Return response
        """

        start_time = time.perf_counter()

        # -------------------------
        # Retrieve
        # -------------------------
        chunks = retrieve_chunks(
            question=question,
            k=k,
            category=category,
        )

        # -------------------------
        # Generate Answer
        # -------------------------
        result = generate_answer(
            question=question,
            chunks=chunks,
        )

        latency = time.perf_counter() - start_time

        # -------------------------
        # Logging
        # -------------------------
        log_query(
            question=question,
            latency=latency,
            retrieved_chunks=len(chunks),
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )

        # -------------------------
        # Final Response
        # -------------------------
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": len(chunks),
            "latency_seconds": round(latency, 3),
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": (
                result["prompt_tokens"]
                + result["completion_tokens"]
            ),
        }


rag_pipeline = RAGPipeline()