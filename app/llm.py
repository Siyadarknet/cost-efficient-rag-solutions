from groq import Groq
from app.config import settings


client = Groq(api_key=settings.GROQ_API_KEY)


def generate_answer(question: str, chunks: list[dict]):
    """
    Generate grounded answer using retrieved chunks.
    """

    if not chunks:
        return {
            "answer": "No relevant context found in the document corpus.",
            "sources": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    context = "\n\n".join(
        [
            f"[Source {i+1}: {chunk['citation']}]\n{chunk['text']}"
            for i, chunk in enumerate(chunks)
        ]
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the question ONLY using the provided context.
Cite the sources you used in the answer.
If the answer is not present in the context, say:
"No relevant context found in the document corpus."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You answer strictly from retrieved context and never hallucinate.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    usage = response.usage

    return {
        "answer": answer,
        "sources": [chunk["citation"] for chunk in chunks],
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }