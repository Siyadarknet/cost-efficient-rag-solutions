import json
from groq import Groq

from app.config import settings


client = Groq(api_key=settings.GROQ_API_KEY)


def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass

    return {
        "faithfulness": 0,
        "answer_relevance": 0,
        "overall_score": 0,
        "rationale": "Judge response was not valid JSON.",
    }


def judge_answer(question: str, answer: str, contexts: list[str]) -> dict:
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an expert RAG evaluator.

Evaluate the answer using ONLY the context.

Criteria:
1. faithfulness: Is the answer supported by the context?
2. answer_relevance: Does the answer directly answer the question?

Score each from 0 to 5.

Return ONLY valid JSON in this format:

{{
  "faithfulness": 0,
  "answer_relevance": 0,
  "overall_score": 0,
  "rationale": "short explanation"
}}

Question:
{question}

Context:
{context_text}

Answer:
{answer}
"""

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        # settings.LLM_MODEL
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator. Return only JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content
    parsed = safe_json_parse(raw)

    usage = response.usage

    return {
        "faithfulness": parsed.get("faithfulness", 0),
        "answer_relevance": parsed.get("answer_relevance", 0),
        "overall_score": parsed.get("overall_score", 0),
        "rationale": parsed.get("rationale", ""),
        "judge_prompt_tokens": usage.prompt_tokens if usage else 0,
        "judge_completion_tokens": usage.completion_tokens if usage else 0,
    }