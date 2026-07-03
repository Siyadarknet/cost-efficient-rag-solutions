import math
import re


def normalize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def exact_match(prediction: str, gold: str):
    return int(normalize(prediction) == normalize(gold))


def f1_score(prediction: str, gold: str):
    pred = normalize(prediction).split()
    gold = normalize(gold).split()

    common = set(pred) & set(gold)

    if len(common) == 0:
        return 0

    precision = len(common) / len(pred)
    recall = len(common) / len(gold)

    return 2 * precision * recall / (precision + recall)


def hit_rate(retrieved_texts: list[str], relevant_keywords: list[str]) -> int:
    for text in retrieved_texts:
        text_lower = text.lower()
        for keyword in relevant_keywords:
            if keyword.lower() in text_lower:
                return 1
    return 0


def recall_at_k(retrieved_texts: list[str], relevant_keywords: list[str]) -> float:
    if not relevant_keywords:
        return 0.0

    found = 0

    combined_text = " ".join(retrieved_texts).lower()

    for keyword in relevant_keywords:
        if keyword.lower() in combined_text:
            found += 1

    return found / len(relevant_keywords)


def mrr(retrieved_texts: list[str], relevant_keywords: list[str]) -> float:
    for rank, text in enumerate(retrieved_texts, start=1):
        text_lower = text.lower()

        for keyword in relevant_keywords:
            if keyword.lower() in text_lower:
                return 1 / rank

    return 0.0


def ndcg_at_k(retrieved_texts: list[str], relevant_keywords: list[str]) -> float:
    dcg = 0.0

    for i, text in enumerate(retrieved_texts):
        relevance = 0
        text_lower = text.lower()

        for keyword in relevant_keywords:
            if keyword.lower() in text_lower:
                relevance = 1
                break

        if relevance:
            dcg += 1 / math.log2(i + 2)

    ideal_relevant = min(len(relevant_keywords), len(retrieved_texts))

    idcg = sum(
        1 / math.log2(i + 2)
        for i in range(ideal_relevant)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def context_precision(retrieved_texts: list[str], relevant_keywords: list[str]) -> float:
    if not retrieved_texts:
        return 0.0

    relevant_chunks = 0

    for text in retrieved_texts:
        text_lower = text.lower()

        if any(keyword.lower() in text_lower for keyword in relevant_keywords):
            relevant_chunks += 1

    return relevant_chunks / len(retrieved_texts)


def evaluate_retrieval(retrieved_texts: list[str], relevant_keywords: list[str]) -> dict:
    return {
        "hit_rate": hit_rate(retrieved_texts, relevant_keywords),
        "recall_at_k": recall_at_k(retrieved_texts, relevant_keywords),
        "mrr": mrr(retrieved_texts, relevant_keywords),
        "ndcg_at_k": ndcg_at_k(retrieved_texts, relevant_keywords),
        "context_precision": context_precision(retrieved_texts, relevant_keywords),
    }