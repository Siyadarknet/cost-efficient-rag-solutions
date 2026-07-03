import json
import time
from pathlib import Path
from app.judge import judge_answer
import pandas as pd
import numpy as np
from app.retriever import retrieve_chunks
from app.rag import rag_pipeline
# from app.metrics import evaluate_retrieval
from app.metrics import (
    evaluate_retrieval,
    exact_match,
    f1_score,
)
USE_LLM_JUDGE = True
USE_GENERATOR_LLM = True

def load_questions(path: str = "evaluation/questions.json"):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_evaluation(
    questions_path: str = "evaluation/questions.json",
    output_path: str = "evaluation/results.csv",
    k: int = 5,
    category: str | None = None,
):
    questions = load_questions(questions_path)

    rows = []

    for item in questions:
        start = time.perf_counter()

        # question = item["question"]
        # relevant_keywords = item.get("relevant_keywords", [])
        question = item["question"]
        relevant_keywords = item.get("relevant_keywords", [])

        gold_answer = item.get("gold_answer", "")

        chunks = retrieve_chunks(
            question=question,
            k=k,
            category=category,
        )

        retrieved_texts = [chunk["text"] for chunk in chunks]

        retrieval_scores = evaluate_retrieval(
            retrieved_texts=retrieved_texts,
            relevant_keywords=relevant_keywords,
        )

        # rag_result = rag_pipeline.ask(
        #     question=question,
        #     k=k,
        #     category=category,
        # )
        if USE_GENERATOR_LLM:
            rag_result = rag_pipeline.ask(
                question=question,
                k=k,
                category=category,
            )
        else:
            rag_result = {
                "answer": "LLM generation disabled during evaluation because API quota/rate limit was reached.",
                "retrieved_chunks": len(chunks),
                "sources": [chunk["citation"] for chunk in chunks],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        # judge_result = judge_answer(
        #     question=question,
        #     answer=rag_result["answer"],
        #     contexts=retrieved_texts,
        # )
        if USE_LLM_JUDGE:
            judge_result = judge_answer(
            question=question,
            answer=rag_result["answer"],
            contexts=retrieved_texts,
        )
        else:
            judge_result = {
                "faithfulness": 0,
                "answer_relevance": 0,
                "overall_score": 0,
                "rationale": "LLM Judge Disabled",
                "judge_prompt_tokens": 0,
                "judge_completion_tokens": 0,
            }

        em = exact_match(rag_result["answer"],gold_answer, )

        f1 = f1_score(rag_result["answer"],gold_answer)
        

        latency = time.perf_counter() - start

        rows.append({
            "id": item["id"],
            "question": question,
            "answer": rag_result["answer"],
            "exact_match": em,
            "f1_score": round(f1,3),
            "retrieved_chunks": rag_result["retrieved_chunks"],
            "latency_seconds": round(latency, 3),
            "hit_rate": retrieval_scores["hit_rate"],
            "recall_at_k": retrieval_scores["recall_at_k"],
            "mrr": retrieval_scores["mrr"],
            "ndcg_at_k": retrieval_scores["ndcg_at_k"],
            "context_precision": retrieval_scores["context_precision"],
            "prompt_tokens": rag_result["prompt_tokens"],
            "completion_tokens": rag_result["completion_tokens"],
            "total_tokens": rag_result["total_tokens"],
            "faithfulness": judge_result["faithfulness"],
            "answer_relevance": judge_result["answer_relevance"],
            "answer_overall_score": judge_result["overall_score"],
            "judge_rationale": judge_result["rationale"],
            # "judge_tokens": judge_result["judge_prompt_tokens"] + judge_result["judge_completion_tokens"],
            "judge_prompt_tokens": judge_result["judge_prompt_tokens"],
            "judge_completion_tokens": judge_result["judge_completion_tokens"],
            "judge_total_tokens": judge_result["judge_prompt_tokens"] + judge_result["judge_completion_tokens"],
            "citation_count": len(rag_result["sources"]),
            "sources": "; ".join(rag_result["sources"]),
        })

    df = pd.DataFrame(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # summary = {
    #     "questions": len(df),
    #     "avg_hit_rate": round(df["hit_rate"].mean(), 3),
    #     "avg_recall_at_k": round(df["recall_at_k"].mean(), 3),
    #     "avg_mrr": round(df["mrr"].mean(), 3),
    #     "avg_ndcg_at_k": round(df["ndcg_at_k"].mean(), 3),
    #     "avg_context_precision": round(df["context_precision"].mean(), 3),
    #     "avg_latency_seconds": round(df["latency_seconds"].mean(), 3),
    #     "total_tokens": int(df["total_tokens"].sum()),
    # }


#     summary = {

#     "questions": len(df),

#     "avg_hit_rate":
#         round(df["hit_rate"].mean(),3),

#     "avg_recall_at_k":
#         round(df["recall_at_k"].mean(),3),

#     "avg_mrr":
#         round(df["mrr"].mean(),3),

#     "avg_ndcg_at_k":
#         round(df["ndcg_at_k"].mean(),3),

#     "avg_context_precision":
#         round(df["context_precision"].mean(),3),

#     "avg_faithfulness":
#         round(df["faithfulness"].mean(),3),

#     "avg_answer_relevance":
#         round(df["answer_relevance"].mean(),3),

#     "avg_overall_score":
#         round(df["answer_overall_score"].mean(),3),

#     "avg_latency_seconds":
#         round(df["latency_seconds"].mean(),3),

#     "p50_latency":
#         round(np.percentile(df["latency_seconds"],50),3),

#     "p95_latency":
#         round(np.percentile(df["latency_seconds"],95),3),

#     "min_latency":
#         round(df["latency_seconds"].min(),3),

#     "max_latency":
#         round(df["latency_seconds"].max(),3),

#     "total_llm_tokens":
#         int(df["total_tokens"].sum()),

#     "total_judge_tokens":
#         int(df["judge_total_tokens"].sum())
# }

    summary = {
        "questions": len(df),

        "avg_hit_rate": round(df["hit_rate"].mean(), 3),
        "avg_recall_at_k": round(df["recall_at_k"].mean(), 3),
        "avg_mrr": round(df["mrr"].mean(), 3),
        "avg_ndcg_at_k": round(df["ndcg_at_k"].mean(), 3),
        "avg_context_precision": round(df["context_precision"].mean(), 3),

        "avg_exact_match": round(df["exact_match"].mean(), 3),
        "avg_f1_score": round(df["f1_score"].mean(), 3),

        "avg_faithfulness": round(df["faithfulness"].mean(), 3),
        "avg_answer_relevance": round(df["answer_relevance"].mean(), 3),
        "avg_overall_score": round(df["answer_overall_score"].mean(), 3),

        "avg_latency_seconds": round(df["latency_seconds"].mean(), 3),

        "p50_latency": round(np.percentile(df["latency_seconds"], 50), 3),
        "p95_latency": round(np.percentile(df["latency_seconds"], 95), 3),

        "min_latency": round(df["latency_seconds"].min(), 3),
        "max_latency": round(df["latency_seconds"].max(), 3),

        "total_llm_tokens": int(df["total_tokens"].sum()),
        "total_judge_tokens": int(df["judge_total_tokens"].sum())
    }

    summary_path = "evaluation/summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    return {
        "results_file": output_path,
        "summary_file": summary_path,
        "summary": summary,
    }


if __name__ == "__main__":
    result = run_evaluation(
        questions_path="evaluation/questions.json",
        output_path="evaluation/results.csv",
        k=5,
        category="general",
    )
    summary = result["summary"]
    print("="*60)
    print("Evaluation Completed")
    print("="*60)

    print(f"Questions           : {summary['questions']}")
    print(f"Recall@k            : {summary['avg_recall_at_k']}")
    print(f"MRR                 : {summary['avg_mrr']}")
    print(f"nDCG                : {summary['avg_ndcg_at_k']}")
    print(f"Context Precision   : {summary['avg_context_precision']}")
    print(f"Faithfulness        : {summary['avg_faithfulness']}")
    print(f"Answer Relevance    : {summary['avg_answer_relevance']}")
    print(f"P50 Latency         : {summary['p50_latency']} sec")
    print(f"P95 Latency         : {summary['p95_latency']} sec")
    print(f"Total Tokens        : {summary['total_llm_tokens']}")
    print("="*60)