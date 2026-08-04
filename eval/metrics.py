import re
from collections import Counter

REFUSAL_PHRASE = "I don't know based on the provided documentation."
CITATION_RE = re.compile(r"\[(\d+)\]")


def recall_at_k(retrieved_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in retrieved_ids[:k] else 0.0


def reciprocal_rank(retrieved_ids: list[str], gold_id: str) -> float:
    for i, rid in enumerate(retrieved_ids, 1):
        if rid == gold_id:
            return 1.0 / i
    return 0.0


def is_refusal(answer: str) -> bool:
    if REFUSAL_PHRASE not in answer:
        return False
    return len(answer.strip()) < len(REFUSAL_PHRASE) + 200


def refusal_correct(answer: str, expected_behavior: str) -> float:
    refused = is_refusal(answer)
    if expected_behavior == "refuse":
        return 1.0 if refused else 0.0
    if expected_behavior == "answer":
        return 1.0 if not refused else 0.0
    return 0.0


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords)


def has_citations(answer: str) -> bool:
    return bool(CITATION_RE.search(answer))


def summarize(rows: list[dict]) -> dict:
    by_cat = Counter(r["category"] for r in rows)
    ans_rows = [r for r in rows if r["category"] == "answerable"]
    refuse_rows = [r for r in rows if r["expected_behavior"] == "refuse"]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "n_total": len(rows),
        "n_by_category": dict(by_cat),
    }

    if ans_rows:
        summary["recall_at_1"] = mean([r["recall@1"] for r in ans_rows])
        summary["recall_at_5"] = mean([r["recall@5"] for r in ans_rows])
        summary["mrr"] = mean([r["mrr"] for r in ans_rows])
        summary["answerable_correct_behavior"] = mean([r["refusal_correct"] for r in ans_rows])
        summary["answerable_keyword_hit_rate"] = mean([r["keyword_hit_rate"] for r in ans_rows])
        summary["answerable_has_citations"] = mean([1.0 if r["has_citations"] else 0.0 for r in ans_rows])

    if refuse_rows:
        summary["refusal_accuracy"] = mean([r["refusal_correct"] for r in refuse_rows])

    return summary