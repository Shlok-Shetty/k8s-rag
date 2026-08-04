import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from src.generate import answer as generate_answer
from src.retrieve import build_retriever
from eval.metrics import (
    recall_at_k,
    reciprocal_rank,
    refusal_correct,
    keyword_hit_rate,
    has_citations,
    summarize,
)

GOLDEN_PATH = Path("eval/golden.jsonl")
REPORTS_DIR = Path("eval/reports")
K_RETRIEVE = 10
K_FINAL = 5


def load_golden(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_one(row: dict, retriever, k_retrieve: int, k_final: int) -> dict:
    q = row["question"]

    hits = retriever.search(q, k=k_retrieve)
    retrieved_ids = [h["id"] for h in hits]

    result = generate_answer(q, k=k_final, retriever=retriever)
    resp = result["answer"]

    out = dict(row)
    out["retrieved_ids"] = retrieved_ids
    out["top_scores"] = [round(h["score"], 4) for h in hits[:k_final]]
    out["answer"] = resp
    out["raw_answer"] = result.get("raw_answer", resp)
    out["enforcement_action"] = result.get("enforcement_action", "none")
    out["has_citations"] = has_citations(resp)
    out["refusal_correct"] = refusal_correct(resp, row["expected_behavior"])
    out["keyword_hit_rate"] = keyword_hit_rate(resp, row.get("expected_answer_keywords", []))

    if row.get("gold_chunk_id"):
        gid = row["gold_chunk_id"]
        out["recall@1"] = recall_at_k(retrieved_ids, gid, 1)
        out["recall@5"] = recall_at_k(retrieved_ids, gid, 5)
        out["recall@10"] = recall_at_k(retrieved_ids, gid, 10)
        out["mrr"] = reciprocal_rank(retrieved_ids, gid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--categories", type=str, default=None, help="comma-separated: answerable,out_of_domain,adversarial")
    parser.add_argument("--tag", type=str, default="run", help="report tag")
    args = parser.parse_args()

    rows = load_golden(GOLDEN_PATH)
    if args.categories:
        cats = set(args.categories.split(","))
        rows = [r for r in rows if r["category"] in cats]
    if args.limit:
        rows = rows[: args.limit]

    print(f"eval set: {len(rows)} rows")

    print("building retriever")
    retriever = build_retriever()

    results = []
    t0 = time.time()
    for row in tqdm(rows, desc="eval"):
        try:
            results.append(run_one(row, retriever, K_RETRIEVE, K_FINAL))
        except Exception as e:
            print(f"failed {row['id']}: {e}")
            results.append({**row, "error": str(e)})
    elapsed = time.time() - t0

    summary = summarize([r for r in results if "error" not in r])
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["avg_seconds_per_query"] = round(elapsed / max(len(results), 1), 2)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"{ts}_{args.tag}.jsonl"
    summary_path = REPORTS_DIR / f"{ts}_{args.tag}_summary.json"

    with report_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nfull results -> {report_path}")
    print(f"summary     -> {summary_path}")


if __name__ == "__main__":
    main()