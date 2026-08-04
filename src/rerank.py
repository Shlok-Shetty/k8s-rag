from pathlib import Path

from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
        if not candidates:
            return []
        pool = candidates[:top_n] if top_n else candidates
        pairs = [(query, c["text"]) for c in pool]
        scores = self.model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        for c, s in zip(pool, scores):
            c = dict(c)
        ranked = []
        for c, s in zip(pool, scores):
            r = dict(c)
            r["rerank_score"] = float(s)
            r["retrieval_score"] = c["score"]
            r["score"] = float(s)
            r["source"] = c.get("source", "unknown") + "+rerank"
            ranked.append(r)
        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked