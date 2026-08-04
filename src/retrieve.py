# # import argparse
# # import json
# # from pathlib import Path

# # import chromadb
# # from chromadb.config import Settings
# # from sentence_transformers import SentenceTransformer

# # CHROMA_DIR = Path("data/processed/chroma")
# # COLLECTION_NAME = "k8s_docs"
# # MODEL_NAME = "BAAI/bge-small-en-v1.5"
# # DEFAULT_K = 5


# # class Retriever:
# #     def __init__(self, chroma_dir: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME, model_name: str = MODEL_NAME):
# #         self.model = SentenceTransformer(model_name)
# #         self.client = chromadb.PersistentClient(
# #             path=str(chroma_dir),
# #             settings=Settings(anonymized_telemetry=False),
# #         )
# #         self.collection = self.client.get_collection(collection_name)

# #     def search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
# #         embedding = self.model.encode(
# #             [query],
# #             normalize_embeddings=True,
# #             convert_to_numpy=True,
# #         )[0].tolist()

# #         result = self.collection.query(
# #             query_embeddings=[embedding],
# #             n_results=k,
# #             include=["documents", "metadatas", "distances"],
# #         )

# #         hits = []
# #         for i in range(len(result["ids"][0])):
# #             hits.append({
# #                 "id": result["ids"][0][i],
# #                 "text": result["documents"][0][i],
# #                 "metadata": result["metadatas"][0][i],
# #                 "distance": result["distances"][0][i],
# #                 "score": 1.0 - result["distances"][0][i],
# #             })
# #         return hits


# # def format_hit(hit: dict, i: int) -> str:
# #     md = hit["metadata"]
# #     headings = md.get("headings") or "(no heading)"
# #     preview = hit["text"].strip().replace("\n", " ")
# #     if len(preview) > 300:
# #         preview = preview[:300] + "..."
# #     return (
# #         f"[{i}] score={hit['score']:.4f}  {md['source_path']}\n"
# #         f"    section: {headings}\n"
# #         f"    {preview}"
# #     )


# # def main() -> None:
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("query", type=str, help="query string")
# #     parser.add_argument("-k", type=int, default=DEFAULT_K)
# #     parser.add_argument("--json", action="store_true", help="output raw json")
# #     args = parser.parse_args()

# #     retriever = Retriever()
# #     hits = retriever.search(args.query, k=args.k)

# #     if args.json:
# #         print(json.dumps(hits, indent=2, ensure_ascii=False))
# #         return

# #     print(f"query: {args.query}")
# #     print(f"top {len(hits)} results:\n")
# #     for i, hit in enumerate(hits, 1):
# #         print(format_hit(hit, i))
# #         print()


# # if __name__ == "__main__":
# #     main()
# import argparse
# import json
# import pickle
# from collections import defaultdict
# from pathlib import Path

# import chromadb
# import numpy as np
# import yaml
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer

# from src.build_bm25 import tokenize

# CHROMA_DIR = Path("data/processed/chroma")
# COLLECTION_NAME = "k8s_docs"
# MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BM25_INDEX_PATH = Path("data/processed/bm25.pkl")
# CHUNKS_PATH = Path("data/processed/chunks.jsonl")
# RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")
# DEFAULT_K = 5


# def load_chunks_map(path: Path = CHUNKS_PATH) -> dict[str, dict]:
#     m = {}
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             c = json.loads(line)
#             m[c["id"]] = c
#     return m


# def chunk_to_metadata(c: dict) -> dict:
#     return {
#         "source_path": c["source_path"],
#         "headings": " > ".join(c["headings"]) if c["headings"] else "",
#         "chunk_index": c["chunk_index"],
#         "token_count": c["token_count"],
#     }


# class VectorRetriever:
#     def __init__(self, chroma_dir: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME, model_name: str = MODEL_NAME):
#         self.model = SentenceTransformer(model_name)
#         self.client = chromadb.PersistentClient(
#             path=str(chroma_dir),
#             settings=Settings(anonymized_telemetry=False),
#         )
#         self.collection = self.client.get_collection(collection_name)

#     def search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
#         emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
#         r = self.collection.query(
#             query_embeddings=[emb],
#             n_results=k,
#             include=["documents", "metadatas", "distances"],
#         )
#         hits = []
#         for i in range(len(r["ids"][0])):
#             hits.append({
#                 "id": r["ids"][0][i],
#                 "text": r["documents"][0][i],
#                 "metadata": r["metadatas"][0][i],
#                 "score": 1.0 - r["distances"][0][i],
#                 "source": "vector",
#             })
#         return hits


# class BM25Retriever:
#     def __init__(self, index_path: Path = BM25_INDEX_PATH, chunks_map: dict | None = None):
#         with index_path.open("rb") as f:
#             d = pickle.load(f)
#         self.bm25 = d["bm25"]
#         self.ids = d["ids"]
#         self.chunks_map = chunks_map if chunks_map is not None else load_chunks_map()

#     def search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
#         tokens = tokenize(query)
#         if not tokens:
#             return []
#         scores = self.bm25.get_scores(tokens)
#         top_idx = np.argsort(scores)[::-1][:k]
#         hits = []
#         for i in top_idx:
#             s = float(scores[i])
#             if s <= 0:
#                 continue
#             cid = self.ids[i]
#             c = self.chunks_map[cid]
#             hits.append({
#                 "id": cid,
#                 "text": c["text"],
#                 "metadata": chunk_to_metadata(c),
#                 "score": s,
#                 "source": "bm25",
#             })
#         return hits


# class HybridRetriever:
#     def __init__(self, vector: VectorRetriever, bm25: BM25Retriever, rrf_k: int = 60, vector_k: int = 30, bm25_k: int = 30):
#         self.vector = vector
#         self.bm25 = bm25
#         self.rrf_k = rrf_k
#         self.vector_k = vector_k
#         self.bm25_k = bm25_k

#     def search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
#         v_hits = self.vector.search(query, k=self.vector_k)
#         b_hits = self.bm25.search(query, k=self.bm25_k)

#         rrf: dict[str, float] = defaultdict(float)
#         by_id: dict[str, dict] = {}
#         for rank, hit in enumerate(v_hits, 1):
#             rrf[hit["id"]] += 1.0 / (self.rrf_k + rank)
#             by_id[hit["id"]] = hit
#         for rank, hit in enumerate(b_hits, 1):
#             rrf[hit["id"]] += 1.0 / (self.rrf_k + rank)
#             if hit["id"] not in by_id:
#                 by_id[hit["id"]] = hit

#         ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
#         out = []
#         for cid, score in ranked:
#             h = dict(by_id[cid])
#             h["score"] = score
#             h["source"] = "hybrid"
#             out.append(h)
#         return out


# def load_retrieval_config(path: Path = RETRIEVAL_CONFIG_PATH) -> dict:
#     with path.open("r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def build_retriever(config: dict | None = None):
#     if config is None:
#         config = load_retrieval_config()
#     mode = config.get("mode", "hybrid")
#     if mode == "vector":
#         return VectorRetriever()
#     if mode == "bm25":
#         return BM25Retriever()
#     if mode == "hybrid":
#         chunks_map = load_chunks_map()
#         return HybridRetriever(
#             vector=VectorRetriever(),
#             bm25=BM25Retriever(chunks_map=chunks_map),
#             rrf_k=config["hybrid"]["rrf_k"],
#             vector_k=config["vector"]["top_k"],
#             bm25_k=config["bm25"]["top_k"],
#         )
#     raise ValueError(f"unknown mode: {mode}")


# def format_hit(hit: dict, i: int) -> str:
#     md = hit["metadata"]
#     headings = md.get("headings") or "(no heading)"
#     preview = hit["text"].strip().replace("\n", " ")
#     if len(preview) > 300:
#         preview = preview[:300] + "..."
#     return (
#         f"[{i}] score={hit['score']:.4f} src={hit['source']}  {md['source_path']}\n"
#         f"    section: {headings}\n"
#         f"    {preview}"
#     )


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query", type=str)
#     parser.add_argument("-k", type=int, default=DEFAULT_K)
#     parser.add_argument("--mode", type=str, default=None, help="override config: vector|bm25|hybrid")
#     parser.add_argument("--json", action="store_true")
#     args = parser.parse_args()

#     config = load_retrieval_config()
#     if args.mode:
#         config["mode"] = args.mode
#     retriever = build_retriever(config)

#     hits = retriever.search(args.query, k=args.k)

#     if args.json:
#         print(json.dumps(hits, indent=2, ensure_ascii=False))
#         return

#     print(f"query: {args.query}  (mode={config.get('mode')})")
#     print(f"top {len(hits)} results:\n")
#     for i, hit in enumerate(hits, 1):
#         print(format_hit(hit, i))
#         print()


# if __name__ == "__main__":
#     main()
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
import yaml
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.build_bm25 import tokenize
from src.rerank import Reranker

CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "k8s_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BM25_INDEX_PATH = Path("data/processed/bm25.pkl")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")
DEFAULT_K = 5


def load_chunks_map(path: Path = CHUNKS_PATH) -> dict[str, dict]:
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            m[c["id"]] = c
    return m


def chunk_to_metadata(c: dict) -> dict:
    return {
        "source_path": c["source_path"],
        "headings": " > ".join(c["headings"]) if c["headings"] else "",
        "chunk_index": c["chunk_index"],
        "token_count": c["token_count"],
    }


class VectorRetriever:
    def __init__(self, chroma_dir: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(collection_name)

    def search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
        r = self.collection.query(
            query_embeddings=[emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(r["ids"][0])):
            hits.append({
                "id": r["ids"][0][i],
                "text": r["documents"][0][i],
                "metadata": r["metadatas"][0][i],
                "score": 1.0 - r["distances"][0][i],
                "source": "vector",
            })
        return hits


class BM25Retriever:
    def __init__(self, index_path: Path = BM25_INDEX_PATH, chunks_map: dict | None = None):
        with index_path.open("rb") as f:
            d = pickle.load(f)
        self.bm25 = d["bm25"]
        self.ids = d["ids"]
        self.chunks_map = chunks_map if chunks_map is not None else load_chunks_map()

    def search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        hits = []
        for i in top_idx:
            s = float(scores[i])
            if s <= 0:
                continue
            cid = self.ids[i]
            c = self.chunks_map[cid]
            hits.append({
                "id": cid,
                "text": c["text"],
                "metadata": chunk_to_metadata(c),
                "score": s,
                "source": "bm25",
            })
        return hits


class HybridRetriever:
    def __init__(self, vector: VectorRetriever, bm25: BM25Retriever, rrf_k: int = 60, vector_k: int = 30, bm25_k: int = 30, candidates_k: int | None = None):
        self.vector = vector
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.candidates_k = candidates_k

    def search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        v_hits = self.vector.search(query, k=self.vector_k)
        b_hits = self.bm25.search(query, k=self.bm25_k)

        rrf: dict[str, float] = defaultdict(float)
        by_id: dict[str, dict] = {}
        for rank, hit in enumerate(v_hits, 1):
            rrf[hit["id"]] += 1.0 / (self.rrf_k + rank)
            by_id[hit["id"]] = hit
        for rank, hit in enumerate(b_hits, 1):
            rrf[hit["id"]] += 1.0 / (self.rrf_k + rank)
            if hit["id"] not in by_id:
                by_id[hit["id"]] = hit

        limit = self.candidates_k if self.candidates_k else k
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:limit]
        out = []
        for cid, score in ranked[:k] if not self.candidates_k else ranked:
            h = dict(by_id[cid])
            h["score"] = score
            h["source"] = "hybrid"
            out.append(h)
        return out


class HybridRerankRetriever:
    def __init__(self, hybrid: HybridRetriever, reranker: Reranker, rerank_top_n: int = 25):
        self.hybrid = hybrid
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n

    def search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        candidates = self.hybrid.search(query, k=self.rerank_top_n)
        reranked = self.reranker.rerank(query, candidates, top_n=self.rerank_top_n)
        return reranked[:k]


def load_retrieval_config(path: Path = RETRIEVAL_CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_retriever(config: dict | None = None):
    if config is None:
        config = load_retrieval_config()
    mode = config.get("mode", "hybrid_rerank")
    if mode == "vector":
        return VectorRetriever()
    if mode == "bm25":
        return BM25Retriever()
    if mode in ("hybrid", "hybrid_rerank"):
        chunks_map = load_chunks_map()
        hybrid = HybridRetriever(
            vector=VectorRetriever(),
            bm25=BM25Retriever(chunks_map=chunks_map),
            rrf_k=config["hybrid"]["rrf_k"],
            vector_k=config["vector"]["top_k"],
            bm25_k=config["bm25"]["top_k"],
            candidates_k=config["hybrid"].get("candidates_k") if mode == "hybrid_rerank" else None,
        )
        if mode == "hybrid":
            return hybrid
        reranker = Reranker(model_name=config["rerank"]["model"])
        return HybridRerankRetriever(
            hybrid=hybrid,
            reranker=reranker,
            rerank_top_n=config["rerank"]["top_n"],
        )
    raise ValueError(f"unknown mode: {mode}")


def format_hit(hit: dict, i: int) -> str:
    md = hit["metadata"]
    headings = md.get("headings") or "(no heading)"
    preview = hit["text"].strip().replace("\n", " ")
    if len(preview) > 300:
        preview = preview[:300] + "..."
    extra = ""
    if "rerank_score" in hit:
        extra = f" rerank={hit['rerank_score']:.4f} retrieval={hit['retrieval_score']:.4f}"
    return (
        f"[{i}] score={hit['score']:.4f} src={hit['source']}{extra}  {md['source_path']}\n"
        f"    section: {headings}\n"
        f"    {preview}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_retrieval_config()
    if args.mode:
        config["mode"] = args.mode
    retriever = build_retriever(config)

    hits = retriever.search(args.query, k=args.k)

    if args.json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return

    print(f"query: {args.query}  (mode={config.get('mode')})")
    print(f"top {len(hits)} results:\n")
    for i, hit in enumerate(hits, 1):
        print(format_hit(hit, i))
        print()


if __name__ == "__main__":
    main()