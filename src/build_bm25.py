import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/bm25.pkl")

STOPWORDS = frozenset("""
a an and are as at be by for from has have he i if in is it its of on or
that the this to was were will with you your can how what when where
which who whom whose why do does did been being have had having
""".split())

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"chunks not found at {CHUNKS_PATH}")

    print(f"loading chunks from {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"loaded {len(chunks)} chunks")

    print("tokenizing")
    tokenized_corpus = []
    ids = []
    for c in tqdm(chunks):
        tokenized_corpus.append(tokenize(c["text"]))
        ids.append(c["id"])

    avg_len = sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus)
    print(f"avg tokens per chunk (post-filter): {avg_len:.1f}")

    print("building BM25 index")
    bm25 = BM25Okapi(tokenized_corpus)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": ids,
            "tokenizer_version": "v1",
        }, f)

    size_mb = INDEX_PATH.stat().st_size / 1024 / 1024
    print(f"done: index saved to {INDEX_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()