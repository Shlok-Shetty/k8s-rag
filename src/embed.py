import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "k8s_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def to_metadata(chunk: dict) -> dict:
    return {
        "source_path": chunk["source_path"],
        "headings": " > ".join(chunk["headings"]) if chunk["headings"] else "",
        "chunk_index": chunk["chunk_index"],
        "token_count": chunk["token_count"],
    }


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"chunks not found at {CHUNKS_PATH}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading model {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    device = model.device
    print(f"model loaded on {device}")

    print(f"reading {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"loaded {len(chunks)} chunks")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"deleting existing collection {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="embedding"):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        collection.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[to_metadata(c) for c in batch],
        )

    count = collection.count()
    print(f"done: {count} vectors in collection '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()