import json
import random
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
OUT_PATH = Path("eval/sampled_chunks.jsonl")
N_SAMPLE = 50
MIN_CHUNK_TOKENS = 150
SEED = 48


def main() -> None:
    random.seed(SEED)
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["token_count"] >= MIN_CHUNK_TOKENS:
                chunks.append(c)

    sample = random.sample(chunks, N_SAMPLE)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in sample:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {N_SAMPLE} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()