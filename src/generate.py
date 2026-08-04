# import argparse
# import json
# from pathlib import Path

# import requests
# import yaml

# from src.retrieve import build_retriever

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = "qwen2.5:7b"
# PROMPTS_PATH = Path("configs/prompts.yaml")
# DEFAULT_K = 5
# MAX_CONTEXT_CHARS = 8000


# def load_prompts(path: Path = PROMPTS_PATH) -> dict:
#     with path.open("r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def format_context(hits: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[dict]]:
#     blocks = []
#     used = []
#     total = 0
#     for i, hit in enumerate(hits, 1):
#         md = hit["metadata"]
#         header = f"[{i}] source: {md['source_path']}"
#         if md.get("headings"):
#             header += f" | section: {md['headings']}"
#         block = f"{header}\n{hit['text'].strip()}"
#         if total + len(block) > max_chars and blocks:
#             break
#         blocks.append(block)
#         used.append(hit)
#         total += len(block)
#     return "\n\n---\n\n".join(blocks), used


# def call_ollama(system: str, user: str, model: str = MODEL, temperature: float = 0.0) -> str:
#     payload = {
#         "model": model,
#         "prompt": user,
#         "system": system,
#         "stream": False,
#         "options": {
#             "temperature": temperature,
#             "num_ctx": 8192,
#         },
#     }
#     r = requests.post(OLLAMA_URL, json=payload, timeout=300)
#     r.raise_for_status()
#     return r.json()["response"].strip()


# def answer(question: str, k: int = DEFAULT_K, retriever=None) -> dict:
#     if retriever is None:
#         retriever = build_retriever()
#     hits = retriever.search(question, k=k)
#     prompts = load_prompts()
#     context, used = format_context(hits)
#     user_prompt = prompts["user_template"].format(context=context, question=question)
#     response = call_ollama(prompts["system"], user_prompt)
#     return {
#         "question": question,
#         "answer": response,
#         "prompt_version": prompts.get("version", "unknown"),
#         "citations": [
#             {
#                 "index": i + 1,
#                 "source_path": h["metadata"]["source_path"],
#                 "headings": h["metadata"].get("headings", ""),
#                 "score": h["score"],
#             }
#             for i, h in enumerate(used)
#         ],
#     }


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("question", type=str)
#     parser.add_argument("-k", type=int, default=DEFAULT_K)
#     parser.add_argument("--json", action="store_true")
#     args = parser.parse_args()

#     result = answer(args.question, k=args.k)

#     if args.json:
#         print(json.dumps(result, indent=2, ensure_ascii=False))
#         return

#     print(f"Q: {result['question']}\n")
#     print(f"A: {result['answer']}\n")
#     print("Sources:")
#     for c in result["citations"]:
#         heading = f" | {c['headings']}" if c["headings"] else ""
#         print(f"  [{c['index']}] {c['source_path']}{heading}  (score={c['score']:.3f})")


# if __name__ == "__main__":
#     main()
import argparse
import json
import re
from pathlib import Path

import requests
import yaml

from src.retrieve import build_retriever

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
PROMPTS_PATH = Path("configs/prompts.yaml")
DEFAULT_K = 5
MAX_CONTEXT_CHARS = 8000

CITATION_RE = re.compile(r"\[(\d+)\]")
REFUSAL_PHRASE = "I don't know based on the provided documentation."


def load_prompts(path: Path = PROMPTS_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_context(hits: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[dict]]:
    blocks = []
    used = []
    total = 0
    for i, hit in enumerate(hits, 1):
        md = hit["metadata"]
        header = f"[{i}] source: {md['source_path']}"
        if md.get("headings"):
            header += f" | section: {md['headings']}"
        block = f"{header}\n{hit['text'].strip()}"
        if total + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used.append(hit)
        total += len(block)
    return "\n\n---\n\n".join(blocks), used


def call_ollama(system: str, user: str, model: str = MODEL, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "prompt": user,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"].strip()


def verify_citations(response: str, n_sources: int) -> dict:
    cited = [int(m.group(1)) for m in CITATION_RE.finditer(response)]
    unique_cited = sorted(set(cited))
    invalid = [i for i in unique_cited if i < 1 or i > n_sources]
    is_refusal = REFUSAL_PHRASE in response and len(response.strip()) < len(REFUSAL_PHRASE) + 200
    return {
        "cited_indices": unique_cited,
        "invalid_indices": invalid,
        "has_citations": len(unique_cited) > 0,
        "is_refusal": is_refusal,
    }


def answer(question: str, k: int = DEFAULT_K, retriever=None) -> dict:
    if retriever is None:
        retriever = build_retriever()
    hits = retriever.search(question, k=k)
    RETRIEVAL_SCORE_THRESHOLD = 0.5
    top_score = hits[0]["score"] if hits else 0.0
    if top_score < RETRIEVAL_SCORE_THRESHOLD:
        return {
            "question": question,
            "answer": REFUSAL_PHRASE,
            "raw_answer": REFUSAL_PHRASE,
            "prompt_version": load_prompts().get("version", "unknown"),
            "verification": {"cited_indices": [], "invalid_indices": [], "has_citations": False, "is_refusal": True},
            "enforcement_action": f"forced_refusal_low_score:{top_score:.3f}",
            "citations": [],
        }
    prompts = load_prompts()
    context, used = format_context(hits)
    user_prompt = prompts["user_template"].format(context=context, question=question)
    response = call_ollama(prompts["system"], user_prompt)

    verification = verify_citations(response, n_sources=len(used))
    enforced = response
    enforcement_action = "none"
    if not verification["is_refusal"] and not verification["has_citations"]:
        enforced = REFUSAL_PHRASE + " (refused: answer contained no citations)"
        enforcement_action = "forced_refusal_no_citations"
    elif verification["invalid_indices"]:
        enforcement_action = f"warning_invalid_citations:{verification['invalid_indices']}"

    return {
        "question": question,
        "answer": enforced,
        "raw_answer": response,
        "prompt_version": prompts.get("version", "unknown"),
        "verification": verification,
        "enforcement_action": enforcement_action,
        "citations": [
            {
                "index": i + 1,
                "source_path": h["metadata"]["source_path"],
                "headings": h["metadata"].get("headings", ""),
                "score": h["score"],
            }
            for i, h in enumerate(used)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = answer(args.question, k=args.k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    if result["enforcement_action"] != "none":
        print(f"[enforcement: {result['enforcement_action']}]\n")
    print("Sources:")
    for c in result["citations"]:
        heading = f" | {c['headings']}" if c["headings"] else ""
        print(f"  [{c['index']}] {c['source_path']}{heading}  (score={c['score']:.3f})")


if __name__ == "__main__":
    main()