import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from transformers import AutoTokenizer

CORPUS_DIR = Path("data/raw/k8s-docs-repo/content/en/docs")
OUT_PATH = Path("data/processed/chunks.jsonl")
TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 450
OVERLAP_TOKENS = 80
MIN_CHUNK_TOKENS = 40

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


@dataclass
class Section:
    headings: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines).strip()


def strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text, count=1)


def split_into_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    heading_stack: list[str] = []
    current = Section(headings=[])
    in_code_block = False

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            current.lines.append(line)
            continue

        m = HEADING_RE.match(line) if not in_code_block else None
        if m:
            if current.lines:
                sections.append(current)
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current = Section(headings=list(heading_stack))
        else:
            current.lines.append(line)

    if current.lines:
        sections.append(current)
    return [s for s in sections if s.text()]


def chunk_tokens(tokenizer, text: str, size: int, overlap: int) -> Iterator[str]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if len(ids) <= size:
        yield text
        return
    step = size - overlap
    for start in range(0, len(ids), step):
        end = min(start + size, len(ids))
        window_ids = ids[start:end]
        if len(window_ids) < MIN_CHUNK_TOKENS and start != 0:
            break
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        yield text[char_start:char_end]
        if end >= len(ids):
            break


def iter_markdown_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*.md"):
        if p.is_file():
            yield p


def process_file(tokenizer, path: Path, corpus_root: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    body = strip_frontmatter(raw)
    sections = split_into_sections(body)
    rel = path.relative_to(corpus_root).as_posix()

    out: list[dict] = []
    chunk_idx = 0
    for section in sections:
        section_text = section.text()
        if not section_text:
            continue
        for chunk in chunk_tokens(tokenizer, section_text, CHUNK_TOKENS, OVERLAP_TOKENS):
            token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
            if token_count < MIN_CHUNK_TOKENS:
                continue
            out.append({
                "id": f"{rel}::{chunk_idx}",
                "source_path": rel,
                "headings": section.headings,
                "chunk_index": chunk_idx,
                "token_count": token_count,
                "text": chunk,
            })
            chunk_idx += 1
    return out


def main() -> None:
    if not CORPUS_DIR.exists():
        raise SystemExit(f"corpus not found at {CORPUS_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    files = list(iter_markdown_files(CORPUS_DIR))
    print(f"found {len(files)} markdown files")

    total_chunks = 0
    total_tokens = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, path in enumerate(files, 1):
            try:
                chunks = process_file(tokenizer, path, CORPUS_DIR)
            except Exception as e:
                print(f"skip {path}: {e}")
                continue
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                total_chunks += 1
                total_tokens += c["token_count"]
            if i % 200 == 0:
                print(f"  {i}/{len(files)} files, {total_chunks} chunks")

    avg = total_tokens / total_chunks if total_chunks else 0
    print(f"done: {total_chunks} chunks, avg {avg:.1f} tokens, out -> {OUT_PATH}")


if __name__ == "__main__":
    main()
    