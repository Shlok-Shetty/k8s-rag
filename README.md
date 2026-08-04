# k8s-rag

End-to-end Retrieval-Augmented Generation over the Kubernetes documentation. Hybrid retrieval, cross-encoder reranking, citation enforcement, and a 141-question golden eval set wired into CI.

Built without LangChain or LlamaIndex — every layer is ~400 lines of Python I can defend line by line.

## What it does

Ask a Kubernetes question, get an answer grounded in the official docs with citations to the source sections. Refuses when the docs don't cover the question or when a question smuggles in a false premise.

```
$ python -m src.generate "how do I scale a deployment to 10 replicas"
Q: how do I scale a deployment to 10 replicas

A: You can scale a Deployment to 10 replicas with:
   kubectl scale deployment/nginx-deployment --replicas=10 [1]

Sources:
  [1] concepts/workloads/controllers/deployment.md | Use Case > Scaling a Deployment
  ...
```

## Pipeline

```
┌──────────────┐    ┌─────────┐    ┌───────────────┐    ┌──────────┐
│ k8s markdown │──▶│ chunker  │──▶│ bge-small     │──▶│ Chroma   │
│ (~1.7k files)│    │ 450 tok │    │ embeddings    │    │ vector   │
└──────────────┘    └─────────┘    └───────────────┘    └────┬─────┘
                        │                                    │
                        └──────▶   BM25 index   ─────────────
                                                          ▼
                                                     ┌──────────┐
       query ──▶ vector top-30 ┐                    │   RRF    │
                                ├──────────────────▶│  fusion  │
                   bm25 top-30  ┘                    └────┬─────┘
                                                          │
                                                          ▼
                                       ┌───────────────────────┐
                                       │ bge-reranker-base     │
                                       │ cross-encoder (top-5) │
                                       └────────────┬──────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │ qwen2.5:7b (Ollama)  │
                                         │ + prompt v5 (yaml)   │
                                         │ + citation check     │
                                         └──────────┬───────────┘
                                                    │
                                                    ▼
                                             cited answer
```

## Baseline (141-question golden set)

| Metric | Value | What it measures |
|---|---|---|
| recall@5 | 0.879 | gold chunk present in top-5 retrieved |
| MRR | 0.711 | mean reciprocal rank of gold chunk |
| answerable correct behavior | 0.879 | answered when it should have |
| refusal accuracy | 0.980 | refused when it should have (OOD + adversarial) |
| answer has citations | 0.912 | every claim cited to a passage |

Full report in `eval/baseline/`.

## Eval set composition

- **91 answerable**: LLM-generated Q&A from randomly sampled chunks, human-reviewed. Each maps to a `gold_chunk_id` for recall/MRR.
- **30 out-of-domain**: hand-written questions about Docker, Helm, cloud providers, certifications — things k8s docs don't cover. Tests refusal.
- **20 adversarial**: hand-written questions with false premises ("Since etcd is deprecated...", "In which release was DaemonSet removed?"). Tests refusal under pressure.

## Prompt iteration story

Baseline v1 answered too liberally (refusal 0.62). v2 over-corrected (refusal 1.00, but answered only 15% of legit questions). v3 and v4 rebalanced. v5 added explicit "MUST answer" question templates that distinguish "Does X do Y?" (genuine lookup) from "Since X does Y, ..." (unverified premise). Landed at 98% refusal + 88% legit answer accuracy.

Every prompt version is in `configs/prompts.yaml` with a `version:` field, so eval reports can be joined against it.

## CI

`.github/workflows/eval.yml` runs a 40-question sampled eval on every PR touching `src/`, `configs/`, `eval/`, or `requirements.txt`. Runs on a self-hosted GPU runner in ~4 min. Thresholds are set below full-baseline to absorb sampling variance; any real regression fails the PR.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Corpus | github.com/kubernetes/website (sparse checkout of `content/en/docs`) | Real docs engineers query; single domain |
| Chunking | 450 tokens, 80 overlap, heading-aware | bge-small has max_seq=512; heading-aware keeps chunks topically coherent |
| Embeddings | BAAI/bge-small-en-v1.5, normalized | Cosine-trained; cheap to run on GPU |
| Vector store | Chroma (persistent, no server) | Right-sized; would swap for Milvus/Weaviate at scale |
| Lexical | rank_bm25 | Custom tokenizer preserves `kube-proxy`, `v1.28`, `ConfigMap` |
| Fusion | Reciprocal Rank Fusion (k=60) | Rank-based; robust to score scale mismatch |
| Reranker | BAAI/bge-reranker-base | Trained with bge-small as first-stage |
| LLM | qwen2.5:7b via Ollama, temp=0 | Local, free, GPU-accelerated |
| Eval | Custom metrics + JSONL reports | ragas planned for offline faithfulness |

## Run it locally

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# fetch docs
git clone --filter=blob:none --sparse https://github.com/kubernetes/website.git data/raw/k8s-docs-repo
cd data/raw/k8s-docs-repo && git sparse-checkout set content/en/docs && cd ../../..

# build indexes
python -m src.ingest
python -m src.embed
python -m src.build_bm25

# ollama pull qwen2.5:7b   (once)
python -m src.generate "what is a pod"
```

Run the full eval:
```bash
python -m eval.run_eval --tag local
```

## What's next

- Ragas-based faithfulness scoring in nightly eval
- Streaming token output (perceived latency win, no quality cost)
- Golden set expansion to 200+ questions covering CRDs and API refs

## Structure

```
k8s-rag/
├── configs/
│   ├── prompts.yaml       versioned prompts
│   └── retrieval.yaml     hybrid + rerank params
├── src/
│   ├── ingest.py          heading-aware chunker
│   ├── embed.py           Chroma ingestion
│   ├── build_bm25.py      lexical index
│   ├── retrieve.py        vector + BM25 + RRF + reranker
│   ├── rerank.py          cross-encoder wrapper
│   └── generate.py        prompt + Ollama + citation enforcement
├── eval/
│   ├── golden.jsonl       141 Q&A pairs
│   ├── metrics.py         recall@k, MRR, refusal, keyword hit
│   ├── run_eval.py        eval loop → JSONL + summary
│   ├── check_thresholds.py  CI gate
│   └── baseline/          checked-in ship baseline
└── .github/workflows/
    └── eval.yml           CI: full eval on self-hosted runner
```