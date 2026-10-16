#!/usr/bin/env python3
"""
Exp-Ablation v2 — building the grounded evaluation query set.

Why build a new one:
  Both requested candidates were unusable.
  - `tests/e2e_live/QA/` : does not exist in the repository.
  - `experiments/datasets/mm_openbench_v1` : **all** 480 ground_truth entries are
    `"[verification needed] expected answer for ..."` placeholders, and source_ids are
    unresolvable UUIDs (233/480 are empty altogether), so no 'source chunks' exist.
    As long as faithfulness is defined as "supported by the source chunks", it cannot be measured on this data at all.

What is used:
  `experiments/datasets/raw/text/` : 50 real Korean documents (KLUE news/wiki).
  They are split into chunks, and each query is given **real chunks** as sources.

Measuring faithfulness does not need a gold answer:
  the definition requires only (generated answer, provided source chunks). Hence
  no 'expected answers' are fabricated.

Output: data/evalset.jsonl  (query, source_chunks, doc_id, ...)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "experiments" / "datasets" / "raw" / "text"


def chunk_document(text: str, target_chars: int = 420) -> list[str]:
    """Split into chunks while respecting paragraph boundaries."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= target_chars or not cur:
            cur = f"{cur}\n{p}".strip()
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    # merge chunks that are too short into the previous chunk
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < 120:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    return merged


# Query templates — only forms answerable from the document content are used.
QUERY_TEMPLATES = [
    "{title}에 대해 핵심 내용을 요약해 주세요.",
    "{title}와 관련해 제시된 구체적인 수치나 사실을 정리해 주세요.",
    "{title}의 배경과 주요 경위를 설명해 주세요.",
    "{title}에서 언급된 주체와 그들의 입장을 정리해 주세요.",
    "{title}의 주요 쟁점은 무엇인가요?",
]


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line:
            return line[:60]
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of queries")
    ap.add_argument("--chunks-per-query", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "data" / "evalset.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(RAW.glob("*.txt"))
    if not files:
        raise SystemExit(f"source texts not found: {RAW}")

    docs = []
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        ch = chunk_document(txt)
        if len(ch) >= 2:
            docs.append({"doc_id": f.stem, "title": extract_title(txt, f.stem),
                         "chunks": ch})

    instances = []
    i = 0
    while len(instances) < args.n:
        d = docs[i % len(docs)]
        tmpl = QUERY_TEMPLATES[(i // len(docs)) % len(QUERY_TEMPLATES)]
        k = min(args.chunks_per_query, len(d["chunks"]))
        # deterministic selection (fixed seed)
        idxs = sorted(rng.sample(range(len(d["chunks"])), k))
        instances.append({
            "id": f"abl_{len(instances):04d}",
            "query": tmpl.format(title=d["title"]),
            "doc_id": d["doc_id"],
            "doc_title": d["title"],
            "source_chunks": [d["chunks"][j] for j in idxs],
            "chunk_indices": idxs,
            "n_chunks": k,
        })
        i += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in instances:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "dataset": "AblationEval-Grounded",
        "version": "v1",
        "n_instances": len(instances),
        "source_corpus": "experiments/datasets/raw/text (KLUE Korean documents)",
        "n_source_documents": len(docs),
        "chunks_per_query": args.chunks_per_query,
        "seed": args.seed,
        "builder": "paper_experiments/ablation_v2/build_evalset.py",
        "note": (
            "No gold answer — the faithfulness definition requires only (answer, source chunks), "
            "so no 'expected answers' are generated. MM-OpenBench v1 was not used because all its "
            "ground_truth entries are placeholders and its source_ids do not resolve."
        ),
    }
    (out.parent / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    avg = sum(len(c) for r in instances for c in r["source_chunks"]) / \
        sum(r["n_chunks"] for r in instances)
    print(f"[✓] {out}  n={len(instances)}  docs={len(docs)}  "
          f"mean chunk length={avg:.0f} chars")
    print(f"    example query: {instances[0]['query']}")
    print(f"    example chunk: {instances[0]['source_chunks'][0][:100]}...")


if __name__ == "__main__":
    main()
