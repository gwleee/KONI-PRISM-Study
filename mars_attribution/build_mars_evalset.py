#!/usr/bin/env python3
"""
Evaluation set for MARS — mixes distractor chunks into AblationEval-Grounded.

Why it is needed:
  The original set gives only 3 relevant chunks per query. Then even if MARS cites every source,
  precision is 1.0 (every source actually supports the answer), and the random-citation control
  yields the same value, so the metric cannot distinguish attribution ability (confirmed in a smoke run).
  Only by mixing relevant and unrelated chunks does "which ones to cite" become a real task.

Composition: 3 relevant chunks per query + 3 distractor chunks from other documents (shuffled, fixed seed)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "ablation_v2" / "data" / "evalset.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-distractors", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "data" / "evalset_mars.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(args.seed)
    out = []
    for i, r in enumerate(rows):
        # collect distractor chunks from other documents
        pool = [(o["doc_id"], c) for o in rows if o["doc_id"] != r["doc_id"]
                for c in o["source_chunks"]]
        picked = rng.sample(pool, min(args.n_distractors, len(pool)))
        rel = [{"text": c, "relevant": True, "doc_id": r["doc_id"]} for c in r["source_chunks"]]
        dis = [{"text": c, "relevant": False, "doc_id": d} for d, c in picked]
        allc = rel + dis
        rng.shuffle(allc)
        out.append({
            "id": r["id"], "query": r["query"], "doc_id": r["doc_id"],
            "doc_title": r["doc_title"],
            "source_chunks": [c["text"] for c in allc],
            "relevance_mask": [c["relevant"] for c in allc],
            "n_relevant": len(rel), "n_distractor": len(dis),
        })

    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (p.parent / "metadata.json").write_text(json.dumps({
        "dataset": "AblationEval-Grounded + distractors (MARS attribution)",
        "n_instances": len(out),
        "chunks_per_query": len(out[0]["source_chunks"]),
        "n_relevant_per_query": out[0]["n_relevant"],
        "n_distractor_per_query": out[0]["n_distractor"],
        "seed": args.seed,
        "base": str(SRC),
        "rationale": ("With only relevant chunks, citing everything gives precision 1.0, so attribution "
                      "ability is not measured. Mixing in distractor chunks creates a selection task."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] {p}  n={len(out)}  chunks/query={len(out[0]['source_chunks'])} "
          f"(relevant {out[0]['n_relevant']} + distractor {out[0]['n_distractor']})")


if __name__ == "__main__":
    main()
