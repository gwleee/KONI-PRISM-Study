#!/usr/bin/env python3
"""
MARS 평가용 평가셋 — AblationEval-Grounded 에 방해 청크(distractor)를 섞는다.

왜 필요한가:
  원본 평가셋은 질의당 관련 청크 3개만 준다. 이 경우 MARS 가 소스를 전부 인용해도
  정밀도가 1.0 이 되고(모든 소스가 실제로 지지하므로), 무작위 인용 통제 조건도
  같은 값을 낸다. 즉 지표가 귀속 능력을 전혀 구분하지 못한다(스모크에서 실측 확인).
  관련 청크와 무관 청크를 섞어야 "어느 것을 인용할지" 가 실제 과제가 된다.

구성: 질의당 관련 청크 3개 + 다른 문서의 방해 청크 3개 (순서 섞음, 시드 고정)
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
        # 다른 문서에서 방해 청크 수집
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
        "rationale": ("관련 청크만 주면 전부 인용해도 정밀도 1.0 이라 귀속 능력이 "
                      "측정되지 않는다. 방해 청크를 섞어 선택 과제를 만든다."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] {p}  n={len(out)}  청크/질의={len(out[0]['source_chunks'])} "
          f"(관련 {out[0]['n_relevant']} + 방해 {out[0]['n_distractor']})")


if __name__ == "__main__":
    main()
