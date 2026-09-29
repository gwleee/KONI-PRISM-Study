#!/usr/bin/env python3
"""
Exp-Ablation v2 — 근거 기반(grounded) 평가 질의셋 구성.

왜 새로 만드는가:
  요구된 두 후보가 모두 사용 불가였다.
  - `tests/e2e_live/QA/` : 저장소에 존재하지 않는다.
  - `experiments/datasets/mm_openbench_v1` : ground_truth 480건이 **전부**
    `"[검증 필요] ...에 대한 예상 답변"` 플레이스홀더이고, source_ids 는
    해소되지 않는 UUID(233/480 은 아예 비어 있음)라 '출처 청크'가 존재하지 않는다.
    faithfulness 를 "출처 청크 지지 여부"로 정의하는 한 이 데이터로는 측정 자체가 불가능하다.

무엇을 쓰는가:
  `experiments/datasets/raw/text/` 의 실제 한국어 문서 50건(KLUE 뉴스/위키).
  이를 청크로 자르고, 각 질의에 **실제 청크**를 출처로 붙인다.

faithfulness 측정에 gold answer 는 필요하지 않다:
  정의가 (생성 답변, 제공된 출처 청크) 만을 요구하기 때문이다. 따라서
  '예상 답변'을 지어내지 않는다.

출력: data/evalset.jsonl  (query, source_chunks, doc_id, ...)
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
    """문단 경계를 존중하며 청크로 분할."""
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
    # 너무 짧은 청크는 앞 청크에 병합
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < 120:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    return merged


# 질의 템플릿 — 문서 내용에 근거해 답할 수 있는 형태만 사용한다.
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
    ap.add_argument("--n", type=int, default=60, help="질의 수")
    ap.add_argument("--chunks-per-query", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "data" / "evalset.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(RAW.glob("*.txt"))
    if not files:
        raise SystemExit(f"원문 없음: {RAW}")

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
        # 결정론적 선택 (시드 고정)
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
        "source_corpus": "experiments/datasets/raw/text (KLUE 한국어 문서)",
        "n_source_documents": len(docs),
        "chunks_per_query": args.chunks_per_query,
        "seed": args.seed,
        "builder": "paper_experiments/ablation_v2/build_evalset.py",
        "note": (
            "gold answer 없음 — faithfulness 정의가 (답변, 출처 청크) 만 요구하므로 "
            "'예상 답변'을 생성하지 않는다. MM-OpenBench v1 은 ground_truth 가 전부 "
            "플레이스홀더이고 source_ids 가 해소되지 않아 사용하지 않았다."
        ),
    }
    (out.parent / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    avg = sum(len(c) for r in instances for c in r["source_chunks"]) / \
        sum(r["n_chunks"] for r in instances)
    print(f"[✓] {out}  n={len(instances)}  문서={len(docs)}  "
          f"평균 청크 길이={avg:.0f}자")
    print(f"    예시 질의: {instances[0]['query']}")
    print(f"    예시 청크: {instances[0]['source_chunks'][0][:100]}...")


if __name__ == "__main__":
    main()
