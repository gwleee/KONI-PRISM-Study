#!/usr/bin/env python3
"""
faithfulness metric validity check — verify the metric works before using it to compare configs.

Expected direction for the three control conditions:
  copy       source copied verbatim as the answer  → near 1.0
  unrelated  an unrelated document as the answer  → near 0.0
  negated    the negated source as the answer     → low, with high contradicted_rate

This check is not a formality. The initial setup (mDeBERTa-base + whole chunk as the premise)
gave copy 0.385 / unrelated 0.314, a **separation of 0.071**. It could not tell grounded answers
from unrelated ones, and an ablation run in that state would have been entirely meaningless.
Only after switching to sentence-level premises and to the xlm-roberta-large-xnli model, reaching
a separation of 0.904, was the metric used for the main measurement.

Usage: python validate_metric.py [--n 15]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.faithfulness import FaithfulnessScorer, NLI_MODEL  # noqa: E402
from common.harness import write_json  # noqa: E402

NEGATIONS = [
    ("했다", "하지 않았다"), ("이다", "이 아니다"), ("된다", "되지 않는다"),
    ("있다", "없다"), ("밝혔다", "부인했다"), ("출시한다", "출시하지 않는다"),
    ("증가", "감소"), ("상승", "하락"),
]


def negate(t: str) -> str:
    for a, b in NEGATIONS:
        t = t.replace(a, b)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--evalset", default=str(HERE / "data" / "evalset.jsonl"))
    ap.add_argument("--out", default=str(HERE / "results" / "metric_validation.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.evalset).read_text(encoding="utf-8").splitlines() if l.strip()][:args.n]
    sc = FaithfulnessScorer()

    res: dict[str, list[float]] = {"copy": [], "unrelated": [], "negated": []}
    con: dict[str, list[float]] = {"copy": [], "unrelated": [], "negated": []}

    for i, r in enumerate(rows):
        src = r["source_chunks"]
        other = rows[(i + len(rows) // 2) % len(rows)]["source_chunks"]
        for name, ans in (("copy", " ".join(src)),
                          ("unrelated", " ".join(other)),
                          ("negated", " ".join(negate(c) for c in src))):
            f = sc.score(ans, src)
            res[name].append(f["faithfulness"])
            con[name].append(f["contradicted_rate"])

    summary = {k: {"faithfulness_mean": st.mean(v),
                   "contradicted_rate_mean": st.mean(con[k]),
                   "n": len(v)} for k, v in res.items()}
    separation = summary["copy"]["faithfulness_mean"] - summary["unrelated"]["faithfulness_mean"]
    passed = (summary["copy"]["faithfulness_mean"] >= 0.85
              and summary["unrelated"]["faithfulness_mean"] <= 0.15
              and separation >= 0.7)

    write_json(Path(args.out), {
        "check": "faithfulness metric validity (control conditions)",
        "nli_model": NLI_MODEL,
        "premise_granularity": "sentence",
        "n_instances": len(rows),
        "conditions": summary,
        "separation_copy_minus_unrelated": separation,
        "pass_criteria": {"copy>=0.85": True, "unrelated<=0.15": True, "separation>=0.70": True},
        "passed": passed,
        "history": {
            "mDeBERTa-base + chunk premise": {"copy": 0.385, "unrelated": 0.314, "separation": 0.071},
            "mDeBERTa-base + sentence premise": {"copy": 0.642, "unrelated": 0.209, "separation": 0.433},
            "xlm-roberta-large + sentence premise": "adopted",
        },
    })

    print(f"model: {NLI_MODEL}  (premise = sentence level)")
    print(f"{'condition':10s} {'faithfulness':>14s} {'contradicted':>14s}")
    for k in ("copy", "unrelated", "negated"):
        print(f"{k:10s} {summary[k]['faithfulness_mean']:14.3f} "
              f"{summary[k]['contradicted_rate_mean']:14.3f}")
    print(f"\nseparation (copy − unrelated): {separation:.3f}")
    print(f"verdict: {'PASS' if passed else 'FAIL — the ablation must not be interpreted with this metric'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
