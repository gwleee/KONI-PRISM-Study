#!/usr/bin/env python3
"""
faithfulness 지표 타당성 검사 — 구성 비교에 쓰기 전에 지표가 작동하는지 확인한다.

통제 조건 3종의 기대 방향:
  copy       출처를 그대로 답변으로  → 1.0 근처
  unrelated  무관한 문서를 답변으로  → 0.0 근처
  negated    출처를 부정한 답변으로  → 낮고, contradicted_rate 높음

이 검사는 형식적 절차가 아니다. 초기 설정(mDeBERTa-base + 청크 전체를 전제로 사용)은
copy 0.385 / unrelated 0.314 로 **분리도 0.071** 이었다. 즉 근거 있는 답변과 무관한
답변을 구분하지 못했고, 그 상태로 ablation 을 돌렸다면 결과가 전부 무의미했다.
전제를 문장 단위로 바꾸고 모델을 xlm-roberta-large-xnli 로 교체해 분리도 0.904 를
확보한 뒤에야 본 측정에 사용했다.

사용법: python validate_metric.py [--n 15]
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
            "xlm-roberta-large + sentence premise": "채택",
        },
    })

    print(f"모델: {NLI_MODEL}  (전제 = 문장 단위)")
    print(f"{'조건':10s} {'faithfulness':>14s} {'contradicted':>14s}")
    for k in ("copy", "unrelated", "negated"):
        print(f"{k:10s} {summary[k]['faithfulness_mean']:14.3f} "
              f"{summary[k]['contradicted_rate_mean']:14.3f}")
    print(f"\n분리도 (copy − unrelated): {separation:.3f}")
    print(f"판정: {'PASS' if passed else 'FAIL — 이 지표로 ablation 을 해석하면 안 된다'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
