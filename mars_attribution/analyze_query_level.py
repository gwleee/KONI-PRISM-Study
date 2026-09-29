#!/usr/bin/env python3
"""
인용 평가의 쿼리 단위 집계 (원고 보고 경로).

평가는 쿼리 30개 × 쿼리 순서 3가지다. 순서만 다른 3회는 독립 표본이 아니므로,
지표는 쿼리별로 먼저 평균한 뒤(n=30) 쿼리 부트스트랩 95% CI(10,000회, 시드 고정)를 붙이고,
인용 발생은 "세 순서 모두에서 인용된 쿼리 수 / 하나 이상에서 인용된 쿼리 수"로 센다.

  python mars_attribution/analyze_query_level.py
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.harness import write_json  # noqa: E402

SEED = 20260929
N_BOOT = 10_000
THETAS = ["0.10", "0.15", "0.20", "0.25", "0.30", "0.40", "0.50", "0.60", "0.70"]


def fnum(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return math.nan


def main() -> None:
    rows = list(csv.DictReader((HERE / "results" / "per_query_metrics.csv").open(encoding="utf-8")))
    by: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by[r["condition"]][r["id"]][int(r["run"])] = r
    queries = sorted(by["theta_0.10"])
    runs = sorted({int(r["run"]) for r in rows})
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(queries), size=(N_BOOT, len(queries)))

    def qmean(cond: str, field: str) -> np.ndarray:
        return np.array([np.nanmean([fnum(by[cond][q][r][field]) for r in runs])
                         if not all(math.isnan(fnum(by[cond][q][r][field])) for r in runs)
                         else math.nan for q in queries])

    def summary(v: np.ndarray) -> dict:
        v = v[~np.isnan(v)]
        idx = boot_idx[:, :len(v)] % len(v)
        bm = v[idx].mean(axis=1)
        return {"n_queries": int(len(v)), "mean": float(v.mean()),
                "bootstrap_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}

    cited = {}
    for t in THETAS:
        cond = f"theta_{t}"
        per_q = [[int(by[cond][q][r]["n_cited"]) > 0 for r in runs] for q in queries]
        cited[t] = {"queries_cited_all_orderings": int(sum(all(x) for x in per_q)),
                    "queries_cited_any_ordering": int(sum(any(x) for x in per_q)),
                    "n_queries": len(queries)}
    high = [[by["theta_0.70"][q][r]["reliability_grade"] == "High" for r in runs] for q in queries]
    payload = {
        "analysis": "attribution, query-level aggregation (30 queries x 3 orderings)",
        "n_queries": len(queries), "n_orderings": len(runs), "seed": SEED, "n_bootstrap": N_BOOT,
        "cited_by_theta": cited,
        "shipped_grade_high_all_orderings": int(sum(all(x) for x in high)),
        "at_theta_0.10": {
            "precision_relevance": summary(qmean("theta_0.10", "precision_relevance")),
            "recall_unit": summary(qmean("theta_0.10", "recall_unit")),
            "random_precision_baseline": summary(qmean("theta_0.10", "random_precision_baseline")),
        },
        "recall_unit_by_theta": {t: summary(qmean(f"theta_{t}", "recall_unit"))
                                 for t in ("0.10", "0.15", "0.20")},
        "control_shuffle_within": {
            "precision_relevance": summary(qmean("control_shuffle_within", "precision_relevance")),
        },
    }
    out = write_json(HERE / "results" / "query_level.json", payload)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
