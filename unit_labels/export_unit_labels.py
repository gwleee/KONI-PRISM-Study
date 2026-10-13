#!/usr/bin/env python3
"""
단위별 NLI 라벨 export — 공개 패키지 v1.1.0 (raw_records 공개)용.

저장된 답변·출처 청크에 채점기(common/faithfulness.py, η=0.5)를 **같은 호출 순서와 배치로**
다시 적용해, 집계값만 있던 raw_records 를 단위(문장) 수준으로 풀어 쓴다. 새 생성·새 측정은
없다. 재계산 집계가 저장값과 하나라도 다르면 실패로 끝내고 아무것도 쓰지 않는다.

  ablation_v2/results/unit_labels.jsonl      레코드(id, config, run)별 단위 목록
      unit, ent_max, con_max, supported(ent_max ≥ η), contradicted(con_max ≥ η)
      검증: faith_n_units / faith_n_supported / faith_contradicted_rate / faith_faithfulness
  mars_attribution/results/unit_labels.jsonl  레코드(id, run)별 단위 × 출처 지지 행렬
      units, ent_max[unit][source], support[unit][source]
      검증: per_query_metrics.csv 의 theta_* 행 precision / recall_unit / recall_source / n_supported_units

사용법 (artefact-env 환경, GPU 0):
  python unit_labels/export_unit_labels.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PE = HERE.parent
sys.path.insert(0, str(PE))
sys.path.insert(0, str(PE / "mars_attribution"))

from common.faithfulness import FaithfulnessScorer, split_fact_units  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def close(a, b, tol=1e-9) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(float(a) - float(b)) <= tol


def ablation(scorer: FaithfulnessScorer) -> tuple[list[dict], list[str]]:
    out, bad = [], []
    ei, ci = scorer._label_idx["entailment"], scorer._label_idx["contradiction"]
    for rec in load_jsonl(PE / "ablation_v2/results/raw_records.jsonl"):
        if "answer_raw" not in rec:
            continue
        units = split_fact_units(rec["answer_raw"])
        premises = scorer._premises(rec["source_chunks"])
        key = f"{rec['id']}/{rec['config']}/run{rec['run']}"
        if not units or not premises:
            if not rec.get("faith_undefined"):
                bad.append(f"{key}: undefined mismatch")
            out.append({"id": rec["id"], "config": rec["config"], "run": rec["run"],
                        "undefined": True, "units": []})
            continue
        pairs = [(s, u) for u in units for s in premises]          # scorer.score() 와 같은 순서
        probs = scorer._nli(pairs).reshape(len(units), len(premises), 3)
        ent = probs[:, :, ei].max(axis=1)
        con = probs[:, :, ci].max(axis=1)
        sup = ent >= scorer.ent_threshold
        cont = con >= scorer.ent_threshold
        checks = [("n_units", len(units), rec["faith_n_units"]),
                  ("n_supported", int(sup.sum()), rec["faith_n_supported"]),
                  ("faithfulness", float(sup.mean()), rec["faith_faithfulness"]),
                  ("contradicted_rate", float(cont.mean()), rec["faith_contradicted_rate"])]
        for name, got, want in checks:
            if not close(got, want):
                bad.append(f"{key}: {name} {got} != {want}")
        out.append({
            "id": rec["id"], "config": rec["config"], "run": rec["run"], "undefined": False,
            "eta": scorer.ent_threshold, "n_premises": len(premises),
            "units": [{"unit": u, "ent_max": round(float(e), 6), "con_max": round(float(c), 6),
                       "supported": bool(s), "contradicted": bool(k)}
                      for u, e, c, s, k in zip(units, ent, con, sup, cont)],
        })
    return out, bad


def mars(scorer: FaithfulnessScorer) -> tuple[list[dict], list[str]]:
    from run_mars_eval import metrics_for  # noqa: PLC0415  (동일 지표 함수)
    ei = scorer._label_idx["entailment"]
    stored = {}
    with (PE / "mars_attribution/results/per_query_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["condition"].startswith("theta_"):
                stored[(r["id"], int(r["run"]), r["condition"])] = r
    out, bad = [], []
    for rec in load_jsonl(PE / "mars_attribution/results/raw_records.jsonl"):
        if "answer" not in rec:
            continue
        units = split_fact_units(rec["answer"])
        prem_per_src = [scorer._premises([c]) for c in rec["source_chunks"]]
        ent = np.zeros((len(units), len(rec["source_chunks"])))
        for si, prem in enumerate(prem_per_src):                     # run_mars_eval 과 같은 순서
            if not prem or not units:
                continue
            P = scorer._nli([(p, u) for u in units for p in prem]).reshape(len(units), len(prem), 3)
            ent[:, si] = P[:, :, ei].max(axis=1)
        support = ent >= scorer.ent_threshold
        for th, info in rec["by_theta"].items():
            m = metrics_for(info["cited_indices"], support, rec.get("relevance_mask"))
            s = stored.get((rec["id"], rec["run"], f"theta_{th}"))
            if s is None:
                bad.append(f"{rec['id']}/run{rec['run']}/theta_{th}: no stored row")
                continue
            for k in ("precision", "recall_unit", "recall_source", "n_supported_units"):
                want = float(s[k]) if s[k] not in ("", "nan") else float("nan")
                if not close(float(m[k]), want, 1e-6):
                    bad.append(f"{rec['id']}/run{rec['run']}/theta_{th}: {k} {m[k]} != {s[k]}")
        out.append({
            "id": rec["id"], "run": rec["run"], "eta": scorer.ent_threshold,
            "units": units,
            "ent_max": [[round(float(x), 6) for x in row] for row in ent],
            "support": support.astype(int).tolist(),
        })
    return out, bad


def main() -> None:
    scorer = FaithfulnessScorer()          # η = 0.5, batch 16, max_length 256 — 측정 때와 동일
    scorer._ensure()
    a, bad_a = ablation(scorer)
    m, bad_m = mars(scorer)
    print(f"ablation: {len(a)} records, mismatches {len(bad_a)}")
    print(f"mars    : {len(m)} records, mismatches {len(bad_m)}")
    for b in (bad_a + bad_m)[:20]:
        print("   ", b)
    if bad_a or bad_m:
        sys.exit("재계산이 저장값과 다르다 — 아무것도 쓰지 않는다")
    for rel, rows in (("ablation_v2/results/unit_labels.jsonl", a),
                      ("mars_attribution/results/unit_labels.jsonl", m)):
        (PE / rel).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                              encoding="utf-8")
        print("wrote", rel)


if __name__ == "__main__":
    main()
