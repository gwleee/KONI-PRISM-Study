#!/usr/bin/env python3
"""
Exp E — SynConflict v2 평가 + v1 대조.

측정 항목:
  1. 운영 임계값 τ=0.65 에서의 CMAE 충돌 검출 성능 (P/R/F1/MCC)
  2. τ 전 구간 스윕 → 최적 τ 와 운영값의 괴리
  3. 임계값 비의존 분리도 (ROC-AUC)
  4. 유사도 구간별 레이블 분포 + 중첩도 (R4 편향 진단)
  5. 충돌 유형별 재현율 (R1 — 유형별 난이도)
  6. register 교락 효과 (동일 문체 vs 교차 문체)
  7. v1 데이터셋 동일 절차 재평가 → v1의 인위적 분리 정량화
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "paper_experiments"))

from common.harness import (  # noqa: E402
    bootstrap_gateway, provenance, set_profile, settings_snapshot,
    write_csv, write_json, binary_scores,
)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def normalize_v1(rows: list[dict]) -> list[dict]:
    """v1 스키마를 v2 스키마로 맞춘다."""
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "text_a": r["text_a"],
            "text_b": r["text_b"],
            "label": int(bool(r["ground_truth_conflict"])),
            "conflict_type": r.get("pair_type", "unknown"),
            "register": "mixed_by_design",
            "cross_register": True,
            "domain": "n/a",
        })
    return out


async def analyze(name: str, rows: list[dict], tau_prod: float) -> dict:
    from common.cmae_eval import (  # noqa: PLC0415
        encode_pairs, evaluate_at_threshold, roc_auc, separability_index,
        similarity_bins, sweep_threshold, verify_against_cmae,
    )

    labels = [int(r["label"]) for r in rows]
    sims, probe = await encode_pairs(rows)

    at_prod = evaluate_at_threshold(sims, labels, tau_prod)
    curve, best = sweep_threshold(sims, labels)
    bins = similarity_bins(sims, labels)
    auc = roc_auc(sims, labels)
    verify = await verify_against_cmae(rows, sims, tau_prod, n_check=20)

    # 유형별 성능
    by_type: dict[str, dict] = {}
    types = sorted({r["conflict_type"] for r in rows})
    for t in types:
        idx = [i for i, r in enumerate(rows) if r["conflict_type"] == t]
        s = sims[idx]
        y = [labels[i] for i in idx]
        pred = (s < tau_prod).astype(int).tolist()
        sc = binary_scores(y, pred)
        by_type[t] = {
            "n": len(idx),
            "is_conflict_type": bool(y[0]) if y else None,
            "cos_mean": float(s.mean()),
            "cos_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "cos_min": float(s.min()), "cos_max": float(s.max()),
            # 충돌 유형이면 recall, 비충돌 유형이면 specificity 가 관심 대상
            "detected_rate": float(np.mean(pred)),
            "correct_rate": sc["recall"] if (y and y[0] == 1) else (
                sc["tn"] / len(idx) if len(idx) else 0.0),
        }

    # register 교락 분석
    by_reg: dict[str, dict] = {}
    for key in ("same", "cross"):
        idx = [i for i, r in enumerate(rows) if (r.get("cross_register", False)) == (key == "cross")]
        if not idx:
            continue
        s = sims[idx]
        y = [labels[i] for i in idx]
        pred = (s < tau_prod).astype(int).tolist()
        by_reg[key] = {"n": len(idx), **binary_scores(y, pred),
                       "cos_mean_conflict": float(s[np.array(y) == 1].mean()) if any(y) else None,
                       "cos_mean_clean": float(s[np.array(y) == 0].mean()) if not all(y) else None}

    # 자명한 베이스라인 — "전부 충돌" / "전부 비충돌"
    trivial_all_pos = binary_scores(labels, [1] * len(labels))
    trivial_all_neg = binary_scores(labels, [0] * len(labels))

    return {
        "dataset": name,
        "n": len(rows),
        "n_conflict": int(sum(labels)),
        "encoder_probe": probe,
        "cmae_path_verification": verify,
        "at_production_tau": at_prod,
        "best_threshold": best,
        "tau_gap": float(best["tau"] - tau_prod),
        "f1_gap": float(best["f1"] - at_prod["f1"]),
        "roc_auc": auc,
        "trivial_baselines": {
            "predict_all_conflict": trivial_all_pos,
            "predict_all_clean": trivial_all_neg,
        },
        "separability_overlap": separability_index(bins),
        "similarity_bins": bins,
        "by_type": by_type,
        "by_register": by_reg,
        "_curve": curve,
        "_sims": sims,
        "_labels": labels,
        "_rows": rows,
    }


# 레이블 taxonomy 민감도 — v2 구성 선택에 결과가 얼마나 의존하는지 검증
VARIANTS: dict[str, dict] = {
    "full": {
        "desc": "v2 전체 (충돌 5종 + 비충돌 3종)",
        "exclude": [],
    },
    "no_unrelated": {
        "desc": "저유사도 비충돌(unrelated_consistent) 제외 — 오경보 유형을 뺐을 때",
        "exclude": ["unrelated_consistent"],
    },
    "no_domain_mismatch": {
        "desc": "주제이탈형 충돌(domain_mismatch) 제외 — 유일하게 잘 잡히는 유형을 뺐을 때",
        "exclude": ["domain_mismatch"],
    },
    "strict_contradiction": {
        "desc": ("주제를 고정한 순수 모순만: 충돌=numerical/paraphrastic/temporal/"
                 "terminological, 비충돌=paraphrase/complementary. "
                 "'같은 주제 안의 모순을 코사인이 잡는가'라는 핵심 질문."),
        "exclude": ["domain_mismatch", "unrelated_consistent"],
    },
}


def variant_analysis(rows: list[dict], sims, labels: list[int], tau_prod: float) -> dict:
    """유형 제외 조합별로 지표를 재계산한다 (재인코딩 없이 인덱싱만)."""
    from common.cmae_eval import (  # noqa: PLC0415
        evaluate_at_threshold, roc_auc, separability_index, similarity_bins,
        sweep_threshold,
    )

    out = {}
    for name, spec in VARIANTS.items():
        idx = [i for i, r in enumerate(rows)
               if r["conflict_type"] not in spec["exclude"]]
        if not idx:
            continue
        s = sims[idx]
        y = [labels[i] for i in idx]
        if len(set(y)) < 2:
            continue
        at_prod = evaluate_at_threshold(s, y, tau_prod)
        curve, best = sweep_threshold(s, y)
        out[name] = {
            "desc": spec["desc"],
            "excluded_types": spec["exclude"],
            "n": len(idx),
            "n_conflict": int(sum(y)),
            "at_production_tau": at_prod,
            "best_threshold": best,
            "roc_auc": roc_auc(s, y),
            "separability_overlap": separability_index(similarity_bins(s, y)),
            "trivial_all_conflict_f1": binary_scores(y, [1] * len(y))["f1"],
        }
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=str(HERE / "data" / "instances.jsonl"))
    ap.add_argument("--v1", default=str(HERE.parents[1] / "experiments" / "datasets"
                                        / "syn_conflict_v1" / "instances.jsonl"))
    ap.add_argument("--profile", default="a100_1")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    set_profile(args.profile)
    bootstrap_gateway()
    from core.config import get_settings  # noqa: PLC0415
    tau_prod = get_settings().cmae_tau_conflict

    results = {}
    v2_rows = load_jsonl(Path(args.v2))
    results["v2"] = await analyze("SynConflict_v2", v2_rows, tau_prod)

    v1p = Path(args.v1)
    if v1p.exists():
        results["v1"] = await analyze("SynConflict_v1", normalize_v1(load_jsonl(v1p)), tau_prod)
    else:
        results["v1"] = {"error": f"v1 dataset not found: {v1p}"}

    # v2 레이블 taxonomy 민감도
    r2 = results["v2"]
    results["v2"]["label_taxonomy_sensitivity"] = variant_analysis(
        r2["_rows"], r2["_sims"], r2["_labels"], tau_prod)

    out = Path(args.out)
    curves = {k: v.pop("_curve", []) for k, v in results.items() if isinstance(v, dict)}
    for v in results.values():
        if isinstance(v, dict):
            v.pop("_sims", None); v.pop("_labels", None); v.pop("_rows", None)

    payload = {
        "experiment": "Exp E — SynConflict v2",
        "production_tau_conflict": tau_prod,
        "provenance": provenance(),
        "settings": settings_snapshot(),
        "results": results,
    }
    write_json(out / "results.json", payload)

    for k, c in curves.items():
        if c:
            write_csv(out / f"threshold_curve_{k}.csv", c,
                      columns=["tau", "precision", "recall", "f1", "mcc",
                               "accuracy", "tp", "fp", "fn", "tn"])

    # 콘솔 요약
    for k in ("v1", "v2"):
        r = results.get(k, {})
        if "error" in r:
            print(f"[{k}] {r['error']}")
            continue
        p = r["at_production_tau"]
        b = r["best_threshold"]
        print(f"\n=== {r['dataset']} (n={r['n']}, conflict={r['n_conflict']}) ===")
        print(f"  CMAE 경로 일치: {r['cmae_path_verification']['agreement_rate']:.0%} "
              f"(max |Δcos|={r['cmae_path_verification']['max_abs_cos_diff']:.2e})")
        print(f"  τ=0.65 운영값 : P={p['precision']:.3f} R={p['recall']:.3f} "
              f"F1={p['f1']:.3f} MCC={p['mcc']:.3f}")
        print(f"  최적 τ={b['tau']:.3f} : F1={b['f1']:.3f}  (Δτ={r['tau_gap']:+.3f}, ΔF1={r['f1_gap']:+.3f})")
        print(f"  ROC-AUC       : {r['roc_auc']:.4f}")
        print(f"  구간 중첩도    : {r['separability_overlap']:.3f}  "
              f"(0=유사도만으로 완전분리, 클수록 현실적)")
        tb = r["trivial_baselines"]["predict_all_conflict"]
        print(f"  자명 베이스라인(전부 충돌) F1={tb['f1']:.3f}")

    print("\n=== v2 레이블 taxonomy 민감도 (τ=0.65) ===")
    print(f"{'variant':22s} {'n':>4s} {'F1':>6s} {'MCC':>7s} {'AUC':>6s} {'trivF1':>7s}")
    for k, v in results["v2"]["label_taxonomy_sensitivity"].items():
        p = v["at_production_tau"]
        print(f"{k:22s} {v['n']:4d} {p['f1']:6.3f} {p['mcc']:7.3f} "
              f"{v['roc_auc']:6.3f} {v['trivial_all_conflict_f1']:7.3f}")

    print(f"\n[✓] {out/'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
