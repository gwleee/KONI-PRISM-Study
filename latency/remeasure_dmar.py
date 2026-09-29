#!/usr/bin/env python3
"""
DMAR ILP solve time 재측정 — 본문 반영값(8.36±0.30 ms) 정합 확인.

Exp C 는 k=3 (text/image/audio) 한 조건만 측정했으나, 본문 §4.2 는
"k≤5 modalities, n≤8 specialist candidates" 범위 전체에 대해 8.36ms 를
주장한다. solve time 이 k 에 따라 변하면 그 문장은 부정확해지므로
k=1..5 전 구간을 측정한다.

계측 대상: DMARSolver._solve_ilp() — Exp C 와 동일 경로.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.harness import bootstrap_gateway, provenance, set_profile, summarize, write_csv, write_json  # noqa: E402

# 레지스트리가 지원하는 모달리티 (dmar.py MODEL_REGISTRY 기준)
ALL_MODALITIES = ["text", "image", "audio", "document", "video"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="a100_1")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    set_profile(args.profile)
    bootstrap_gateway()
    from algorithms.dmar import MODEL_REGISTRY, get_dmar_solver  # noqa: PLC0415
    from core.config import HARDWARE_PROFILES, get_settings  # noqa: PLC0415

    s = get_settings()
    v_max = HARDWARE_PROFILES[s.hardware_profile]["vram_gb"]
    solver = get_dmar_solver()
    alpha = solver._compute_alpha(s.hardware_profile)

    print(f"profile={s.hardware_profile} v_max={v_max}GB alpha={alpha:.3f} "
          f"n_models={len(MODEL_REGISTRY)}")
    print(f"reps={args.reps} (warmup {args.warmup} excluded)\n")

    rows, raw = [], []
    for k in range(1, len(ALL_MODALITIES) + 1):
        mods = ALL_MODALITIES[:k]
        samples = []
        for i in range(args.warmup + args.reps):
            t = time.perf_counter()
            decisions = solver._solve_ilp(mods, alpha, v_max)
            dt = (time.perf_counter() - t) * 1000.0
            if i >= args.warmup:
                samples.append(dt)
                raw.append({"k": k, "iter": i - args.warmup, "ms": round(dt, 4),
                            "solved": decisions is not None})
        st = summarize(samples)
        rows.append({"k": k, "modalities": "+".join(mods), **st})
        print(f"  k={k} ({'+'.join(mods):32s}) "
              f"{st['mean']:8.3f} ± {st['std']:6.3f} ms   "
              f"p50={st['p50']:7.3f} p95={st['p95']:7.3f}")

    k3 = next(r for r in rows if r["k"] == 3)
    paper_mean, paper_std = 8.36, 0.30
    expc_mean, expc_std = 8.355033333333333, 0.30178600357541924

    # k≤5 전 구간 통합 (본문 주장 범위)
    all_samples = [r["ms"] for r in raw]
    overall = summarize(all_samples)

    verdict = {
        "paper_text_value_ms": {"mean": paper_mean, "std": paper_std,
                                "location": "KONI-Prism.tex §4.2 and Results (DMAR)"},
        "expc_stored_k3_ms": {"mean": expc_mean, "std": expc_std, "n": 30},
        "remeasured_k3_ms": {"mean": k3["mean"], "std": k3["std"], "n": k3["n"]},
        "remeasured_k1_to_k5_ms": overall,
        "delta_vs_paper_k3": k3["mean"] - paper_mean,
        "delta_vs_expc_k3": k3["mean"] - expc_mean,
        "k_dependence": {
            "min_k_mean": min(r["mean"] for r in rows),
            "max_k_mean": max(r["mean"] for r in rows),
            "spread_ms": max(r["mean"] for r in rows) - min(r["mean"] for r in rows),
        },
    }

    out = Path(args.out)
    write_csv(out / "dmar_solve_by_k.csv", rows,
              columns=["k", "modalities", "n", "mean", "std", "min", "p50", "p95", "max"])
    write_csv(out / "dmar_solve_raw.csv", raw, columns=["k", "iter", "ms", "solved"])
    write_json(out / "dmar_remeasure.json", {
        "experiment": "DMAR ILP solve time 재측정 (본문 정합 확인)",
        "provenance": provenance(),
        "alpha": alpha, "v_max_gb": v_max,
        "n_model_candidates": len(MODEL_REGISTRY),
        "reps_per_k": args.reps, "warmup_excluded": args.warmup,
        "by_k": rows,
        "verdict": verdict,
    })

    print(f"\n── 정합 확인 ──")
    print(f"  본문 반영값        : {paper_mean:.2f} ± {paper_std:.2f} ms")
    print(f"  Exp C 저장값 (k=3) : {expc_mean:.3f} ± {expc_std:.3f} ms (n=30)")
    print(f"  재측정값     (k=3) : {k3['mean']:.3f} ± {k3['std']:.3f} ms (n={k3['n']})")
    print(f"  Δ(재측정 − 본문)   : {verdict['delta_vs_paper_k3']:+.3f} ms")
    print(f"  Δ(재측정 − Exp C)  : {verdict['delta_vs_expc_k3']:+.3f} ms")
    print(f"\n  k=1..5 통합        : {overall['mean']:.3f} ± {overall['std']:.3f} ms "
          f"(p95={overall['p95']:.3f})")
    print(f"  k 의존 편차        : {verdict['k_dependence']['spread_ms']:.3f} ms "
          f"({verdict['k_dependence']['min_k_mean']:.3f} ~ "
          f"{verdict['k_dependence']['max_k_mean']:.3f})")
    print(f"\n[✓] {out/'dmar_remeasure.json'}")


if __name__ == "__main__":
    main()
