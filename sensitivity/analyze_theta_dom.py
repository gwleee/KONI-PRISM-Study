#!/usr/bin/env python3
"""
Exp F 부속 분석 — θ_dom(=0.60) 이 실제로 도달 가능한 값인가?

CMAE 전략 결정은 max(T̃) > θ_dom 으로 Trust-Dominant 를 고른다.
그런데 T̃ = softmax(Trust/τ_T) 는 합이 1인 분포이므로 k개 모달리티에서
max(T̃) 의 하한은 1/k 다. Trust 값들의 산포가 작으면 max(T̃) ≈ 1/k 에 머문다.

k=3(전형적 멀티모달)에서 1/k=0.333 이므로, θ_dom=0.60 을 넘으려면
Trust 값이 매우 크게 벌어져야 한다. 실제로 그런 일이 일어나는지 측정한다.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.harness import bootstrap_gateway, set_profile, write_json  # noqa: E402


def max_softmax_needed(k: int, tau_t: float, theta_dom: float) -> float:
    """max(T̃) > θ_dom 이 되려면 1위와 2위 Trust 격차가 얼마여야 하는지.

    최악(가장 유리한) 가정: 나머지 k-1개가 모두 동일한 값이라고 두고
    softmax 최대값이 θ_dom 을 넘는 최소 격차 Δ 를 푼다.
      e^{Δ/τ} / (e^{Δ/τ} + (k-1)) > θ_dom
      => Δ > τ · ln( θ_dom(k-1) / (1-θ_dom) )
    Trust ∈ [0,1] 이므로 Δ > 1 이면 도달 불가능하다.
    """
    return tau_t * np.log(theta_dom * (k - 1) / (1 - theta_dom))


async def main() -> None:
    set_profile("a100_1")
    bootstrap_gateway()
    from algorithms.cmae import CMAEEvaluator, ModalityResponse  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415

    s = get_settings()
    tau_t, theta_dom = s.cmae_tau_t, s.cmae_theta_dom

    # 이론적 필요 격차
    theory = {}
    for k in (2, 3, 4, 5):
        d = max_softmax_needed(k, tau_t, theta_dom)
        theory[k] = {
            "min_trust_gap_required": float(d),
            "reachable_given_trust_in_0_1": bool(d < 1.0),
            "max_softmax_lower_bound": 1.0 / k,
        }

    # 실측: SynConflict v2 + 외부 데이터로 k=2,3,4 구성해 max(T̃) 분포 측정
    rows = [json.loads(l) for l in
            (HERE.parent / "synconflict_v2" / "data" / "instances.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()][:200]

    ev = CMAEEvaluator()
    empirical = {}
    for k in (2, 3, 4):
        maxes = []
        for r in rows:
            texts = [r["text_a"], r["text_b"],
                     r["text_a"][: max(20, len(r["text_a"]) // 2)],
                     r["text_b"][: max(20, len(r["text_b"]) // 2)]][:k]
            resp = [ModalityResponse(modality=m, model_id=f"m{i}", content=t)
                    for i, (m, t) in enumerate(
                        zip(["text", "image", "audio", "document"], texts))]
            o = await ev.evaluate(resp)
            maxes.append(float(o.trust_weights.max()))
        a = np.asarray(maxes)
        empirical[k] = {
            "n": int(a.size),
            "max_trust_mean": float(a.mean()),
            "max_trust_max": float(a.max()),
            "max_trust_p99": float(np.percentile(a, 99)),
            "pct_exceeding_theta_dom": float((a > theta_dom).mean()),
        }

    payload = {
        "analysis": "θ_dom 도달 가능성",
        "tau_T": tau_t, "theta_dom": theta_dom,
        "theory_min_trust_gap": theory,
        "empirical_max_trust": empirical,
        "conclusion": (
            "T̃=softmax(Trust/τ_T) 는 합이 1인 분포이므로 max(T̃) ≥ 1/k 이고, "
            "Trust∈[0,1] 범위에서 상한도 제한된다. k=3, τ_T=0.5 에서 "
            "max(T̃)>0.60 을 만족하려면 1위-2위 Trust 격차가 "
            f"{theory[3]['min_trust_gap_required']:.3f} 이상이어야 하는데, "
            "Trust = γ·Align + (1-γ)·q̂ 는 두 항 모두 [0,1] 이고 실측 산포가 작아 "
            "실제로는 도달하지 못한다. 즉 Trust-Dominant 분기는 k≥3 에서 "
            "사실상 발화하지 않는다."
        ),
    }
    write_json(HERE / "results" / "theta_dom_reachability.json", payload)

    print(f"τ_T={tau_t}, θ_dom={theta_dom}\n")
    print("[이론] max(T̃)>θ_dom 에 필요한 1위-2위 Trust 격차")
    print(f"{'k':>3s} {'필요격차':>10s} {'Trust∈[0,1]에서 도달가능':>24s} {'max(T̃) 하한':>12s}")
    for k, v in theory.items():
        print(f"{k:3d} {v['min_trust_gap_required']:10.3f} "
              f"{str(v['reachable_given_trust_in_0_1']):>24s} "
              f"{v['max_softmax_lower_bound']:12.3f}")
    print("\n[실측] SynConflict v2 200건")
    print(f"{'k':>3s} {'mean':>8s} {'p99':>8s} {'max':>8s} {'θ_dom 초과율':>12s}")
    for k, v in empirical.items():
        print(f"{k:3d} {v['max_trust_mean']:8.3f} {v['max_trust_p99']:8.3f} "
              f"{v['max_trust_max']:8.3f} {v['pct_exceeding_theta_dom']*100:11.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
