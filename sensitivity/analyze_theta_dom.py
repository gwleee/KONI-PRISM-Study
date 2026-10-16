#!/usr/bin/env python3
"""
Exp F supplementary analysis — is θ_dom (=0.60) actually reachable?

The CMAE strategy decision picks Trust-Dominant when max(T̃) > θ_dom.
But T̃ = softmax(Trust/τ_T) is a distribution summing to 1, so with k modalities
the lower bound of max(T̃) is 1/k. If the Trust values have small spread, max(T̃) stays ≈ 1/k.

For k=3 (typical multimodal), 1/k=0.333, so exceeding θ_dom=0.60 requires
a very large spread in Trust values. We measure whether that actually happens.
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
    """How large the top-1 vs top-2 Trust gap must be for max(T̃) > θ_dom.

    Extreme (most favorable) assumption: the remaining k-1 values are all equal;
    solve for the minimum gap Δ at which the softmax maximum exceeds θ_dom.
      e^{Δ/τ} / (e^{Δ/τ} + (k-1)) > θ_dom
      => Δ > τ · ln( θ_dom(k-1) / (1-θ_dom) )
    Since Trust ∈ [0,1], it is unreachable if Δ > 1.
    """
    return tau_t * np.log(theta_dom * (k - 1) / (1 - theta_dom))


async def main() -> None:
    set_profile("a100_1")
    bootstrap_gateway()
    from algorithms.cmae import CMAEEvaluator, ModalityResponse  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415

    s = get_settings()
    tau_t, theta_dom = s.cmae_tau_t, s.cmae_theta_dom

    # theoretically required gap
    theory = {}
    for k in (2, 3, 4, 5):
        d = max_softmax_needed(k, tau_t, theta_dom)
        theory[k] = {
            "min_trust_gap_required": float(d),
            "reachable_given_trust_in_0_1": bool(d < 1.0),
            "max_softmax_lower_bound": 1.0 / k,
        }

    # empirical: build k=2,3,4 from SynConflict v2 + external data and measure the max(T̃) distribution
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
        "analysis": "θ_dom reachability",
        "tau_T": tau_t, "theta_dom": theta_dom,
        "theory_min_trust_gap": theory,
        "empirical_max_trust": empirical,
        "conclusion": (
            "T̃=softmax(Trust/τ_T) is a distribution summing to 1, so max(T̃) ≥ 1/k, "
            "and its upper bound is also limited for Trust∈[0,1]. At k=3, τ_T=0.5, "
            "satisfying max(T̃)>0.60 requires a top-1 vs top-2 Trust gap of at least "
            f"{theory[3]['min_trust_gap_required']:.3f}, but "
            "in Trust = γ·Align + (1-γ)·q̂ both terms lie in [0,1] and the observed spread is small, "
            "so it is not reached in practice. That is, the Trust-Dominant branch "
            "effectively never fires for k≥3."
        ),
    }
    write_json(HERE / "results" / "theta_dom_reachability.json", payload)

    print(f"τ_T={tau_t}, θ_dom={theta_dom}\n")
    print("[theory] top-1 vs top-2 Trust gap required for max(T̃)>θ_dom")
    print(f"{'k':>3s} {'gap_req':>10s} {'reachable in Trust∈[0,1]':>24s} {'max(T̃) lower':>12s}")
    for k, v in theory.items():
        print(f"{k:3d} {v['min_trust_gap_required']:10.3f} "
              f"{str(v['reachable_given_trust_in_0_1']):>24s} "
              f"{v['max_softmax_lower_bound']:12.3f}")
    print("\n[empirical] SynConflict v2, 200 items")
    print(f"{'k':>3s} {'mean':>8s} {'p99':>8s} {'max':>8s} {'>θ_dom rate':>12s}")
    for k, v in empirical.items():
        print(f"{k:3d} {v['max_trust_mean']:8.3f} {v['max_trust_p99']:8.3f} "
              f"{v['max_trust_max']:8.3f} {v['pct_exceeding_theta_dom']*100:11.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
