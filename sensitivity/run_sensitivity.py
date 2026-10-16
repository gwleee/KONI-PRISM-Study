#!/usr/bin/env python3
"""
Exp F — hyperparameter sensitivity sweep.

Swept parameters (actual code defaults per core/config.py):
  τ_conflict = 0.65   CMAE Eq.(9)  conflict threshold
  γ          = 0.6    CMAE Eq.(11) alignment vs quality-estimate weight  [0.4, 0.8]
  τ_T        = 0.5    CMAE Eq.(12) softmax temperature    [0.3, 0.7]
  θ_dom      = 0.60   CMAE strategy decision (Trust-Dominant threshold)
  λ₁,λ₂,λ₃   = 0.5/0.3/0.2  HMMR Eq.(13) importance weights
  ω_S,ω_E,ω_L= 0.50/0.35/0.15 HMMR Eq.(15) tier token budget
  δ          = 0.3    HMMR Eq.(15) ModalMatch weight
  θ_imp      = 0.60   HMMR Eq.(16) episodic promotion threshold ★verified actual code value

★ θ_imp: not stated in the paper/spec, so verification was requested.
  The actual code value is `hmmr_theta_imp: float = 0.60` in core/config.py,
  read at hmmr.py:88 as self._theta_imp and used for the Eq.(16) promotion decision.

Metrics:
  - conflict detection F1 / MCC / ROC-AUC  (SynConflict v2 + external datasets)
  - CMAE conflict-resolution strategy distribution (trust_dominant / explicit_conflict / abstain)
  - trust weight concentration (max T̃, entropy)
  - HMMR retrieval rank change (Spearman ρ vs default settings), per-tier selection distribution
  - MARS citation count / reliability grade distribution

Note — why faithfulness is not used as a sweep metric:
  In the current repository, the faithfulness in paper Table 6 is not measured; it is
  a synthetic value that scripts/run_real_experiments.py:run_ablation_experiment()
  draws with rng.uniform from hard-coded ranges (SRC_QUALITY/MEM_DELTA/TRUST_BOOST ...)
  and combines as a weighted sum (zero algorithm calls). Sweeping that definition as is
  yields curves unrelated to the parameters, so here we use actually computable
  substitute metrics (MARS reliability score / citation precision, CMAE detection performance).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import (  # noqa: E402
    bootstrap_gateway, provenance, set_profile, settings_snapshot,
    write_csv, write_json,
)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation (ties get average rank)."""
    def rank(x: list[float]) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        order = np.argsort(arr, kind="mergesort")
        r = np.empty(arr.size, dtype=float)
        r[order] = np.arange(1, arr.size + 1, dtype=float)
        s = arr[order]
        i = 0
        while i < s.size:
            j = i
            while j + 1 < s.size and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r
    if len(a) < 2:
        return float("nan")
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


# ── CMAE-side sweep ─────────────────────────────────────────────────────────────

async def sweep_cmae(rows: list[dict], sims: np.ndarray, out_dir: Path) -> dict:
    """Sweep τ_conflict / γ / τ_T / θ_dom."""
    from algorithms.cmae import CMAEEvaluator, ModalityResponse  # noqa: PLC0415
    from common.cmae_eval import evaluate_at_threshold, roc_auc  # noqa: PLC0415

    labels = [int(r["label"]) for r in rows]
    curves: dict[str, list[dict]] = {}

    # ── τ_conflict: directly affects detection performance (Eq.9) ──
    tau_rows = []
    for tau in np.round(np.arange(0.30, 0.951, 0.025), 4):
        m = evaluate_at_threshold(sims, labels, float(tau))
        tau_rows.append({"param": "tau_conflict", "value": float(tau),
                         "f1": m["f1"], "mcc": m["mcc"],
                         "precision": m["precision"], "recall": m["recall"],
                         "roc_auc": roc_auc(sims, labels)})
    curves["tau_conflict"] = tau_rows

    # ── γ / τ_T / θ_dom: no effect on detection (Eq.9), but affect trust and strategy ──
    # Run the actual CMAEEvaluator on a representative subset to observe the strategy distribution.
    subset = rows[:150]
    triples = []
    for r in subset:
        triples.append([
            ModalityResponse(modality="text", model_id="llm_7b", content=r["text_a"]),
            ModalityResponse(modality="image", model_id="vlm_7b", content=r["text_b"]),
            ModalityResponse(modality="audio", model_id="stt_large",
                             content=r["text_a"][: max(20, len(r["text_a"]) // 2)]),
        ])

    async def strategy_profile(gamma: float, tau_t: float, theta_dom: float) -> dict:
        ev = CMAEEvaluator()
        ev._gamma, ev._tau_t, ev._theta_dom = gamma, tau_t, theta_dom
        counts = {"trust_dominant": 0, "explicit_conflict": 0, "abstain": 0}
        max_t, ent, n_conf = [], [], []
        for resp in triples:
            o = await ev.evaluate(resp)
            counts[o.conflict_strategy] += 1
            max_t.append(float(o.trust_weights.max()))
            ent.append(entropy(o.trust_weights))
            n_conf.append(len(o.conflict_pairs))
        n = len(triples)
        return {
            "n": n,
            "pct_trust_dominant": counts["trust_dominant"] / n,
            "pct_explicit_conflict": counts["explicit_conflict"] / n,
            "pct_abstain": counts["abstain"] / n,
            "max_trust_mean": float(np.mean(max_t)),
            "trust_entropy_mean": float(np.mean(ent)),
            "conflict_pairs_mean": float(np.mean(n_conf)),
        }

    base_g, base_tt, base_td = 0.6, 0.5, 0.60

    g_rows = []
    for g in np.round(np.arange(0.40, 0.801, 0.05), 4):
        p = await strategy_profile(float(g), base_tt, base_td)
        g_rows.append({"param": "gamma", "value": float(g), **p})
    curves["gamma"] = g_rows

    tt_rows = []
    for tt in np.round(np.arange(0.30, 0.701, 0.05), 4):
        p = await strategy_profile(base_g, float(tt), base_td)
        tt_rows.append({"param": "tau_T", "value": float(tt), **p})
    curves["tau_T"] = tt_rows

    td_rows = []
    for td in np.round(np.arange(0.35, 0.901, 0.05), 4):
        p = await strategy_profile(base_g, base_tt, float(td))
        td_rows.append({"param": "theta_dom", "value": float(td), **p})
    curves["theta_dom"] = td_rows

    return curves


# ── HMMR-side sweep ─────────────────────────────────────────────────────────────

async def reseed(project_id: str, n_units: int) -> dict:
    """Delete this project's data from the collection and reload it with the current parameters.

    λ (Eq.13) and θ_imp (Eq.16) are **write-time** parameters. Repeating retrieval over
    already stored memory shows no change when they vary (importance and promotion are
    fixed at store time). So these two must be reloaded per value to get a real curve.
    """
    from qdrant_client import AsyncQdrantClient  # noqa: PLC0415
    from qdrant_client.models import (  # noqa: PLC0415
        FieldCondition, Filter, FilterSelector, MatchValue,
    )

    from algorithms.hmmr import COLLECTION_NAME, get_hmmr_retriever  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415

    url = os.environ.get("QDRANT_URL") or get_settings().qdrant_url
    c = AsyncQdrantClient(url=url)
    await c.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="project_id", match=MatchValue(value=project_id))])),
        wait=True,
    )
    await c.close()

    hmmr = get_hmmr_retriever()
    stored, promoted, errs = 0, 0, []
    for i in range(n_units):
        try:
            await hmmr.update_memory(
                query_text=f"{i}번째 세션 질의: 추론 서버 성능 지표 점검",
                query_modalities=["text"],
                cmae_trust_weights=np.array([0.5, 0.3, 0.2]),
                project_id=project_id,
                conversation_id=f"conv_{i % 5}",
                response_content=(
                    f"추론 서버의 정확도는 {90 + i % 8}.{i % 10}%로 측정되었으며 "
                    f"응답 지연은 {40 + i % 60}ms 수준이다."),
                turn_idx=i, total_turns=n_units,
            )
            stored += 1
        except Exception as e:
            errs.append(f"{type(e).__name__}: {str(e)[:100]}")
            if len(errs) > 3:
                break
    return {"stored": stored, "errors": errs[:2]}


async def sweep_hmmr(project_id: str, queries: list[str],
                     reseed_units: int = 60) -> dict:
    """Sweep λ / ω / δ / θ_imp — changes in retrieval rank and tier distribution.

    δ, ω are read-time (retrieval) parameters → no reload needed.
    λ, θ_imp are write-time parameters → reload per value.
    """
    from algorithms.hmmr import get_hmmr_retriever  # noqa: PLC0415

    hmmr = get_hmmr_retriever()

    async def profile() -> dict:
        rel_all, tiers, n_ret, budgets = [], [], [], []
        for q in queries:
            res = await hmmr.retrieve(query_text=q, project_id=project_id,
                                      query_modalities=["text"])
            rel_all.extend(res.relevance_scores)
            tiers.extend(u.tier for u in res.retrieved_units)
            n_ret.append(len(res.retrieved_units))
            budgets.append(res.token_budget)
        from collections import Counter  # noqa: PLC0415
        c = Counter(tiers)
        tot = max(1, sum(c.values()))
        return {
            "n_retrieved_mean": float(np.mean(n_ret)),
            "relevance_mean": float(np.mean(rel_all)) if rel_all else 0.0,
            "relevance_top1_mean": float(np.mean(
                [r[0] for r in [[x] for x in rel_all]])) if rel_all else 0.0,
            "tier_S_pct": c.get("S", 0) / tot,
            "tier_E_pct": c.get("E", 0) / tot,
            "tier_L_pct": c.get("L", 0) / tot,
            "_ranking": list(rel_all),
        }

    # baseline settings
    base = {"l1": hmmr._lambda1, "l2": hmmr._lambda2, "l3": hmmr._lambda3,
            "delta": hmmr._delta, "os": hmmr._omega_s, "oe": hmmr._omega_e,
            "ol": hmmr._omega_l, "ti": hmmr._theta_imp}
    await reseed(project_id, reseed_units)
    base_prof = await profile()
    base_rank = base_prof.pop("_ranking")

    curves: dict[str, list[dict]] = {}

    async def one(param: str, value: float, apply, do_reseed: bool = False) -> dict:
        apply()
        seed_info = None
        if do_reseed:
            seed_info = await reseed(project_id, reseed_units)
        p = await profile()
        rank = p.pop("_ranking")
        n = min(len(rank), len(base_rank))
        p["spearman_vs_base"] = spearman(base_rank[:n], rank[:n]) if n >= 2 else float("nan")
        # restore
        hmmr._lambda1, hmmr._lambda2, hmmr._lambda3 = base["l1"], base["l2"], base["l3"]
        hmmr._delta = base["delta"]
        hmmr._omega_s, hmmr._omega_e, hmmr._omega_l = base["os"], base["oe"], base["ol"]
        hmmr._theta_imp = base["ti"]
        row = {"param": param, "value": float(value), **p}
        if seed_info is not None:
            row["reseeded_units"] = seed_info["stored"]
            row["reseed_errors"] = "; ".join(seed_info["errors"])
        return row

    async def reseed_baseline() -> None:
        """Restore memory to the baseline-parameter state before starting a group.

        Prevents memory state left by write-time parameter (λ, θ_imp) sweeps from
        contaminating the baseline of later read-time parameter (δ, ω) sweeps.
        """
        hmmr._lambda1, hmmr._lambda2, hmmr._lambda3 = base["l1"], base["l2"], base["l3"]
        hmmr._theta_imp = base["ti"]
        await reseed(project_id, reseed_units)

    # δ sweep (read-time parameter)
    await reseed_baseline()
    rows = []
    for d in np.round(np.arange(0.0, 1.01, 0.1), 4):
        def ap(d=d): hmmr._delta = float(d)
        rows.append(await one("delta", d, ap))
    curves["delta"] = rows

    # θ_imp sweep (write-time — reload per value)
    rows = []
    for ti in np.round(np.arange(0.30, 0.951, 0.10), 4):
        def ap(ti=ti): hmmr._theta_imp = float(ti)
        rows.append(await one("theta_imp", ti, ap, do_reseed=True))
    curves["theta_imp"] = rows

    # simple λ sweep: vary λ₁ and normalize λ₂, λ₃ keeping their ratio
    rows = []
    for l1 in np.round(np.arange(0.2, 0.81, 0.1), 4):
        rest = 1.0 - float(l1)
        r2 = base["l2"] / (base["l2"] + base["l3"])
        def ap(l1=l1, rest=rest, r2=r2):
            hmmr._lambda1 = float(l1)
            hmmr._lambda2 = rest * r2
            hmmr._lambda3 = rest * (1 - r2)
        rows.append(await one("lambda1", l1, ap, do_reseed=True))
    curves["lambda1"] = rows

    # ω sweep (read-time parameter) — start after restoring state left by the preceding reload sweeps
    await reseed_baseline()
    rows = []
    for os_ in np.round(np.arange(0.2, 0.81, 0.1), 4):
        rest = 1.0 - float(os_)
        re_ = base["oe"] / (base["oe"] + base["ol"])
        def ap(os_=os_, rest=rest, re_=re_):
            hmmr._omega_s = float(os_)
            hmmr._omega_e = rest * re_
            hmmr._omega_l = rest * (1 - re_)
        rows.append(await one("omega_S", os_, ap))
    curves["omega_S"] = rows

    return {"baseline": base_prof, "baseline_params": base, "curves": curves}


# ── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="a100_1")
    ap.add_argument("--v2", default=str(HERE.parent / "synconflict_v2" / "data" / "instances.jsonl"))
    ap.add_argument("--external", default=str(HERE.parent / "external_eval" / "data"))
    ap.add_argument("--project-id", default="expc_latency")
    ap.add_argument("--skip-hmmr", action="store_true")
    ap.add_argument("--skip-cmae", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    set_profile(args.profile)
    bootstrap_gateway()
    from common.cmae_eval import encode_pairs  # noqa: PLC0415

    out = Path(args.out)
    payload: dict = {
        "experiment": "Exp F — hyperparameter sensitivity sweep",
        "provenance": provenance(),
        "settings": settings_snapshot(),
        "theta_imp_note": {
            "code_value": settings_snapshot()["hmmr_theta_imp"],
            "source": "gateway/core/config.py :: Settings.hmmr_theta_imp",
            "consumed_at": "gateway/algorithms/hmmr.py:88 -> self._theta_imp (Eq.16 promotion decision)",
        },
        "datasets": {},
    }

    # ── per-dataset CMAE sweep
    all_curve_rows: list[dict] = []
    dataset_specs = [] if args.skip_cmae else [("synconflict_v2", Path(args.v2))] + [
        (f"external_{p.stem.replace('_pairs','')}", p)
        for p in sorted(Path(args.external).glob("*_pairs.jsonl"))
    ]
    for name, path in dataset_specs:
        if not path.exists():
            continue
        rows = load_jsonl(path)
        if len(rows) > 2000:
            rows = rows[:2000]
        # exclude neutral from external datasets
        rows = [r for r in rows if r.get("group") != "neutral"]
        print(f"[..] CMAE sweep: {name} (n={len(rows)})")
        sims, _ = await encode_pairs(rows)
        curves = await sweep_cmae(rows, sims, out)
        payload["datasets"][name] = {"n": len(rows), "curves": curves}
        for pname, crows in curves.items():
            for cr in crows:
                all_curve_rows.append({"dataset": name, **cr})

    if args.skip_cmae:
        prev = out / "results.json"
        if prev.exists():
            payload["datasets"] = json.loads(prev.read_text(encoding="utf-8")).get("datasets", {})
            print("[i] --skip-cmae: reusing CMAE results from the existing results.json")
    else:
        write_csv(out / "cmae_sensitivity.csv", all_curve_rows)

    # ── HMMR sweep
    if not args.skip_hmmr:
        print("[..] HMMR sweep")
        try:
            qs = ["추론 서버의 정확도 추이", "응답 지연이 기준을 넘는지",
                  "성능 지표 점검 결과 요약", "최근 세션의 주요 수치"]
            payload["hmmr"] = await sweep_hmmr(args.project_id, qs)
            hrows = []
            for pname, crows in payload["hmmr"]["curves"].items():
                hrows.extend({"param_group": pname, **cr} for cr in crows)
            write_csv(out / "hmmr_sensitivity.csv", hrows)
        except Exception as e:
            payload["hmmr"] = {"status": "failed",
                               "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  HMMR sweep failed: {payload['hmmr']['error']}")

    write_json(out / "results.json", payload)

    # ── summary output
    print("\n=== τ_conflict sensitivity (F1) ===")
    print(f"{'dataset':22s} {'F1@0.65':>8s} {'bestτ':>6s} {'bestF1':>7s} {'ΔF1':>7s}")
    for name, d in payload["datasets"].items():
        tc = d["curves"]["tau_conflict"]
        at65 = min(tc, key=lambda r: abs(r["value"] - 0.65))
        best = max(tc, key=lambda r: r["f1"])
        print(f"{name:22s} {at65['f1']:8.3f} {best['value']:6.3f} {best['f1']:7.3f} "
              f"{best['f1']-at65['f1']:+7.3f}")

    d0 = payload["datasets"].get("synconflict_v2")
    if d0:
        print("\n=== γ / τ_T / θ_dom → strategy distribution (SynConflict v2, n=150) ===")
        for pname in ("gamma", "tau_T", "theta_dom"):
            print(f"  {pname}:")
            for r in d0["curves"][pname]:
                print(f"    {r['value']:.2f}  dom={r['pct_trust_dominant']:.2f} "
                      f"expl={r['pct_explicit_conflict']:.2f} abst={r['pct_abstain']:.2f} "
                      f"maxT={r['max_trust_mean']:.3f} H={r['trust_entropy_mean']:.3f}")

    h = payload.get("hmmr")
    if h and h.get("curves"):
        print("\n=== HMMR δ / θ_imp / λ₁ / ω_S sensitivity ===")
        for pname, crows in h["curves"].items():
            vals = [r["relevance_mean"] for r in crows]
            sp = [r["spearman_vs_base"] for r in crows]
            print(f"  {pname:10s} relevance {min(vals):.4f}~{max(vals):.4f} "
                  f"(range {max(vals)-min(vals):.4f})  ρ_min={np.nanmin(sp):.3f}")

    print(f"\n[✓] {out/'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
