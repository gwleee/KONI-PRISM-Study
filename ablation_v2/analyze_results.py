#!/usr/bin/env python3
"""
Statistical analysis of Ablation v2 results — paired comparison + verdict on whether super-additivity reproduces.

Configs are compared on **identical (run, query) pairs**. With greedy decoding,
the same input yields the same answer, so paired differences carry only the config effect.

query_level block (the path reported in the manuscript): the 3 runs differ only in query order, and order changes
memory accumulation and hence values. The 3 runs of a query are not independent samples, so the unit of analysis is the query (n=30).
Per-query mean over 3 runs → paired difference (ablated − full) → query bootstrap 95% CI (10,000 resamples, fixed seed,
percentile) + sign-flip permutation test (100,000 permutations). Ordering variance is reported separately as ICC(1).
The n=90 path (paired_vs_full) is kept for comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.harness import write_json, write_csv  # noqa: E402

BASE = "full"
OTHERS = ["no_trust_flow", "no_hmmr", "no_mars"]
SEED = 20260929
N_BOOT = 10_000
N_PERM = 100_000


def icc1(mat: np.ndarray) -> float:
    """One-way random-effects ICC(1); rows = queries, columns = orderings."""
    n, k = mat.shape
    grand = mat.mean()
    msb = k * ((mat.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((mat - mat.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    den = msb + (k - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def query_level(scores: dict, answers: dict, queries: list, runs: list) -> dict:
    """scores[(run, id)][config] -> faithfulness; answers likewise -> answer text."""
    rng = np.random.default_rng(SEED)
    q = {c: np.array([np.mean([scores[(r, i)][c] for r in runs]) for i in queries])
         for c in [BASE] + OTHERS}
    out = {"n_queries": len(queries), "n_orderings": len(runs), "seed": SEED,
           "n_bootstrap": N_BOOT, "n_permutation": N_PERM,
           "full_mean": float(q[BASE].mean()),
           "full_icc1_across_orderings": icc1(np.array(
               [[scores[(r, i)][BASE] for r in runs] for i in queries])),
           "full_within_query_sd_median": float(np.median(
               [np.std([scores[(r, i)][BASE] for r in runs], ddof=1) for i in queries])),
           "contrasts": {}}
    boot_idx = rng.integers(0, len(queries), size=(N_BOOT, len(queries)))
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(queries)))
    for c in OTHERS:
        d = q[c] - q[BASE]
        e = {"mean_delta": float(d.mean()),
             "n_scores_identical_instances": int(sum(
                 scores[(r, i)][c] == scores[(r, i)][BASE] for r in runs for i in queries)),
             "n_answers_byte_identical_instances": int(sum(
                 answers[(r, i)][c] == answers[(r, i)][BASE] for r in runs for i in queries)),
             "n_queries_worse": int((d < 0).sum()), "n_queries_better": int((d > 0).sum())}
        if np.allclose(d, 0):
            e["all_identical"] = True
        else:
            bm = d[boot_idx].mean(axis=1)
            e["bootstrap_ci95"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
            perm = np.abs((signs * d).mean(axis=1))
            e["permutation_p"] = float(((perm >= abs(d.mean()) - 1e-12).sum() + 1) / (N_PERM + 1))
            e["wilcoxon_pratt_p"] = float(stats.wilcoxon(d, zero_method="pratt").pvalue)
            e["per_ordering_mean_delta"] = [float(np.mean(
                [scores[(r, i)][c] - scores[(r, i)][BASE] for i in queries])) for r in runs]
            e["ci_includes_zero"] = bool(e["bootstrap_ci95"][0] <= 0 <= e["bootstrap_ci95"][1])
        out["contrasts"][c] = e
    out["multiplicity"] = ("two confirmatory contrasts (no_trust_flow, no_hmmr); no correction "
                           "applied; no_mars is structurally zero and not tested")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results" / "results.json"))
    ap.add_argument("--raw", default=str(HERE / "results" / "raw_records.jsonl"))
    ap.add_argument("--out", default=str(HERE / "results" / "analysis.json"))
    args = ap.parse_args()

    d = json.load(open(args.results, encoding="utf-8"))
    recs = [json.loads(l) for l in Path(args.raw).read_text(encoding="utf-8").splitlines() if l.strip()]

    by: dict[tuple, dict[str, float]] = defaultdict(dict)
    ans: dict[tuple, dict[str, str]] = defaultdict(dict)
    for r in recs:
        if "faith_faithfulness" in r:
            by[(r["run"], r["id"])][r["config"]] = r["faith_faithfulness"]
            ans[(r["run"], r["id"])][r["config"]] = r.get("answer_raw", "")
    keys = [k for k, v in by.items() if len(v) == 1 + len(OTHERS)]
    full = np.array([by[k][BASE] for k in keys])

    paired = {}
    for c in OTHERS:
        x = np.array([by[k][c] for k in keys])
        dlt = x - full
        identical = int((dlt == 0).sum())
        entry = {
            "mean_delta": float(dlt.mean()),
            "std_delta": float(dlt.std(ddof=1)),
            "n_pairs": len(keys),
            "n_identical_to_full": identical,
            "all_identical": bool(np.allclose(dlt, 0)),
        }
        if not entry["all_identical"]:
            ci = stats.t.interval(0.95, len(dlt) - 1, loc=dlt.mean(), scale=stats.sem(dlt))
            entry["ci95"] = [float(ci[0]), float(ci[1])]
            entry["wilcoxon_p"] = float(stats.wilcoxon(dlt).pvalue)
            entry["significant_at_0.05"] = bool(entry["wilcoxon_p"] < 0.05)
            entry["ci_includes_zero"] = bool(ci[0] <= 0 <= ci[1])
        paired[c] = entry

    # super-additive verdict: is |Δ(trust_flow)| larger than the sum of the individual component |Δ|?
    tf = abs(paired["no_trust_flow"]["mean_delta"])
    parts = {c: abs(paired[c]["mean_delta"]) for c in ("no_hmmr", "no_mars")}
    super_add = {
        "claim_in_paper": ("Removing trust flow causes a 34.5pp drop, larger than the sum of "
                           "removing the two individual components (super-additive)"),
        "measured_trust_flow_drop_pp": paired["no_trust_flow"]["mean_delta"] * 100,
        "measured_component_drops_pp": {k: -v * 100 for k, v in parts.items()},
        "sum_of_components_pp": -sum(parts.values()) * 100,
        "trust_flow_exceeds_sum": tf > sum(parts.values()),
        "reproduced": bool(tf > sum(parts.values())
                           and paired["no_trust_flow"].get("significant_at_0.05", False)),
        "verdict": None,
    }
    if not paired["no_trust_flow"].get("significant_at_0.05", True):
        super_add["verdict"] = (
            "Not reproduced. The effect of removing trust flow is not statistically distinguishable from 0 "
            f"(p={paired['no_trust_flow'].get('wilcoxon_p'):.3f}, "
            f"95% CI {paired['no_trust_flow'].get('ci95')} includes 0).")
    elif not super_add["trust_flow_exceeds_sum"]:
        super_add["verdict"] = "Not reproduced. The trust-flow drop is smaller than the sum of the individual components."
    else:
        super_add["verdict"] = "Reproduced."

    # deployed strategy selection: explicit-conflict is chosen even when no conflict pair exists
    strat: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for r in recs:
        if "conflict_strategy" in r:
            strat[(r["run"], r["id"])][r["config"]] = (r["conflict_strategy"], r.get("n_conflict_pairs"))
    base_strat = [v[BASE] for v in strat.values() if BASE in v]
    strategy_audit = {
        "n_instances_per_config": len(base_strat),
        "n_configs": len({c for v in strat.values() for c in v}),
        "n_empty_P_explicit_conflict": sum(1 for s_, n in base_strat if n == 0 and s_ == "explicit_conflict"),
        "share_empty_P_explicit_conflict": (sum(1 for s_, n in base_strat if n == 0 and s_ == "explicit_conflict")
                                            / len(base_strat)) if base_strat else None,
        "n_pairs_strategy_or_P_differs_across_configs": sum(1 for v in strat.values() if len(set(v.values())) > 1),
    }
    runs = sorted({k[0] for k in keys})
    queries = sorted({k[1] for k in keys if all((r, k[1]) in by for r in runs)})
    payload = {
        "analysis": "Ablation v2 — paired comparison and super-additive verdict",
        "n_pairs": len(keys),
        "query_level": query_level(by, ans, queries, runs),
        "strategy_audit": strategy_audit,
        "aggregated_from_results_json": d["aggregated"],
        "paired_vs_full": paired,
        "super_additive": super_add,
        "structural_note": (
            "If no_mars equals full on every pair, the structure in which MARS cannot affect the answer body "
            "(answer_final = answer_raw + citations_text) is confirmed empirically."),
    }
    write_json(Path(args.out), payload)

    rows = []
    for c in [BASE] + OTHERS:
        a = d["aggregated"][c]
        p = paired.get(c, {})
        rows.append({
            "config": c, "mean": a["mean"], "std": a["std"],
            "delta_vs_full": a["delta_vs_full"],
            "paired_mean_delta": p.get("mean_delta"),
            "ci95_low": (p.get("ci95") or [None, None])[0],
            "ci95_high": (p.get("ci95") or [None, None])[1],
            "wilcoxon_p": p.get("wilcoxon_p"),
        })
    write_csv(Path(args.out).parent / "analysis_table.csv", rows)

    print(f"paired n={len(keys)}\n")
    print(f"{'config':16s} {'mean':>8s} {'std':>7s} {'paired Δ':>10s} {'95% CI':>20s} {'p':>8s}")
    for c in [BASE] + OTHERS:
        a = d["aggregated"][c]; p = paired.get(c, {})
        ci = p.get("ci95")
        cis = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else ("identical" if c != BASE else "—")
        ps = f"{p['wilcoxon_p']:.4f}" if p.get("wilcoxon_p") is not None else "—"
        pd = f"{p['mean_delta']:+.4f}" if p else "—"
        print(f"{c:16s} {a['mean']:8.4f} {a['std']:7.4f} {pd:>10s} {cis:>20s} {ps:>8s}")
    print(f"\nsuper-additive: {super_add['verdict']}")
    print(f"[✓] {args.out}")


if __name__ == "__main__":
    main()
