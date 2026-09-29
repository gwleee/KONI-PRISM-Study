"""
Paired analysis of HMMR v2 records (same statistics as ablation_v2/analyze_results.py):
mean over probes per run, std over run means, paired delta over (seed, probe) pairs,
t-interval 95% CI of the paired delta, Wilcoxon signed-rank p.

target_level block (reported path): the three seeds re-deal the same pool of turns into new
conversation sets, so most target turns are probed once per seed. Probes of the same target are
repeated measurements, not independent ones. The unit is therefore the target turn: each target's
reciprocal rank is averaged over the seeds in which it is probed, paired differences are taken per
target, and we report a percentile bootstrap 95% CI over targets (10,000 resamples, fixed seed)
and a two-sided sign-flip permutation test (100,000 draws). Seed variation is reported as ICC(1)
over targets probed in all three seeds. The (seed, probe) path above is kept for comparison.

  python analyze.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
METRICS = ("rr", "r@1", "r@3", "r@5", "rr_lenient")
SEED = 20260929
N_BOOT = 10_000
N_PERM = 100_000


def per_target(rows: dict, metric: str = "rr") -> dict:
    acc = defaultdict(list)
    for (_, _, target), r in rows.items():
        acc[target].append(r[metric])
    return {t: float(np.mean(v)) for t, v in acc.items()}


def icc1(mat: np.ndarray) -> float:
    n, k = mat.shape
    grand = mat.mean()
    msb = k * ((mat.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((mat - mat.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    den = msb + (k - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def target_level(names: dict, comps: list) -> dict:
    rng = np.random.default_rng(SEED)
    targets = sorted(per_target(names["flat_rag"]))
    n = len(targets)
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, n))
    means = {c: per_target(names[c]) for c in {x for pair in comps for x in pair}}
    seeds_per_target = defaultdict(set)
    for (seed, _, target) in names["flat_rag"]:
        seeds_per_target[target].add(seed)
    all3 = [t for t in targets if len(seeds_per_target[t]) == 3]

    def seed_matrix(rows: dict, other: dict | None = None) -> np.ndarray:
        cell = defaultdict(dict)
        for (seed, _, target), r in rows.items():
            v = r["rr"] - (other[(seed, _, target)]["rr"] if other is not None else 0.0)
            cell[target][seed] = v
        return np.array([[cell[t][s] for s in sorted(cell[t])] for t in all3])

    out = {"unit": "target turn", "n_targets": n, "n_targets_all_seeds": len(all3),
           "n_probes": len(names["flat_rag"]), "seed": SEED, "n_bootstrap": N_BOOT,
           "n_permutation": N_PERM,
           "mrr": {c: float(np.mean([m[t] for t in targets])) for c, m in means.items()},
           "icc1_flat_rr": icc1(seed_matrix(names["flat_rag"])),
           "icc1_full_minus_flat_rr": icc1(seed_matrix(names["hmmr@default"], names["flat_rag"])),
           "contrasts": {}}
    for a, b in comps:
        d = np.array([means[a][t] - means[b][t] for t in targets])
        e = {"mean_delta": float(d.mean()), "n_targets_nonzero": int(np.count_nonzero(d)),
             "n_targets_worse": int((d < 0).sum()), "n_targets_better": int((d > 0).sum())}
        if np.count_nonzero(d) == 0:
            e["identical"] = True
        else:
            bm = d[boot_idx].mean(axis=1)
            e["bootstrap_ci95"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
            e["permutation_p"] = float(((np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12).sum() + 1)
                                       / (N_PERM + 1))
            e["ci_includes_zero"] = bool(e["bootstrap_ci95"][0] <= 0 <= e["bootstrap_ci95"][1])
        out["contrasts"][f"{a} - {b}"] = e
    return out


def load():
    recs = [json.loads(l) for l in (RES / "records.jsonl").open(encoding="utf-8")]
    table = defaultdict(dict)   # (config, budget, variant) -> {(seed, project, target): rec}
    for r in recs:
        table[(r["config"], r["budget"], r["variant"])][(r["seed"], r["project_id"], r["target"])] = r
    return table


def summary(rows: dict, metric: str) -> dict:
    by_seed = defaultdict(list)
    for (seed, _, _), r in rows.items():
        by_seed[seed].append(r[metric])
    run_means = [float(np.mean(v)) for _, v in sorted(by_seed.items())]
    return {"mean": float(np.mean([r[metric] for r in rows.values()])),
            "std_runs": float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0,
            "run_means": run_means, "n": len(rows)}


def paired(a: dict, b: dict, metric: str) -> dict:
    keys = sorted(set(a) & set(b))
    d = np.array([a[k][metric] - b[k][metric] for k in keys])
    out = {"n_pairs": len(keys), "delta": float(d.mean()), "n_nonzero": int(np.count_nonzero(d)),
           "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}
    if np.count_nonzero(d) == 0:
        out.update({"ci95": None, "wilcoxon_p": None, "identical": True})
        return out
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    out.update({"ci95": [float(ci[0]), float(ci[1])], "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
                "identical": False})
    return out


def main() -> None:
    t = load()
    F = t[("flat_rag", "none", "-")]
    out: dict = {"configs": {}, "paired": {}, "by_target_modality": {}, "by_promotion": {}, "coverage": {}}
    names = {
        "flat_rag": F,
        "hmmr_nodelta@default": t[("hmmr_nodelta", "default", "-")],
        "hmmr@default": t[("hmmr", "default", "modalities")],
        "hmmr_nodelta@unbounded": t[("hmmr_nodelta", "unbounded", "-")],
        "hmmr@unbounded": t[("hmmr", "unbounded", "modalities")],
        "hmmr_rho50@default": t[("hmmr", "default", "modalities_rho50")],
        "hmmr_rho100@default": t[("hmmr", "default", "modalities_rho100")],
        "hmmr_rho50@unbounded": t[("hmmr", "unbounded", "modalities_rho50")],
        "hmmr_rho100@unbounded": t[("hmmr", "unbounded", "modalities_rho100")],
    }
    for n, rows in names.items():
        out["configs"][n] = {m: summary(rows, m) for m in METRICS}
        out["coverage"][n] = {
            "target_not_returned": float(np.mean([r["rank"] is None for r in rows.values()])),
            "mean_n_returned": float(np.mean([r["n_returned"] for r in rows.values()]))}
    comps = [
        ("hmmr@default", "hmmr_nodelta@default"), ("hmmr_nodelta@default", "flat_rag"),
        ("hmmr@default", "flat_rag"),
        ("hmmr@unbounded", "hmmr_nodelta@unbounded"), ("hmmr_nodelta@unbounded", "flat_rag"),
        ("hmmr@unbounded", "flat_rag"),
        ("hmmr_rho50@default", "hmmr_nodelta@default"), ("hmmr_rho100@default", "hmmr_nodelta@default"),
        ("hmmr_rho50@unbounded", "hmmr_nodelta@unbounded"), ("hmmr_rho100@unbounded", "hmmr_nodelta@unbounded"),
    ]
    for a, b in comps:
        out["paired"][f"{a} - {b}"] = {m: paired(names[a], names[b], m) for m in METRICS}
    out["target_level"] = target_level(names, [("hmmr@default", "flat_rag"),
                                               ("hmmr_nodelta@default", "flat_rag"),
                                               ("hmmr@default", "hmmr_nodelta@default")])
    for mod in ("text", "document", "image"):
        sel = lambda rows: {k: r for k, r in rows.items() if r["target_modalities"][-1] == mod}  # noqa: E731
        out["by_target_modality"][mod] = {n: summary(sel(names[n]), "rr")["mean"]
                                          for n in ("flat_rag", "hmmr_nodelta@default", "hmmr@default",
                                                    "hmmr_nodelta@unbounded", "hmmr@unbounded")}
        out["by_target_modality"][mod]["n"] = len(sel(F))
    for flag in (True, False):
        sel = lambda rows: {k: r for k, r in rows.items() if r["target_promoted_to_E"] == flag}  # noqa: E731
        out["by_promotion"][f"promoted={flag}"] = {n: summary(sel(names[n]), "rr")["mean"]
                                                   for n in ("flat_rag", "hmmr_nodelta@default", "hmmr@default",
                                                             "hmmr_nodelta@unbounded", "hmmr@unbounded")}
        out["by_promotion"][f"promoted={flag}"]["n"] = len(sel(F))
    (RES / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"{'config':26s} {'MRR':>7s} {'±runs':>6s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'notret':>7s} {'nret':>5s}")
    for n in names:
        c = out["configs"][n]
        print(f"{n:26s} {c['rr']['mean']:.4f} {c['rr']['std_runs']:.4f} {c['r@1']['mean']:.3f} "
              f"{c['r@3']['mean']:.3f} {c['r@5']['mean']:.3f} {out['coverage'][n]['target_not_returned']:.3f} "
              f"{out['coverage'][n]['mean_n_returned']:5.1f}")
    print()
    for k, v in out["paired"].items():
        p = v["rr"]
        ci = f"[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]" if p["ci95"] else "identical"
        pv = f"{p['wilcoxon_p']:.2e}" if p["wilcoxon_p"] is not None else "-"
        print(f"{k:50s} dMRR {p['delta']:+.4f} {ci} p={pv}  (+{p['n_pos']}/-{p['n_neg']})")
    print()
    print("by target modality (MRR):", json.dumps(out["by_target_modality"], indent=0))
    print("by promotion (MRR):", json.dumps(out["by_promotion"], indent=0))


if __name__ == "__main__":
    main()
