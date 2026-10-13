#!/usr/bin/env python3
"""
Fig. 2 (operational profile, measured on A100) and Fig. 3 (shipped versus corrected routing).

Both read their numbers from the frozen export (`export/paper_numbers.csv`), so a figure cannot
drift from the manuscript. Nothing is typed in.

Figures are drawn at the printed width (the 14.59 cm text block of the single-column
manuscript, FINAL_W) without tight cropping, so the point sizes below are the printed sizes;
no text is smaller than 8 pt.

  python regen_figs.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker  # noqa: F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
EXPORT = HERE.parent / "export" / "paper_numbers.csv"

BLUE, ORANGE, GREEN, GRAY = "#2E5E8E", "#D98C3F", "#4E8B5A", "#8A8A8A"
TC = "#222222"
FINAL_W = 14.59 / 2.54          # printed width in inches
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.edgecolor": TC, "text.color": TC,
    "axes.labelcolor": TC, "xtick.color": TC, "ytick.color": TC,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

V = {r["name"]: float(r["value"]) for r in csv.DictReader(EXPORT.open(encoding="utf-8"))}


def fig2() -> None:
    stages = [("Routing", "lat.dmar_ms", "lat.dmar_p95_ms", BLUE),
              ("Conflict", "lat.cmae_ms", "lat.cmae_p95_ms", BLUE),
              ("Memory", "lat.hmmr_ms", "lat.hmmr_p95_ms", BLUE),
              ("Attrib.", "lat.mars_ms", "lat.mars_p95_ms", BLUE),
              ("Specialist", "lat.specialist_ms", None, ORANGE),
              ("Generation", "lat.generate_ms", None, ORANGE)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(FINAL_W, 3.1), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.7, 1.0]})
    x = np.arange(len(stages))
    means = [V[k] for _, k, _, _ in stages]
    ax.bar(x, means, color=[c for *_, c in stages], width=0.62)
    for i, (_, k, pk, _) in enumerate(stages):
        if pk:
            ax.plot([i, i], [V[k], V[pk]], color=TC, lw=1.0)
            ax.plot([i - 0.12, i + 0.12], [V[pk]] * 2, color=TC, lw=1.0)
        ax.text(i, means[i] * 1.5, f"{means[i]:.0f}" if means[i] >= 100 else f"{means[i]:.1f}",
                ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e4)
    # plain tick labels: log-axis exponents would print below 8 pt
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, *_ in stages], fontsize=8.5, rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("latency (ms, log scale)")
    ax.set_title("(a) per-stage cost,\nmean with p95 whisker", loc="left")

    # the exported share itself, not a re-rounding of it, so the panel matches the text
    share = V["lat.analytic_share"]
    parts = [V["lat.analytic_sum_ms"], V["lat.specialist_ms"], V["lat.generate_ms"]]
    rest = V["lat.total_ms"] - sum(parts)
    ax2.barh([0], [parts[0]], color=BLUE, label="four analytical stages")
    ax2.barh([0], [parts[1]], left=[parts[0]], color=ORANGE, label="specialist inference")
    ax2.barh([0], [parts[2]], left=[sum(parts[:2])], color=GREEN, label="answer generation")
    ax2.barh([0], [rest], left=[sum(parts)], color=GRAY, label="other")
    ax2.set_yticks([])
    ax2.set_xlabel("end-to-end latency (ms)", labelpad=2)
    ax2.set_title(f"(b) budget: analytical\nstages {share*100:.2f}%", loc="left")
    ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4000))
    fig.legend(*ax2.get_legend_handles_labels(), loc="outside lower center", ncol=4,
               frameon=False, handlelength=1.2, columnspacing=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_operational.{ext}")
    plt.close(fig)


def fig3() -> None:
    """(a) when the shipped path discards the solution, (b) what that costs in budget
    violations at each profile, (c) what it buys in quality -- repeated-modality family."""
    profs = [("a100", "A100 80 GB\n(measured)"), ("a6000", "48 GB\u2021\n(profile)"),
             ("4090", "24 GB\u2021\n(profile)")]
    fig = plt.figure(figsize=(FINAL_W, 4.9), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[0.75, 1.25], height_ratios=[1.0, 0.95])
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    w = 0.34

    fams = [("distinct", "distinct\nmodalities"), ("repeated", "repeated\nmodalities")]
    fb = [V[f"dmar.{f}.4090.fallback_rate"] for f, _ in fams]
    # single series: neutral grey, kept apart from the shipped/corrected pair of (b) and (c)
    a1.bar([0, 1], fb, 0.55, color=GRAY)
    for i, v in enumerate(fb):
        a1.text(i, v + 0.03, f"{v:.0%}", ha="center", fontsize=9)
    a1.set_xticks([0, 1])
    a1.set_xticklabels([t for _, t in fams], fontsize=8.5)
    a1.set_ylim(0, 1.18)
    a1.set_ylabel("greedy fallback rate")
    a1.set_title("(a) solution discarded", loc="left", fontsize=9)

    x = np.arange(len(profs))
    # (c) paired shipped/corrected estimates per profile, joined by a line: no truncated bars
    sh_q = [V[f"dmar.repeated.{p}.shipped_qbar"] for p, _ in profs]
    co_q = [V[f"dmar.repeated.{p}.corrected_qbar"] for p, _ in profs]
    for i, (u, v) in enumerate(zip(sh_q, co_q)):
        if abs(u - v) < 1e-9:
            # identical: an orange ring around the blue point keeps both series visible
            a3.scatter([i], [u], s=110, facecolors="none", edgecolors=ORANGE, linewidths=1.6, zorder=2)
            a3.scatter([i], [v], s=28, color=BLUE, zorder=3)
            a3.text(i + 0.09, u, f"{u:.3f}", fontsize=8.5, va="center")
        else:
            a3.plot([i, i], [u, v], color=GRAY, lw=1.2, zorder=1)
            a3.scatter([i], [u], s=34, color=ORANGE, zorder=3)
            a3.scatter([i], [v], s=34, color=BLUE, zorder=3)
            a3.text(i + 0.07, u, f"{u:.3f}", fontsize=8.5, va="center")
            a3.text(i + 0.07, v, f"{v:.3f}", fontsize=8.5, va="center")
    k = len(profs) - 1
    dq = co_q[k] - sh_q[k]
    a3.annotate(f"\u0394$\\bar{{Q}}$ = {dq:.3f}".replace("-", "\u2212"),
                xy=(k - 0.03, (sh_q[k] + co_q[k]) / 2), xytext=(k - 0.75, co_q[k] - 0.006),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
    a3.set_xticks(x)
    a3.set_xticklabels([t.replace("\n", " ") for _, t in profs], fontsize=8.5)
    a3.set_xlim(-0.5, len(profs) - 0.5)
    a3.set_ylim(0.835, 0.885)
    a3.set_ylabel("mean routing\nquality $\\bar{Q}$", fontsize=9)
    a3.set_title("(c) benefit: routing quality", loc="left", fontsize=9)
    for ax, key, lab, lim, ttl, fmt in (
            (a2, "violation_rate", "VRAM budget violation rate", (0, 0.42),
             "(b) cost: budget violations", "{:.3f}"),):
        sh = [V[f"dmar.repeated.{p}.shipped_{key}"] for p, _ in profs]
        co = [V[f"dmar.repeated.{p}.corrected_{key}"] for p, _ in profs]
        ax.bar(x - w / 2, sh, w, color=ORANGE, label="as shipped")
        ax.bar(x + w / 2, co, w, color=BLUE, label="corrected")
        off = (lim[1] - lim[0]) * 0.03
        for i, (u, v) in enumerate(zip(sh, co)):
            # label both bars in (b): a corrected rate of zero has no visible bar otherwise
            same = abs(u - v) < 1e-9 and key != "violation_rate"
            if same:
                ax.text(i, max(u, lim[0]) + off, fmt.format(u), ha="center", fontsize=8)
            else:
                # (b) labels stand upright so adjacent zero labels do not run together
                kw = dict(rotation=90, va="bottom", fontsize=8) if key == "violation_rate" \
                    else dict(fontsize=8)
                ax.text(i - w / 2, max(u, lim[0]) + off, fmt.format(u), ha="center", **kw)
                ax.text(i + w / 2, max(v, lim[0]) + off, fmt.format(v), ha="center", **kw)
        ax.set_xticks(x)
        ax.set_xticklabels([t for _, t in profs], fontsize=8.5)
        ax.set_ylim(*lim)
        ax.set_ylabel(lab, fontsize=9)
        ax.set_title(ttl, loc="left", fontsize=9)
    # one legend for the shipped/corrected pair, shared by (b) and (c)
    h, lab = a2.get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper right", ncol=2, frameon=False, fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_routing.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig2()
    fig3()
    print(f"-> {OUT}/fig2_operational.pdf, {OUT}/fig3_routing.pdf")
