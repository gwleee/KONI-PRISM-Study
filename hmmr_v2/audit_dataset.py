"""
MultiTurnSet v2 self-audit: can the dataset structure alone guarantee an advantage for the
modality-match term (delta)?  Run before any retrieval measurement.

Checks
  A. Probe modality vs target modality: independence (chi-square) and match rates against
     the target and against a random distractor from the same project.
  B. Expected ModalMatch factor (Eq. 17, delta from product settings) for target vs distractors.
  C. Target placement: position of targets within sessions and projects.
  D. Probe difficulty by modality: BGE-M3 cosine(q_probe, q_store) of the target and the best
     distractor, per target modality — a generator artefact that made one modality easier to
     retrieve would show up here.
  E. Builder sources: no expected-outcome language.

  conda run -n artefact-env python audit_dataset.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.harness import assert_real_encoder, bootstrap_gateway, set_profile  # noqa: E402

DELTA = 0.3


def mm(unit, query):
    return 1.0 + DELTA * len(set(unit) & set(query)) / len(set(query))


async def main() -> None:
    set_profile("a100_1")
    bootstrap_gateway()
    from core.config import get_settings  # noqa: PLC0415
    from services.embedding_service import get_embedding_service  # noqa: PLC0415
    assert abs(get_settings().hmmr_delta - DELTA) < 1e-9
    emb = get_embedding_service()
    await assert_real_encoder(emb)

    report: dict = {}
    runs = sorted((HERE / "data" / "runs").glob("run_[0-9].json"))
    for rf in runs:
        run = json.loads(rf.read_text(encoding="utf-8"))
        rows, mm_t, mm_d, pos, match_t, match_d = [], [], [], Counter(), [], []
        diff = {"text": [], "document": [], "image": []}
        for proj in run["projects"]:
            turns = [t for s in proj["sessions"] for t in s["turns"]]
            store_vecs = np.asarray(await emb.encode([t["q_store"] for t in turns]), dtype=np.float64)
            store_vecs /= np.linalg.norm(store_vecs, axis=1, keepdims=True)
            idx = {t["turn_key"]: i for i, t in enumerate(turns)}
            probe_vecs = np.asarray(await emb.encode([p["query"] for p in proj["probes"]]), dtype=np.float64)
            probe_vecs /= np.linalg.norm(probe_vecs, axis=1, keepdims=True)
            for si, s in enumerate(proj["sessions"]):
                for t in s["turns"]:
                    pos[(si, t["turn_idx"] // 5)] += 1
            rng = np.random.default_rng(run["seed"])
            for p, pv in zip(proj["probes"], probe_vecs):
                tgt = tuple(p["target_modalities"])
                q = tuple(p["modalities"])
                rows.append((tgt, q))
                others = [t for t in turns if t["turn_key"] != p["target_turn_key"]]
                d = others[rng.integers(len(others))]
                match_t.append(q == tgt)
                match_d.append(q == tuple(d["modalities"]))
                mm_t.append(mm(tgt, q))
                mm_d.append(float(np.mean([mm(o["modalities"], q) for o in others])))
                sims = store_vecs @ pv
                ti = idx[p["target_turn_key"]]
                best_other = float(np.max(np.delete(sims, ti)))
                diff[p["target_modalities"][-1] if len(p["target_modalities"]) > 1 else "text"].append(
                    (float(sims[ti]), best_other, float(sims[ti] > best_other)))
        labels_t = sorted({r[0] for r in rows})
        labels_q = sorted({r[1] for r in rows})
        table = np.array([[sum(1 for r in rows if r == (a, b)) for b in labels_q] for a in labels_t])
        chi2, pval, dof, _ = chi2_contingency(table)
        report[rf.name] = {
            "A_contingency": {"rows_target": [list(x) for x in labels_t], "cols_probe": [list(x) for x in labels_q],
                              "table": table.tolist(), "chi2": chi2, "dof": dof, "p": pval},
            "A_match_rate_target": float(np.mean(match_t)),
            "A_match_rate_random_distractor": float(np.mean(match_d)),
            "B_mean_modalmatch_target": float(np.mean(mm_t)),
            "B_mean_modalmatch_distractors": float(np.mean(mm_d)),
            "C_target_positions_session_x_half": {f"s{k[0]}h{k[1]}": v for k, v in sorted(pos.items())},
            "D_probe_difficulty_by_target_modality": {
                m: {"n": len(v), "cos_target": float(np.mean([x[0] for x in v])),
                    "cos_best_distractor": float(np.mean([x[1] for x in v])),
                    "target_beats_all_distractors": float(np.mean([x[2] for x in v]))}
                for m, v in diff.items() if v},
        }
    pat = re.compile(r"\b(win|wins|beat|beats|outperform|expected result|upper bound|gain|HMMR|delta)\b|δ", re.I)
    report["E_builder_outcome_language"] = {
        f: [ln.strip() for ln in (HERE / f).read_text(encoding="utf-8").splitlines() if pat.search(ln)]
        for f in ("build_segments.py", "gen_llm.py", "assemble.py")}
    out = HERE / "results" / "dataset_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for k, v in report.items():
        if k.startswith("run_"):
            print(k, "chi2 p=%.3f" % v["A_contingency"]["p"],
                  "match tgt=%.3f dist=%.3f" % (v["A_match_rate_target"], v["A_match_rate_random_distractor"]),
                  "MM tgt=%.4f dist=%.4f" % (v["B_mean_modalmatch_target"], v["B_mean_modalmatch_distractors"]))
            for m, d in v["D_probe_difficulty_by_target_modality"].items():
                print("   ", m, d)
    print("E:", report["E_builder_outcome_language"])


if __name__ == "__main__":
    asyncio.run(main())
