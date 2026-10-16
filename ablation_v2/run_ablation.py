#!/usr/bin/env python3
"""
Exp-Ablation v2 — ablation re-measurement based on real pipeline calls.

The original `scripts/run_real_experiments.py::run_ablation_experiment()` never called
the algorithms; it drew rng.uniform from hard-coded ranges and took a weighted sum.
This script rules that approach out entirely:

  - no hard-coded quality ranges
  - no weighted sums of random numbers
  - no code checking an "expected ordering" (e.g. no_trust_flow must be lowest)
  - measured values are recorded as-is, whether or not super-additivity reproduces

Four configurations (matching entries of ABLATION_CONFIGS in the original paper):
  full           DMAR → CMAE → HMMR → LLM → MARS (full)
  no_trust_flow  replace CMAE trust with uniform (1/k), cutting propagation to HMMR/MARS
  no_hmmr        skip HMMR retrieval (no memory context)
  no_mars        skip MARS (no citations)

Pipeline stages follow the structure of gateway/algorithms/pipeline.py::run_pipeline
exactly, and prompts are built by **directly importing** the product code's
_build_system_prompt/_build_user_message (not re-implemented).

See common/faithfulness.py for the faithfulness definition.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import (  # noqa: E402
    bootstrap_gateway, provenance, set_profile, settings_snapshot,
    summarize, write_csv, write_json, write_jsonl,
)

CONFIGS = [
    ("full",          "Full KONI-Prism (DMAR+CMAE+HMMR+MARS)"),
    ("no_trust_flow", "w/o CMAE→HMMR/MARS trust flow (uniform trust)"),
    ("no_hmmr",       "w/o HMMR (no memory retrieval)"),
    ("no_mars",       "w/o MARS (no attribution/citations)"),
]


class SpecialistCallFailed(RuntimeError):
    pass


class MemoryRetrievalSilentlyEmpty(RuntimeError):
    """HMMR returns only empty results without raising."""


async def assert_qdrant_healthy() -> None:
    """Check that Qdrant is up and the collection is healthy.

    HMMRRetriever.retrieve() swallows per-tier exceptions and returns empty results,
    so the experiment silently continues even if Qdrant is down. Memory-using configs
    then produce the same values as no_hmmr and the whole ablation becomes meaningless
    (one run was actually measured with Qdrant dead from 'Too many open files').
    """
    import os as _os  # noqa: PLC0415

    from qdrant_client import AsyncQdrantClient  # noqa: PLC0415
    from qdrant_client.models import Distance, VectorParams  # noqa: PLC0415

    from algorithms.hmmr import COLLECTION_NAME  # noqa: PLC0415
    from services.embedding_service import EMBED_DIM  # noqa: PLC0415

    url = _os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        c = AsyncQdrantClient(url=url)
        names = {x.name for x in (await c.get_collections()).collections}
        if COLLECTION_NAME not in names:
            # Create it if missing. The product _get_client() assigns self._client and then
            # awaits _ensure_collection(), so a single creation failure leaves the
            # singleton permanently stuck without a collection. Ensure it here up front.
            await c.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))
        info = await c.get_collection(COLLECTION_NAME)
        st = str(info.status)
        await c.close()
    except Exception as e:  # noqa: BLE001
        raise MemoryRetrievalSilentlyEmpty(
            f"Qdrant health check failed ({url}): {type(e).__name__}: {str(e)[:150]}") from e
    if "green" not in st.lower():
        raise MemoryRetrievalSilentlyEmpty(f"Qdrant collection status={st}")


def _check_llm(res: object) -> str:
    """llm_client.generate() swallows exceptions and returns '[LLM error: ...]'.
    Check so that failed responses are not aggregated as valid results."""
    text = res if isinstance(res, str) else (
        res.get("text", "") if isinstance(res, dict) else "")
    if not isinstance(text, str) or not text.strip():
        raise SpecialistCallFailed(f"empty response: {res!r:.120}")
    if text.startswith("[LLM error:") or text.startswith("[STT error:"):
        raise SpecialistCallFailed(text[:200])
    return text


async def run_one(cfg: str, inst: dict, project_id: str, conv_id: str,
                  turn_idx: int, total_turns: int, seed_memory: bool) -> dict:
    """Run a single query under the given ablation config; return the answer and intermediate outputs."""
    from algorithms.cmae import ModalityResponse, get_cmae_evaluator  # noqa: PLC0415
    from algorithms.dmar import ModalityInput, get_dmar_solver  # noqa: PLC0415
    from algorithms.hmmr import get_hmmr_retriever  # noqa: PLC0415
    from algorithms.mars import SourceNode, get_mars_scorer  # noqa: PLC0415
    from algorithms.pipeline import _build_system_prompt, _build_user_message  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415
    from services.specialist.llm_client import LLMClient  # noqa: PLC0415

    s = get_settings()
    t0 = time.perf_counter()

    # ── 1. DMAR: route source chunks as modality inputs (text → llm_7b pass-through)
    inputs = [ModalityInput(modality="text", content=c) for c in inst["source_chunks"]]
    dmar = get_dmar_solver()
    routing = await dmar.route_and_execute(inputs, s.hardware_profile)

    # ── 2. CMAE
    cmae = get_cmae_evaluator()
    responses = [
        ModalityResponse(modality=o.modality, model_id=o.model_id, content=o.content)
        for o in routing.outputs
    ]
    cmae_out = await cmae.evaluate(responses, quality_estimates=routing.quality_estimates)

    trust = cmae_out.trust_weights
    if cfg == "no_trust_flow":
        # block the trust signal: replace with uniform weights (CMAE context, HMMR and MARS)
        trust = np.full(len(responses), 1.0 / len(responses), dtype=float)
        cmae_out.trust_weights = trust

    cmae_context = cmae.build_context(cmae_out)

    # ── 3. HMMR
    memory_context, n_retrieved = "", 0
    if cfg != "no_hmmr":
        hmmr = get_hmmr_retriever()
        hres = await hmmr.retrieve(
            query_text=inst["query"], project_id=project_id,
            query_modalities=["text"], cmae_trust_weights=trust)
        memory_context = hres.context_text
        n_retrieved = len(hres.retrieved_units)

    # ── 4. LLM generation (using the product prompt builders)
    system_prompt = _build_system_prompt(cmae_out.conflict_strategy)
    user_msg = _build_user_message(inst["query"], cmae_context, memory_context)
    llm = LLMClient()
    # temperature=0.0 → do_sample=False in llm/app.py (greedy).
    # Sampling noise must be removed so config differences are not buried in decoding randomness.
    res = await llm.generate(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_msg}],
        max_new_tokens=512, temperature=0.0)
    await llm.aclose()
    answer_raw = _check_llm(res)

    # ── 5. MARS
    citations, reliability, grade, n_cited = "", None, None, 0
    if cfg != "no_mars":
        mars = get_mars_scorer()
        nodes = [
            SourceNode(node_id=f"c{i}", source_type=r.modality, content=r.content,
                       embedding=cmae_out.embeddings[i], trust_weight=float(trust[i]))
            for i, r in enumerate(responses)
        ]
        mout = await mars.score(answer_raw, nodes)
        citations = mars.format_citations(mout, nodes)
        reliability = float(mout.reliability_score)
        grade = mout.reliability_grade
        n_cited = len(mout.cited_indices)

    answer_final = answer_raw + citations

    # ── 6. memory update (used by HMMR on later turns) — done identically for no_hmmr
    #        too, keeping the memory state identical across configs.
    if seed_memory:
        try:
            hmmr = get_hmmr_retriever()
            await hmmr.update_memory(
                query_text=inst["query"], query_modalities=["text"],
                cmae_trust_weights=trust, project_id=project_id,
                conversation_id=conv_id, response_content=answer_raw,
                turn_idx=turn_idx, total_turns=total_turns)
        except Exception as e:  # noqa: BLE001
            return {"memory_update_error": f"{type(e).__name__}: {str(e)[:100]}"} | {
                "id": inst["id"], "config": cfg, "answer_raw": answer_raw,
                "answer_final": answer_final, "elapsed_ms": (time.perf_counter()-t0)*1000}

    return {
        "id": inst["id"], "config": cfg,
        "query": inst["query"],
        "answer_raw": answer_raw,
        "answer_final": answer_final,
        "citations": citations,
        "conflict_strategy": cmae_out.conflict_strategy,
        "trust_weights": [round(float(x), 4) for x in trust],
        "n_conflict_pairs": len(cmae_out.conflict_pairs),
        "n_memory_retrieved": n_retrieved,
        "mars_reliability": reliability,
        "mars_grade": grade,
        "n_cited": n_cited,
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", default=str(HERE / "data" / "evalset.jsonl"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--profile", default="a100_1")
    ap.add_argument("--ent-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    set_profile(args.profile)
    bootstrap_gateway()

    # Specialist container URL fix-up: the settings default is a docker hostname (http://llm:8004),
    # which fails DNS resolution when run on the host. Use the localhost mapping if it is up.
    import httpx  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415
    for env_key, url in (("LLM_URL", "http://localhost:8004"),):
        try:
            if httpx.get(f"{url}/health", timeout=5.0).status_code == 200:
                os.environ[env_key] = url
        except Exception:  # noqa: BLE001
            pass
    get_settings.cache_clear()  # type: ignore[attr-defined]
    _s = get_settings()
    print(f"LLM endpoint: {_s.llm_url}")

    from common.faithfulness import FaithfulnessScorer  # noqa: PLC0415

    insts = [json.loads(l) for l in Path(args.evalset).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n:
        insts = insts[:args.n]
    await assert_qdrant_healthy()
    print(f"evalset {len(insts)} items × {len(CONFIGS)} configs × {args.runs} runs "
          f"= {len(insts)*len(CONFIGS)*args.runs} generations\n")

    # Scoring is done in one pass after all generation finishes.
    # Running NLI inside the generation loop occupies the same GPU as the LLM, slowing generation,
    # and makes measurement conditions differ between runs.
    all_records: list[dict] = []
    per_run: dict[str, list[dict]] = {c: [] for c, _ in CONFIGS}

    import random as _random  # noqa: PLC0415
    for run in range(args.runs):
        # Separate project_id per run so memory states are independent.
        # Shuffle query order per run so that runs vary even under greedy decoding
        # (source of variation = accumulated HMMR memory state, not decoding randomness).
        # Note on the memory isolation unit:
        #   HMMRRetriever.retrieve() filters **only by project_id** and
        #   does not use conversation_id (hmmr.py:242). So separating configs
        #   only by conversation_id makes the four configs share one memory pool and
        #   contaminate each other in execution order (the first measurement did this;
        #   it surfaced when full and no_mars, with identical prompts, gave different values).
        #   project_id must be separated per config.
        order = list(range(len(insts)))
        _random.Random(1000 + run).shuffle(order)
        run_insts = [insts[i] for i in order]
        print(f"── run {run+1}/{args.runs} (query order seed={1000+run})")
        run_scores: dict[str, list[float]] = {c: [] for c, _ in CONFIGS}
        for cfg, _desc in CONFIGS:
            pid = f"ablv2_run{run}_{cfg}"        # per-config memory isolation
            cid = f"{pid}_conv"
            t_cfg = time.perf_counter()
            fails = 0
            for i, inst in enumerate(run_insts):
                try:
                    rec = await run_one(cfg, inst, pid, cid, i, len(run_insts),
                                        seed_memory=True)
                except SpecialistCallFailed as e:
                    fails += 1
                    all_records.append({"id": inst["id"], "config": cfg, "run": run,
                                        "error": str(e)[:200]})
                    continue
                rec.update({"run": run, "source_chunks": inst["source_chunks"]})
                all_records.append(rec)
                run_scores[cfg].append(rec)
            mem_users = [r for r in run_scores[cfg] if r.get("n_memory_retrieved") is not None]
            with_mem = sum(1 for r in mem_users if r.get("n_memory_retrieved", 0) > 0)
            if cfg != "no_hmmr" and mem_users and with_mem == 0:
                raise MemoryRetrievalSilentlyEmpty(
                    f"[{cfg}] run{run}: a memory-using config retrieved 0 results on every query. "
                    f"Most likely a Qdrant failure or a corrupted HMMR singleton. "
                    f"Values in this state are indistinguishable from no_hmmr, so the measurement is aborted.")
            per_run[cfg].append({"run": run, "n": len(run_scores[cfg]),
                                 "failures": fails,
                                 "queries_with_memory": with_mem})
            print(f"   {cfg:15s} generated {len(run_scores[cfg])} fail={fails} "
                  f"({time.perf_counter()-t_cfg:.0f}s)")

    # ── batch scoring (after generation completes)
    print("\nScoring (NLI)...")
    scorer = FaithfulnessScorer(ent_threshold=args.ent_threshold)
    t_score = time.perf_counter()
    scored = 0
    for rec in all_records:
        if "answer_raw" not in rec:
            continue
        f = scorer.score(rec["answer_raw"], rec["source_chunks"])
        rec.update({f"faith_{k}": v for k, v in f.items()})
        scored += 1
    print(f"  {scored} scored ({time.perf_counter()-t_score:.0f}s)")

    # rebuild per run×config means
    for cfg, _ in CONFIGS:
        for entry in per_run[cfg]:
            vals = [r["faith_faithfulness"] for r in all_records
                    if r.get("config") == cfg and r.get("run") == entry["run"]
                    and "faith_faithfulness" in r]
            entry["mean"] = float(np.mean(vals)) if vals else float("nan")

    # ── aggregation
    out = Path(args.out)
    write_jsonl(out / "raw_records.jsonl", all_records)

    agg = {}
    for cfg, desc in CONFIGS:
        means = [r["mean"] for r in per_run[cfg] if not np.isnan(r["mean"])]
        agg[cfg] = {
            "description": desc,
            "per_run_means": means,
            "mean": float(np.mean(means)) if means else None,
            "std": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "n_runs": len(means),
            "n_per_run": per_run[cfg][0]["n"] if per_run[cfg] else 0,
            "total_failures": sum(r["failures"] for r in per_run[cfg]),
        }
    full_mean = agg["full"]["mean"]
    for cfg in agg:
        agg[cfg]["delta_vs_full"] = (
            None if full_mean is None or agg[cfg]["mean"] is None
            else agg[cfg]["mean"] - full_mean)

    diag = {}
    for cfg, _ in CONFIGS:
        recs = [r for r in all_records if r.get("config") == cfg and "faith_faithfulness" in r]
        if not recs:
            continue
        diag[cfg] = {
            "n_units_mean": float(np.mean([r["faith_n_units"] for r in recs])),
            "contradicted_rate_mean": float(np.mean([r["faith_contradicted_rate"] for r in recs])),
            "answer_chars_mean": float(np.mean([len(r["answer_raw"]) for r in recs])),
            "n_memory_retrieved_mean": float(np.mean([r.get("n_memory_retrieved", 0) for r in recs])),
            "mars_reliability_mean": (
                float(np.mean([r["mars_reliability"] for r in recs
                               if r.get("mars_reliability") is not None]))
                if any(r.get("mars_reliability") is not None for r in recs) else None),
            "strategy_distribution": {
                k: sum(1 for r in recs if r.get("conflict_strategy") == k) / len(recs)
                for k in ("trust_dominant", "explicit_conflict", "abstain")},
        }

    write_json(out / "results.json", {
        "experiment": "Ablation v2 — re-measurement based on real pipeline calls",
        "provenance": provenance(),
        "settings": settings_snapshot(),
        "faithfulness_definition": scorer.definition(),
        "evalset": {"path": args.evalset, "n": len(insts)},
        "runs": args.runs,
        "decoding": {"temperature": 0.0, "mode": "greedy (do_sample=False)",
                     "max_new_tokens": 512,
                     "note": ("Greedy decoding, so identical prompts yield identical answers. "
                              "full and no_mars share the same prompt, so their answer bodies "
                              "must match; this also verifies the structural fact that MARS "
                              "is a post-hoc stage.")},
        "run_variance_source": ("Query order is shuffled per run so the accumulated HMMR memory "
                                "state differs. Decoding randomness is removed, so variation "
                                "across runs comes only from the memory state."),
        "memory_isolation": ("project_id is separated per run×config. "
                             "HMMR retrieve filters only by project_id "
                             "(conversation_id unused), so separating configs requires "
                             "project_id."),
        "configs": {c: d for c, d in CONFIGS},
        "aggregated": agg,
        "per_run": per_run,
        "diagnostics": diag,
    })

    rows = [{"config": c, "description": agg[c]["description"],
             "mean": agg[c]["mean"], "std": agg[c]["std"],
             "delta_vs_full": agg[c]["delta_vs_full"],
             "per_run": ";".join(f"{x:.4f}" for x in agg[c]["per_run_means"])}
            for c, _ in CONFIGS]
    write_csv(out / "ablation_summary.csv", rows)

    print(f"\n{'config':16s} {'mean':>8s} {'std':>8s} {'Δ vs full':>10s}  per-run")
    for c, _ in CONFIGS:
        a = agg[c]
        if a["mean"] is None:
            print(f"{c:16s} {'FAILED':>8s} — all runs failed "
                  f"({a['total_failures']} failures)")
            continue
        d = "---" if a["delta_vs_full"] is None else f"{a['delta_vs_full']:+.4f}"
        print(f"{c:16s} {a['mean']:8.4f} {a['std']:8.4f} {d:>10s}  "
              f"{['%.4f' % x for x in a['per_run_means']]}")
    print(f"\n[✓] {out/'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
