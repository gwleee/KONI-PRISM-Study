"""
HMMR retrieval re-measurement on MultiTurnSet v2 — production code path.

Write path per turn (mirrors gateway/algorithms/pipeline.py):
  DMAR routing decision (_solve_ilp, greedy fallback on failure, as in route_and_execute)
  -> CMAE evaluate() on the turn's specialist outputs  (algorithms/cmae.py)
  -> HMMRRetriever.update_memory()                      (tier-S store, theta_imp promotion to E)
  -> every 10 user turns: HMMRRetriever.store_episodic_summary()  (services/conversation_service.py)

Read path per probe:
  flat_rag       Qdrant cosine ranking over the project's tier-S units (no tier weight, no modality term)
  hmmr_nodelta   HMMRRetriever.retrieve() with delta = 0
  hmmr           HMMRRetriever.retrieve() with the configured delta
Each HMMR configuration is read twice: with the production default token budget and with an
unbounded budget, so that ranking quality can be separated from budget truncation.

Specialist outputs (LLM answer, parsed passage, VLM description) were generated offline with the
same model weights as the product containers (see gen_llm.py); every memory operation and score
above is computed by product code imported read-only.

  conda run -n artefact-env python run_hmmr_v2.py --runs data/runs/run_0.json data/runs/run_1.json data/runs/run_2.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import assert_real_encoder, bootstrap_gateway, provenance, set_profile  # noqa: E402

SUMMARIZE_EVERY = 10          # services/conversation_service.py::_SUMMARIZE_EVERY
TOTAL_TURNS = 100             # routers/conversations.py passes total_turns=100
UNBOUNDED = 10 ** 9
FLAT_LIMIT = 200
MOD_VARIANTS = ("modalities", "modalities_rho50", "modalities_rho100")


class SilentFailure(RuntimeError):
    pass


class LogTap:
    """Wraps a structlog logger and records warning events that product code swallows."""

    WATCH = {"hmmr_search_failed", "novelty_search_failed"}

    def __init__(self, inner):
        self._inner = inner
        self.events: list[tuple[str, dict]] = []

    def warning(self, event, **kw):
        if event in self.WATCH:
            self.events.append((event, kw))
        return self._inner.warning(event, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def rank_metrics(ranked: list[tuple], target_key: str, target_conv: str, target_source: str,
                 source_of: dict) -> dict:
    """ranked: list of ("turn", turn_key) or ("summary", conversation_id) in retrieval order."""
    def first(pred):
        for i, r in enumerate(ranked, 1):
            if pred(r):
                return i
        return None
    strict = first(lambda r: r == ("turn", target_key))
    lenient = first(lambda r: r == ("turn", target_key) or r == ("summary", target_conv))
    same_src = first(lambda r: r[0] == "turn" and source_of.get(r[1]) == target_source)
    out = {"rank": strict, "rank_lenient": lenient, "rank_same_source": same_src, "n_returned": len(ranked)}
    for name, rk in (("", strict), ("_lenient", lenient)):
        out[f"rr{name}"] = 1.0 / rk if rk else 0.0
        for k in (1, 3, 5):
            out[f"r@{k}{name}"] = 1.0 if rk and rk <= k else 0.0
    return out


async def replay_and_probe(run: dict, summaries: dict, nonce: str, log: list) -> list[dict]:
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

    from algorithms import hmmr as hmmr_mod
    from algorithms.cmae import ModalityResponse, get_cmae_evaluator
    from algorithms.dmar import get_dmar_solver
    from algorithms.hmmr import COLLECTION_NAME, HMMRRetriever
    from core.config import HARDWARE_PROFILES, get_settings
    from services.embedding_service import get_embedding_service

    settings = get_settings()
    tap = hmmr_mod.logger if isinstance(hmmr_mod.logger, LogTap) else LogTap(hmmr_mod.logger)
    hmmr_mod.logger = tap

    dmar = get_dmar_solver()
    cmae = get_cmae_evaluator()
    emb = get_embedding_service()
    writer = HMMRRetriever()
    full = HMMRRetriever()
    nodelta = HMMRRetriever()
    nodelta._delta = 0.0
    hw = settings.hardware_profile
    v_max = HARDWARE_PROFILES[hw]["vram_gb"]
    alpha = dmar._compute_alpha(hw)

    records = []
    for proj in run["projects"]:
        project_id = f"hmmrv2-{nonce}-s{run['seed']}-p{proj['project_index']}"
        key_of_query = {}
        source_of = {}
        n_events_before = len(tap.events)
        t0 = time.perf_counter()
        # ── write path ──────────────────────────────────────────────────────
        for sess in proj["sessions"]:
            conv = sess["conversation_id"]
            for n_turn, t in enumerate(sess["turns"], 1):
                key_of_query[t["q_store"]] = t["turn_key"]
                source_of[t["turn_key"]] = t["source"]
                mods = t["modalities"]
                decisions = dmar._solve_ilp(mods, alpha, v_max) or dmar._greedy_fallback(mods, v_max)
                content_for = {"text": t["answer"], "document": t["content"], "image": t["content"]}
                responses = [ModalityResponse(modality=m, model_id=decisions[m], content=content_for[m],
                                              quality_estimate=dmar._q_hat.get((decisions[m], m), 0.8))
                             for m in mods]
                q_est = {r.model_id: r.quality_estimate for r in responses}
                cmae_out = await cmae.evaluate(responses, quality_estimates=q_est)
                await writer.update_memory(
                    query_text=t["q_store"], query_modalities=mods,
                    cmae_trust_weights=cmae_out.trust_weights, project_id=project_id,
                    conversation_id=conv, response_content=t["answer"],
                    turn_idx=t["turn_idx"], total_turns=TOTAL_TURNS)
                if n_turn % SUMMARIZE_EVERY == 0 and (conv, n_turn) in summaries:
                    await writer.store_episodic_summary(
                        summary_text=summaries[(conv, n_turn)], project_id=project_id,
                        conversation_id=conv, importance_score=0.7)
        client = await writer._get_client()
        pts, off = [], None
        while True:
            batch, off = await client.scroll(COLLECTION_NAME, scroll_filter=Filter(must=[
                FieldCondition(key="project_id", match=MatchValue(value=project_id))]),
                limit=256, offset=off, with_payload=True, with_vectors=False)
            pts += batch
            if off is None:
                break
        stored = [{"tier": p.payload["tier"], "turn_key": key_of_query.get(p.payload["query_text"]),
                   "conversation_id": p.payload["conversation_id"], "modalities": p.payload["modalities"],
                   "importance": p.payload["importance_score"]} for p in pts]
        promoted = {s["turn_key"] for s in stored if s["tier"] == "E" and s["turn_key"]}
        log.append({"project_id": project_id, "stored": stored, "write_s": time.perf_counter() - t0})

        def to_ranked(units) -> list[tuple]:
            out = []
            for u in units:
                k = key_of_query.get(u.query_text)
                out.append(("turn", k) if k else ("summary", u.conversation_id))
            return out

        # ── read path ───────────────────────────────────────────────────────
        for pr in proj["probes"]:
            q_vec = await emb.encode_single(pr["query"])
            hits = await client.search(COLLECTION_NAME, query_vector=q_vec.tolist(), limit=FLAT_LIMIT,
                                       query_filter=Filter(must=[
                                           FieldCondition(key="project_id", match=MatchValue(value=project_id)),
                                           FieldCondition(key="tier", match=MatchValue(value="S"))]),
                                       with_payload=True)
            flat = [("turn", key_of_query[h.payload["query_text"]]) for h in hits]
            base = {"seed": run["seed"], "project_id": project_id, "target": pr["target_turn_key"],
                    "target_modalities": pr["target_modalities"],
                    "target_promoted_to_E": pr["target_turn_key"] in promoted}
            args = (pr["target_turn_key"], pr["target_conversation_id"], pr["target_source"], source_of)
            rec = {**base, "config": "flat_rag", "budget": "none", "variant": "-",
                   **rank_metrics(flat, *args)}
            records.append(rec)
            for budget_name, budget in (("default", 2000), ("unbounded", UNBOUNDED)):
                r0 = await nodelta.retrieve(pr["query"], project_id, pr["modalities"], total_token_budget=budget)
                records.append({**base, "config": "hmmr_nodelta", "budget": budget_name, "variant": "-",
                                **rank_metrics(to_ranked(r0.retrieved_units), *args)})
                for var in MOD_VARIANTS:
                    r1 = await full.retrieve(pr["query"], project_id, pr[var], total_token_budget=budget)
                    records.append({**base, "config": "hmmr", "budget": budget_name, "variant": var,
                                    "probe_modalities": pr[var],
                                    **rank_metrics(to_ranked(r1.retrieved_units), *args)})
        if len(tap.events) > n_events_before:
            raise SilentFailure(f"{project_id}: {tap.events[n_events_before:][:3]}")
        await client.delete(COLLECTION_NAME, points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="project_id", match=MatchValue(value=project_id))])))
    return records


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--profile", default="a100_1")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    set_profile(a.profile)
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    bootstrap_gateway()
    from ablation_v2.run_ablation import assert_qdrant_healthy  # noqa: PLC0415
    from services.embedding_service import get_embedding_service  # noqa: PLC0415

    probe = await assert_real_encoder(get_embedding_service())
    await assert_qdrant_healthy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex[:8]
    all_records, status = [], []
    for rf in a.runs:
        run = json.loads(Path(rf).read_text(encoding="utf-8"))
        sp = Path(rf).with_name(Path(rf).stem + "_summaries.json")
        summaries = {(s["conversation_id"], s["after_turn"]): s["summary"]
                     for s in json.loads(sp.read_text(encoding="utf-8"))}
        log: list = []
        t0 = time.perf_counter()
        try:
            recs = await replay_and_probe(run, summaries, nonce, log)
            all_records += recs
            status.append({"run": rf, "seed": run["seed"], "ok": True, "n_records": len(recs),
                           "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as e:  # noqa: BLE001 — failed runs are preserved with their cause
            status.append({"run": rf, "seed": run["seed"], "ok": False,
                           "error": f"{type(e).__name__}: {str(e)[:500]}"})
        (out / f"memory_state_seed{run['seed']}.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        print(status[-1])
    with (out / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "run_status.json").write_text(json.dumps({
        "provenance": provenance({"encoder_probe": probe, "nonce": nonce}), "runs": status},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
