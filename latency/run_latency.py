#!/usr/bin/env python3
"""
Exp C — per-stage end-to-end latency + measured cold-start / warm-start.

Measured stages (paper pipeline order):
  1. dmar_route          routing decision (ILP solve) — excludes the specialist model call
  2. specialist_infer    specialist model inference (per-modality LLM/VLM/STT/DOC)
  3. cmae_encode         CMAE — BGE-M3 encoding + alignment/trust (Eq.7~12)
  4. hmmr_retrieve       HMMR — Qdrant 3-tier retrieval (Eq.13~17)
  5. llm_generate        final answer generation
  6. mars_score          MARS — source attribution/reliability (Eq.18~22)
  7. total               whole-query wall-clock

Principles:
  - Warmup iterations are excluded from the statistics (5 by default).
  - A stage that could not actually be called is never filled with an estimate; it is
    left as status="unavailable" with the reason recorded.
  - VRAM is measured with nvidia-smi before and after each stage.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import (  # noqa: E402
    bootstrap_gateway, provenance, set_profile, settings_snapshot,
    summarize, write_csv, write_json,
)

STAGES = ["dmar_route", "specialist_infer", "cmae_encode", "hmmr_retrieve",
          "llm_generate", "mars_score", "total"]


class SpecialistCallFailed(RuntimeError):
    """The specialist model call failed but no exception was raised."""


def _check_llm_response(res: object) -> str:
    """Verify that the LLMClient.generate() result is a genuine generation.

    generate() in the product code gateway/services/specialist/llm_client.py
    swallows every exception and returns the string f"[LLM error: {e}]". So
    "no exception was raised" is no evidence that the call succeeded. Recording a
    failure response as a valid measurement disguises the DNS failure time (~7ms)
    as inference latency (this was in fact recorded once).
    """
    text = res if isinstance(res, str) else (
        res.get("text", "") if isinstance(res, dict) else "")
    if not isinstance(text, str) or not text.strip():
        raise SpecialistCallFailed(f"empty response: {res!r:.120}")
    if text.startswith("[LLM error:") or text.startswith("[STT error:"):
        raise SpecialistCallFailed(text[:200])
    return text


# ── VRAM measurement ──────────────────────────────────────────────────────────

def vram_used_mib() -> dict[int, int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        d = {}
        for line in out.strip().splitlines():
            i, m = [x.strip() for x in line.split(",")]
            d[int(i)] = int(m)
        return d
    except Exception:
        return {}


# ── Service availability check ────────────────────────────────────────────────────

async def probe_services() -> dict[str, dict]:
    """Check availability of the specialist containers / infrastructure.

    Unavailable services are recorded as such and left as status=unavailable in the results.
    """
    import httpx  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415

    s = get_settings()
    targets = {
        "llm": s.llm_url, "vlm": s.vlm_url, "stt": s.stt_url,
        "doc": s.doc_url, "ffmpeg": s.ffmpeg_url,
    }
    # In-container hostnames are mapped to localhost ports for local runs
    localmap = {
        "llm": "http://localhost:8004", "vlm": "http://localhost:8001",
        "stt": "http://localhost:8002", "doc": "http://localhost:8003",
        "ffmpeg": "http://localhost:8007",
    }
    out: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=5.0) as c:
        for name, url in targets.items():
            for candidate in (localmap[name], url):
                try:
                    r = await c.get(f"{candidate}/health")
                    out[name] = {"available": r.status_code == 200,
                                 "url": candidate, "status": r.status_code,
                                 "body": r.text[:200]}
                    break
                except Exception as e:
                    out[name] = {"available": False, "url": candidate,
                                 "error": f"{type(e).__name__}: {str(e)[:120]}"}
    # Qdrant
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get("http://localhost:6333/readyz")
            out["qdrant"] = {"available": r.status_code == 200, "status": r.status_code}
    except Exception as e:
        out["qdrant"] = {"available": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    return out


# ── Memory seeding (HMMR retrieval is meaningless on an empty collection) ──────────────────────

async def ensure_collection_ready(retries: int = 5) -> dict:
    """Make sure the HMMR collection exists beforehand.

    The product code HMMRRetriever._get_client() awaits _ensure_collection()
    **after** assigning self._client. If collection creation fails transiently,
    later calls short-circuit on `self._client is not None` and never retry the
    creation, so the singleton stays in the 'no collection' state permanently.
    We guarantee it here to avoid skew (empty-collection retrieval latency).
    """
    from qdrant_client import AsyncQdrantClient  # noqa: PLC0415
    from qdrant_client.models import Distance, VectorParams  # noqa: PLC0415

    from algorithms.hmmr import COLLECTION_NAME  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415
    from services.embedding_service import EMBED_DIM  # noqa: PLC0415

    url = os.environ.get("QDRANT_URL") or get_settings().qdrant_url
    last = ""
    for attempt in range(retries):
        try:
            c = AsyncQdrantClient(url=url)
            existing = {x.name for x in (await c.get_collections()).collections}
            if COLLECTION_NAME not in existing:
                await c.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=EMBED_DIM,
                                                distance=Distance.COSINE))
            info = await c.get_collection(COLLECTION_NAME)
            await c.close()
            return {"status": "ready", "url": url, "attempts": attempt + 1,
                    "points": info.points_count}
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:150]}"
            await asyncio.sleep(2)
    return {"status": "failed", "url": url, "error": last}


async def seed_memory(project_id: str, n_units: int) -> dict:
    """Load memory units into the 3 HMMR tiers, so retrieval latency is realistic."""
    from algorithms.hmmr import get_hmmr_retriever  # noqa: PLC0415

    hmmr = get_hmmr_retriever()
    t0 = time.perf_counter()
    stored = 0
    errors: list[str] = []
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
            errors.append(f"{type(e).__name__}: {str(e)[:120]}")
            if len(errors) > 3:
                break
    return {"requested": n_units, "stored": stored,
            "elapsed_s": time.perf_counter() - t0,
            "errors": errors[:3]}


# ── Single-query measurement ─────────────────────────────────────────────────────────

async def timed_query(project_id: str, services: dict, q_text: str,
                      nonce: str = "") -> dict:
    """Measure each pipeline stage separately. Failed stages are left as None."""
    from algorithms.cmae import ModalityResponse, get_cmae_evaluator  # noqa: PLC0415
    from algorithms.dmar import get_dmar_solver  # noqa: PLC0415
    from algorithms.hmmr import get_hmmr_retriever  # noqa: PLC0415
    from algorithms.mars import SourceNode, get_mars_scorer  # noqa: PLC0415
    from core.config import HARDWARE_PROFILES, get_settings  # noqa: PLC0415

    s = get_settings()
    hw = HARDWARE_PROFILES[s.hardware_profile]
    marks: dict[str, float | None] = {k: None for k in STAGES}
    errors: dict[str, str] = {}
    t_total = time.perf_counter()

    # 1. DMAR — measure ILP routing only (the specialist call is split into stage 2)
    #    route_and_execute() combines routing + specialist execution, so to measure
    #    the pure routing cost of Eq.(1)~(5) we call _solve_ilp directly.
    solver = get_dmar_solver()
    modalities = ["text", "image", "audio"]
    try:
        alpha = solver._compute_alpha(s.hardware_profile)
        t = time.perf_counter()
        decisions = solver._solve_ilp(modalities, alpha, hw["vram_gb"])
        marks["dmar_route"] = (time.perf_counter() - t) * 1000
        if decisions is None:
            errors["dmar_route_note"] = "ILP failed → greedy fallback path"
    except Exception as e:
        errors["dmar_route"] = f"{type(e).__name__}: {str(e)[:150]}"

    # 2. Specialist model inference (LLM text path)
    if services.get("llm", {}).get("available"):
        try:
            from services.specialist.llm_client import LLMClient  # noqa: PLC0415
            t = time.perf_counter()
            cl = LLMClient()
            res = await cl.generate(
                [{"role": "user", "content": q_text}], max_new_tokens=64)
            await cl.aclose()
            _check_llm_response(res)          # do not record failure responses as measurements
            marks["specialist_infer"] = (time.perf_counter() - t) * 1000
        except Exception as e:
            errors["specialist_infer"] = f"{type(e).__name__}: {str(e)[:150]}"
    else:
        errors["specialist_infer"] = "llm service unavailable"

    # 3. CMAE
    # The nonce makes the text unique per iteration to bypass the internal
    # EmbeddingService cache. Measuring cache hits makes the BGE-M3 encoding cost
    # near zero and underestimates CMAE/MARS stage latency by more than 100x.
    responses = [
        ModalityResponse(modality="text", model_id="llm_7b",
                         content=f"[{nonce}] {q_text} 에 대한 응답: 정확도는 93.2%로 확인된다."),
        ModalityResponse(modality="image", model_id="vlm_7b",
                         content=f"[{nonce}] 이미지에서 확인된 정확도 지표는 93.0% 부근이다."),
        ModalityResponse(modality="audio", model_id="stt_large",
                         content=f"[{nonce}] 음성 기록상 정확도는 대략 93퍼센트로 언급되었다."),
    ]
    cmae_out = None
    try:
        ev = get_cmae_evaluator()
        t = time.perf_counter()
        cmae_out = await ev.evaluate(responses)
        marks["cmae_encode"] = (time.perf_counter() - t) * 1000
    except Exception as e:
        errors["cmae_encode"] = f"{type(e).__name__}: {str(e)[:150]}"

    # 4. HMMR
    if services.get("qdrant", {}).get("available"):
        try:
            hmmr = get_hmmr_retriever()
            t = time.perf_counter()
            await hmmr.retrieve(
                query_text=q_text, project_id=project_id,
                query_modalities=["text"],
                cmae_trust_weights=cmae_out.trust_weights if cmae_out else None)
            marks["hmmr_retrieve"] = (time.perf_counter() - t) * 1000
        except Exception as e:
            errors["hmmr_retrieve"] = f"{type(e).__name__}: {str(e)[:150]}"
    else:
        errors["hmmr_retrieve"] = "qdrant unavailable"

    # 5. Final answer generation
    answer = (f"[{nonce}] 정확도는 93.2%로 측정되었으며 세 모달리티가 "
              f"대체로 일치한다. 응답 지연은 기준 범위 안에 있다.")
    if services.get("llm", {}).get("available"):
        try:
            from services.specialist.llm_client import LLMClient  # noqa: PLC0415
            t = time.perf_counter()
            cl = LLMClient()
            res = await cl.generate(
                [{"role": "user", "content": q_text}], max_new_tokens=256)
            await cl.aclose()
            text = _check_llm_response(res)   # do not record failure responses as measurements
            marks["llm_generate"] = (time.perf_counter() - t) * 1000
            answer = text
        except Exception as e:
            errors["llm_generate"] = f"{type(e).__name__}: {str(e)[:150]}"
    else:
        errors["llm_generate"] = "llm service unavailable"

    # 6. MARS
    if cmae_out is not None:
        try:
            mars = get_mars_scorer()
            nodes = [
                SourceNode(node_id=f"n{i}", source_type=r.modality,
                           content=r.content, embedding=cmae_out.embeddings[i],
                           trust_weight=float(cmae_out.trust_weights[i]))
                for i, r in enumerate(responses)
            ]
            t = time.perf_counter()
            await mars.score(answer, nodes)
            marks["mars_score"] = (time.perf_counter() - t) * 1000
        except Exception as e:
            errors["mars_score"] = f"{type(e).__name__}: {str(e)[:150]}"

    marks["total"] = (time.perf_counter() - t_total) * 1000
    return {"marks": marks, "errors": errors}


# ── cold-start / warm-start ──────────────────────────────────────────────────

async def measure_cold_start() -> dict:
    """Model load time + first inference vs resident-state inference.

    BGE-M3 (the gateway-resident encoder) can be measured directly in-process.
    Specialist container load time is measured separately from container startup
    (measure_container_cold_start).
    """
    from services.embedding_service import EmbeddingService  # noqa: PLC0415

    texts = ["추론 서버의 정확도는 93.2%로 측정되었다."] * 4

    v0 = vram_used_mib()
    svc = EmbeddingService()          # new instance — not yet loaded
    t = time.perf_counter()
    await svc.encode(texts[:1])       # load + first encoding
    load_plus_first = (time.perf_counter() - t) * 1000
    v1 = vram_used_mib()

    impl = type(svc._model).__name__
    if impl == "_FallbackEncoder":
        return {"status": "invalid", "reason": "BGE-M3 load failed (fallback encoder)"}

    # resident-state encoding with different text each time to bypass the cache
    warm = []
    for i in range(10):
        t = time.perf_counter()
        await svc.encode([f"추론 서버의 정확도는 {90+i}.{i}%로 측정되었다."])
        warm.append((time.perf_counter() - t) * 1000)

    return {
        "status": "measured",
        "component": "bge_m3_encoder",
        "load_plus_first_infer_ms": load_plus_first,
        "warm_infer_ms": summarize(warm),
        "cold_warm_ratio": load_plus_first / float(np.mean(warm)),
        "vram_before_mib": v0,
        "vram_after_mib": v1,
        "vram_delta_mib": {k: v1.get(k, 0) - v0.get(k, 0) for k in v1},
    }


def measure_container_cold_start(service: str, container: str) -> dict:
    """Measure the time from container restart until /health returns 200."""
    import urllib.request  # noqa: PLC0415

    port = {"llm": 8004, "vlm": 8001, "stt": 8002, "doc": 8003}[service]
    url = f"http://localhost:{port}/health"

    def up() -> bool:
        try:
            return urllib.request.urlopen(url, timeout=3).status == 200
        except Exception:
            return False

    if not up():
        return {"status": "unavailable",
                "reason": f"{service} container is not running; cold-start cannot be measured"}

    r = subprocess.run(["docker", "restart", container],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return {"status": "error", "reason": r.stderr[:200]}

    t = time.perf_counter()
    deadline = t + 1800
    while time.perf_counter() < deadline:
        if up():
            return {"status": "measured", "service": service,
                    "restart_to_healthy_s": time.perf_counter() - t}
        time.sleep(2)
    return {"status": "timeout", "waited_s": time.perf_counter() - t}


# ── Main ───────────────────────────────────────────────────────────────────

async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("HARDWARE_PROFILE", "a100_1"))
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed-memory", type=int, default=60)
    ap.add_argument("--project-id", default="expc_latency")
    ap.add_argument("--cold-start-containers", nargs="*", default=[])
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    set_profile(args.profile)
    bootstrap_gateway()
    from core.config import get_settings  # noqa: PLC0415

    services = await probe_services()
    # If the probe connected via localhost, make the clients use the same URL.
    # (Container hostnames do not resolve on the host, so the configured values mismatch)
    for _svc, _key in (("llm", "llm_url"), ("stt", "stt_url")):
        info = services.get(_svc, {})
        if info.get("available") and info.get("url"):
            os.environ[_key.upper()] = info["url"]
    get_settings.cache_clear()  # type: ignore[attr-defined]
    print("=== Service availability ===")
    for k, v in services.items():
        print(f"  {k:8s} {'OK' if v.get('available') else 'UNAVAILABLE'}"
              f"  {v.get('error', v.get('body',''))[:70]}")

    cold = await measure_cold_start()
    print(f"\n=== cold-start (BGE-M3) === {cold.get('status')}")
    if cold.get("status") == "measured":
        print(f"  load+first infer {cold['load_plus_first_infer_ms']:.0f}ms  "
              f"resident infer {cold['warm_infer_ms']['mean']:.1f}±"
              f"{cold['warm_infer_ms']['std']:.1f}ms  "
              f"ratio {cold['cold_warm_ratio']:.0f}×")

    coll = {}
    if services.get("qdrant", {}).get("available"):
        coll = await ensure_collection_ready()
        print(f"\nQdrant collection: {coll}")

    seed = {}
    if coll.get("status") == "ready" and args.seed_memory > 0:
        seed = await seed_memory(args.project_id, args.seed_memory)
        print(f"\nMemory seeding: {seed['stored']}/{seed['requested']} "
              f"({seed['elapsed_s']:.1f}s) errors={seed['errors']}")

    queries = [
        "추론 서버의 최근 정확도와 응답 지연을 알려주세요.",
        "냉각 시스템 가동 온도가 기준을 넘었는지 확인해 주세요.",
        "배송 라우터의 정시율 추이를 요약해 주세요.",
    ]

    print(f"\n=== Repeated measurement (warmup={args.warmup}, reps={args.reps}) ===")
    rows: list[dict] = []
    all_errors: dict[str, int] = {}
    for i in range(args.warmup + args.reps):
        r = await timed_query(args.project_id, services,
                              queries[i % len(queries)],
                              nonce=f"it{i}-{time.time_ns()}")
        is_warm = i >= args.warmup
        row = {"iter": i, "phase": "measure" if is_warm else "warmup"}
        row.update({k: (round(v, 3) if v is not None else "") for k, v in r["marks"].items()})
        rows.append(row)
        for k in r["errors"]:
            all_errors[k] = all_errors.get(k, 0) + 1
        if i == 0 and r["errors"]:
            print("  [first-iteration errors]", json.dumps(r["errors"], ensure_ascii=False)[:400])

    measured = [r for r in rows if r["phase"] == "measure"]
    stats = {}
    for st in STAGES:
        vals = [float(r[st]) for r in measured if r[st] != ""]
        if not vals:
            stats[st] = {"status": "unavailable",
                         "reason": f"measurement failed in every iteration ({all_errors.get(st,0)} times)"}
            continue
        a = np.asarray(vals)
        p50 = float(np.percentile(a, 50))
        # Samples above 5x p50 are treated as client retries/timeouts and
        # reported separately (not discarded; their count and values are kept).
        out_mask = a > max(5.0 * p50, p50 + 1000.0)
        stats[st] = {
            "status": "measured",
            **summarize(a[~out_mask].tolist()),
            "outliers_excluded": {
                "n": int(out_mask.sum()),
                "values_ms": [round(x, 1) for x in a[out_mask].tolist()],
                "note": "above 5x p50 — classified as client retry/timeout",
            },
            "all_samples_summary": summarize(vals),
        }

    container_cold = {}
    for svc in args.cold_start_containers:
        cname = {"llm": "koni_llm", "vlm": "koni_vlm",
                 "stt": "koni_stt", "doc": "koni_doc"}.get(svc)
        if cname:
            print(f"\n[..] {svc} container cold-start measurement")
            container_cold[svc] = measure_container_cold_start(svc, cname)
            print(f"     {container_cold[svc]}")

    out = Path(args.out)
    write_csv(out / f"latency_raw_{args.profile}.csv", rows,
              columns=["iter", "phase"] + STAGES)
    write_json(out / f"results_{args.profile}.json", {
        "experiment": "Exp C — end-to-end latency + cold-start",
        "hardware_profile": args.profile,
        "reps": args.reps, "warmup_excluded": args.warmup,
        "provenance": provenance(),
        "settings": settings_snapshot(),
        "service_availability": services,
        "qdrant_collection": coll,
        "memory_seeding": seed,
        "stage_latency_ms": stats,
        "cold_start_encoder": cold,
        "cold_start_containers": container_cold,
        "stage_error_counts": all_errors,
    })

    print(f"\n{'stage':18s} {'status':12s} {'mean±std (ms)':>22s} {'p95':>9s}")
    for st in STAGES:
        s = stats[st]
        if s["status"] == "measured":
            print(f"{st:18s} {'measured':12s} "
                  f"{s['mean']:12.2f} ± {s['std']:6.2f} {s['p95']:9.2f}")
        else:
            print(f"{st:18s} {'UNAVAILABLE':12s} {s['reason'][:40]:>22s}")
    print(f"\n[✓] {out}/results_{args.profile}.json")


if __name__ == "__main__":
    asyncio.run(main())
