#!/usr/bin/env python3
"""
Exp-Ablation v2 — 실제 파이프라인 호출 기반 ablation 재측정.

기존 `scripts/run_real_experiments.py::run_ablation_experiment()` 는 알고리즘을
한 번도 호출하지 않고 하드코딩 구간에서 rng.uniform 을 뽑아 가중합했다.
본 스크립트는 그 방식을 전면 배제한다:

  - 하드코딩된 품질 구간 없음
  - 난수 가중합 없음
  - "기대 순서"(예: no_trust_flow 가 최저여야 함) 검사 코드 없음
  - super-additive 재현 여부와 무관하게 실측값을 그대로 기록

구성 4종 (원 논문 ABLATION_CONFIGS 중 대응 항목):
  full           DMAR → CMAE → HMMR → LLM → MARS (전체)
  no_trust_flow  CMAE trust 를 균등(1/k)으로 대체해 HMMR·MARS 로의 전파를 차단
  no_hmmr        HMMR 검색 생략 (메모리 컨텍스트 없음)
  no_mars        MARS 생략 (인용 없음)

파이프라인 단계는 gateway/algorithms/pipeline.py::run_pipeline 의 구조를 그대로
따르며, 프롬프트 구성은 제품 코드의 _build_system_prompt/_build_user_message 를
**직접 import 해 사용**한다(재구현하지 않음).

faithfulness 정의는 common/faithfulness.py 참조.
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
    ("no_trust_flow", "w/o CMAE→HMMR/MARS trust flow (trust 균등화)"),
    ("no_hmmr",       "w/o HMMR (메모리 검색 없음)"),
    ("no_mars",       "w/o MARS (귀속·인용 없음)"),
]


class SpecialistCallFailed(RuntimeError):
    pass


class MemoryRetrievalSilentlyEmpty(RuntimeError):
    """HMMR 이 예외 없이 빈 결과만 돌려주는 상태."""


async def assert_qdrant_healthy() -> None:
    """Qdrant 가 살아 있고 컬렉션이 정상인지 확인한다.

    HMMRRetriever.retrieve() 는 tier 별 예외를 삼키고 빈 결과를 반환하므로,
    Qdrant 가 죽어도 실험은 조용히 진행된다. 그러면 메모리를 쓰는 구성이
    no_hmmr 과 동일한 값을 내면서 ablation 이 통째로 무의미해진다
    (실제로 Qdrant 가 'Too many open files' 로 죽은 채 한 번 측정됐다).
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
            # 없으면 만든다. 제품 _get_client() 는 self._client 를 대입한 뒤
            # _ensure_collection() 을 await 하므로, 생성이 한 번 실패하면
            # 싱글톤이 영구히 '컬렉션 없음' 상태로 굳는다. 여기서 선제 보장한다.
            await c.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))
        info = await c.get_collection(COLLECTION_NAME)
        st = str(info.status)
        await c.close()
    except Exception as e:  # noqa: BLE001
        raise MemoryRetrievalSilentlyEmpty(
            f"Qdrant 상태 확인 실패 ({url}): {type(e).__name__}: {str(e)[:150]}") from e
    if "green" not in st.lower():
        raise MemoryRetrievalSilentlyEmpty(f"Qdrant collection status={st}")


def _check_llm(res: object) -> str:
    """llm_client.generate() 는 예외를 삼키고 '[LLM error: ...]' 를 반환한다.
    실패 응답을 정상 결과로 집계하지 않도록 검사한다."""
    text = res if isinstance(res, str) else (
        res.get("text", "") if isinstance(res, dict) else "")
    if not isinstance(text, str) or not text.strip():
        raise SpecialistCallFailed(f"빈 응답: {res!r:.120}")
    if text.startswith("[LLM error:") or text.startswith("[STT error:"):
        raise SpecialistCallFailed(text[:200])
    return text


async def run_one(cfg: str, inst: dict, project_id: str, conv_id: str,
                  turn_idx: int, total_turns: int, seed_memory: bool) -> dict:
    """단일 질의를 지정된 ablation 구성으로 실행하고 답변·중간산출을 반환."""
    from algorithms.cmae import ModalityResponse, get_cmae_evaluator  # noqa: PLC0415
    from algorithms.dmar import ModalityInput, get_dmar_solver  # noqa: PLC0415
    from algorithms.hmmr import get_hmmr_retriever  # noqa: PLC0415
    from algorithms.mars import SourceNode, get_mars_scorer  # noqa: PLC0415
    from algorithms.pipeline import _build_system_prompt, _build_user_message  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415
    from services.specialist.llm_client import LLMClient  # noqa: PLC0415

    s = get_settings()
    t0 = time.perf_counter()

    # ── 1. DMAR: 출처 청크를 모달리티 입력으로 라우팅 (text → llm_7b 패스스루)
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
        # trust 신호 차단: 균등 가중으로 대체 (CMAE 컨텍스트·HMMR·MARS 모두)
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

    # ── 4. LLM 생성 (제품 프롬프트 빌더 사용)
    system_prompt = _build_system_prompt(cmae_out.conflict_strategy)
    user_msg = _build_user_message(inst["query"], cmae_context, memory_context)
    llm = LLMClient()
    # temperature=0.0 → llm/app.py 에서 do_sample=False (greedy).
    # 샘플링 노이즈를 제거해야 구성 간 차이가 디코딩 난수에 묻히지 않는다.
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

    # ── 6. 메모리 갱신 (다음 턴이 HMMR 로 활용) — no_hmmr 구성에서도 동일하게
    #        적재해 구성 간 메모리 상태를 동일하게 유지한다.
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
    ap.add_argument("--n", type=int, default=0, help="0 = 전체")
    ap.add_argument("--profile", default="a100_1")
    ap.add_argument("--ent-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    set_profile(args.profile)
    bootstrap_gateway()

    # 전문 컨테이너 URL 정합: 설정 기본값은 docker 호스트명(http://llm:8004)이라
    # 호스트에서 실행하면 DNS 해석에 실패한다. localhost 매핑이 살아 있으면 그것을 쓴다.
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
    print(f"평가셋 {len(insts)}건 × 구성 {len(CONFIGS)}종 × {args.runs} runs "
          f"= {len(insts)*len(CONFIGS)*args.runs} 회 생성\n")

    # 채점은 생성이 모두 끝난 뒤 한 번에 수행한다.
    # 생성 루프 안에서 NLI 를 돌리면 LLM 과 같은 GPU 를 점유해 생성이 느려지고,
    # 측정 조건도 실행마다 달라진다.
    all_records: list[dict] = []
    per_run: dict[str, list[dict]] = {c: [] for c, _ in CONFIGS}

    import random as _random  # noqa: PLC0415
    for run in range(args.runs):
        # run 마다 project_id 를 분리해 메모리 상태를 독립시킨다.
        # 질의 순서를 run 마다 섞어, greedy 디코딩에서도 run 간 변동이 생기도록 한다
        # (변동의 원천 = HMMR 메모리 축적 상태. 디코딩 난수가 아님).
        # 메모리 격리 단위 주의:
        #   HMMRRetriever.retrieve() 는 **project_id 로만** 필터링하고
        #   conversation_id 는 쓰지 않는다(hmmr.py:242). 따라서 구성을
        #   conversation_id 로만 분리하면 네 구성이 같은 메모리 풀을 공유해
        #   실행 순서대로 서로를 오염시킨다(실제로 처음에 그렇게 측정됐고,
        #   프롬프트가 동일한 full 과 no_mars 가 다른 값을 내면서 드러났다).
        #   구성마다 project_id 를 분리해야 한다.
        order = list(range(len(insts)))
        _random.Random(1000 + run).shuffle(order)
        run_insts = [insts[i] for i in order]
        print(f"── run {run+1}/{args.runs} (질의순서 seed={1000+run})")
        run_scores: dict[str, list[float]] = {c: [] for c, _ in CONFIGS}
        for cfg, _desc in CONFIGS:
            pid = f"ablv2_run{run}_{cfg}"        # 구성별 메모리 격리
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
                    f"[{cfg}] run{run}: 메모리를 쓰는 구성인데 전 질의에서 검색 결과가 0건이다. "
                    f"Qdrant 장애이거나 HMMR 싱글톤이 오염됐을 가능성이 높다. "
                    f"이 상태의 수치는 no_hmmr 과 구분되지 않으므로 측정을 중단한다.")
            per_run[cfg].append({"run": run, "n": len(run_scores[cfg]),
                                 "failures": fails,
                                 "queries_with_memory": with_mem})
            print(f"   {cfg:15s} 생성 {len(run_scores[cfg])}건 fail={fails} "
                  f"({time.perf_counter()-t_cfg:.0f}s)")

    # ── 일괄 채점 (생성 완료 후)
    print("\n채점 중 (NLI)...")
    scorer = FaithfulnessScorer(ent_threshold=args.ent_threshold)
    t_score = time.perf_counter()
    scored = 0
    for rec in all_records:
        if "answer_raw" not in rec:
            continue
        f = scorer.score(rec["answer_raw"], rec["source_chunks"])
        rec.update({f"faith_{k}": v for k, v in f.items()})
        scored += 1
    print(f"  {scored}건 채점 완료 ({time.perf_counter()-t_score:.0f}s)")

    # run×config 별 평균 재구성
    for cfg, _ in CONFIGS:
        for entry in per_run[cfg]:
            vals = [r["faith_faithfulness"] for r in all_records
                    if r.get("config") == cfg and r.get("run") == entry["run"]
                    and "faith_faithfulness" in r]
            entry["mean"] = float(np.mean(vals)) if vals else float("nan")

    # ── 집계
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
        "experiment": "Ablation v2 — 실제 파이프라인 호출 기반 재측정",
        "provenance": provenance(),
        "settings": settings_snapshot(),
        "faithfulness_definition": scorer.definition(),
        "evalset": {"path": args.evalset, "n": len(insts)},
        "runs": args.runs,
        "decoding": {"temperature": 0.0, "mode": "greedy (do_sample=False)",
                     "max_new_tokens": 512,
                     "note": ("greedy 이므로 동일 프롬프트는 동일 답변을 낸다. "
                              "full 과 no_mars 는 프롬프트가 같으므로 답변 본문이 "
                              "일치해야 하며, 이는 MARS 가 사후 단계라는 구조적 사실의 "
                              "검증이기도 하다.")},
        "run_variance_source": ("run 마다 질의 순서를 섞어 HMMR 메모리 축적 상태를 "
                                "다르게 했다. 디코딩 난수는 제거돼 있으므로 run 간 "
                                "변동은 메모리 상태에서만 온다."),
        "memory_isolation": ("project_id 를 run×config 단위로 분리한다. "
                             "HMMR retrieve 는 project_id 로만 필터링하므로 "
                             "(conversation_id 미사용) 구성별 분리에는 project_id 가 "
                             "필요하다."),
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
            print(f"{c:16s} {'FAILED':>8s} — 전 실행 실패 "
                  f"({a['total_failures']}건)")
            continue
        d = "---" if a["delta_vs_full"] is None else f"{a['delta_vs_full']:+.4f}"
        print(f"{c:16s} {a['mean']:8.4f} {a['std']:8.4f} {d:>10s}  "
              f"{['%.4f' % x for x in a['per_run_means']]}")
    print(f"\n[✓] {out/'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
