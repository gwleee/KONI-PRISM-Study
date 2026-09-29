#!/usr/bin/env python3
"""
MARS attribution 실측 — tab:mars 대체용.

MARS 는 답변 본문에 관여하지 않는 attribution 레이어임이 구조적으로 확정됐으므로
(ablation_v2: no_mars 가 full 과 90/90 쌍 동일), 평가 축은 인용 정확성이다.

정의 (지지 판정은 faithfulness 와 동일 NLI 게이트 재사용)
  support(s, u) = 1  iff  max_{s' ∈ sentences(s)} P_entail(s' → u) ≥ 0.5
  supports(s)   = 1  iff  ∃u ∈ U(A) : support(s, u) = 1

  citation precision = |{s ∈ Cited : supports(s)}| / |Cited|
  citation recall    = |{u : supported(u) ∧ ∃s ∈ Cited, support(s,u)}| / |{u : supported(u)}|
                       (사실 단위 기준 — 채택한 정의)
  source-level recall= |{s : supports(s) ∧ s ∈ Cited}| / |{s : supports(s)}|  (보조)

θ_sent 스윕
  운영값 θ_sent=0.70 에서는 인용이 **구조적으로 불가능**하다.
  Eq.(21) 은 max_p G(p,n) ≥ θ_sent 를 요구하는데 G = cos·T̃_n·φ 이고
  T̃ 는 softmax 출력(합=1)이라 k개 소스에서 max T̃ ≤ 1/k 부근에 머문다.
  k=2 에서도 G ≤ 1.0×0.5×1.2 = 0.60 < 0.70 이다.
  따라서 운영값 결과와 함께 θ_sent 를 낮춘 구간을 스윕해 인용이 발생하는
  지점과 그때의 정밀도/재현율을 함께 보고한다.

통제 조건 (ablation_v2 의 지표 검증 방식 준수)
  shuffle_within : 같은 질의의 소스 집합에서 같은 개수를 무작위 선택
  shuffle_across : 다른 질의의 소스 집합을 인용으로 사용 (정밀도 ~0 이어야 함)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import (  # noqa: E402
    bootstrap_gateway, provenance, set_profile, settings_snapshot,
    write_csv, write_json, write_jsonl,
)

THETA_SWEEP = [0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]


class SpecialistCallFailed(RuntimeError):
    pass


def _check_llm(res: object) -> str:
    text = res if isinstance(res, str) else (
        res.get("text", "") if isinstance(res, dict) else "")
    if not isinstance(text, str) or not text.strip():
        raise SpecialistCallFailed(f"빈 응답: {res!r:.120}")
    if text.startswith("[LLM error:"):
        raise SpecialistCallFailed(text[:200])
    return text


async def generate_one(inst: dict, project_id: str, conv_id: str,
                       turn_idx: int, total: int) -> dict:
    """full 구성으로 한 질의를 실행하고 MARS 입력/출력을 모두 보존한다."""
    from algorithms.cmae import ModalityResponse, get_cmae_evaluator  # noqa: PLC0415
    from algorithms.dmar import ModalityInput, get_dmar_solver  # noqa: PLC0415
    from algorithms.hmmr import get_hmmr_retriever  # noqa: PLC0415
    from algorithms.mars import SourceNode, get_mars_scorer  # noqa: PLC0415
    from algorithms.pipeline import _build_system_prompt, _build_user_message  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415
    from services.specialist.llm_client import LLMClient  # noqa: PLC0415

    s = get_settings()
    inputs = [ModalityInput(modality="text", content=c) for c in inst["source_chunks"]]
    routing = await get_dmar_solver().route_and_execute(inputs, s.hardware_profile)

    cmae = get_cmae_evaluator()
    responses = [ModalityResponse(modality=o.modality, model_id=o.model_id,
                                  content=o.content) for o in routing.outputs]
    cmae_out = await cmae.evaluate(responses, quality_estimates=routing.quality_estimates)
    trust = cmae_out.trust_weights

    hres = await get_hmmr_retriever().retrieve(
        query_text=inst["query"], project_id=project_id,
        query_modalities=["text"], cmae_trust_weights=trust)

    llm = LLMClient()
    res = await llm.generate(
        [{"role": "system", "content": _build_system_prompt(cmae_out.conflict_strategy)},
         {"role": "user", "content": _build_user_message(
             inst["query"], cmae.build_context(cmae_out), hres.context_text)}],
        max_new_tokens=512, temperature=0.0)
    await llm.aclose()
    answer = _check_llm(res)

    mars = get_mars_scorer()
    nodes = [SourceNode(node_id=f"c{i}", source_type=r.modality, content=r.content,
                        embedding=cmae_out.embeddings[i], trust_weight=float(trust[i]))
             for i, r in enumerate(responses)]

    # θ_sent 를 바꿔가며 재채점 — 임베딩·답변 재사용(재생성 없음)
    by_theta = {}
    for th in THETA_SWEEP:
        mars._theta_sent = th
        out = await mars.score(answer, nodes)
        by_theta[f"{th:.2f}"] = {
            "cited_indices": list(out.cited_indices),
            "n_cited": len(out.cited_indices),
            "contributions": [round(float(c), 5) for c in out.contributions],
            "reliability_score": float(out.reliability_score),
            "reliability_grade": out.reliability_grade,
            "citations_text_len": len(mars.format_citations(out, nodes)),
        }
    mars._theta_sent = get_settings().mars_theta_sent  # 원복

    await get_hmmr_retriever().update_memory(
        query_text=inst["query"], query_modalities=["text"], cmae_trust_weights=trust,
        project_id=project_id, conversation_id=conv_id,
        response_content=answer, turn_idx=turn_idx, total_turns=total)

    return {
        "id": inst["id"], "query": inst["query"], "answer": answer,
        "source_chunks": inst["source_chunks"],
        "relevance_mask": inst.get("relevance_mask"),
        "trust_weights": [round(float(x), 4) for x in trust],
        "n_sources": len(nodes),
        "by_theta": by_theta,
    }


def metrics_for(cited: list[int], support: np.ndarray,
                relevance: list[bool] | None = None) -> dict:
    """support: (n_units, n_sources) 불리언 행렬.

    relevance 가 주어지면 NLI 와 독립적인 정답(해당 질의의 문서에서 온 청크인가)에
    대한 정밀도/재현율도 함께 계산한다. 무작위 인용의 기대 정밀도는
    n_relevant / n_sources 이므로 이 값이 비교 기준선이 된다.
    """
    n_units, n_src = support.shape
    supports_src = support.any(axis=0)               # 소스가 무언가를 지지하는가
    supported_units = support.any(axis=1)            # 단위가 어떤 소스로든 지지되는가

    if cited:
        prec = float(supports_src[cited].mean())
        cov = support[:, cited].any(axis=1)          # 인용된 소스가 지지하는 단위
        rec_unit = (float((cov & supported_units).sum() / supported_units.sum())
                    if supported_units.sum() else float("nan"))
        rec_src = (float(np.isin(np.where(supports_src)[0], cited).mean())
                   if supports_src.sum() else float("nan"))
    else:
        prec = float("nan")                          # 인용이 없으면 정밀도 미정의
        rec_unit = 0.0 if supported_units.sum() else float("nan")
        rec_src = 0.0 if supports_src.sum() else float("nan")
    out = {"precision": prec, "recall_unit": rec_unit, "recall_source": rec_src,
           "n_cited": len(cited), "n_sources": n_src,
           "n_units": int(n_units), "n_supported_units": int(supported_units.sum()),
           "n_supporting_sources": int(supports_src.sum())}
    if relevance is not None:
        rel = np.asarray(relevance, dtype=bool)
        out["precision_relevance"] = (float(rel[cited].mean()) if cited else float("nan"))
        out["recall_relevance"] = (float(np.isin(np.where(rel)[0], cited).mean())
                                   if rel.sum() else float("nan"))
        out["random_precision_baseline"] = float(rel.mean())
    return out


def agg(vals: list[float]) -> dict:
    a = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    return {"mean": float(a.mean()) if a.size else None,
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", default=str(HERE.parent / "ablation_v2" / "data" / "evalset.jsonl"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--profile", default="a100_1")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    set_profile(args.profile)
    bootstrap_gateway()
    import httpx  # noqa: PLC0415
    from core.config import get_settings  # noqa: PLC0415
    try:
        if httpx.get("http://localhost:8004/health", timeout=5).status_code == 200:
            os.environ["LLM_URL"] = "http://localhost:8004"
    except Exception:  # noqa: BLE001
        pass
    get_settings.cache_clear()  # type: ignore[attr-defined]

    sys.path.insert(0, str(HERE.parent / "ablation_v2"))
    from run_ablation import assert_qdrant_healthy  # noqa: PLC0415
    await assert_qdrant_healthy()

    from common.faithfulness import FaithfulnessScorer, split_fact_units  # noqa: PLC0415
    scorer = FaithfulnessScorer()

    insts = [json.loads(l) for l in Path(args.evalset).read_text(encoding="utf-8").splitlines() if l.strip()][:args.n]
    print(f"질의 {len(insts)} × {args.runs} runs = {len(insts)*args.runs} 생성\n"
          f"운영 θ_sent={get_settings().mars_theta_sent}, θ_cite={get_settings().mars_theta_cite}\n")

    records: list[dict] = []
    for run in range(args.runs):
        order = list(range(len(insts)))
        random.Random(1000 + run).shuffle(order)
        pid = f"marsattr_run{run}"
        t0 = time.perf_counter()
        for i, idx in enumerate(order):
            try:
                r = await generate_one(insts[idx], pid, f"{pid}_conv", i, len(insts))
            except SpecialistCallFailed as e:
                records.append({"run": run, "id": insts[idx]["id"], "error": str(e)[:200]})
                continue
            r["run"] = run
            records.append(r)
        print(f"  run {run+1}/{args.runs}: {time.perf_counter()-t0:.0f}s")

    # ── NLI 지지 행렬 계산 (질의당 1회) + 지표 산출
    print("\n지지 행렬 계산 (NLI)...")
    rng = random.Random(4242)
    ok = [r for r in records if "answer" in r]
    rows: list[dict] = []
    t0 = time.perf_counter()
    for k, r in enumerate(ok):
        units = split_fact_units(r["answer"])
        prem_per_src = [scorer._premises([c]) for c in r["source_chunks"]]
        ei = scorer._label_idx["entailment"] if scorer._label_idx else None
        scorer._ensure(); ei = scorer._label_idx["entailment"]
        support = np.zeros((len(units), len(r["source_chunks"])), dtype=bool)
        for si, prem in enumerate(prem_per_src):
            if not prem or not units:
                continue
            P = scorer._nli([(p, u) for u in units for p in prem]).reshape(len(units), len(prem), 3)
            support[:, si] = P[:, :, ei].max(axis=1) >= scorer.ent_threshold
        r["_support"] = support

        for th, info in r["by_theta"].items():
            rows.append({"run": r["run"], "id": r["id"], "condition": f"theta_{th}",
                         "theta_sent": float(th),
                         **metrics_for(info["cited_indices"], support, r.get("relevance_mask")),
                         "reliability_score": info["reliability_score"],
                         "reliability_grade": info["reliability_grade"]})

        # 통제 조건 — 인용이 실제로 발생하는 최저 θ 기준
        working = next((t for t in sorted(r["by_theta"], key=float)
                        if r["by_theta"][t]["n_cited"] > 0), None)
        if working:
            n_c = r["by_theta"][working]["n_cited"]
            n_s = len(r["source_chunks"])
            rows.append({"run": r["run"], "id": r["id"], "condition": "control_shuffle_within",
                         "theta_sent": float(working),
                         **metrics_for(rng.sample(range(n_s), n_c), support,
                                       r.get("relevance_mask")),
                         "reliability_score": None, "reliability_grade": None})
            other = ok[(k + len(ok) // 2) % len(ok)]
            o_units = split_fact_units(other["answer"])
            o_prem = [scorer._premises([c]) for c in other["source_chunks"]]
            cross = np.zeros((len(units), len(other["source_chunks"])), dtype=bool)
            for si, prem in enumerate(o_prem):
                if prem and units:
                    P = scorer._nli([(p, u) for u in units for p in prem]).reshape(len(units), len(prem), 3)
                    cross[:, si] = P[:, :, ei].max(axis=1) >= scorer.ent_threshold
            rows.append({"run": r["run"], "id": r["id"], "condition": "control_shuffle_across",
                         "theta_sent": float(working),
                         **metrics_for(list(range(min(n_c, cross.shape[1]))), cross),
                         "reliability_score": None, "reliability_grade": None})
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(ok)}")
    print(f"  완료 ({time.perf_counter()-t0:.0f}s)")

    for r in records:
        r.pop("_support", None)
    out = Path(args.out)
    write_jsonl(out / "raw_records.jsonl", records)
    write_csv(out / "per_query_metrics.csv", rows)

    conds = sorted({r["condition"] for r in rows},
                   key=lambda c: (not c.startswith("theta"), c))
    summary = {}
    for c in conds:
        sub = [r for r in rows if r["condition"] == c]
        summary[c] = {
            "precision": agg([r["precision"] for r in sub]),
            "recall_unit": agg([r["recall_unit"] for r in sub]),
            "recall_source": agg([r["recall_source"] for r in sub]),
            "precision_relevance": agg([r.get("precision_relevance", float("nan")) for r in sub]),
            "recall_relevance": agg([r.get("recall_relevance", float("nan")) for r in sub]),
            "random_precision_baseline": agg([r.get("random_precision_baseline", float("nan")) for r in sub]),
            "n_cited_mean": float(np.mean([r["n_cited"] for r in sub])),
            "queries_with_any_citation": int(sum(1 for r in sub if r["n_cited"] > 0)),
            "n_queries": len(sub),
            "grade_distribution": {
                g: sum(1 for r in sub if r["reliability_grade"] == g)
                for g in ("High", "Medium", "Low")},
        }

    write_json(out / "results.json", {
        "experiment": "MARS attribution 실측 (tab:mars 대체)",
        "provenance": provenance(),
        "settings": settings_snapshot(),
        "definition": {
            "support": "max_{s' in sentences(s)} P_entail(s'->u) >= 0.5",
            "precision": "|{s in Cited : supports(s)}| / |Cited|",
            "recall_unit": "|{u supported and covered by a cited source}| / |{u supported}|",
            "recall_source": "|{s supporting and cited}| / |{s supporting}|",
            "nli_model": scorer.definition()["nli_model"],
        },
        "structural_note": (
            "Eq.(21) 의 max_p G(p,n) >= θ_sent 는 G = cos·T̃·φ 이고 T̃ 가 softmax "
            "출력(합=1)이므로 k>=2 에서 상한이 1.0×0.5×1.2=0.60 < 0.70 이다. "
            "운영값 θ_sent=0.70 에서는 인용이 구조적으로 발생할 수 없다."),
        "n_queries": len(insts), "runs": args.runs,
        "summary": summary,
    })

    def f3(d):
        return f"{d['mean']:.3f}±{d['std']:.3f}" if d["mean"] is not None else "정의불가"
    print(f"\n{'condition':24s} {'prec(NLI)':>13s} {'prec(rel)':>13s} "
          f"{'rec(unit)':>13s} {'n_cited':>8s} {'인용>0':>8s}")
    for c in conds:
        s_ = summary[c]
        print(f"{c:24s} {f3(s_['precision']):>13s} {f3(s_['precision_relevance']):>13s} "
              f"{f3(s_['recall_unit']):>13s} {s_['n_cited_mean']:8.2f} "
              f"{s_['queries_with_any_citation']:4d}/{s_['n_queries']}")
    rb = summary[conds[0]]["random_precision_baseline"]["mean"]
    if rb is not None:
        print(f"\n무작위 인용의 기대 정밀도(관련성 기준): {rb:.3f}")
    print(f"\n[✓] {out/'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
