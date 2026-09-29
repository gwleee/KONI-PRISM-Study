#!/usr/bin/env python3
"""
Exp D — 외부 contradiction 데이터셋에서 CMAE 충돌 검출 평가.

목적: 자체 제작 SynConflict 에서의 성능이 사람이 주석한 외부 일반 데이터셋으로
일반화되는지 검증한다 (리뷰어 요구: 외부 일반화 증명).

데이터셋 3종 (요구 2종 이상 충족):
  - SNLI       (stanfordnlp/snli)      contradiction ↔ entailment
  - MultiNLI   (nyu-mll/multi_nli)     contradiction ↔ entailment (장르 다양)
  - VitaminC   (tals/vitaminc)         REFUTES ↔ SUPPORTS
                                       FEVER 기반 + 위키 개정 대조쌍.
                                       FEVER 원본은 근거 텍스트가 위키 덤프에
                                       분리돼 있어 즉시 쌍 구성이 불가능하므로,
                                       근거가 인라인된 FEVER 파생 데이터셋인
                                       VitaminC를 사용한다.

매핑: CMAE 는 두 응답의 쌍에 대해 Eq.(9) conflict = 1[cos < τ] 로 판정한다.
      NLI 의 (premise, hypothesis) / 사실검증의 (evidence, claim) 을
      두 모달리티 응답 쌍으로 간주한다.

neutral / NOT ENOUGH INFO 는 '모순 아님'이지만 '지지'도 아니어서 기본 평가에서
제외하고, --include-neutral 로 비충돌에 포함한 변형도 별도 보고한다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import (  # noqa: E402
    binary_scores, bootstrap_gateway, provenance, set_profile,
    settings_snapshot, write_csv, write_json, write_jsonl,
)

DATASETS = {
    "snli": {
        "hf": ("stanfordnlp/snli",),
        "split": "validation",
        "a": "premise", "b": "hypothesis",
        "conflict_labels": {2},          # ClassLabel: 0=ent,1=neu,2=contradiction
        "clean_labels": {0},
        "neutral_labels": {1},
        "label_key": "label",
    },
    "mnli": {
        "hf": ("nyu-mll/multi_nli",),
        "split": "validation_matched",
        "a": "premise", "b": "hypothesis",
        "conflict_labels": {2},
        "clean_labels": {0},
        "neutral_labels": {1},
        "label_key": "label",
    },
    "vitaminc": {
        "hf": ("tals/vitaminc",),
        "split": "validation",
        "a": "evidence", "b": "claim",
        "conflict_labels": {"REFUTES"},
        "clean_labels": {"SUPPORTS"},
        "neutral_labels": {"NOT ENOUGH INFO"},
        "label_key": "label",
    },
}


def build_pairs(name: str, n_per_class: int, seed: int) -> list[dict]:
    """HF 데이터셋을 CMAE 쌍 형식으로 변환. 클래스 균형 + 시드 고정 샘플링."""
    from datasets import load_dataset  # noqa: PLC0415

    spec = DATASETS[name]
    ds = load_dataset(*spec["hf"], split=spec["split"])
    lk = spec["label_key"]

    buckets: dict[str, list[dict]] = {"conflict": [], "clean": [], "neutral": []}
    for row in ds:
        lab = row[lk]
        ta, tb = row[spec["a"]], row[spec["b"]]
        if not isinstance(ta, str) or not isinstance(tb, str):
            continue
        ta, tb = ta.strip(), tb.strip()
        if not ta or not tb:
            continue
        if lab in spec["conflict_labels"]:
            grp = "conflict"
        elif lab in spec["clean_labels"]:
            grp = "clean"
        elif lab in spec["neutral_labels"]:
            grp = "neutral"
        else:
            continue                      # SNLI 의 -1 (주석자 불일치) 등
        buckets[grp].append({"text_a": ta, "text_b": tb, "raw_label": str(lab)})

    rng = random.Random(seed)
    out: list[dict] = []
    for grp, label in (("conflict", 1), ("clean", 0), ("neutral", 0)):
        pool = buckets[grp]
        rng.shuffle(pool)
        take = pool[:n_per_class]
        for i, r in enumerate(take):
            out.append({
                "id": f"{name}_{grp}_{i:05d}",
                "dataset": name,
                "text_a": r["text_a"],
                "text_b": r["text_b"],
                "label": label,
                "group": grp,
                "conflict_type": grp,       # 공용 평가기 호환
                "raw_label": r["raw_label"],
            })
    rng.shuffle(out)
    return out


async def evaluate(name: str, rows: list[dict], tau_prod: float) -> dict:
    from common.cmae_eval import (  # noqa: PLC0415
        encode_pairs, evaluate_at_threshold, roc_auc, separability_index,
        similarity_bins, sweep_threshold, verify_against_cmae,
    )
    import numpy as np  # noqa: PLC0415

    def block(sub: list[dict]) -> dict | None:
        if not sub:
            return None
        idx = [rows.index(r) for r in sub]
        return idx

    # 기본 평가: neutral 제외
    primary = [r for r in rows if r["group"] in ("conflict", "clean")]
    sims_all, probe = await encode_pairs(rows)
    sim_by_id = {r["id"]: float(s) for r, s in zip(rows, sims_all)}

    def metrics(subset: list[dict]) -> dict:
        s = np.array([sim_by_id[r["id"]] for r in subset])
        y = [r["label"] for r in subset]
        at = evaluate_at_threshold(s, y, tau_prod)
        curve, best = sweep_threshold(s, y)
        return {
            "n": len(subset),
            "n_conflict": int(sum(y)),
            "at_production_tau": at,
            "best_threshold": best,
            "tau_gap": float(best["tau"] - tau_prod),
            "f1_gap": float(best["f1"] - at["f1"]),
            "roc_auc": roc_auc(s, y),
            "separability_overlap": separability_index(similarity_bins(s, y)),
            "trivial_all_conflict_f1": binary_scores(y, [1] * len(y))["f1"],
            "cos_mean_conflict": float(s[np.array(y) == 1].mean()) if any(y) else None,
            "cos_mean_clean": float(s[np.array(y) == 0].mean()) if not all(y) else None,
            "_curve": curve,
        }

    prim = metrics(primary)
    with_neutral = metrics(rows) if any(r["group"] == "neutral" for r in rows) else None
    verify = await verify_against_cmae(primary[:20],
                                       np.array([sim_by_id[r["id"]] for r in primary[:20]]),
                                       tau_prod, n_check=20)

    curve = prim.pop("_curve")
    if with_neutral:
        with_neutral.pop("_curve", None)

    return {
        "dataset": name,
        "encoder_probe": probe,
        "cmae_path_verification": verify,
        "primary_contradiction_vs_entailment": prim,
        "with_neutral_as_clean": with_neutral,
        "_curve": curve,
        "_sims": sim_by_id,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["snli", "mnli", "vitaminc"])
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", default="a100_1")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    set_profile(args.profile)
    bootstrap_gateway()
    from core.config import get_settings  # noqa: PLC0415
    tau_prod = get_settings().cmae_tau_conflict

    out = Path(args.out)
    results = {}
    for name in args.datasets:
        print(f"[..] {name} 로딩/변환")
        rows = build_pairs(name, args.n_per_class, args.seed)
        write_jsonl(out.parent / "data" / f"{name}_pairs.jsonl", rows)
        print(f"[..] {name} 인코딩/평가 (n={len(rows)})")
        r = await evaluate(name, rows, tau_prod)
        curve = r.pop("_curve")
        r.pop("_sims", None)
        write_csv(out / f"threshold_curve_{name}.csv", curve,
                  columns=["tau", "precision", "recall", "f1", "mcc",
                           "accuracy", "tp", "fp", "fn", "tn"])
        results[name] = r

    write_json(out / "results.json", {
        "experiment": "Exp D — 외부 contradiction 데이터셋 CMAE 평가",
        "production_tau_conflict": tau_prod,
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "provenance": provenance(),
        "settings": settings_snapshot(),
        "results": results,
    })

    print(f"\n{'dataset':10s} {'n':>5s} {'F1@0.65':>8s} {'MCC':>7s} {'AUC':>6s} "
          f"{'bestτ':>6s} {'bestF1':>7s} {'trivF1':>7s} {'cos(C)':>7s} {'cos(¬C)':>8s}")
    for name, r in results.items():
        p = r["primary_contradiction_vs_entailment"]
        a = p["at_production_tau"]
        print(f"{name:10s} {p['n']:5d} {a['f1']:8.3f} {a['mcc']:7.3f} {p['roc_auc']:6.3f} "
              f"{p['best_threshold']['tau']:6.3f} {p['best_threshold']['f1']:7.3f} "
              f"{p['trivial_all_conflict_f1']:7.3f} {p['cos_mean_conflict']:7.3f} "
              f"{p['cos_mean_clean']:8.3f}")
    print(f"\n[✓] {out/'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
