#!/usr/bin/env python3
"""
Exp E — SynConflict v2 builder.

Fixes two structural flaws of v1 (scripts/build_conflict_set.py).

Narrow similarity band near the threshold
  The v1 docstring explicitly states "designed for optimal F1 around τ=0.65", i.e. conflicts
  were deliberately placed at cos<0.65 and clean pairs at cos≥0.66. τ=0.65 being optimal is then
  a consequence of how the dataset was built, not a property of the algorithm (circular validation).
  v2 (a) covers the full similarity range for each label and (b) always includes **low-similarity
  non-conflicts** (unrelated_consistent) and **high-similarity conflicts** (paraphrastic), exposing
  the region that no single cosine threshold can separate.

Narrow conflict taxonomy
  v1 effectively has one conflict type (official-report figures vs colloquial complaints). It is
  also fully confounded with **register** — one cannot tell whether a low cosine comes from the
  'contradiction' or from the 'register difference'.
  v2 extends to 5 conflict types (numerical / domain_mismatch / paraphrastic / temporal /
  terminological), records the register of each pair, and makes **same-register conflicts** the
  default. The register confound can be measured post hoc via the register field.

Output:
  instances.jsonl  — instances (id, text_a, text_b, label, conflict_type, register, domain, lang)
  metadata.json    — configuration/seed/version
  (similarity is measured and attached separately in evaluate_cmae_v2.py — generation and measurement are decoupled)

Usage:
  python build_synconflict_v2.py --n 600 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data"

# ── Domain definitions ──────────────────────────────────────────────────────────────
# Each domain provides (subject, metric, unit, high-value range, low-value range).

DOMAINS: dict[str, dict] = {
    "system_perf": {
        "label": "시스템 성능",
        "subjects": ["추론 서버", "검색 엔진", "배치 파이프라인", "API 게이트웨이", "스트리밍 처리기"],
        "metrics": [
            ("평균 응답 지연", "ms", (40, 120), (600, 1800)),
            ("초당 처리량", "건", (1500, 3000), (120, 400)),
            ("가동률", "%", (99.0, 99.9), (82.0, 93.0)),
        ],
    },
    "model_eval": {
        "label": "모델 평가",
        "subjects": ["분류 모델", "판독 모델", "예측 모델", "탐지 모델", "요약 모델"],
        "metrics": [
            ("정확도", "%", (91.0, 97.5), (58.0, 72.0)),
            ("F1 점수", "", (0.90, 0.96), (0.55, 0.70)),
            ("오탐률", "%", (0.5, 2.0), (12.0, 25.0)),
        ],
    },
    "clinical": {
        "label": "임상",
        "subjects": ["투여군", "대조군", "코호트 A", "3상 시험군", "추적 관찰군"],
        "metrics": [
            ("치료 반응률", "%", (72.0, 88.0), (28.0, 44.0)),
            ("부작용 발생률", "%", (2.0, 6.0), (18.0, 31.0)),
            ("평균 회복 기간", "일", (7, 12), (28, 45)),
        ],
    },
    "energy": {
        "label": "에너지",
        "subjects": ["신규 설비", "열병합 발전기", "냉각 시스템", "태양광 어레이", "축전 모듈"],
        "metrics": [
            ("전력 효율", "%", (88.0, 96.0), (54.0, 68.0)),
            ("연간 절감액", "만원", (1800, 3200), (150, 480)),
            ("가동 온도", "℃", (38, 46), (72, 88)),
        ],
    },
    "logistics": {
        "label": "물류",
        "subjects": ["자동 분류기", "재고 관리 시스템", "배송 라우터", "창고 로봇", "수요 예측기"],
        "metrics": [
            ("배송 정시율", "%", (94.0, 99.0), (61.0, 77.0)),
            ("재고 회전율", "회", (11.0, 16.0), (3.0, 6.0)),
            ("오분류율", "%", (0.3, 1.2), (7.0, 15.0)),
        ],
    },
    "finance": {
        "label": "금융",
        "subjects": ["신용 평가 모형", "이상거래 탐지기", "포트폴리오 엔진", "리스크 산출기", "결제 승인 시스템"],
        "metrics": [
            ("승인 정확도", "%", (93.0, 98.5), (64.0, 78.0)),
            ("월간 손실률", "%", (0.2, 0.9), (4.5, 9.0)),
            ("처리 건수", "만건", (240, 420), (35, 90)),
        ],
    },
}

PERIODS = ["2024년 상반기", "2024년 4분기", "2025년 1분기", "지난 회계연도", "이번 분기"]
PERIODS_LATER = ["2025년 상반기", "2025년 3분기", "2026년 1분기", "올해", "다음 분기"]

# Register templates — render the same content in different registers.
# To decouple conflict from register, v2 matches the register of both sides by default.


def _fmt_val(v: float, unit: str) -> str:
    if unit == "":
        return f"{v:.3f}"
    if isinstance(v, int) or unit in ("일", "건", "회", "℃", "만원", "만건"):
        return f"{round(v):,}{unit}"
    return f"{v:.1f}{unit}"


def _sample(rng: random.Random, rng_range: tuple[float, float]) -> float:
    lo, hi = rng_range
    return rng.uniform(lo, hi)


def _render(register: str, subject: str, metric: str, val: str, period: str) -> str:
    """Render the same proposition in the given register."""
    if register == "formal":
        return (f"{period} {subject}의 {metric}은(는) {val}로 측정되었다. "
                f"해당 값은 정기 점검 보고서에 기록되어 있다.")
    if register == "casual":
        return (f"{period}에 {subject} {metric} 보니까 {val} 정도 나오더라고요. "
                f"점검 결과에 그렇게 적혀 있었어요.")
    # technical
    return (f"[{period}] {subject} :: {metric} = {val} "
            f"(source: periodic_inspection_report)")


REGISTERS = ["formal", "casual", "technical"]


# ── 5 conflict types ────────────────────────────────────────────────────────────

def make_numerical(rng: random.Random, reg: str) -> dict:
    """Contradictory values for the same subject, metric and period. (Inherited from v1, with matched register.)"""
    dom = rng.choice(list(DOMAINS))
    d = DOMAINS[dom]
    subj = rng.choice(d["subjects"])
    metric, unit, hi, lo = rng.choice(d["metrics"])
    period = rng.choice(PERIODS)
    a = _render(reg, subj, metric, _fmt_val(_sample(rng, hi), unit), period)
    b = _render(reg, subj, metric, _fmt_val(_sample(rng, lo), unit), period)
    return {"text_a": a, "text_b": b, "conflict_type": "numerical", "domain": dom}


def make_domain_mismatch(rng: random.Random, reg: str) -> dict:
    """Answers the same query with subjects from different domains — the topic itself diverges."""
    d1, d2 = rng.sample(list(DOMAINS), 2)
    dd1, dd2 = DOMAINS[d1], DOMAINS[d2]
    s1, s2 = rng.choice(dd1["subjects"]), rng.choice(dd2["subjects"])
    m1, u1, hi1, _ = rng.choice(dd1["metrics"])
    m2, u2, hi2, _ = rng.choice(dd2["metrics"])
    period = rng.choice(PERIODS)
    a = _render(reg, s1, m1, _fmt_val(_sample(rng, hi1), u1), period)
    b = _render(reg, s2, m2, _fmt_val(_sample(rng, hi2), u2), period)
    return {"text_a": a, "text_b": b, "conflict_type": "domain_mismatch",
            "domain": f"{d1}|{d2}"}


NEGATION_PAIRS = [
    ("기준을 충족하였다", "기준을 충족하지 못하였다"),
    ("유의미하게 개선되었다", "유의미하게 악화되었다"),
    ("승인되었다", "반려되었다"),
    ("안정적으로 유지되고 있다", "불안정한 상태가 지속되고 있다"),
    ("목표치를 초과 달성하였다", "목표치에 미달하였다"),
    ("이상이 발견되지 않았다", "다수의 이상이 발견되었다"),
    ("도입이 권고된다", "도입이 권고되지 않는다"),
    ("통계적으로 유의하였다", "통계적으로 유의하지 않았다"),
]


def make_paraphrastic(rng: random.Random, reg: str) -> dict:
    """Nearly identical surface form with only the predicate reversed — a 'hard' conflict with very high cosine similarity.

    Cases that a single-cosine-threshold detector misses in principle. v1 had none.
    """
    dom = rng.choice(list(DOMAINS))
    d = DOMAINS[dom]
    subj = rng.choice(d["subjects"])
    metric, _, _, _ = rng.choice(d["metrics"])
    period = rng.choice(PERIODS)
    pos, neg = rng.choice(NEGATION_PAIRS)
    if reg == "technical":
        a = f"[{period}] {subj} :: {metric} — {pos}"
        b = f"[{period}] {subj} :: {metric} — {neg}"
    elif reg == "casual":
        a = f"{period} {subj} {metric} 관련해서는 {pos}고 하네요."
        b = f"{period} {subj} {metric} 관련해서는 {neg}고 하네요."
    else:
        a = f"{period} {subj}의 {metric} 평가 결과, {pos}."
        b = f"{period} {subj}의 {metric} 평가 결과, {neg}."
    return {"text_a": a, "text_b": b, "conflict_type": "paraphrastic", "domain": dom}


def make_temporal(rng: random.Random, reg: str) -> dict:
    """Attributes the same proposition to incompatible points in time (temporal order/time-point contradiction)."""
    dom = rng.choice(list(DOMAINS))
    d = DOMAINS[dom]
    subj = rng.choice(d["subjects"])
    metric, unit, hi, _ = rng.choice(d["metrics"])
    val = _fmt_val(_sample(rng, hi), unit)
    i = rng.randrange(len(PERIODS))
    p_early, p_late = PERIODS[i], PERIODS_LATER[i]
    if reg == "technical":
        a = f"{subj} :: {metric} = {val} :: first_achieved_at = {p_early}"
        b = f"{subj} :: {metric} = {val} :: first_achieved_at = {p_late}"
    elif reg == "casual":
        a = f"{subj} {metric}이 {val} 찍은 게 {p_early}부터였어요. 그 전엔 그 수준이 아니었고요."
        b = f"{subj} {metric}이 {val} 찍은 건 {p_late}이 처음이에요. 그 전엔 그 수준이 아니었고요."
    else:
        a = f"{subj}의 {metric}이 {val}에 최초 도달한 시점은 {p_early}이며, 이전 기간에는 해당 수준에 이르지 못하였다."
        b = f"{subj}의 {metric}이 {val}에 최초 도달한 시점은 {p_late}이며, 이전 기간에는 해당 수준에 이르지 못하였다."
    return {"text_a": a, "text_b": b, "conflict_type": "temporal", "domain": dom}


# (term, unit system A, conversion factor, unit system B) — the same quantity stated in different units with inconsistent values
TERM_PAIRS = [
    ("응답 지연", "밀리초", 1.0, "초", 1000.0),
    ("저장 용량", "메가바이트", 1.0, "기가바이트", 1024.0),
    ("처리 주기", "분", 1.0, "시간", 60.0),
]

SYNONYM_CONFUSIONS = [
    ("정밀도", "재현율"),
    ("정확도", "일치율"),
    ("처리량", "대역폭"),
    ("가동률", "가용률"),
]


def make_terminological(rng: random.Random, reg: str) -> dict:
    """The same concept stated with a different term/unit system where the conversion does not match, yielding a contradiction."""
    dom = rng.choice(list(DOMAINS))
    d = DOMAINS[dom]
    subj = rng.choice(d["subjects"])
    period = rng.choice(PERIODS)

    if rng.random() < 0.5:
        # Unit-system mismatch — values disagree after conversion
        name, ua, _, ub, factor = rng.choice(TERM_PAIRS)
        base = rng.uniform(200, 900)
        va = f"{base:.0f}{ua}"
        # The correct conversion is base/factor. Deliberately insert a value that is far off.
        wrong = (base / factor) * rng.uniform(4.0, 9.0)
        vb = f"{wrong:.2f}{ub}"
        if reg == "technical":
            a = f"[{period}] {subj} :: {name} = {va}"
            b = f"[{period}] {subj} :: {name} = {vb}"
        elif reg == "casual":
            a = f"{period} {subj} {name}은 {va} 정도예요."
            b = f"{period} {subj} {name}은 {vb} 정도예요."
        else:
            a = f"{period} {subj}의 {name}은(는) {va}로 측정되었다."
            b = f"{period} {subj}의 {name}은(는) {vb}로 측정되었다."
        sub = "unit_mismatch"
    else:
        # Near-synonym confusion — different metrics referred to as the same, with opposite conclusions
        t1, t2 = rng.choice(SYNONYM_CONFUSIONS)
        v1 = rng.uniform(90, 97)
        v2 = rng.uniform(55, 70)
        if reg == "technical":
            a = f"[{period}] {subj} :: primary_metric({t1}) = {v1:.1f}% :: verdict = PASS"
            b = f"[{period}] {subj} :: primary_metric({t2}) = {v2:.1f}% :: verdict = FAIL"
        elif reg == "casual":
            a = f"{period} {subj}는 {t1} 기준으로 {v1:.1f}%라 합격이라고 보면 돼요."
            b = f"{period} {subj}는 {t2} 기준으로 {v2:.1f}%라 불합격이라고 보면 돼요."
        else:
            a = f"{period} {subj}는 핵심 지표인 {t1} {v1:.1f}%를 근거로 합격 판정되었다."
            b = f"{period} {subj}는 핵심 지표인 {t2} {v2:.1f}%를 근거로 불합격 판정되었다."
        sub = "synonym_confusion"
    return {"text_a": a, "text_b": b, "conflict_type": "terminological",
            "domain": dom, "subtype": sub}


# ── 3 non-conflict types ──────────────────────────────────────────────────────────

def make_paraphrase(rng: random.Random, reg: str) -> dict:
    """Restatement of the same fact — upper similarity range."""
    dom = rng.choice(list(DOMAINS))
    d = DOMAINS[dom]
    subj = rng.choice(d["subjects"])
    metric, unit, hi, _ = rng.choice(d["metrics"])
    val = _fmt_val(_sample(rng, hi), unit)
    period = rng.choice(PERIODS)
    a = _render(reg, subj, metric, val, period)
    if reg == "technical":
        b = f"[{period}] measurement({subj}, {metric}) -> {val} // per periodic inspection"
    elif reg == "casual":
        b = f"{period} {subj} {metric}는 {val}였어요. 점검 기록에 남아 있고요."
    else:
        b = (f"정기 점검 기록에 따르면 {period} {subj}의 {metric}은(는) {val}로 확인된다.")
    return {"text_a": a, "text_b": b, "conflict_type": "paraphrase", "domain": dom}


def make_complementary(rng: random.Random, reg: str) -> dict:
    """Different but compatible metrics of the same subject — mid similarity range."""
    dom = rng.choice(list(DOMAINS))
    d = DOMAINS[dom]
    subj = rng.choice(d["subjects"])
    (m1, u1, hi1, _), (m2, u2, hi2, _) = rng.sample(d["metrics"], 2)
    period = rng.choice(PERIODS)
    a = _render(reg, subj, m1, _fmt_val(_sample(rng, hi1), u1), period)
    b = _render(reg, subj, m2, _fmt_val(_sample(rng, hi2), u2), period)
    return {"text_a": a, "text_b": b, "conflict_type": "complementary", "domain": dom}


UNRELATED_FACTS = [
    "회의실 예약은 사내 포털에서 처리한다.",
    "주차 등록은 매월 1일에 갱신된다.",
    "구내식당은 11시 30분에 문을 연다.",
    "연차 신청은 결재선 승인 후 반영된다.",
    "보안 교육은 연 2회 실시된다.",
    "출입 카드 재발급은 총무팀이 담당한다.",
    "사내 도서관은 3층에 위치한다.",
    "정기 소방 점검은 분기마다 진행된다.",
]


def make_unrelated_consistent(rng: random.Random, reg: str) -> dict:
    """Unrelated but **non-contradictory** statements — lower similarity range.

    In v1 every clean pair had cos≥0.66, so no non-conflict existed below τ=0.65,
    which made perfect separation possible with a single threshold. This type breaks that artificial separation.
    """
    dom = rng.choice(list(DOMAINS))
    d = DOMAINS[dom]
    subj = rng.choice(d["subjects"])
    metric, unit, hi, _ = rng.choice(d["metrics"])
    period = rng.choice(PERIODS)
    a = _render(reg, subj, metric, _fmt_val(_sample(rng, hi), unit), period)
    b = rng.choice(UNRELATED_FACTS)
    if reg == "technical":
        b = f"[note] {b}"
    return {"text_a": a, "text_b": b, "conflict_type": "unrelated_consistent",
            "domain": dom}


CONFLICT_BUILDERS = {
    "numerical": make_numerical,
    "domain_mismatch": make_domain_mismatch,
    "paraphrastic": make_paraphrastic,
    "temporal": make_temporal,
    "terminological": make_terminological,
}
CLEAN_BUILDERS = {
    "paraphrase": make_paraphrase,
    "complementary": make_complementary,
    "unrelated_consistent": make_unrelated_consistent,
}


# ── Dataset assembly ────────────────────────────────────────────────────────────

def build(n: int, rng: random.Random, cross_register_ratio: float) -> list[dict]:
    """conflict:non-conflict = 1:1, with types allocated evenly.

    cross_register_ratio: fraction of pairs whose two sides deliberately use different registers.
      A minority group for measuring the register confound post hoc;
      the default is kept low (v1 was effectively 1.0).
    """
    n_conf = n // 2
    n_clean = n - n_conf

    instances: list[dict] = []

    def emit(builders: dict, count: int, label: int) -> None:
        types = list(builders)
        per = count // len(types)
        rem = count - per * len(types)
        alloc = {t: per for t in types}
        for t in types[:rem]:
            alloc[t] += 1
        for t, c in alloc.items():
            for i in range(c):
                cross = rng.random() < cross_register_ratio
                if cross:
                    ra, rb = rng.sample(REGISTERS, 2)
                    reg_field = f"{ra}|{rb}"
                    rec = builders[t](rng, ra)
                    # Since regenerating only text_b in another register is hard,
                    # cross cases take b from a pair rebuilt in register rb.
                    rec_b = builders[t](rng, rb)
                    rec["text_b"] = rec_b["text_b"]
                else:
                    ra = rng.choice(REGISTERS)
                    reg_field = f"{ra}|{ra}"
                    rec = builders[t](rng, ra)
                rec.update({
                    "id": f"{t}_{i:04d}",
                    "label": label,                       # 1 = conflict
                    "ground_truth_conflict": bool(label),
                    "register": reg_field,
                    "cross_register": cross,
                    "lang": "ko",
                })
                instances.append(rec)

    emit(CONFLICT_BUILDERS, n_conf, 1)
    emit(CLEAN_BUILDERS, n_clean, 0)
    rng.shuffle(instances)
    return instances


def main() -> None:
    ap = argparse.ArgumentParser(description="SynConflict v2 builder (Exp E)")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cross-register-ratio", type=float, default=0.25)
    ap.add_argument("--output", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    inst = build(args.n, rng, args.cross_register_ratio)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "instances.jsonl").open("w", encoding="utf-8") as f:
        for r in inst:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_type = Counter(r["conflict_type"] for r in inst)
    by_reg = Counter(r["register"] for r in inst)

    meta = {
        "dataset": "SynConflict",
        "version": "v2",
        "n_instances": len(inst),
        "n_conflict": sum(r["label"] for r in inst),
        "n_clean": sum(1 - r["label"] for r in inst),
        "conflict_types": ["numerical", "domain_mismatch", "paraphrastic",
                           "temporal", "terminological"],
        "clean_types": ["paraphrase", "complementary", "unrelated_consistent"],
        "counts_by_type": dict(by_type),
        "counts_by_register": dict(by_reg),
        "cross_register_ratio": args.cross_register_ratio,
        "random_seed": args.seed,
        "generator": "paper_experiments/synconflict_v2/build_synconflict_v2.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design_notes": {
            "similarity_coverage": (
                "Covers the full similarity range for each label. Includes both"
                " high-similarity conflicts (paraphrastic) and low-similarity non-conflicts"
                " (unrelated_consistent), exposing the region no single cosine threshold can separate."
                " Unlike v1, it is not designed so that a particular τ is optimal."
            ),
            "conflict_taxonomy": (
                "Extended to 5 conflict types (existing numerical/domain_mismatch + new"
                " paraphrastic/temporal/terminological)."
            ),
            "register_deconfounding": (
                "In v1, register and label were confounded (conflict = formal register vs"
                " non-conflict = colloquial). v2 matches the register of both sides by default and"
                " measures the register effect separately post hoc with a minority cross_register group."
            ),
        },
    }
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[✓] {out/'instances.jsonl'}  n={len(inst)} "
          f"(conflict={meta['n_conflict']}, clean={meta['n_clean']})")
    for t, c in sorted(by_type.items()):
        print(f"    {t:24s} {c}")


if __name__ == "__main__":
    main()
