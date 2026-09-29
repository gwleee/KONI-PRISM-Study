#!/usr/bin/env python3
"""
Exp E — SynConflict v2 빌더.

v1(scripts/build_conflict_set.py)의 구조적 결함 두 가지를 해소한다.

임계 근처 좁은 유사도 구간 편향
  v1은 docstring에 "τ=0.65 근방 최적 F1 설계"라고 명시돼 있다. 즉 conflict를
  cos<0.65, clean을 cos≥0.66에 의도적으로 배치했다. 이러면 τ=0.65가 최적인 것은
  데이터셋 구성의 결과이지 알고리즘의 성질이 아니다(순환 검증).
  v2는 (a) 레이블별로 유사도 전 구간을 커버하고, (b) **저유사도 비충돌**
  (unrelated_consistent)과 **고유사도 충돌**(paraphrastic)을 반드시 포함해
  단일 코사인 임계값으로는 분리 불가능한 영역을 노출한다.

충돌 유형 협소
  v1의 충돌은 사실상 1종(공식 보고서 수치 vs 구어체 불만)이다. 게다가 이는
  **문체(register) 교란**과 완전히 교락돼 있다 — 낮은 코사인이 '모순' 때문인지
  '문체 차이' 때문인지 구분 불가능하다.
  v2는 충돌 5종(numerical / domain_mismatch / paraphrastic / temporal /
  terminological)으로 확장하고, 각 쌍에 register를 명시해 **동일 문체 내 충돌**을
  기본값으로 둔다. 문체 교락은 register 필드로 사후 측정 가능하다.

출력:
  instances.jsonl  — 인스턴스 (id, text_a, text_b, label, conflict_type, register, domain, lang)
  metadata.json    — 구성/시드/버전
  (유사도는 별도 evaluate_cmae_v2.py 에서 실측해 부착한다 — 생성과 측정을 분리)

사용법:
  python build_synconflict_v2.py --n 600 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data"

# ── 도메인 정의 ──────────────────────────────────────────────────────────────
# 각 도메인은 (주체, 지표, 단위, 고값범위, 저값범위) 를 제공한다.

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

# 문체(register) 템플릿 — 동일 내용을 다른 문체로 렌더한다.
# 충돌 여부와 문체를 분리하기 위해 v2는 기본적으로 양쪽 문체를 일치시킨다.


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
    """동일 명제를 지정된 문체로 렌더."""
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


# ── 충돌 유형 5종 ────────────────────────────────────────────────────────────

def make_numerical(rng: random.Random, reg: str) -> dict:
    """같은 주체·지표·기간에 대해 모순되는 수치. (v1에서 계승, 문체는 일치시킴)"""
    dom = rng.choice(list(DOMAINS))
    d = DOMAINS[dom]
    subj = rng.choice(d["subjects"])
    metric, unit, hi, lo = rng.choice(d["metrics"])
    period = rng.choice(PERIODS)
    a = _render(reg, subj, metric, _fmt_val(_sample(rng, hi), unit), period)
    b = _render(reg, subj, metric, _fmt_val(_sample(rng, lo), unit), period)
    return {"text_a": a, "text_b": b, "conflict_type": "numerical", "domain": dom}


def make_domain_mismatch(rng: random.Random, reg: str) -> dict:
    """동일 질의에 대해 서로 다른 도메인의 주체를 답한다 — 주제 자체가 어긋남."""
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
    """표층은 거의 동일하고 서술어만 반대 — 코사인 유사도가 매우 높은 '어려운' 충돌.

    단일 코사인 임계값 기반 검출기가 원리적으로 놓치는 사례. v1에는 전무했다.
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
    """동일 명제를 양립 불가능한 시점에 귀속시킨다 (시간 순서·시점 모순)."""
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


# (용어, 단위계 A, 환산계수, 단위계 B) — 같은 물리량을 다른 단위로 말하되 값이 불일치
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
    """동일 개념을 다른 용어/단위계로 진술하되 환산이 맞지 않아 모순이 되는 경우."""
    dom = rng.choice(list(DOMAINS))
    d = DOMAINS[dom]
    subj = rng.choice(d["subjects"])
    period = rng.choice(PERIODS)

    if rng.random() < 0.5:
        # 단위계 불일치 — 환산하면 값이 어긋난다
        name, ua, _, ub, factor = rng.choice(TERM_PAIRS)
        base = rng.uniform(200, 900)
        va = f"{base:.0f}{ua}"
        # 올바른 환산은 base/factor. 의도적으로 크게 어긋난 값을 넣는다.
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
        # 유사 용어 혼동 — 서로 다른 지표를 같은 것으로 지칭하며 상반된 결론
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


# ── 비충돌 유형 3종 ──────────────────────────────────────────────────────────

def make_paraphrase(rng: random.Random, reg: str) -> dict:
    """동일 사실의 재진술 — 유사도 상단."""
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
    """같은 주체의 서로 다른, 그러나 양립 가능한 지표 — 유사도 중간."""
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
    """서로 무관하지만 **모순되지는 않는** 진술 — 유사도 하단.

    v1의 clean 집합은 전부 cos≥0.66이라 τ=0.65 아래에 비충돌이 존재하지 않았다.
    그래서 임계값 하나로 완벽 분리가 가능했다. 이 유형이 그 인위적 분리를 깬다.
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


# ── 데이터셋 조립 ────────────────────────────────────────────────────────────

def build(n: int, rng: random.Random, cross_register_ratio: float) -> list[dict]:
    """충돌:비충돌 = 1:1, 각 유형 균등 배분.

    cross_register_ratio: 양쪽 문체를 일부러 다르게 만들 비율.
      문체 교락(register confound)을 사후 측정하기 위한 소수 그룹이며,
      기본값은 낮게 둔다(v1은 사실상 1.0이었다).
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
                    # text_b 만 다른 문체로 재생성하기 어려우므로,
                    # cross 케이스는 b를 rb 문체로 다시 만든 쌍에서 가져온다.
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
    ap = argparse.ArgumentParser(description="SynConflict v2 빌더 (Exp E)")
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
                "레이블별로 유사도 전 구간을 커버한다. 고유사도 충돌"
                "(paraphrastic)과 저유사도 비충돌(unrelated_consistent)을 모두"
                " 포함해 단일 코사인 임계값으로 분리 불가능한 영역을 노출한다."
                " v1처럼 특정 τ가 최적이 되도록 설계하지 않는다."
            ),
            "conflict_taxonomy": (
                "충돌 5종으로 확장 (기존 numerical/domain_mismatch + 신규"
                " paraphrastic/temporal/terminological)."
            ),
            "register_deconfounding": (
                "v1은 충돌=공식문체 vs 비충돌=구어체로 문체와 레이블이 교락돼"
                " 있었다. v2는 기본적으로 양쪽 문체를 일치시키고, cross_register"
                " 소수 그룹으로 문체 효과를 사후 분리 측정한다."
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
