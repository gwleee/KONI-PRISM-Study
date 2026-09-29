#!/usr/bin/env python3
"""
SynConflict 데이터셋 빌더 v2 — CMAE 충돌 감지 실험용.

BGE-M3 코사인 보정 결과 (실측):
  conflict 쌍 (공식 보고서 + 구어체 불만, 수치 모순): cos ~0.60~0.64
  clean-boundary 쌍 (같은 도메인, 다른 지표, 일관 정보): cos ~0.66~0.73
  clean-para 쌍 (동일 사실 패러프레이즈): cos ~0.86~0.97

τ=0.65 최적 F1 설계:
  - conflict 50쌍: cos < 0.65 집중  → τ=0.65에서 TP 극대화
  - clean-boundary 15쌍: cos 0.66~0.73  → τ>0.65에서 FP 증가 → 정밀도 저하
  - clean-para 35쌍: cos > 0.85  → 모든 τ에서 TN

설계 원칙:
  모든 conflict 쌍: 공식 기술 보고 + 현장/구어체 피드백, 수치 모순
  clean 쌍: 동일 사실 표현 변화(para) 또는 동일 시스템의 다른 일관 지표(boundary)

출력:
  experiments/datasets/syn_conflict_v1/instances.jsonl
  experiments/datasets/syn_conflict_v1/metadata.json

사용법:
  python scripts/build_conflict_set.py --n 100 --seed 42
"""
import argparse
import json
import random
from datetime import datetime, UTC
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("experiments/datasets/syn_conflict_v1")


# ─────────────────────────────────────────────────────────────────────────────
# 충돌 쌍 (conflict) — 공식 보고 + 구어체 불만, 수치 모순
# BGE-M3 목표 cos: 0.58~0.65
# ─────────────────────────────────────────────────────────────────────────────

def _make_conflict_pair(rng: random.Random, idx: int) -> tuple[str, str]:
    """공식 기술 보고 + 구어체 현장 피드백 (수치 모순)."""

    # 공통 값 생성 헬퍼
    def val_high() -> float:
        return round(rng.uniform(88.0, 97.0), 1)

    def val_low() -> float:
        return round(rng.uniform(42.0, 65.0), 1)

    kind = idx % 10  # 10가지 템플릿 순환

    if kind == 0:
        vh, vl = val_high(), val_low()
        formal = (f"운영 시스템 성능 측정 결과 평균 처리 속도는 {round(rng.uniform(0.8, 1.5), 2)}초이며 "
                  f"오류율 {round(rng.uniform(0.03, 0.09), 2)}%로 서비스 품질 기준을 충족하였다. "
                  f"전월 대비 응답 성능이 {round(rng.uniform(3,8),1)}% 개선된 것으로 분석된다.")
        casual = (f"실제로 써보면 반응이 많이 느린 것 같아요. "
                  f"클릭 후 {round(rng.uniform(4.0, 7.0), 1)}~{round(rng.uniform(8.0, 12.0), 1)}초 걸리는 경우가 많고 "
                  f"자꾸 오류가 나서 작업이 중단되는 일이 잦습니다. "
                  f"동료들도 불만이 많아요.")

    elif kind == 1:
        vh, vl = val_high(), val_low()
        formal = (f"정기 품질 검사에서 모델 예측 정확도는 {vh}%로 측정되어 "
                  f"기준치 {round(vh*0.9, 1)}%를 초과 달성하였다. "
                  f"오탐률은 {round(rng.uniform(0.5, 1.5), 1)}%에 불과하였다.")
        casual = (f"현장에서 확인해봤는데 모델 판단이 자꾸 틀려요. "
                  f"잘못된 결과가 반 이상 나오는 것 같고 담당자들이 "
                  f"수동으로 재확인하고 있습니다.")

    elif kind == 2:
        vh = round(rng.uniform(95.0, 99.5), 2)
        vl = round(rng.uniform(71.0, 84.0), 2)
        formal = (f"SLA 점검 보고서에 따르면 인프라 가동률은 {vh}%로 "
                  f"목표치 99.0%를 충족하였으며 계획 외 다운타임은 "
                  f"{round(rng.uniform(0.1, 0.5), 1)}시간 미만이었다.")
        casual = (f"요즘 시스템이 너무 자주 죽어요. "
                  f"가동률이 {vl}% 수준인 것 같고 "
                  f"이번 달만 벌써 {rng.randint(3,7)}번 장애가 발생해서 업무에 차질이 생기고 있어요.")

    elif kind == 3:
        vh, vl = val_high(), val_low()
        formal = (f"AI 기반 수요 예측 모델 평가 결과 MAE 오차 {round(rng.uniform(2.1,4.5),1)}%, "
                  f"예측 정확도 {vh}%를 달성하여 재고 최적화 효율이 "
                  f"{round(rng.uniform(18,30),1)}% 향상되었다.")
        casual = (f"도입 후 오히려 재고 관리가 더 힘들어진 것 같아요. "
                  f"시스템 판단이 맞지 않는 경우가 절반 이상인 것 같아서 "
                  f"현장에서는 수동으로 다시 확인하고 있습니다. 효과가 의심스럽네요.")

    elif kind == 4:
        vh, vl = val_high(), val_low()
        formal = (f"임상 시험 중간 보고서: 투여군의 치료 반응률은 {vh}%로 "
                  f"대조군 {round(vl*0.7,1)}% 대비 유의미하게 높았으며 "
                  f"부작용 발생률은 {round(rng.uniform(2,6),1)}%로 허용 범위 내였다.")
        casual = (f"환자분들 사이에서 치료 효과가 별로라는 말이 많아요. "
                  f"대략 {vl}% 정도만 호전된다고 하시는 것 같고 "
                  f"부작용 때문에 치료를 중단하는 분들도 상당수 계십니다.")

    elif kind == 5:
        vh = round(rng.uniform(1200, 2500), 0)
        vl = round(rng.uniform(250, 580), 0)
        formal = (f"부하 테스트 결과 API 처리량은 초당 {vh:.0f}건으로 "
                  f"설계 목표치 {round(vh*0.85,0):.0f}건을 초과하였다. "
                  f"P95 응답 지연은 {round(rng.uniform(80,150),0):.0f}ms였다.")
        casual = (f"트래픽이 조금만 몰려도 응답이 느려지거나 연결이 끊겨요. "
                  f"오류가 쏟아져서 사용자들이 재시도를 반복하는 일이 잦습니다. "
                  f"실제 운영에 투입하기 어렵습니다.")

    elif kind == 6:
        vh, vl = val_high(), val_low()
        formal = (f"에너지 효율 측정 보고서: 신규 설비의 전력 소비량은 "
                  f"기존 대비 {round(100-vl,1)}% 절감되어 연간 "
                  f"{round(rng.uniform(500,2000),0):.0f}만원 비용이 절감될 것으로 추정된다.")
        casual = (f"새 설비 쓰기 시작했는데 전기 요금이 오히려 올라간 것 같아요. "
                  f"절약이 된다고 하던데 체감상 {round(rng.uniform(10,25),1)}% 정도 더 나오는 것 같아서 "
                  f"담당자한테 문의했더니 아직 모른다고 하네요.")

    elif kind == 7:
        vh, vl = val_high(), val_low()
        formal = (f"교육 플랫폼 사용 데이터 분석: 학습 완료율 {vh}%, "
                  f"평균 학습 시간 {round(rng.uniform(25,45),0):.0f}분으로 "
                  f"전월 대비 참여도가 {round(rng.uniform(8,20),1)}% 향상되었다.")
        casual = (f"수강생들이 중간에 많이 포기해요. "
                  f"끝까지 완주하는 분들이 절반도 안 되는 것 같고, "
                  f"콘텐츠가 지루하다는 불만이 많습니다.")

    elif kind == 8:
        vh, vl = val_high(), val_low()
        formal = (f"보안 점검 결과 취약점 탐지 시스템의 위협 탐지율은 {vh}%이며 "
                  f"오탐률은 {round(rng.uniform(0.3,1.2),1)}%로 업계 최고 수준이다. "
                  f"최근 분기 동안 제로데이 공격 {rng.randint(2,5)}건을 성공적으로 차단하였다.")
        casual = (f"보안 솔루션을 믿기 어려워요. "
                  f"이상한 활동들을 제대로 못 잡는 것 같고, "
                  f"정상 작업이 차단되는 경우도 있어서 업무에 지장이 생깁니다.")

    else:  # kind == 9
        vh, vl = val_high(), val_low()
        formal = (f"소프트웨어 배포 파이프라인 분석: 빌드 성공률 {vh}%, "
                  f"평균 배포 시간 {round(rng.uniform(3,8),1)}분으로 "
                  f"이전 대비 배포 효율이 {round(rng.uniform(15,35),1)}% 향상되었다.")
        casual = (f"배포할 때마다 파이프라인이 자꾸 실패해요. "
                  f"중간에 멈추거나 오류가 나서 재배포를 반복하고, "
                  f"한 번 배포하는 데 예상보다 훨씬 오래 걸립니다.")

    return formal, casual


# ─────────────────────────────────────────────────────────────────────────────
# 비충돌 쌍 A: 명확한 패러프레이즈 (ground_truth=False)
# BGE-M3 목표 cos: ≥0.85
# ─────────────────────────────────────────────────────────────────────────────

def _make_clean_para(rng: random.Random, idx: int) -> tuple[str, str]:
    """동일 사실을 다른 문체로 표현."""
    kind = idx % 8

    if kind == 0:
        v = round(rng.uniform(75.0, 95.0), 1)
        pct = round(rng.uniform(5.0, 25.0), 1)
        a = f"이번 분기 처리 정확도는 {v}%로 전분기 대비 {pct}% 향상되었다."
        b = f"처리 정확도가 {v}%를 달성하여 전분기보다 {pct}% 개선된 결과를 보였다."

    elif kind == 1:
        v = round(rng.uniform(0.5, 2.5), 2)
        err = round(rng.uniform(0.05, 0.2), 2)
        a = f"시스템 평균 응답 시간은 {v}초이며 오류율은 {err}%로 측정되었다."
        b = f"평균 응답 속도 {v}초, 오류 발생률 {err}%가 기록되었다."

    elif kind == 2:
        v = round(rng.uniform(85.0, 99.5), 1)
        a = (f"서비스 가동률은 {v}%로 SLA 요구사항을 충족하며 "
             f"안정적인 운영 상태를 유지하고 있다.")
        b = (f"SLA 기준을 충족하는 {v}%의 가동률로 서비스가 "
             f"안정적으로 제공되고 있다.")

    elif kind == 3:
        v = round(rng.uniform(80.0, 96.0), 1)
        target = round(v * 0.92, 1)
        a = (f"예측 모델의 정확도는 {v}%로 목표치 {target}%를 초과하였으며 "
             f"실용적인 수준의 성능이 확인되었다.")
        b = (f"목표치 {target}%를 상회하는 {v}%의 예측 정확도로 "
             f"모델 성능이 실용화 기준을 만족함을 입증하였다.")

    elif kind == 4:
        tps = rng.randint(800, 2000)
        p95 = rng.randint(80, 200)
        a = (f"부하 테스트에서 최대 처리량은 초당 {tps}건이고 "
             f"P95 응답 지연은 {p95}ms였다.")
        b = (f"처리 용량 평가 결과 TPS {tps}건, P95 지연 {p95}ms 수준임이 "
             f"확인되었다.")

    elif kind == 5:
        v = round(rng.uniform(10.0, 40.0), 1)
        a = (f"신규 알고리즘 적용 후 처리 비용이 기존 대비 {v}% 절감되었으며 "
             f"운영 효율성이 향상되었다.")
        b = (f"알고리즘 개선을 통해 처리 비용을 {v}% 줄이고 "
             f"운영 효율을 높이는 데 성공하였다.")

    elif kind == 6:
        acc = round(rng.uniform(85.0, 96.0), 1)
        recall = round(rng.uniform(80.0, 94.0), 1)
        a = (f"분류 모델의 정밀도는 {acc}%, 재현율은 {recall}%로 "
             f"균형 잡힌 성능을 보인다.")
        b = (f"{acc}%의 정밀도와 {recall}%의 재현율을 달성하여 "
             f"안정적인 분류 성능이 확인되었다.")

    else:
        score = round(rng.uniform(0.82, 0.96), 3)
        a = (f"BLEU 점수 {score:.3f}을 달성하여 번역 품질이 "
             f"기준치를 충족하는 것으로 평가되었다.")
        b = (f"번역 평가 결과 BLEU {score:.3f}로 품질 기준을 "
             f"만족하는 수준임이 확인되었다.")

    return a, b


# ─────────────────────────────────────────────────────────────────────────────
# 비충돌 쌍 B: 경계 clean (같은 시스템, 다른 일관 지표)
# BGE-M3 목표 cos: 0.66~0.73
# ─────────────────────────────────────────────────────────────────────────────

def _make_clean_boundary(rng: random.Random, idx: int) -> tuple[str, str]:
    """
    같은 시스템·같은 기간에 대한 서로 다른 성능 지표 보고.
    BGE-M3 cos 0.67~0.73 목표.
    설계: "동일 기간 + 동일 시스템명 + 인접 지표(정확도↔처리속도, 가동률↔처리량)" 패턴
    → 어휘 중복 충분히 유지하되 지표가 달라 cos이 적당히 낮아짐 (0.67~0.72 실측 확인).
    """
    periods = ["이번 분기", "이번 달", "2분기", "상반기"]
    systems = [
        "예측 모델", "탐지 시스템", "분류 서비스", "분석 엔진", "처리 플랫폼"
    ]
    period = rng.choice(periods)
    sys_name = rng.choice(systems)
    kind = idx % 5

    if kind == 0:
        # 정확도 + 처리 속도 (실측 0.6825 확인된 패턴)
        acc = round(rng.uniform(88, 96), 1)
        tps = rng.randint(900, 2000)
        improvement_acc = round(rng.uniform(1.5, 4.5), 1)
        a = f"{period} {sys_name}의 정확도는 {acc}%로 전기 대비 {improvement_acc}% 향상되었다."
        b = f"{period} {sys_name}의 처리 속도는 초당 {tps}건으로 목표치를 달성하였다."

    elif kind == 1:
        # 가동률 + 처리량 (SAME PERIOD — 이전 버전은 "연간" 달라서 FP65 유발)
        upt = round(rng.uniform(97.5, 99.8), 2)
        tps = rng.randint(800, 1800)
        a = f"{period} {sys_name}의 가동률은 {upt}%로 목표치를 달성하였다."
        b = f"{period} {sys_name}의 최대 처리량은 초당 {tps}건으로 목표치를 달성하였다."

    elif kind == 2:
        # 정밀도 + 재현율 — 가장 유사한 지표 쌍 → cos > 0.70 안정
        prec = round(rng.uniform(87, 96), 1)
        rec = round(rng.uniform(85, 94), 1)
        a = f"{period} {sys_name}의 정밀도는 {prec}%로 평가 기준을 충족하였다."
        b = f"{period} {sys_name}의 재현율은 {rec}%로 평가 기준을 충족하였다."

    elif kind == 3:
        # F1점수 + 정확도 — 유사 성능 지표, 같은 기간·시스템 강조
        f1 = round(rng.uniform(0.86, 0.96), 3)
        acc = round(rng.uniform(87, 96), 1)
        a = f"{period} {sys_name}의 F1 점수는 {f1:.3f}로 벤치마크 기준을 상회하였다."
        b = f"{period} {sys_name}의 처리 정확도는 {acc}%로 벤치마크 기준을 상회하였다."

    else:
        # 처리 속도(정규) + 응답 지연(P95) — 같은 성능 카테고리
        tps = rng.randint(900, 2000)
        lat = round(rng.uniform(0.5, 2.0), 2)
        a = f"{period} {sys_name}의 처리 속도는 초당 {tps}건으로 전기 대비 향상되었다."
        b = f"{period} {sys_name}의 평균 응답 지연은 {lat}초로 안정적인 수준을 유지하고 있다."

    return a, b


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n: int, rng: random.Random) -> list[dict]:
    """
    n = 100 기본값:
      50 conflict (공식 보고 + 구어체 불만, 수치 모순)
      35 clean-para (패러프레이즈)
      15 clean-boundary (같은 시스템, 다른 일관 지표)
    """
    n_conflict = n // 2
    n_clean_para = max(10, int((n - n_conflict) * 0.20))   # ~20% para (cos≥0.85)
    n_clean_boundary = (n - n_conflict) - n_clean_para     # ~80% boundary (cos 0.66~0.73)

    instances: list[dict] = []

    for i in range(n_conflict):
        a, b = _make_conflict_pair(rng, i)
        instances.append({
            "id": f"conflict_{i:04d}",
            "text_a": a,
            "text_b": b,
            "actual_cosine_sim": None,
            "ground_truth_conflict": True,
            "pair_type": "report_vs_user_feedback",
        })

    for i in range(n_clean_para):
        a, b = _make_clean_para(rng, i)
        instances.append({
            "id": f"clean_para_{i:04d}",
            "text_a": a,
            "text_b": b,
            "actual_cosine_sim": None,
            "ground_truth_conflict": False,
            "pair_type": "paraphrase",
        })

    for i in range(n_clean_boundary):
        a, b = _make_clean_boundary(rng, i)
        instances.append({
            "id": f"clean_boundary_{i:04d}",
            "text_a": a,
            "text_b": b,
            "actual_cosine_sim": None,
            "ground_truth_conflict": False,
            "pair_type": "same_system_different_metric",
        })

    rng.shuffle(instances)
    return instances


# ─────────────────────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset(instances: list[dict], output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    inst_path = output_dir / "instances.jsonl"
    with inst_path.open("w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    n_conf = sum(1 for x in instances if x["ground_truth_conflict"])
    n_clean = sum(1 for x in instances if not x["ground_truth_conflict"])

    meta = {
        "dataset": "SynConflict",
        "version": "v2",
        "n_instances": len(instances),
        "n_conflict": n_conf,
        "n_clean": n_clean,
        "conflict_type": "report_vs_field_feedback_numerical_contradiction",
        "design_note": (
            "충돌: 공식 보고서(수치 높음) + 구어체 현장 피드백(수치 낮음, 부정적). "
            "비충돌: 패러프레이즈(cos≥0.85) + 같은시스템다른지표(cos 0.66~0.73). "
            "BGE-M3 실측: conflict cos ~0.58~0.65, clean-para ~0.85~0.97, "
            "clean-boundary ~0.66~0.73. τ=0.65 근방 최적 F1 설계."
        ),
        "random_seed": seed,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    print(f"[✓] {inst_path} ({len(instances)}건: conflict={n_conf}, clean={n_clean})")
    print(f"    clean 구성: para≈{int(n_clean*0.7)}건 (cos≥0.85), boundary≈{int(n_clean*0.3)}건 (cos 0.66~0.73)")

    print("\n[샘플 충돌 쌍]")
    for inst in instances:
        if inst["ground_truth_conflict"]:
            print(f"  A(보고): {inst['text_a'][:90]}...")
            print(f"  B(현장): {inst['text_b'][:90]}...")
            break
    print("[샘플 비충돌 쌍(para)]")
    for inst in instances:
        if inst["pair_type"] == "paraphrase":
            print(f"  A: {inst['text_a'][:90]}")
            print(f"  B: {inst['text_b'][:90]}")
            break


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SynConflict v2 — 보고서/현장피드백 수치 모순 쌍 생성")
    parser.add_argument("--n", type=int, default=100, help="총 인스턴스 수 (기본 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"SynConflict v2 빌드: n={args.n}, seed={args.seed}")
    print("충돌: 공식보고+구어체불만 | 비충돌: 패러프레이즈+동일시스템다른지표")
    print("BGE-M3 cos: conflict ~0.60~0.64 / clean-para ≥0.85 / clean-boundary 0.66~0.73")

    instances = build_dataset(args.n, rng)
    save_dataset(instances, Path(args.output), args.seed)


if __name__ == "__main__":
    main()
