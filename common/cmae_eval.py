"""
CMAE 쌍(pair) 단위 충돌 검출 평가 — Exp D/E/F 공용.

CMAE 실 구현(gateway/algorithms/cmae.py)의 충돌 판정은 Eq.(9):
    conflict(i,j) = 1[ A(i,j) < τ_conflict ],  A = cos_sim(Enc(r_i), Enc(r_j))
따라서 k=2 쌍에 대해서는 "코사인 < τ" 와 정확히 동치다.
이 동치는 verify_against_cmae() 에서 실제 CMAEEvaluator 를 돌려 확인한다
(수식 재구현이 아니라 제품 코드 경로를 검증한다).
"""
from __future__ import annotations

import numpy as np

from .harness import assert_real_encoder, binary_scores


async def encode_pairs(
    instances: list[dict],
    batch_size: int = 64,
    text_a_key: str = "text_a",
    text_b_key: str = "text_b",
) -> tuple[np.ndarray, dict]:
    """인스턴스 쌍을 BGE-M3로 인코딩해 코사인 유사도 배열을 만든다.

    반환: (sims shape (N,), encoder_probe dict)
    """
    from services.embedding_service import get_embedding_service  # noqa: PLC0415

    svc = get_embedding_service()
    probe = await assert_real_encoder(svc)   # 폴백 인코더면 여기서 중단

    texts: list[str] = []
    for r in instances:
        texts.append(r[text_a_key])
        texts.append(r[text_b_key])

    vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        vecs.append(await svc.encode(texts[i:i + batch_size]))
    emb = np.concatenate(vecs, axis=0)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    a = emb[0::2]
    b = emb[1::2]
    sims = np.sum(a * b, axis=1)
    return sims.astype(np.float64), probe


def evaluate_at_threshold(sims: np.ndarray, labels: list[int], tau: float) -> dict:
    """Eq.(9) 판정: cos < τ 이면 충돌로 예측."""
    pred = (sims < tau).astype(int).tolist()
    out = binary_scores(labels, pred)
    out["tau"] = float(tau)
    return out


def sweep_threshold(
    sims: np.ndarray,
    labels: list[int],
    lo: float = 0.0,
    hi: float = 1.0,
    step: float = 0.005,
) -> tuple[list[dict], dict]:
    """τ 전 구간 스윕. 반환: (곡선, 최적 F1 지점)"""
    curve = []
    for tau in np.arange(lo, hi + 1e-9, step):
        curve.append(evaluate_at_threshold(sims, labels, float(tau)))
    best = max(curve, key=lambda r: (r["f1"], r["mcc"]))
    return curve, best


def roc_auc(sims: np.ndarray, labels: list[int]) -> float:
    """충돌 점수 = -cos (낮은 유사도일수록 충돌). Mann-Whitney U 기반 AUC.

    임계값 선택과 무관한 '분리 가능성' 자체를 측정한다.
    """
    y = np.asarray(labels, dtype=int)
    score = -np.asarray(sims, dtype=np.float64)
    pos, neg = score[y == 1], score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    s_sorted = score[order]
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    # 동점 처리: 평균 순위
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    n1, n0 = float(pos.size), float(neg.size)
    return float((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def similarity_bins(
    sims: np.ndarray, labels: list[int], edges: list[float] | None = None
) -> list[dict]:
    """유사도 구간별 레이블 분포 — 데이터셋이 특정 구간에 몰려 있는지 진단."""
    if edges is None:
        edges = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]
    y = np.asarray(labels, dtype=int)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sims >= lo) & (sims < hi if hi < 1.0 else sims <= 1.0)
        rows.append({
            "bin_lo": lo, "bin_hi": hi,
            "n": int(m.sum()),
            "n_conflict": int(y[m].sum()) if m.any() else 0,
            "n_clean": int((1 - y[m]).sum()) if m.any() else 0,
        })
    return rows


def separability_index(bins: list[dict]) -> float:
    """구간 중첩도: 두 레이블이 동시에 존재하는 구간이 담는 표본 비율.

    0에 가까울수록 '유사도만으로 완전 분리 가능'(= v1 같은 인위적 구성),
    클수록 임계값 하나로는 분리 불가능한 현실적 데이터셋.
    """
    total = sum(b["n"] for b in bins)
    if total == 0:
        return 0.0
    overlap = sum(b["n"] for b in bins if b["n_conflict"] > 0 and b["n_clean"] > 0)
    return overlap / total


async def verify_against_cmae(
    instances: list[dict], sims: np.ndarray, tau: float, n_check: int = 20
) -> dict:
    """제품 CMAEEvaluator 를 직접 돌려 쌍 판정이 일치하는지 확인한다.

    코사인 재구현이 실 코드와 어긋나지 않았음을 보증하기 위한 게이트.
    """
    from algorithms.cmae import CMAEEvaluator, ModalityResponse  # noqa: PLC0415

    ev = CMAEEvaluator()
    ev._tau_conflict = tau  # 평가 대상 임계값으로 고정
    n_check = min(n_check, len(instances))
    agree = 0
    details = []
    for idx in range(n_check):
        r = instances[idx]
        out = await ev.evaluate([
            ModalityResponse(modality="text", model_id="a", content=r["text_a"]),
            ModalityResponse(modality="text", model_id="b", content=r["text_b"]),
        ])
        cmae_conflict = len(out.conflict_pairs) > 0
        ours = bool(sims[idx] < tau)
        ok = cmae_conflict == ours
        agree += ok
        details.append({
            "id": r.get("id"), "cmae_A01": float(out.alignment_matrix[0, 1]),
            "our_cos": float(sims[idx]), "cmae_conflict": cmae_conflict,
            "our_conflict": ours, "agree": ok,
        })
    return {
        "n_checked": n_check,
        "n_agree": agree,
        "agreement_rate": agree / n_check if n_check else 0.0,
        "max_abs_cos_diff": max(
            (abs(d["cmae_A01"] - d["our_cos"]) for d in details), default=0.0),
        "samples": details[:5],
    }
