"""
Pair-level CMAE conflict detection evaluation — shared by Exp D/E/F.

The conflict decision in the actual CMAE implementation (gateway/algorithms/cmae.py) is Eq.(9):
    conflict(i,j) = 1[ A(i,j) < τ_conflict ],  A = cos_sim(Enc(r_i), Enc(r_j))
Hence for a k=2 pair it is exactly equivalent to "cosine < τ".
This equivalence is checked in verify_against_cmae() by running the actual CMAEEvaluator
(it verifies the product code path, not a re-implementation of the formula).
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
    """Encode instance pairs with BGE-M3 and build an array of cosine similarities.

    Returns: (sims shape (N,), encoder_probe dict)
    """
    from services.embedding_service import get_embedding_service  # noqa: PLC0415

    svc = get_embedding_service()
    probe = await assert_real_encoder(svc)   # abort here if the fallback encoder is active

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
    """Eq.(9) decision: predict conflict if cos < τ."""
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
    """Sweep over the full τ range. Returns: (curve, best-F1 point)"""
    curve = []
    for tau in np.arange(lo, hi + 1e-9, step):
        curve.append(evaluate_at_threshold(sims, labels, float(tau)))
    best = max(curve, key=lambda r: (r["f1"], r["mcc"]))
    return curve, best


def roc_auc(sims: np.ndarray, labels: list[int]) -> float:
    """Conflict score = -cos (lower similarity means conflict). AUC via Mann-Whitney U.

    Measures 'separability' itself, independent of threshold choice.
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
    # ties: average rank
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
    """Label distribution per similarity bin — diagnoses whether the dataset is concentrated in certain bins."""
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
    """Bin overlap: fraction of samples in bins where both labels are present.

    Near 0 means 'fully separable by similarity alone' (= an artificial construction like v1);
    larger values mean a realistic dataset that a single threshold cannot separate.
    """
    total = sum(b["n"] for b in bins)
    if total == 0:
        return 0.0
    overlap = sum(b["n"] for b in bins if b["n_conflict"] > 0 and b["n_clean"] > 0)
    return overlap / total


async def verify_against_cmae(
    instances: list[dict], sims: np.ndarray, tau: float, n_check: int = 20
) -> dict:
    """Run the product CMAEEvaluator directly and check that pair decisions match.

    A gate guaranteeing that the cosine re-implementation does not diverge from the actual code.
    """
    from algorithms.cmae import CMAEEvaluator, ModalityResponse  # noqa: PLC0415

    ev = CMAEEvaluator()
    ev._tau_conflict = tau  # pin to the threshold under evaluation
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
