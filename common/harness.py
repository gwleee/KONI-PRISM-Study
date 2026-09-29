"""
공용 실험 하네스.

설계 원칙:
  1. 제품 코드(gateway/) 무변경. sys.path 주입으로 read-only import만 수행.
  2. BGE-M3 폴백(랜덤 임베딩) 사용 시 즉시 실패 — 조용한 무효 실험 방지.
  3. 모든 결과에 환경 provenance(git sha, GPU, 파라미터) 동봉.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_DIR = REPO_ROOT / "gateway"
PAPER_EXP_DIR = REPO_ROOT / "paper_experiments"


# ── gateway 알고리즘 read-only import ────────────────────────────────────────

def bootstrap_gateway() -> None:
    """gateway/ 를 sys.path 에 올려 알고리즘 모듈을 import 가능하게 한다.

    gateway 모듈은 `from core.config import ...` 형태의 절대 import를 쓰므로
    gateway 디렉터리 자체가 sys.path에 있어야 한다. 제품 코드는 건드리지 않는다.
    """
    p = str(GATEWAY_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def set_profile(profile: str) -> None:
    """하드웨어 프로파일을 환경변수로 고정한다 (get_settings 캐시 전에 호출).

    .env 파일 위치에 따라 프로파일이 달라지는 것을 막기 위해 실험에서는
    항상 명시적으로 지정한다.
    """
    os.environ["HARDWARE_PROFILE"] = profile


# ── 임베딩 진위 게이트 ────────────────────────────────────────────────────────

class FallbackEncoderError(RuntimeError):
    """BGE-M3 대신 랜덤 폴백 인코더가 활성화됐을 때 발생."""


_PROBE = [
    "The cat sat on the mat.",
    "A cat is sitting on a mat.",
    "The stock market crashed in 1929.",
]


async def assert_real_encoder(embedding_service: Any) -> dict[str, float]:
    """임베딩 서비스가 진짜 BGE-M3 인지 검증한다.

    services.embedding_service 는 로딩 실패 시 _FallbackEncoder(시드 고정 랜덤)
    로 조용히 대체된다. 그 상태로 실험을 돌리면 전부 무의미한 숫자가 나오므로
    여기서 강제로 실패시킨다.

    판정: 패러프레이즈 쌍 cos > 0.70 AND 무관 쌍 cos < 0.50.
    """
    vecs = await embedding_service.encode(_PROBE)
    v = np.asarray(vecs, dtype=np.float64)
    n = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    sim = n @ n.T
    para = float(sim[0, 1])
    unrel = float(sim[0, 2])

    impl = type(getattr(embedding_service, "_model", None)).__name__
    if impl == "_FallbackEncoder":
        raise FallbackEncoderError(
            "BGE-M3 로딩 실패 → _FallbackEncoder(랜덤 임베딩) 활성화됨. "
            "실험 결과가 전부 무효가 되므로 중단한다. "
            "FlagEmbedding 설치와 HF 캐시(BAAI/bge-m3)를 확인할 것."
        )
    if not (para > 0.70 and unrel < 0.50):
        raise FallbackEncoderError(
            f"임베딩 sanity check 실패: paraphrase_cos={para:.4f} (>0.70 기대), "
            f"unrelated_cos={unrel:.4f} (<0.50 기대). 인코더={impl}"
        )
    return {"paraphrase_cos": para, "unrelated_cos": unrel, "encoder_impl": impl}


# ── provenance ───────────────────────────────────────────────────────────────

def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, cwd=REPO_ROOT
        ).stdout.strip()
    except Exception:
        return ""


def gpu_info() -> list[dict[str, Any]]:
    out = _sh([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "vram_total_mib": int(parts[2]),
                "driver": parts[3],
            })
    return gpus


def provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _sh(["git", "rev-parse", "HEAD"]),
        "git_branch": _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_sh(["git", "status", "--porcelain"])),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpus": gpu_info(),
        "hardware_profile_env": os.environ.get("HARDWARE_PROFILE"),
    }
    if extra:
        prov.update(extra)
    return prov


def settings_snapshot() -> dict[str, Any]:
    """실험 시점의 알고리즘 하이퍼파라미터 실값을 기록한다."""
    from core.config import get_settings  # noqa: PLC0415

    s = get_settings()
    return {
        "hardware_profile": s.hardware_profile,
        "vram_gb": s.vram_gb,
        "alpha": s.alpha,
        "dmar_alpha_ref": s.dmar_alpha_ref,
        "dmar_beta": s.dmar_beta,
        "dmar_vram_reserve_gb": s.dmar_vram_reserve_gb,
        "dmar_qmin_default": s.dmar_qmin_default,
        "cmae_tau_conflict": s.cmae_tau_conflict,
        "cmae_gamma": s.cmae_gamma,
        "cmae_tau_t": s.cmae_tau_t,
        "cmae_theta_dom": s.cmae_theta_dom,
        "hmmr_lambda1": s.hmmr_lambda1,
        "hmmr_lambda2": s.hmmr_lambda2,
        "hmmr_lambda3": s.hmmr_lambda3,
        "hmmr_delta": s.hmmr_delta,
        "hmmr_omega_s": s.hmmr_omega_s,
        "hmmr_omega_e": s.hmmr_omega_e,
        "hmmr_omega_l": s.hmmr_omega_l,
        "hmmr_theta_imp": s.hmmr_theta_imp,
        "mars_theta_cite": s.mars_theta_cite,
        "mars_theta_sent": s.mars_theta_sent,
    }


# ── 통계 유틸 ────────────────────────────────────────────────────────────────

def summarize(samples: list[float]) -> dict[str, float]:
    """mean ± std + 분위수. 표본이 비면 NaN 대신 None-safe 값을 낸다."""
    if not samples:
        return {"n": 0}
    a = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def binary_scores(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """이진 분류 지표. sklearn 의존 없이 직접 계산(제로 division 안전)."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(yt) if len(yt) else 0.0
    # Matthews 상관계수 — 클래스 불균형에 강건
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": acc, "mcc": float(mcc),
    }


# ── 결과 저장 ────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        return super().default(o)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, cls=_NpEncoder),
        encoding="utf-8",
    )
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, cls=_NpEncoder) + "\n")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> Path:
    import csv  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    cols = columns or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


class Timer:
    """구간 wall-clock 계측 (perf_counter 기반)."""

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self._t0: float | None = None
        self._label: str | None = None

    def __call__(self, label: str) -> "Timer":
        self._label = label
        return self

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        assert self._t0 is not None and self._label is not None
        self.marks[self._label] = (time.perf_counter() - self._t0) * 1000.0  # ms
        self._t0 = None
        self._label = None
