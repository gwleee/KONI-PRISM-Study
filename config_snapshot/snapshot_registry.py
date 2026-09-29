#!/usr/bin/env python3
"""
배포 코드의 라우팅 레지스트리와 설정값을 그대로 기록한다 (제품 코드는 read-only import).

원고의 Table 1과 모델 크기 서술(§3.3, §5.2)이 이 파일을 거쳐 export에 들어간다.
설정 스냅샷(settings_snapshot)에는 없는 값 — 모델별 VRAM, 모달리티별 품질 하한 Q_min,
품질 추정 초기값, 비용 — 이 레지스트리 상수로만 존재하기 때문이다.

  python config_snapshot/snapshot_registry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.harness import (bootstrap_gateway, provenance, set_profile,  # noqa: E402
                            settings_snapshot, write_json)


def main() -> None:
    set_profile("a100_1")
    bootstrap_gateway()
    from algorithms import dmar  # noqa: PLC0415

    registry = {
        mid: {
            "vram_gb": spec.vram_gb,
            "supported_modalities": list(spec.supported_modalities),
            "cost_hat": dmar._COST_HAT.get(mid),
            "cost_scale": dict(spec.cost_scale),
            "q_hat_init": dict(spec.q_hat_init),
        }
        for mid, spec in dmar.MODEL_REGISTRY.items()
    }
    payload = {
        "experiment": "deployed routing registry and configuration snapshot",
        "provenance": provenance(),
        "settings": settings_snapshot(),
        "model_registry": registry,
        "q_min": dict(dmar.Q_MIN),
        "q_min_range": [min(dmar.Q_MIN.values()), max(dmar.Q_MIN.values())],
        "rel_base": dict(dmar._REL_BASE),
        "specialist_vram_gb_sum": sum(r["vram_gb"] for m, r in registry.items()
                                      if m in ("vlm_7b", "stt_large", "doc_processor", "llm_7b")),
    }
    out = write_json(HERE / "results" / "registry_snapshot.json", payload)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
