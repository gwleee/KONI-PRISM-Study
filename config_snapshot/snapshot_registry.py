#!/usr/bin/env python3
"""
Record the deployed code's routing registry and settings verbatim (product code is imported read-only).

The manuscript's Table 1 and model-size statements (§3.3, §5.2) reach the export through this file,
because values absent from the settings snapshot (settings_snapshot) — per-model VRAM, per-modality
quality floor Q_min, initial quality estimates, cost — exist only as these registry constants.

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
