"""
MultiTurnSet v2 — stage 1: cut product assets into content segments.

Sources (experiments/datasets/raw/):
  text/  KLUE news and encyclopedia articles  -> modality "text"
  pdf/   arXiv papers                         -> modality "document"
  image/ PNG charts                           -> modality "image"

Each segment later becomes one conversation turn. Selection is deterministic
(position-based) so that the segment set does not depend on any retrieval
outcome.

Run with an interpreter that has pypdf (e.g. the base conda env):
  python paper_experiments/hmmr_v2/build_segments.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "experiments" / "datasets" / "raw"
OUT = Path(__file__).resolve().parent / "data" / "segments.jsonl"

TEXT_PER_ARTICLE = 2
TEXT_MIN_CHARS = 150
PDF_PER_PAPER = 6
PDF_WINDOW_CHARS = 900
PDF_MAX_PAGES = 10


def text_segments(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    paras = [ln for ln in lines if ln and not ln.startswith("#")]
    out = [p for p in paras if len(p) >= TEXT_MIN_CHARS]
    if not out and paras:
        out = [" ".join(paras)]
    return out[:TEXT_PER_ARTICLE]


def pdf_segments(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    text = " ".join(
        (reader.pages[i].extract_text() or "") for i in range(min(PDF_MAX_PAGES, len(reader.pages)))
    )
    text = re.sub(r"\s+", " ", text)
    m = re.search(r"\bReferences\b", text)
    if m and m.start() > PDF_WINDOW_CHARS * PDF_PER_PAPER:
        text = text[: m.start()]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    windows, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) > PDF_WINDOW_CHARS and cur:
            windows.append(cur.strip())
            cur = ""
        cur += s + " "
    if cur.strip():
        windows.append(cur.strip())
    windows = [w for w in windows if len(w) >= PDF_WINDOW_CHARS // 2]
    if len(windows) <= PDF_PER_PAPER:
        return windows
    step = len(windows) / PDF_PER_PAPER
    return [windows[int(i * step)] for i in range(PDF_PER_PAPER)]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted((RAW / "text").glob("*.txt")):
        for k, seg in enumerate(text_segments(p)):
            rows.append({"source": f"text/{p.name}", "modality": "text", "part": k, "text": seg})
    for p in sorted((RAW / "pdf").glob("*.pdf")):
        for k, seg in enumerate(pdf_segments(p)):
            rows.append({"source": f"pdf/{p.name}", "modality": "document", "part": k, "text": seg})
    for p in sorted((RAW / "image").glob("*.png")):
        rows.append({"source": f"image/{p.name}", "modality": "image", "part": 0,
                     "image_path": str(p.relative_to(REPO))})
    with OUT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"seg_id": f"seg{i:04d}", **r}, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["modality"]] = counts.get(r["modality"], 0) + 1
    print(f"{len(rows)} segments -> {OUT}  {counts}")


if __name__ == "__main__":
    main()
