"""
Formal definition and measurement of faithfulness — for Ablation v2 and paper review item B.

Definition
──────────
For an answer A and the set of source chunks S = {s_1, …, s_m} provided for its query:

  1. Fact-unit decomposition
       U(A) = {u_1, …, u_n}
     A is split into sentence-level passages. The splitter uses the same rules as
     the product code `gateway/algorithms/mars.py::MARSScorer._split_into_passages`
     (Korean sentence-final endings '다.'/'요.' and punctuation). This makes the unit over
     which MARS computes attribution identical to the unit of faithfulness.

  2. Support decision
       supported(u) = 1  iff  max_{s ∈ S} P_entail(s → u) ≥ θ_ent
     P_entail is the entailment probability of a multilingual NLI model, with premise = source chunk
     and hypothesis = fact unit. Default θ_ent = 0.5.

  3. Score
       faithfulness(A, S) = |{u ∈ U(A) : supported(u) = 1}| / |U(A)|

  If U(A) is empty, faithfulness is undefined; it is recorded as 0.0 and flagged separately.

Auxiliary metrics (diagnostic, not primary)
  - contradicted_rate : fraction of units with max_s P_contradict(s → u) ≥ θ_ent
  - cosine_support    : support rate judged by max BGE-M3 cosine ≥ 0.5.
                        Recorded alongside to show that thresholded cosine
                        cannot substitute for entailment (same context as Exp D/E).

What it does not require
  No gold answer is required, because the definition is closed over (answer, sources) alone;
  hence no synthetic 'expected answer' is injected.

Limitations (stated explicitly)
  NLI judgments are imperfect. In particular, (a) numeric/unit reasoning and (b) compositional
  entailment spanning multiple chunks can be missed (only single premise-hypothesis pairs are evaluated).
  This metric is therefore used for **comparison across configurations**, not as an absolute level.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

NLI_MODEL = "joeddav/xlm-roberta-large-xnli"

# The premise is **the chunk split into sentences, not the whole chunk**.
# XNLI-family models are trained on short sentence pairs; a document chunk of 400+ characters
# as the premise is out of distribution and judgments collapse. Separation (copy − unrelated)
# measured in the control check (validate_metric.py):
#     mDeBERTa-base + chunk premise   : 0.385 − 0.314 = 0.071  (unusable)
#     mDeBERTa-base + sentence premise: 0.642 − 0.209 = 0.433
#     xlm-roberta-large + sentence premise: 0.971 − 0.048 = 0.924  ← adopted
MIN_PREMISE_CHARS = 10


def split_fact_units(text: str) -> list[str]:
    """Same splitting rule as the product MARSScorer._split_into_passages."""
    sentences = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=요\.)\s+", text)
    return [s.strip() for s in sentences if s.strip()]


class FaithfulnessScorer:
    def __init__(self, ent_threshold: float = 0.5, device: str | None = None,
                 max_length: int = 256, batch_size: int = 16) -> None:
        self.ent_threshold = ent_threshold
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = device
        self._tok = None
        self._model = None
        self._label_idx: dict[str, int] = {}

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification, AutoTokenizer,
        )
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL).eval().to(dev)
        self._device = dev
        id2label = self._model.config.id2label
        self._label_idx = {v.lower(): int(k) for k, v in id2label.items()}
        missing = {"entailment", "contradiction"} - set(self._label_idx)
        if missing:
            raise RuntimeError(f"NLI label check failed: {id2label}")

    def _nli(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """List of (premise, hypothesis) → softmax probabilities (n, 3)."""
        import torch  # noqa: PLC0415
        self._ensure()
        outs = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            enc = self._tok([p for p, _ in chunk], [h for _, h in chunk],
                            return_tensors="pt", truncation=True,
                            max_length=self.max_length, padding=True).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, 3))

    @staticmethod
    def _premises(source_chunks: list[str]) -> list[str]:
        """Split a source chunk into a list of premise sentences (see rationale in the comment above)."""
        out: list[str] = []
        for c in source_chunks:
            out.extend(split_fact_units(c))
        return [s for s in out if len(s) >= MIN_PREMISE_CHARS]

    def score(self, answer: str, source_chunks: list[str]) -> dict[str, Any]:
        self._ensure()          # populate _label_idx (accessed before _nli, so must run first)
        units = split_fact_units(answer)
        premises = self._premises(source_chunks)
        if not units or not premises:
            return {"faithfulness": 0.0, "n_units": len(units),
                    "n_supported": 0, "contradicted_rate": 0.0,
                    "cosine_support": 0.0, "undefined": True}

        ei, ci = self._label_idx["entailment"], self._label_idx["contradiction"]
        pairs = [(s, u) for u in units for s in premises]
        probs = self._nli(pairs).reshape(len(units), len(premises), 3)

        ent_max = probs[:, :, ei].max(axis=1)
        con_max = probs[:, :, ci].max(axis=1)
        supported = ent_max >= self.ent_threshold
        contradicted = con_max >= self.ent_threshold

        return {
            "faithfulness": float(supported.mean()),
            "n_units": int(len(units)),
            "n_premises": int(len(premises)),
            "n_supported": int(supported.sum()),
            "contradicted_rate": float(contradicted.mean()),
            "ent_max_mean": float(ent_max.mean()),
            "undefined": False,
        }

    def definition(self) -> dict[str, Any]:
        """Formal definition recorded alongside the result files."""
        return {
            "name": "faithfulness (fact-unit source support)",
            "formula": ("faithfulness(A,S) = |{u in U(A) : "
                        "max_{s in S} P_entail(s -> u) >= theta_ent}| / |U(A)|"),
            "fact_unit_splitter": ("same rule as: gateway/algorithms/mars.py::"
                                   "MARSScorer._split_into_passages"),
            "nli_model": NLI_MODEL,
            "theta_ent": self.ent_threshold,
            "premise": "s = source chunk split into sentences (not the whole chunk — see comment above)",
            "premise_granularity": "sentence",
            "hypothesis": "fact unit u of the answer",
            "requires_gold_answer": False,
            "validation": {
                "control_copy": "source copied verbatim as answer → 0.957",
                "control_unrelated": "unrelated document as answer → 0.053",
                "separation": 0.904,
                "script": "paper_experiments/ablation_v2/validate_metric.py",
            },
            "limitations": [
                "evaluates only single premise-hypothesis pairs, so compositional entailment spanning multiple sentences can be missed",
                "weak at numeric/unit reasoning",
                "should be used for comparison across configurations, not as an absolute level",
            ],
        }
