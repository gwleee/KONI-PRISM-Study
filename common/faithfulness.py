"""
faithfulness 형식 정의 및 측정 — Ablation v2 및 논문 리뷰 B 대응.

정의
────
답변 A 와 그 질의에 제공된 출처 청크 집합 S = {s_1, …, s_m} 에 대해:

  1. 사실 단위 분해
       U(A) = {u_1, …, u_n}
     A 를 문장 단위 패시지로 분해한다. 분해기는 제품 코드
     `gateway/algorithms/mars.py::MARSScorer._split_into_passages` 와 동일한
     규칙을 쓴다(한국어 종결어미 '다.'/'요.' 및 문장부호 기준). 이는 MARS 가
     귀속을 계산하는 단위와 faithfulness 의 단위를 일치시키기 위함이다.

  2. 지지 판정
       supported(u) = 1  iff  max_{s ∈ S} P_entail(s → u) ≥ θ_ent
     P_entail 은 다국어 NLI 모델의 entailment 확률이며, 전제=출처 청크,
     가설=사실 단위로 둔다. 기본 θ_ent = 0.5.

  3. 점수
       faithfulness(A, S) = |{u ∈ U(A) : supported(u) = 1}| / |U(A)|

  U(A) 가 비면 faithfulness 는 정의되지 않으며 0.0 으로 기록하고 별도 표기한다.

부수 지표(진단용, 주 지표 아님)
  - contradicted_rate : max_s P_contradict(s → u) ≥ θ_ent 인 단위 비율
  - cosine_support    : BGE-M3 코사인 최대값 ≥ 0.5 로 판정한 지지 비율.
                        임계 기반 코사인이 함의를 대체할 수 없음을 보이기 위해
                        함께 기록한다(Exp D/E 결과와 같은 맥락).

무엇을 요구하지 않는가
  gold answer 를 요구하지 않는다. 정의가 (답변, 출처) 만으로 닫혀 있기 때문이며,
  따라서 '예상 답변'을 합성해 넣지 않는다.

한계 (명시)
  NLI 모델의 판정은 완전하지 않다. 특히 (a) 수치·단위 추론, (b) 다중 청크에
  걸친 조합적 함의는 놓칠 수 있다(단일 전제-가설 쌍으로만 평가하므로).
  따라서 본 지표는 절대 수준이 아니라 **구성 간 비교**에 사용한다.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

NLI_MODEL = "joeddav/xlm-roberta-large-xnli"

# 전제(premise)는 **청크 전체가 아니라 청크를 문장으로 분해한 것**을 쓴다.
# XNLI 계열 모델은 짧은 문장 쌍에 학습돼 있어, 400자 이상의 문서 청크를 전제로
# 주면 분포 밖으로 나가 판정이 무너진다. 통제 검사(validate_metric.py)에서
# 실측한 분리도(copy − unrelated):
#     mDeBERTa-base + 청크 전제   : 0.385 − 0.314 = 0.071  (사용 불가)
#     mDeBERTa-base + 문장 전제   : 0.642 − 0.209 = 0.433
#     xlm-roberta-large + 문장 전제: 0.957 − 0.053 = 0.904  ← 채택
MIN_PREMISE_CHARS = 10


def split_fact_units(text: str) -> list[str]:
    """제품 MARSScorer._split_into_passages 와 동일한 분해 규칙."""
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
            raise RuntimeError(f"NLI 라벨 확인 실패: {id2label}")

    def _nli(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """(premise, hypothesis) 목록 → softmax 확률 (n, 3)."""
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
        """출처 청크를 전제 문장 목록으로 분해한다 (위 주석의 근거 참조)."""
        out: list[str] = []
        for c in source_chunks:
            out.extend(split_fact_units(c))
        return [s for s in out if len(s) >= MIN_PREMISE_CHARS]

    def score(self, answer: str, source_chunks: list[str]) -> dict[str, Any]:
        self._ensure()          # _label_idx 채우기 (_nli 이전에 접근하므로 선행 필요)
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
        """결과 파일에 함께 기록되는 형식 정의."""
        return {
            "name": "faithfulness (fact-unit source support)",
            "formula": ("faithfulness(A,S) = |{u in U(A) : "
                        "max_{s in S} P_entail(s -> u) >= theta_ent}| / |U(A)|"),
            "fact_unit_splitter": ("동일 규칙: gateway/algorithms/mars.py::"
                                   "MARSScorer._split_into_passages"),
            "nli_model": NLI_MODEL,
            "theta_ent": self.ent_threshold,
            "premise": "출처 청크를 문장 단위로 분해한 s (청크 전체가 아님 — 위 주석 참조)",
            "premise_granularity": "sentence",
            "hypothesis": "답변의 사실 단위 u",
            "requires_gold_answer": False,
            "validation": {
                "control_copy": "출처를 그대로 답변 → 0.957",
                "control_unrelated": "무관 문서를 답변 → 0.053",
                "separation": 0.904,
                "script": "paper_experiments/ablation_v2/validate_metric.py",
            },
            "limitations": [
                "단일 전제-가설 쌍으로만 평가하므로 여러 문장에 걸친 조합적 함의를 놓칠 수 있다",
                "수치·단위 추론에 약하다",
                "절대 수준이 아니라 구성 간 비교에 사용해야 한다",
            ],
        }
