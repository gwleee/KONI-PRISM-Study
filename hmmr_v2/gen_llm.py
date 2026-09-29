"""
MultiTurnSet v2 — LLM/VLM generation stages (offline vLLM, same weights as the
product specialist containers: Qwen2-VL-7B-Instruct, Qwen2.5-7B-Instruct).

Stages
  caption    image segments -> VLM description, using the product DMAR prompt
             ("이 image를 분석해 주세요.", gateway/algorithms/dmar.py).
  questions  every segment -> question pairs (q_store, q_probe) + an answer to q_store.
  summaries  assembled sessions -> 10-turn session summaries, using the product
             prompt and message format of services/conversation_service.py.

Run in the vLLM env:
  python gen_llm.py caption
  python gen_llm.py questions
  python gen_llm.py summaries --runs data/runs/run_*.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
HF = Path.home() / ".cache" / "huggingface" / "hub"
VLM = "Qwen/Qwen2-VL-7B-Instruct"
LLM = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260922

PAIRS_PER_MODALITY = {"text": 1, "document": 1, "image": 3}

Q_SYSTEM = (
    "You write realistic user questions for a research assistant that keeps a memory of "
    "past conversations. Reply with JSON only."
)
Q_USER = """Material (a {kind} the user shared with the assistant):
<<<
{content}
>>>

Write {n} question pair(s) about this material. For each pair:
- "q_store": a question the user asks now while working with this material.
- "q_probe": a question the same user asks days later, in a different session, to recall the
  same specific information. Use different wording from q_store and make it self-contained
  (name the subject explicitly; do not say "this document" or "the image").
{distinct}Write both questions in the language of the material ({lang}).
Return: {{"pairs": [{{"q_store": "...", "q_probe": "..."}}]}}"""

A_SYSTEM = "주어진 자료에 근거해 사용자의 질문에 간결하고 정확하게 답하세요."
A_USER = "자료:\n{content}\n\n질문: {q}"

# services/conversation_service.py::_generate_summary
SUM_SYSTEM = "다음 대화를 3-5문장으로 간결하게 요약하세요. 핵심 주제, 결론, 중요 정보를 포함하세요."


def _lang(seg: dict) -> str:
    return "English" if seg["modality"] == "document" else "Korean"


def _kind(seg: dict) -> str:
    return {"text": "text passage", "document": "PDF paper excerpt", "image": "image (described below)"}[seg["modality"]]


def _content(seg: dict) -> str:
    return seg["caption"] if seg["modality"] == "image" else seg["text"]


def _parse_pairs(text: str, n: int) -> list[dict] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        pairs = json.loads(m.group(0)).get("pairs", [])
    except json.JSONDecodeError:
        return None
    ok = [p for p in pairs if isinstance(p, dict) and p.get("q_store") and p.get("q_probe")
          and p["q_store"].strip() != p["q_probe"].strip()]
    return ok[:n] if len(ok) >= n else None


def load_segments(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / name).open(encoding="utf-8")]


def write(name: str, rows: list[dict]) -> None:
    with (DATA / name).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage_caption() -> None:
    from PIL import Image
    from vllm import LLM as VLLM, SamplingParams

    segs = load_segments("segments.jsonl")
    llm = VLLM(model=VLM, max_model_len=8192, limit_mm_per_prompt={"image": 1}, seed=SEED)
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    prompt = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              "이 image를 분석해 주세요.<|im_end|>\n<|im_start|>assistant\n")
    imgs = [s for s in segs if s["modality"] == "image"]
    outs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": Image.open(REPO / s["image_path"]).convert("RGB")}}
         for s in imgs], sp)
    cap = {s["seg_id"]: o.outputs[0].text.strip() for s, o in zip(imgs, outs)}
    for s in segs:
        if s["seg_id"] in cap:
            s["caption"] = cap[s["seg_id"]]
    write("segments_captioned.jsonl", segs)
    print(f"captioned {len(cap)} images")


def stage_questions() -> None:
    from vllm import LLM as VLLM, SamplingParams

    segs = load_segments("segments_captioned.jsonl")
    llm = VLLM(model=LLM, max_model_len=8192, seed=SEED)
    tok = llm.get_tokenizer()

    def chat(system: str, user: str) -> str:
        return tok.apply_chat_template([{"role": "system", "content": system},
                                        {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)

    pending = list(range(len(segs)))
    pairs: dict[int, list[dict]] = {}
    for attempt, temp in enumerate([0.0, 0.7, 0.7, 0.9]):
        if not pending:
            break
        prompts = []
        for i in pending:
            s = segs[i]
            n = PAIRS_PER_MODALITY[s["modality"]]
            distinct = "- Different pairs must ask about different details.\n" if n > 1 else ""
            prompts.append(chat(Q_SYSTEM, Q_USER.format(kind=_kind(s), content=_content(s), n=n,
                                                        distinct=distinct, lang=_lang(s))))
        outs = llm.generate(prompts, SamplingParams(temperature=temp, max_tokens=768, seed=SEED + attempt))
        still = []
        for i, o in zip(pending, outs):
            p = _parse_pairs(o.outputs[0].text, PAIRS_PER_MODALITY[segs[i]["modality"]])
            if p is None:
                still.append(i)
            else:
                pairs[i] = p
        print(f"attempt {attempt}: parsed {len(pending) - len(still)}/{len(pending)}")
        pending = still

    turns, a_prompts = [], []
    for i, s in enumerate(segs):
        for k, p in enumerate(pairs.get(i, [])):
            turns.append({"turn_key": f"{s['seg_id']}_{k}", "seg_id": s["seg_id"], "source": s["source"],
                          "modality": s["modality"], "content": _content(s),
                          "q_store": p["q_store"].strip(), "q_probe": p["q_probe"].strip()})
            a_prompts.append(chat(A_SYSTEM, A_USER.format(content=_content(s), q=p["q_store"].strip())))
    outs = llm.generate(a_prompts, SamplingParams(temperature=0.0, max_tokens=384))
    for t, o in zip(turns, outs):
        t["answer"] = o.outputs[0].text.strip()
    write("turns.jsonl", turns)
    print(f"{len(turns)} turns; unparsed segments: {[segs[i]['seg_id'] for i in pending]}")


def stage_summaries(run_files: list[str]) -> None:
    from vllm import LLM as VLLM, SamplingParams

    llm = VLLM(model=LLM, max_model_len=8192, seed=SEED)
    tok = llm.get_tokenizer()
    for rf in run_files:
        run = json.loads(Path(rf).read_text(encoding="utf-8"))
        jobs, prompts = [], []
        for proj in run["projects"]:
            for sess in proj["sessions"]:
                msgs = []
                for t in sess["turns"]:
                    msgs.append(f"[USER] {t['q_store'][:300]}")
                    msgs.append(f"[ASSISTANT] {t['answer'][:300]}")
                    if len(msgs) // 2 % 10 == 0:
                        dialog = "\n".join(msgs[-20:])
                        prompts.append(tok.apply_chat_template(
                            [{"role": "system", "content": SUM_SYSTEM}, {"role": "user", "content": dialog}],
                            tokenize=False, add_generation_prompt=True))
                        jobs.append((sess["conversation_id"], len(msgs) // 2))
        outs = llm.generate(prompts, SamplingParams(temperature=0.3, max_tokens=512, seed=run["seed"]))
        summaries = [{"conversation_id": c, "after_turn": n, "summary": o.outputs[0].text.strip()}
                     for (c, n), o in zip(jobs, outs)]
        out = Path(rf).with_name(Path(rf).stem + "_summaries.json")
        out.write_text(json.dumps(summaries, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{rf}: {len(summaries)} summaries -> {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["caption", "questions", "summaries"])
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args()
    {"caption": stage_caption, "questions": stage_questions}.get(a.stage, lambda: stage_summaries(a.runs))()
