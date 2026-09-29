"""
MultiTurnSet v2 — stage 3: assemble projects, sessions and probe queries.

For each run seed:
  * all turns are shuffled and dealt into projects of SESSIONS x TURNS_PER_SESSION turns;
  * a turn's modality set follows its source asset:
      text -> ["text"], document -> ["text", "document"], image -> ["text", "image"];
  * every stored turn contributes one probe (its q_probe); the probe's target is that turn;
  * every other turn in the project is a candidate distractor — distractors come from the same
    pool as targets and are never selected or filtered by modality;
  * the probe's modality set is drawn from the empirical distribution of turn modality sets in
    the same project, independently of the target turn (field "modalities").

Two additional probe-modality variants are stored for a sensitivity analysis only:
  "modalities_rho50"  — with probability 0.5 copy the target's set, otherwise the independent draw;
  "modalities_rho100" — the target's set.
They share the same random draws as the primary field so comparisons are paired.

  python assemble.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

PROJECTS = 3
SESSIONS = 5
TURNS_PER_SESSION = 10
MODS = {"text": ["text"], "document": ["text", "document"], "image": ["text", "image"]}


def build(seed: int, turns: list[dict]) -> dict:
    rng = random.Random(seed)
    mod_rng = random.Random(10_000 + seed)
    pool = turns[:]
    rng.shuffle(pool)
    need = PROJECTS * SESSIONS * TURNS_PER_SESSION
    used, unused = pool[:need], pool[need:]
    ns = uuid.UUID(int=seed)
    projects = []
    for p in range(PROJECTS):
        chunk = used[p * SESSIONS * TURNS_PER_SESSION:(p + 1) * SESSIONS * TURNS_PER_SESSION]
        sessions = []
        for s in range(SESSIONS):
            conv_id = str(uuid.uuid5(ns, f"p{p}s{s}"))
            sess_turns = []
            for j, t in enumerate(chunk[s * TURNS_PER_SESSION:(s + 1) * TURNS_PER_SESSION]):
                sess_turns.append({**t, "modalities": MODS[t["modality"]], "turn_idx": j})
            sessions.append({"conversation_id": conv_id, "turns": sess_turns})
        all_turns = [t for se in sessions for t in se["turns"]]
        combos = [t["modalities"] for t in all_turns]
        probes = []
        for se in sessions:
            for t in se["turns"]:
                indep = list(mod_rng.choice(combos))
                coin = mod_rng.random()
                probes.append({
                    "target_turn_key": t["turn_key"],
                    "target_conversation_id": se["conversation_id"],
                    "target_source": t["source"],
                    "query": t["q_probe"],
                    "target_modalities": t["modalities"],
                    "modalities": indep,
                    "modalities_rho50": t["modalities"] if coin < 0.5 else indep,
                    "modalities_rho100": t["modalities"],
                })
        mod_rng.shuffle(probes)
        projects.append({"project_index": p, "sessions": sessions, "probes": probes})
    return {"seed": seed, "projects": projects, "unused_turn_keys": [t["turn_key"] for t in unused]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    turns = [json.loads(l) for l in (DATA / "turns.jsonl").open(encoding="utf-8")]
    keys = [t["q_store"] for t in turns]
    assert len(set(keys)) == len(keys), "q_store must be unique (turn identity is recovered from it)"
    (DATA / "runs").mkdir(exist_ok=True)
    for seed in a.seeds:
        run = build(seed, turns)
        out = DATA / "runs" / f"run_{seed}.json"
        out.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
        n_probe = sum(len(p["probes"]) for p in run["projects"])
        print(f"seed {seed}: {PROJECTS} projects x {SESSIONS} sessions x {TURNS_PER_SESSION} turns, "
              f"{n_probe} probes -> {out.name}")


if __name__ == "__main__":
    main()
