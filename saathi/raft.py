"""RAFT-style fine-tuning dataset builder.

RAFT (retrieval-augmented fine-tuning) trains a model on question / context / answer triples where each question
came with one 'oracle' chunk (holds the answer) and 'distractor' chunks (do not). That
teaches a small open model to answer only from the right document and cite it. This
script builds the same kind of JSONL from the knowledge base and the RM's call notes, so
the bank can later fine-tune an in-house model (no customer data leaves the bank).

Run:  python -m saathi.raft            (template questions, no key)
      python -m saathi.raft --llm      (questions written by the LLM in Settings / keys file)
"""
from __future__ import annotations

import argparse
import json
import random

from . import config as C, keys
from .kb import load_chunks

TEMPLATES = ["What should I know about {t}?", "{t} ke baare mein batao", "Explain {t} simply for a connector",
             "Customer asks about {t}: what do I say?"]


def build(use_llm: bool = False, n_distract: int = 2, seed: int = 7) -> str:
    rng = random.Random(seed)
    chunks = load_chunks()
    llm = None
    if use_llm:
        from .llm import LLM
        k = keys.load()
        llm = LLM(k.get("LLM_PROVIDER", "Offline (no LLM)"), k.get("LLM_API_KEY", ""), k.get("LLM_MODEL", ""), k.get("LLM_BASE_URL", ""))
    out = C.DATA / "raft_dataset.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            qs = [t.format(t=c.title) for t in rng.sample(TEMPLATES, 2)]
            if llm is not None and llm.online:
                try:
                    txt = llm.complete(f"Write 3 short questions a loan connector in India might ask that this text answers. "
                                       f"One line each, no numbering.\n\n{c.text}", max_tokens=200)
                    qs = [q.strip("-• ").strip() for q in txt.splitlines() if q.strip()][:3] or qs
                except Exception:
                    pass
            for q in qs:
                distract = rng.sample([d for d in chunks if d is not c], k=min(n_distract, len(chunks) - 1))
                docs = [c] + distract
                rng.shuffle(docs)
                rec = dict(question=q,
                           context=[dict(source=d.cite, text=d.text) for d in docs],
                           oracle=c.cite,
                           answer=f"##begin_quote## {c.text[:400]} ##end_quote## (source: {c.cite})",
                           instruction="Answer only from the oracle document and quote it. Say you do not know otherwise.")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    print("Wrote", build(ap.parse_args().llm))
