"""Knowledge base and retrieval (RAG).

Standard RAG: documents are chunked, indexed and the top chunks are passed to the model
with the question. Production would use embeddings (e.g. BGE) and a vector DB; this demo uses a TF-IDF index (word + character n-grams, so Hinglish spellings
still match) to run offline with no GPU and no vector DB. Swap `KB.search` for a vector
store in production.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config as C


@dataclass
class Chunk:
    doc: str
    title: str
    text: str

    @property
    def cite(self) -> str:
        return f"{self.doc} > {self.title}"


def load_chunks() -> list[Chunk]:
    out = []
    for f in sorted(C.KB_DIR.glob("*.md")):
        doc = f.stem.replace("_", " ").title()
        parts = re.split(r"^## ", f.read_text(encoding="utf-8"), flags=re.M)
        for p in parts[1:]:
            title, _, body = p.partition("\n")
            body = " ".join(body.split())
            if body:
                out.append(Chunk(doc=doc, title=title.strip(), text=body))
    return out


class KB:
    def __init__(self):
        self.chunks = load_chunks()
        docs = [c.title + ". " + c.text for c in self.chunks]
        self.w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.M = hstack([self.w.fit_transform(docs), 0.6 * self.c.fit_transform(docs)]).tocsr()

    def search(self, q: str, k: int = 3, min_score: float = 0.05) -> list[tuple[Chunk, float]]:
        v = hstack([self.w.transform([q]), 0.6 * self.c.transform([q])]).tocsr()
        sims = (self.M @ v.T).toarray().ravel()
        norms = np.sqrt(self.M.multiply(self.M).sum(1)).A.ravel() * np.sqrt(v.multiply(v).sum()) + 1e-9
        sims = sims / norms
        idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx if sims[i] >= min_score]

    def context(self, q: str, k: int = 3) -> tuple[str, list[str]]:
        hits = self.search(q, k)
        ctx = "\n\n".join(f"[{c.cite}] {c.text}" for c, _ in hits)
        return ctx, [c.cite for c, _ in hits]
