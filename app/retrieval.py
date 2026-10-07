"""Tokeniser, BM25 and the persisted chunk index. Pure standard library."""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

STOP = set("""a an the of in on at to for from by with and or is are was were be been being what which who whom
whose when where why how does do did has have had it its this that these those as into than then there their
they them i me my we our you your can could should would will shall may might must not no yes tell give list
name value number please also about per""".split())

_NUM_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9_#\-./]*")
_SPLIT = re.compile(r"[_\-./#]+")


def tokens(text: str) -> list[str]:
    """Identifier-friendly tokens: 'AB-12' -> ab-12, ab12, ab, 12; 'SIG_ONE#' -> ... sigone."""
    text = _NUM_COMMA.sub("", text)
    out: list[str] = []
    for m in _TOK.finditer(text):
        t = m.group(0).lower().rstrip(".-_/")
        if not t:
            continue
        out.append(t)
        parts = [p for p in _SPLIT.split(t) if p]
        if len(parts) > 1:
            out.append("".join(parts))
            out.extend(parts)
    return out


def query_terms(text: str, weight: float = 1.0, into: dict | None = None) -> dict[str, float]:
    terms = into if into is not None else {}
    for t in tokens(text):
        if t in STOP:
            continue
        terms[t] = max(terms.get(t, 0.0), weight)
    return terms


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        """
        BM25 scores each chunk by how well it matches the query words. It rewards chunks that contain rare query words, many times, without letting very long chunks win just by being long.
        It tokenizes every chunk (here docs is path + text).
        self.dl[i] stores each chunk's length in tokens, and self.avg is the average.
        self.inv is an inverted index: term → [(chunk_id, count), ...]. For example, "orr-1847" maps to the chunks containing it and how many times. A search then looks at only the chunks that contain a query term, not the whole corpus.
        """
        self.k1, self.b, self.n = k1, b, len(docs)
        self.inv: dict[str, list[tuple[int, int]]] = {}
        self.dl: list[int] = []
        for i, d in enumerate(docs):
            toks = tokens(d)
            self.dl.append(len(toks))
            for t, c in Counter(toks).items():
                self.inv.setdefault(t, []).append((i, c))
        self.avg = (sum(self.dl) / self.n) if self.n else 1.0

    def search(self, terms: dict[str, float], k: int = 10, mult: list[float] | None = None):
        """For each query term t with weight w, and each chunk i that contains it:

            score += w × idf × tf × (k1 + 1) / (tf + k1 × (1 − b + b × len_i / avg_len))

            The parts:

            idf is how rare the term is: log(1 + (N − df + 0.5) / (df + 0.5)). N is the number of chunks and df is how many contain the term. A word in almost every chunk gets a score near 0. A word in one chunk, like a ticket id, gets a high score. This is why exact identifiers retrieve so well.
            tf is how many times the term appears in the chunk, but with diminishing returns. The tf / (tf + k1…) shape means 10 occurrences are not worth 10 times one occurrence.
            k1 = 1.4 sets how quickly that saturation happens. A higher value lets repeated words count for longer.
            b = 0.75 is the length penalty. A chunk longer than average gets a larger denominator and so a lower score. b = 0 turns the penalty off, and b = 1 applies it fully.
            w is the query-term weight. The question's own words use 1.0 and the model's suggested keywords use 0.5.
            mult is an optional per-chunk multiplier. I use 0.4 for chunks from withdrawn files, so they rank lower but are not removed.
            The method returns the top k pairs (chunk_id, score), best first.
        
        """
        scores: dict[int, float] = {}
        for t, w in terms.items():
            plist = self.inv.get(t)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg)
                scores[i] = scores.get(i, 0.0) + w * idf * tf * (self.k1 + 1) / denom
        if mult:
            for i in scores:
                scores[i] *= mult[i]
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


class Index:
    """chunks: [{file, kind, status, loc, text}] + a BM25 over 'path + text'."""

    def __init__(self, chunks: list[dict] | None = None, skipped: list[str] | None = None):
        self.chunks = chunks or []
        self.skipped = skipped or []
        self.root: Path | None = None
        self.bm25 = BM25([c["file"] + "\n" + c["text"] for c in self.chunks])
        # superseded / withdrawn documents are down-weighted, never the preferred source
        self.mult = [0.4 if c["status"] == "withdrawn" else 1.0 for c in self.chunks]
        self.status = {c["file"]: c["status"] for c in self.chunks}

    def search(self, terms: dict[str, float], k: int = 10):
        return self.bm25.search(terms, k, self.mult)

    def save(self, directory: Path) -> None:
        """save writes three things into /app/index:
            chunks.jsonl: one JSON chunk per line. It is written to chunks.jsonl.tmp first, then os.replace swaps it in. That rename is atomic, so a crash can't leave a half-written file.
            skipped.json: the list of skipped files, for debugging.
            root.txt: the corpus path.
        """
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "chunks.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        os.replace(tmp, directory / "chunks.jsonl")
        (directory / "skipped.json").write_text(json.dumps(self.skipped, indent=1))
        (directory / "root.txt").write_text(str(self.root or ""))

    @classmethod
    def load(cls, directory: Path) -> "Index":
        """
        No file means an empty Index, so the server starts fine before any indexing.
        Otherwise it reads the chunk file and calls cls(chunks), which rebuilds the BM25 index in memory.

        Only the chunks are saved, not the BM25 tables. Rebuilding takes seconds at this size, and it keeps the saved format simple and hard to corrupt.
        The point of saving at all is that a server restart (or a crash) keeps the index, so you don't have to re-transcribe the images."""
        p = directory / "chunks.jsonl"
        if not p.exists():
            return cls()  # we use class method cls to return empty index
        chunks = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        idx = cls(chunks) #  if index exist then return an index with chunks, bm25 
        r = directory / "root.txt"
        if r.exists() and r.read_text().strip():
            idx.root = Path(r.read_text().strip())
        return idx
