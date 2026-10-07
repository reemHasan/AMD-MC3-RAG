"""Answer one question from the index.
Purpose: answer one question from the index, then check the answer with two deterministic guards (grounding and necessity).
It returns (answer, citations, confidence), or a refusal ("", [], 0.0).

Flow: expand keywords -> BM25 -> excerpts -> LLM says answer | search | none.
'search' is the multi-hop step (an identifier found in one file is looked up in
another). Image files in the context are read directly by the vision model.
Afterwards two deterministic guards run:

  * grounding  - the answer must literally occur in the evidence shown to the model,
                 otherwise it is a guess or a recollection and we refuse;
  * necessity  - a file stays in the citations only if it contains the answer or
                 supplied the identifier that led to it.
"""
from __future__ import annotations

import json
import logging
import re
import time

from retrieval import Index, query_terms

log = logging.getLogger("mc3.agent")
# 	The refusal value: empty answer, no citations, confidence 0.
REFUSE = ("", [], 0.0)
# MAX_ROUNDS = 4	The maximum number of ask/search loops per question.
MAX_ROUNDS = 4
#MAX_VISION_CALLS = 3	The maximum number of images read per question, to protect the time limit.
MAX_VISION_CALLS = 3
#ANSWER_PROMPT	Tells the model to use only the excerpts, give a bare value, ignore withdrawn files, and reply with JSON: status (answer, search or none), answer, search and sources.
ANSWER_PROMPT = """You answer questions using ONLY the numbered document excerpts below. The documents describe fictional products: never use anything you remember, only what the excerpts say.

Rules:
1. "answer" is the bare VALUE only (a number, part number, version, quarter, code, pin). No sentence, no explanation. Keep the complete identifying qualifier: "Q2 FY31" not "Q2", "REV-D7" not "D7", "XYZ-PRT-0042-K" in full. Do not add units.
2. Excerpts marked WITHDRAWN/SUPERSEDED are out of date. Never answer from them when a current document exists.
3. Watch for near-misses: a similar product, an old value mentioned in a comment, a different revision. Use the row/line that matches every detail of the question.
4. If the question needs a value for an identifier (ticket, error code, part number) that an excerpt mentions, but the value itself is not in the excerpts yet, ask for a search for that identifier.
5. If the excerpts do not contain the answer, do not guess.

Reply with ONE JSON object and nothing else:
{{"status": "answer" | "search" | "none",
 "answer": "<value, only when status is answer>",
 "search": "<identifier(s) to look up, only when status is search>",
 "sources": [<excerpt numbers>]}}
"sources": for "answer", list ONLY the excerpts that are necessary: the one holding the value, plus (if you used a search step) the one that gave you the identifier. Never list an excerpt that merely discusses the same topic. For "search", list the excerpt(s) that contained the identifier.

QUESTION: {question}

EXCERPTS:
{excerpts}
"""
#EXPAND_PROMPT	Asks for up to 6 extra search phrases.
EXPAND_PROMPT = """Question: {question}

Write up to 6 short search phrases likely to appear verbatim in the documents that answer this question: product names, identifiers, technical terms, synonyms and likely field/column names. One per line, no numbering, no commentary."""
#VISION_PROMPT	Asks the model to transcribe what is relevant in an image and end with ANSWER: <value> or ANSWER: NONE.
VISION_PROMPT = """This image is the file "{file}".
Question: {question}
First transcribe the text in the image that is relevant to the question (labels, pin numbers with their signal names, table cells, codes) exactly as printed. Then finish with a line "ANSWER: <value>" giving only the value, or "ANSWER: NONE" if the image does not contain the answer."""


# ------------------------------------------------------------------ small helpers

def norm(s: str) -> str:
    """The grader's normalisation: uppercase, drop whitespace and - . · _"""
    return re.sub(r"[\s\-\.·_]", "", (s or "").upper())


_UNIT = (r"(?:°\s*[CF]|deg(?:rees?)?(?:\s*[CF])?|celsius|seconds?|secs?|s|ms|milliseconds?|minutes?|"
         r"mins?|hours?|hrs?|days?|[kmg]?hz|[kmgt]i?b|watts?|w|volts?|v|amps?|a|c|f|%|percent)")


def clean_answer(a) -> str:
    """
    Cleans the model's answer. It keeps the first line, strips quotes,
    a trailing period and Answer: prefixes, and turns 94 °C into 94. NONE or N/A become empty.
    """
    a = str(a or "").strip()
    a = a.splitlines()[0].strip() if a else ""
    a = re.sub(r"(?i)^(final\s+)?(answer|value)\s*[:\-]\s*", "", a).strip()
    a = a.strip("`\"' ").rstrip(".").strip()
    m = re.fullmatch(rf"(-?\d[\d,]*\.?\d*)\s*{_UNIT}", a, re.I)
    if m:
        a = m.group(1)
    if a.upper() in {"NONE", "N/A", "NULL", "UNKNOWN", "NOT FOUND"}:
        return ""
    return a


def parse_json(text: str) -> dict | None:
    """
    Extracts the JSON object from the model's reply.
    If the JSON is broken, it falls back to regex for each field.
    """
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    # lenient fallback for slightly broken JSON
    out: dict = {}
    for key in ("status", "answer", "search"):
        mm = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if mm:
            out[key] = mm.group(1)
    ms = re.search(r'"sources"\s*:\s*\[([^\]]*)\]', text)
    if ms:
        out["sources"] = re.findall(r"\d+", ms.group(1))
    return out or None


def _ordered_unique(xs):
    """
    Removes duplicates and keeps the original order.
    """
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class Context:
    """The excerpts shown to the model, numbered E1.. in insertion order."""

    def __init__(self, index: Index, max_chars: int = 26000):
        """
        Stores the index, a character cap (26,000), the list of chunk ids,
        and the per-image vision readings."""
        self.index, self.max_chars = index, max_chars
        self.ids: list[int] = []
        self.chars = 0
        self.vision: dict[str, str] = {}

    def files(self) -> list[str]:
        """	The distinct files currently in the context, in order. """
        return _ordered_unique(self.index.chunks[i]["file"] for i in self.ids)

    def add_hits(self, hits, limit: int, per_file: int = 3, min_rel: float = 0.0, always: int = 0) -> int:
        """Add chunks best-first. The top `always` are taken regardless; after that a chunk
        must score at least min_rel * best, so tie-scored noise never crowds out a file."""
        added = 0
        best = hits[0][1] if hits else 0.0
        counts: dict[str, int] = {}
        for i in self.ids:
            f = self.index.chunks[i]["file"]
            counts[f] = counts.get(f, 0) + 1
        for ci, _score in hits:
            if ci in self.ids:
                continue
            if added >= always and _score < min_rel * best:
                break
            c = self.index.chunks[ci]
            if counts.get(c["file"], 0) >= per_file:
                continue
            if self.chars + len(c["text"]) > self.max_chars:
                break
            self.ids.append(ci)
            self.chars += len(c["text"])
            counts[c["file"]] = counts.get(c["file"], 0) + 1
            added += 1
            if added >= limit:
                break
        return added

    def render(self) -> str:
        """	Formats the excerpts as [E1] path | location,
          tagged WITHDRAWN where relevant. Images also show their vision reading."""
        blocks = []
        for n, ci in enumerate(self.ids, 1):
            c = self.index.chunks[ci]
            tag = "  [WITHDRAWN / SUPERSEDED - DO NOT USE]" if c["status"] == "withdrawn" else ""
            body = c["text"]
            v = self.vision.get(c["file"])
            if c["kind"] == "image" and v:
                body += "\nVision model reading of this image for the question:\n" + v
            blocks.append(f"[E{n}] {c['file']} | {c['loc']}{tag}\n{body}")
        return "\n\n".join(blocks)

    def files_for(self, ids) -> list[str]:
        """Converts the model's excerpt numbers (E3, 3) into file paths."""
        out = []
        for x in ids or []:
            m = re.search(r"\d+", str(x))
            if m and 1 <= int(m.group(0)) <= len(self.ids):
                out.append(self.index.chunks[self.ids[int(m.group(0)) - 1]]["file"])
        return _ordered_unique(out)

    def evidence(self) -> dict[str, str]:
        """Maps each file to all the text the model saw from it, including vision readings. The guards use this."""
        ev: dict[str, str] = {}
        for ci in self.ids:
            c = self.index.chunks[ci]
            ev[c["file"]] = ev.get(c["file"], "") + "\n" + c["text"]
        for f, v in self.vision.items():
            if v:
                ev[f] = ev.get(f, "") + "\n" + v
        return ev


class Agent:
    def __init__(self, index: Index, llm):
        self.index, self.llm = index, llm

    # ---------------------------------------------------------------- steps
    def _expand(self, question: str) -> list[str]:
        """Asks the model for extra keywords. Any failure returns []."""
        try:
            raw = self.llm.generate(EXPAND_PROMPT.format(question=question), max_new_tokens=70)
        except Exception as e:  # noqa: BLE001
            log.warning("expand failed: %s", e)
            return []
        out = []
        for ln in raw.splitlines():
            ln = re.sub(r"^[\s\-\*\d\.\)]+", "", ln).strip()
            if 2 <= len(ln) <= 80:
                out.append(ln)
        return out[:6]

    def _vision(self, ctx: Context, question: str, used: list[int]) -> None:
        from parsers import load_image  # local import keeps PIL out of module import time
        """For each image in the context, passes the real picture and the question to the model.
          It stores the reading, or "" if the answer is NONE.
        """
        for ci in list(ctx.ids):
            c = self.index.chunks[ci]
            if c["kind"] != "image" or c["file"] in ctx.vision or used[0] >= MAX_VISION_CALLS:
                continue
            used[0] += 1
            path = self.index.root / c["file"] if getattr(self.index, "root", None) else None
            try:
                raw = self.llm.generate(VISION_PROMPT.format(file=c["file"], question=question),
                                        image=load_image(path), max_new_tokens=220, tag=c["file"])
            except Exception as e:  # noqa: BLE001
                log.warning("vision failed for %s: %s", c["file"], e)
                ctx.vision[c["file"]] = ""
                continue
            m = re.search(r"(?im)^\s*ANSWER\s*:\s*(.*)$", raw)
            if m and m.group(1).strip().strip("`\"'. ").upper() in {"NONE", ""}:
                ctx.vision[c["file"]] = ""
            else:
                ctx.vision[c["file"]] = raw.strip()

    def _ask(self, question: str, ctx: Context) -> dict | None:
        """	Sends the prompt with the excerpts and returns the parsed JSON."""
        prompt = ANSWER_PROMPT.format(question=question, excerpts=ctx.render())
        return parse_json(self.llm.generate(prompt, max_new_tokens=140))

    # ---------------------------------------------------------------- main
    def answer(self, question: str, budget: float = 24.0):
        """ The main loop:
            1. Retrieve. Extra keywords from _expand are added to the query at half weight. BM25 returns hits. The top ones go into the context, and the best two images are always added.
            2.Loop (up to 4 rounds, stopping if less than 3 seconds remain):
                Read any new images.
                Ask the model, then branch on status:
                answer: go to _finalize.
                search: record the hop (search string plus the files that gave the identifier), run BM25 on the search string alone, and add results. If nothing new is found, stop.
                none: widen once by adding deeper hits. If that adds nothing, or it already widened, stop.
            3.If the loop ends without an answer, return REFUSE.
                """
        t_end = time.time() + budget
        idx = self.index
        if not idx.chunks:
            return REFUSE

        terms = query_terms(question, 1.0)
        for kw in self._expand(question):
            query_terms(kw, 0.5, into=terms)
        hits = idx.search(terms, k=40) # ask the index to do search by using bm25 logic
        # create a new Context object and pass the index to it, then the best found files to be used in answer
        ctx = Context(idx)
        ctx.add_hits(hits, limit=8, per_file=3, min_rel=0.25, always=3)
        # always let the best two image files in: their text may be unreadable to BM25
        imgs = [h for h in hits if idx.chunks[h[0]]["kind"] == "image"][:2]
        ctx.add_hits(imgs, limit=2, per_file=3)

        hops: list[tuple[str, list[str]]] = []
        tried = {question.lower()}
        widened = False
        vision_used = [0]
        for _round in range(MAX_ROUNDS):
            if time.time() > t_end - 3:
                log.warning("out of time budget")
                break
            self._vision(ctx, question, vision_used)
            resp = self._ask(question, ctx)
            if not resp:
                break
            status = str(resp.get("status", "")).lower()
            if status == "answer":
                return self._finalize(resp, ctx, hops, hits)
            if status == "search":
                s = str(resp.get("search", "")).strip()
                if not s or s.lower() in tried:
                    break
                tried.add(s.lower())
                hops.append((s, ctx.files_for(resp.get("sources"))))
                if not ctx.add_hits(idx.search(query_terms(s, 1.0), k=12), limit=5, per_file=8, always=5):
                    break
                continue
            # "none": once, look deeper into the original ranking before giving up
            if widened or not ctx.add_hits(hits, limit=8, per_file=4, min_rel=0.1, always=2):
                break
            widened = True
        return REFUSE

    def _finalize(self, resp: dict, ctx: Context, hops, hits):
        """Applies the guards and picks the citations.
           the two guards
            Grounding: the cleaned answer must appear in the evidence. If it doesn't, return REFUSE.
            Necessity: keep a file only if it contains the answer, or it was declared as the source of an identifier used in a search hop.
            Cleanup: drop withdrawn files. If nothing remains, return REFUSE. Otherwise return (answer, files, 0.9).
        """
        ans = clean_answer(resp.get("answer"))
        if not ans:
            return REFUSE
        ev = ctx.evidence()
        na = norm(ans)
        ans_files = [f for f, t in ev.items() if na and na in norm(t)]
        if not ans_files:
            log.info("refusing ungrounded answer %r", ans)
            return REFUSE

        cited = ctx.files_for(resp.get("sources"))
        chain: list[str] = []
        for s, declared in hops:
            keys = [norm(w) for w in s.split() if len(norm(w)) >= 3] + [norm(s)]
            for f in declared:
                if f in ev and any(k and k in norm(ev[f]) for k in keys):
                    chain.append(f)

        needed = set(ans_files) | set(chain)
        final = [f for f in _ordered_unique(cited + chain) if f in needed]
        if not final:
            rank = {f: r for r, f in enumerate(ctx.files())}
            final = [min(ans_files, key=lambda f: rank.get(f, 1e9))]
        final = [f for f in final if self.index.status.get(f) != "withdrawn"]
        if not final:  # the only support is a withdrawn document
            return REFUSE
        return ans, final, 0.9
