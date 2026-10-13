"""Answer one question from the index.

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
import os
import re
import time

from retrieval import Index, query_terms

log = logging.getLogger("mc3.agent")

REFUSE = ("", [], 0.0)
MAX_ROUNDS = 4
MAX_VISION_CALLS = 3
CONTEXT_CHARS = int(os.environ.get("MC3_CONTEXT_CHARS", "30000"))   # ~7.5k tokens of excerpts
DEBUG = bool(os.environ.get("MC3_DEBUG"))  # also log the full prompts
BRIDGE = os.environ.get("MC3_BRIDGE", "1") != "0"
VISION_GATE = os.environ.get("MC3_VISION_GATE", "1") != "0"

ANSWER_PROMPT = """You answer questions using ONLY the numbered document excerpts below. The documents describe fictional products: never use anything you remember, only what the excerpts say.

Rules:
1. "answer" is the bare VALUE only (a number, part number, version, quarter, code, pin). No sentence, no explanation. Keep the complete identifying qualifier: "Q2 FY31" not "Q2", "REV-D7" not "D7", "XYZ-PRT-0042-K" in full. Do not add units.
2. Excerpts marked WITHDRAWN/SUPERSEDED are out of date. Never answer from them when a current document exists.
3. Watch for near-misses: a similar product, an old value mentioned in a comment, a different revision. Use the row/line that matches every detail of the question. The value must belong to exactly the product/part, the attribute and the conditions the question names (volume, revision, date, load...). The kind of value must match the kind of thing asked: an "error code" comes from a log or a document that logs errors, not from a pin or signal name; a "part number" is not a ticket number. A number for a different part, a different attribute, or a condition that is not stated in the excerpts is NOT an answer: use status "none". Example: if the question asks for a price at a given volume and the excerpts only give a cost for some other item, or no price at that volume, the status is "none".
4. If the question needs a value for an identifier (ticket, error code, part number) that an excerpt mentions, but the value itself is not in the excerpts yet, ask for a search for that identifier.
5. If the excerpts do not contain the answer, do not guess.

Reply with ONE JSON object and nothing else:
{{"status": "answer" | "search" | "none",
 "answer": "<value, only when status is answer>",
 "search": "<identifier(s) to look up, only when status is search>",
 "sources": [<excerpt numbers>]}}
"sources": for "answer", list ONLY the excerpts that are necessary: the one holding the value, PLUS every excerpt you needed to find out which item the question is about. Example: if the question says "the incident in the log" and one excerpt (the log) names the ticket or code while another excerpt holds the value for that ticket, list BOTH, even if all excerpts were already in front of you and no search was needed. Never list an excerpt that merely discusses the same topic. For "search", list the excerpt(s) that contained the identifier.

QUESTION: {question}

EXCERPTS:
{excerpts}
"""

EXPAND_PROMPT = """Question: {question}

Write up to 6 short search phrases likely to appear verbatim in the documents that answer this question: product names, identifiers, technical terms, synonyms and likely field/column names. One per line, no numbering, no commentary."""

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


def _words(text: str) -> list[str]:
    """Lower-case alphanumeric words of 3+ characters, minus stop words."""
    from retrieval import STOP
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 3 and w not in STOP]


def parse_json(text: str) -> dict | None:
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

    def __init__(self, index: Index, max_chars: int = CONTEXT_CHARS):
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
            if VISION_GATE:
                # an image the question is not about must not get to "answer" it: a VLM will happily
                # name a signal when asked for an error code. Need >= 2 shared words with the question.
                q_words = set(_words(question))
                shared = q_words & set(_words(c["file"] + " " + c["text"]))
                if len(shared) < 2:
                    log.info("vision skipped for %s (shared words with question: %s)", c["file"], sorted(shared))
                    ctx.vision[c["file"]] = ""
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
        if DEBUG:
            log.info("PROMPT >>>\n%s\n<<<", prompt)
        raw = self.llm.generate(prompt, max_new_tokens=140)
        log.info("model reply: %s", raw.replace("\n", " ")[:500])
        return parse_json(raw)

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
        hits = idx.search(terms, k=40)

        ctx = Context(idx)
        total = sum(len(c["text"]) for c in idx.chunks)
        if total <= ctx.max_chars:
            # the whole corpus fits: read it all, most relevant first, instead of risking a retrieval miss
            ranked = [ci for ci, _ in idx.search(terms, k=len(idx.chunks))]
            seen = set(ranked)
            order = ranked + [i for i in range(len(idx.chunks)) if i not in seen]
            ctx.add_hits([(i, 1.0) for i in order], limit=len(order), per_file=10 ** 9)
            log.info("small corpus (%d chars): all %d chunks in context", total, len(order))
        else:
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
                return self._finalize(question, resp, ctx, hops, hits)
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

    def _finalize(self, question: str, resp: dict, ctx: Context, hops, hits):
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
            log.info("REFUSE: answer %r is not in the evidence shown", ans)
            return REFUSE

        cited = ctx.files_for(resp.get("sources"))
        chain: list[str] = []
        for s, declared in hops:
            keys = [norm(w) for w in s.split() if len(norm(w)) >= 3] + [norm(s)]
            for f in declared:
                if f in ev and any(k and k in norm(ev[f]) for k in keys):
                    chain.append(f)

        # the row the model relied on = cited files that state the answer (not every file that happens to)
        core = [f for f in cited if f in ans_files] or ans_files
        needed = set(ans_files) | set(chain)
        ids, _ = self._link_ids(question, ans, core, ev)
        for f in cited:  # a cited file holding the identifier that leads to the answer row is needed too
            if f in ev and any(i.lower() in ev[f].lower() for i in ids):
                needed.add(f)
        final = [f for f in _ordered_unique(cited + chain) if f in needed]
        if not final:
            rank = {f: r for r, f in enumerate(ctx.files())}
            final = [min(ans_files, key=lambda f: rank.get(f, 1e9))]
        if BRIDGE:
            for f in self._bridge_files(question, ans, core, ans_files, final, ev):
                log.info("bridge file added: %s", f)
                final.append(f)
        final = [f for f in final if self.index.status.get(f) != "withdrawn"]
        if not final:  # the only support is a withdrawn document
            log.info("REFUSE: only withdrawn documents support %r", ans)
            return REFUSE
        # every number the question imposes as a condition (e.g. "10,000 unit volume") must be
        # visible in the files we cite; otherwise the value is for something else
        blob = norm(" ".join(ev.get(f, "") for f in final))
        for cond in _condition_numbers(question):
            if cond not in blob:
                log.info("REFUSE: condition %r from the question is not in %s", cond, final)
                return REFUSE
        log.info("ANSWER %r  cited=%s  chain=%s", ans, final, chain)
        return ans, final, 0.9

    @staticmethod
    def _answer_lines(ans, ans_files, ev) -> list[str]:
        """Lines that state the answer. Whole-token match first ('4.3.2' must not match 'ORR-1432')."""
        pat = re.compile(r"(?<![\w.])" + re.escape(ans) + r"(?![\w])", re.I)
        lines = [ln for f in ans_files for ln in ev[f].splitlines() if pat.search(ln)]
        if lines:
            return lines
        na = norm(ans)
        return [ln for f in ans_files for ln in ev[f].splitlines() if na in norm(ln)]

    def _link_ids(self, question, ans, ans_files, ev) -> tuple[set, list]:
        """(identifiers on the answer row that the question does not mention, the answer lines)."""
        na = norm(ans)
        ans_lines = self._answer_lines(ans, ans_files, ev)
        q_norm = norm(question)
        ids = set()
        for ln in ans_lines:
            for m in _ID_RE.findall(ln):
                if norm(m) != na and norm(m) not in q_norm and norm(m) not in na:
                    ids.add(m)
        return ids, ans_lines

    def _bridge_files(self, question, ans, core, ans_files, final, ev) -> list[str]:
        """Files that explain WHICH item the question means, when the answer row never says so.

        The answer row holds an identifier (a ticket, a code) that the question does not mention.
        Another file contains that identifier next to wording from the question that the answer
        row lacks (the log line about the incident). That file is needed: without it you could not
        get from the question to the row. Per identifier, only the best-matching file is returned."""
        ids, ans_lines = self._link_ids(question, ans, core, ev)
        if not ans_lines or not ids:
            return []
        row_text = " ".join(ans_lines).lower()
        qwords = {w for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in row_text and w not in _Q_NOISE}
        if not qwords:
            return []
        out = []
        for ident in sorted(ids):
            key = norm(ident)
            holders = [f for f, t in ev.items() if key in norm(t)]
            if len(holders) > 4:
                continue  # a shared product/family name, not a specific link
            scores = {}
            for f in holders:
                if f in final or f in ans_files or self.index.status.get(f) == "withdrawn":
                    continue
                lines = ev[f].splitlines()
                for j, ln in enumerate(lines):
                    if key in norm(ln):
                        window = " ".join(lines[max(0, j - 2): j + 3]).lower()
                        scores[f] = max(scores.get(f, 0), sum(1 for w in qwords if w in window))
            log.info("bridge candidates for %s: %s (question words not in the answer row: %s)",
                     ident, scores, sorted(qwords))
            if not scores:
                continue
            best = max(scores.values())
            top = [f for f, v in scores.items() if v == best]
            if best >= 2 or (best == 1 and sum(1 for v in scores.values() if v >= 1) == 1):
                out.append(top[0])
        return _ordered_unique(out)


# identifiers such as ORR-1847, E7731, REV-C2, ORR-FAN-2214-B
_ID_RE = re.compile(r"\b[A-Za-z]{2,}(?:[-_][A-Za-z0-9]+)*[-_]\d[A-Za-z0-9-]*\b|\b[A-Z]\d{3,}\b")
_Q_NOISE = {"which", "what", "where", "shows", "whose", "there", "their", "these", "those", "about",
            "release", "version", "number", "value", "fixed", "underlying", "defect"}


def _condition_numbers(question: str) -> list[str]:
    """Stand-alone numbers in the question, such as '10,000' (not dates, not parts of ids)."""
    out = []
    for m in re.finditer(r"(?<![\w.\-])(\d{1,3}(?:,\d{3})+|\d{3,})(?![\w\-])", question):
        out.append(norm(m.group(1).replace(",", "")))
    return out
