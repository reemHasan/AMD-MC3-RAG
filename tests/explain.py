#!/usr/bin/env python3
"""Explain, OFFLINE (no model, no server), why the guards would keep or drop files.

Reads the index saved by --index and shows, for a question and the answer the model gave:
  * which files contain the answer, and the exact answer lines;
  * the identifiers on those lines that lead to other files (e.g. a ticket id);
  * how every candidate "bridge" file scores, and which one would be cited;
  * the numbers the question imposes as conditions, and where they appear;
  * for each image, how many words it shares with the question (the vision gate).

    python tests/explain.py --index-dir work/index \\
        --question "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?" \\
        --answer 4.3.2 --cited support/bug_database.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from agent import Agent, _condition_numbers, _words, norm  # noqa: E402
from retrieval import Index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", default="work/index")
    ap.add_argument("--question", required=True)
    ap.add_argument("--answer", required=True, help="the answer the model gave (or the expected one)")
    ap.add_argument("--cited", nargs="*", default=[], help="files the model cited, if you know them")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="   %(message)s")

    idx = Index.load(Path(a.index_dir))
    if not idx.chunks:
        print(f"no index in {a.index_dir}; run --index first")
        return 2
    ev: dict[str, str] = {}
    for c in idx.chunks:
        ev[c["file"]] = ev.get(c["file"], "") + "\n" + c["text"]
    ag = Agent(idx, None)

    na = norm(a.answer)
    ans_files = [f for f, t in ev.items() if na in norm(t)]
    core = [f for f in a.cited if f in ans_files] or ans_files
    print(f"files that contain the answer {a.answer!r}: {ans_files or 'NONE'}")
    for ln in ag._answer_lines(a.answer, core, ev)[:6]:
        print(f"   answer line: {ln[:200]}")

    ids, _ = ag._link_ids(a.question, a.answer, core, ev)
    print(f"\nidentifiers on the answer line(s) that the question does not mention: {sorted(ids) or 'none'}")
    final = list(dict.fromkeys(a.cited or core))
    print("bridge analysis:")
    out = ag._bridge_files(a.question, a.answer, core, ans_files, final, ev)
    print(f"=> bridge files that would be added: {out or 'none'}")

    print("\ncondition numbers in the question (must appear in the cited files):")
    blob = norm(" ".join(ev.get(f, "") for f in set(final) | set(out)))
    for cond in _condition_numbers(a.question) or []:
        print(f"   {cond}: {'found' if cond in blob else 'NOT FOUND -> the answer would be refused'}")
    if not _condition_numbers(a.question):
        print("   none")

    print("\nvision gate (an image is read against the question only if it shares >= 2 words):")
    qw = set(_words(a.question))
    for c in idx.chunks:
        if c["kind"] == "image":
            shared = sorted(qw & set(_words(c["file"] + " " + c["text"])))
            print(f"   {c['file']}: shares {shared} -> {'READ' if len(shared) >= 2 else 'skipped'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
