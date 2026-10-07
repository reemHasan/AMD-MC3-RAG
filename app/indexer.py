"""Walk the corpus defensively and build the Index.

Survives: empty dirs, unknown types, unreadable files, encrypted files, corrupt
files, and files that make a parser hang. Nothing here may raise out of the walk.
"""
from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

from parsers import IMAGE_PROMPT, kind_of, parse_file, file_status
from retrieval import Index

log = logging.getLogger("mc3.index")
PER_FILE_SECONDS = 90


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout("parser timed out")


def build_index(corpus: Path, llm=None, image_deadline: float | None = None) -> Index:
    """image_deadline: wall-clock time after which images are no longer transcribed at
    index time (they still get a filename chunk and are read directly at query time)."""
    corpus = Path(corpus)
    chunks: list[dict] = []
    skipped: list[str] = []

    def describe(img, tag=""):
        if llm is None:
            return ""
        if image_deadline is not None and time.time() > image_deadline:
            return ""
        try:
            return llm.generate(IMAGE_PROMPT, image=img, max_new_tokens=450, tag=tag)
        except Exception as e:  # noqa: BLE001  (e.g. OOM): keep the image as a filename-only chunk
            log.warning("transcription failed for %s: %s", tag, e)
            return ""

    files: list[tuple[str, Path, str]] = []

    def on_walk_error(err):
        skipped.append(f"{getattr(err, 'filename', '?')}: {err}")

    try:
        for dirpath, dirnames, filenames in os.walk(corpus, onerror=on_walk_error):
            dirnames.sort()
            # prune directories with no read/search bits (root would otherwise bypass them locally)
            keep = []
            for dn in dirnames:
                try:
                    m = (Path(dirpath) / dn).stat().st_mode
                    if m & 0o444 and m & 0o111:
                        keep.append(dn)
                    else:
                        skipped.append(f"{dn}/: no read permission")
                except Exception as e:  # noqa: BLE001
                    skipped.append(f"{dn}/: {e}")
            dirnames[:] = keep
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                try:
                    rel = p.relative_to(corpus).as_posix()
                    kind = kind_of(p)
                    if kind is None:
                        skipped.append(f"{rel}: unknown type")
                        continue
                    st = p.stat()
                    # Treat "no read bits" as unreadable even when running as root, so local
                    # tests behave like the evaluation (which drops DAC_OVERRIDE).
                    if not (st.st_mode & 0o444):
                        skipped.append(f"{rel}: no read permission")
                        continue
                    files.append((rel, p, kind))
                except Exception as e:  # noqa: BLE001
                    skipped.append(f"{p}: {e}")
    except Exception as e:  # noqa: BLE001
        skipped.append(f"walk: {e}")

    # images last, so a deadline only ever costs image transcriptions
    files.sort(key=lambda t: (t[2] == "image", t[0]))

    use_alarm = hasattr(signal, "SIGALRM")
    """rel, the path relative to the corpus root, with forward slashes. This is the exact string used later as the citation.
    kind, from kind_of(p). If it's None (unknown extension such as .dat), the file is skipped.
    A permission check: st_mode & 0o444 tests the read bits. If none are set, the file is skipped.
    """
    for rel, p, kind in files:
        try:
            if use_alarm:
                try:
                    signal.signal(signal.SIGALRM, _alarm)
                    signal.alarm(PER_FILE_SECONDS if kind != "image" else 240)
                except ValueError:  # not main thread
                    use_alarm = False
            try:
                parts = parse_file(p, kind, describe)
            finally:
                if use_alarm:
                    signal.alarm(0)
            if not parts:
                skipped.append(f"{rel}: no extractable content")
                continue
            status = file_status(rel, parts[0]["text"])
            for c in parts:
                chunks.append({"file": rel, "kind": kind, "status": status,
                               "loc": c["loc"], "text": c["text"]})
        except BaseException as e:  # noqa: BLE001  (incl. timeouts, RecursionError)
            if isinstance(e, KeyboardInterrupt):
                raise
            skipped.append(f"{rel}: {type(e).__name__}: {e}")
            log.warning("skipped %s: %s", rel, e)
    log.info("indexed %d chunks from %d files; skipped %d", len(chunks), len(files), len(skipped))
    idx = Index(chunks, skipped)
    idx.root = corpus
    return idx
