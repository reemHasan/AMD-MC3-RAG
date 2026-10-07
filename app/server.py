"""Long-running process: loads the model once, holds the index, answers over a unix socket.

Started by the container CMD (`app.py --serve`). `app.py --index` / `--query` are thin
clients of this process, and will start it themselves if it is not running.

Protocol: one JSON object per connection, newline-terminated, one JSON reply.
    {"op": "ping"}
    {"op": "index", "corpus": "/app/corpus"}
    {"op": "query", "corpus": "...", "query": "...", "budget": 24}
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

#LOCK becomes /tmp/mc3_rag.sock.lock, a small file used to make sure only one server runs at a time.
# flock asks the operating system for an exclusive lock on that file.
# LOCK_EX means exclusive, so only one process can hold it.
# LOCK_NB means non-blocking. If the lock is taken, it fails immediately with OSError instead of waiting.

SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3_rag.sock")
LOCK = SOCK + ".lock"
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
T0 = time.time()
# The startup budget (10 min) covers model load AND indexing. Stop transcribing images
# at index time after this many seconds since process start; they are still read directly
# at query time.
IMAGE_CUTOFF_S = float(os.environ.get("MC3_IMAGE_CUTOFF_S", "420"))

log = logging.getLogger("mc3.server")


def _recv_line(conn: socket.socket) -> bytes:
    """What it does: it reads from a connection until it has received one full line, meaning everything up to a newline.

    Why it's needed: a stream socket has no message boundaries. If a client sends one 100 KB message, recv may return it in several pieces, say 64 KB and then 36 KB. TCP-style streams only guarantee the bytes arrive in order, not in the same chunks they were sent. So we agree on a rule: one message = one line ending in \n. This is called "framing". The reader keeps appending chunks until it sees the newline.

    The if not chunk: break line handles a closed connection. When the other side hangs up, recv returns empty bytes, and without this the loop would spin forever.
    """
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf


def serve() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.info("another server already holds the lock; exiting")
        return 0

    # bind FIRST, so clients queue on the socket while the (slow) model loads
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    # Create a phone. AF_UNIX means a file-path address on this machine. SOCK_STREAM means a continuous flow of bytes.
    """socket(...) → create a Unix socket.
       bind(SOCK) → attach it to a socket path, e.g. /tmp/app.sock.
      listen(16) → make it a server socket and allow up to 16 pending connection requests.
    """
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(16)

    from agent import Agent
    from indexer import build_index
    from llm import make_llm
    from retrieval import Index
# 1. Load model into VRAM
    llm = None
    try:
        llm = make_llm()
    except Exception:  # noqa: BLE001
        log.exception("model failed to load; every question will be refused")

    index = Index.load(INDEX_DIR)
    if index.root is None:
        index.root = Path(os.environ.get("MC3_CORPUS", "/app/corpus"))
    log.info("ready: %d chunks (model load took %.0fs)", len(index.chunks), time.time() - T0)
    tried_index = bool(index.chunks)

    while True:
        conn, _ = srv.accept()
        try:
            conn.settimeout(700)
            req = json.loads(_recv_line(conn).decode("utf-8") or "{}")
            op = req.get("op")
            if op == "ping":
                rep = {"ok": True, "chunks": len(index.chunks)}
            elif op == "shutdown" and os.environ.get("MC3_ALLOW_SHUTDOWN"):
                conn.sendall(b'{"ok": true}\n'); conn.close(); os._exit(0)
            elif op == "index":
                t = time.time()
                index = build_index(Path(req["corpus"]), llm, image_deadline=T0 + IMAGE_CUTOFF_S)
                index.save(INDEX_DIR)
                tried_index = True
                files = len({c["file"] for c in index.chunks})
                log.info("index built in %.1fs: %d chunks, %d files, skipped %s",
                         time.time() - t, len(index.chunks), files, index.skipped)
                rep = {"ok": True, "chunks": len(index.chunks), "files": files, "skipped": len(index.skipped)}
                # rep never returned
            elif op == "query":
                if not tried_index and Path(req.get("corpus", "")).is_dir():
                    log.warning("query before --index: indexing lazily")
                    index = build_index(Path(req["corpus"]), llm)
                    index.save(INDEX_DIR)
                    tried_index = True
                if index.root is None or str(index.root) != req.get("corpus", str(index.root)):
                    if req.get("corpus"):
                        index.root = Path(req["corpus"])
                if llm is None:
                    ans, cites, conf = "", [], 0.0
                else:
                    ans, cites, conf = Agent(index, llm).answer(req["query"], float(req.get("budget", 24)))
                rep = {"answer": ans, "citations": cites, "confidence": conf}
            else:
                rep = {"error": f"unknown op {op!r}"}
        except Exception as e:  # noqa: BLE001
            log.exception("request failed")
            rep = {"error": f"{type(e).__name__}: {e}"}
        try:
            conn.sendall((json.dumps(rep, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError:
            pass
        finally:
            conn.close()


if __name__ == "__main__":
    sys.exit(serve())
