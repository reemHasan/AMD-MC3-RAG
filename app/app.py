#!/usr/bin/env python3
"""MC-3 RAG submission: thin client for the resident server (see server.py).

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

The model and the index live in a long-running process (server.py, started by the container CMD,
or by this script if it is missing), so a per-question invocation costs a socket call,
not a model load.

Output: /app/output/<query-id>_output.json  {"answer", "citations", "confidence"}.
A valid file is ALWAYS written, even if everything else fails (as a refusal).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3_rag.sock")
SERVER_LOG = os.environ.get("MC3_SERVER_LOG", "/tmp/mc3_server.log")

QUERY_HARD_LIMIT_S = 28.0   # the harness allows 30s from exec to file


def _connect(wait: float):
    """Connect to the server socket, retrying for `wait` seconds. None if it never appears."""
    """What it does: it tries to dial the server, and keeps retrying for up to wait seconds. It returns the connected socket, or None if the server never appeared.
        Why retries are needed: when the container starts, the server may still be importing torch and may not have created the socket file yet. The two errors mean:
        FileNotFoundError: the socket file doesn't exist yet.
        ConnectionRefusedError: the file exists but nobody is listening.
        Each failed attempt closes its socket and tries again after 0.25s, so no sockets leak.
    """
    end = time.time() + wait
    while True:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(SOCK)
            return s
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            s.close()
            if time.time() >= end:
                return None
            time.sleep(0.25)


def _spawn_server() -> None:
    """
    What it does: it starts a new server process, as a fallback if the container's CMD didn't start one.
    The arguments matter:
    sys.executable is the same Python that is running now.
    stdout=log, stderr=log sends the server's output to a log file. Without this, the server would hold the caller's terminal or pipe open, which can make the harness wait for it forever.
    stdin=DEVNULL gives it no input.
    start_new_session=True detaches it from the client's process group. This lets the server survive after the client exits, which is the whole point, since the client is short-lived.
    close_fds=True stops the server inheriting the client's open file handles.

    Popen returns immediately without waiting for the server. It's "start and walk away".
    """
    with open(SERVER_LOG, "ab") as log:
        subprocess.Popen([sys.executable, str(HERE / "server.py")],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         start_new_session=True, close_fds=True)


def rpc(payload: dict, timeout: float, connect_wait: float, spawn: bool = True) -> dict | None:
    """`timeout` is the TOTAL budget in seconds, connect/spawn waiting included."""
    """
    RPC means "remote procedure call". Here it means: send one request, get one reply.
    Step by step:
    Fix the deadline with t_end. timeout is the total budget, including waiting to connect, so the query can't exceed the 30s limit.
    Connect. If that fails, start a server and try once more.
    settimeout makes recv raise an error instead of waiting forever. It is set to the time left.
    Send the request as JSON text plus \n, encoded to bytes. Sockets carry bytes, not Python objects.
    Receive in a loop until the reply ends with \n.
    Parse the JSON back into a dictionary.
    finally: s.close() always hangs up, even after an error.
    Any failure (OSError for network or timeout problems, ValueError for bad JSON) returns None. main() treats None as a refusal and still writes a valid file.
    """

    t_end = time.time() + timeout
    s = _connect(min(connect_wait, timeout))
    if s is None and spawn:
        _spawn_server()
        s = _connect(max(0.0, min(connect_wait, t_end - time.time())))
    if s is None:
        return None
    try:
        s.settimeout(max(0.5, t_end - time.time()))
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8")) if buf else None
    except (OSError, ValueError):
        return None
    finally:
        s.close()


def index(corpus: Path) -> None:
    """Build and persist the index (in the resident server). Charged to the startup budget."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # the CMD-started server may still be importing torch; give it time before starting our own
    rep = rpc({"op": "index", "corpus": str(corpus)}, timeout=590, connect_wait=20)
    if rep is None or "error" in rep:
        print(f"index failed: {rep}", file=sys.stderr)
        # exit 0 on purpose: a queries-time lazy index is still a chance; a crash is not


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    rep = rpc({"op": "query", "corpus": str(corpus), "query": query, "budget": 23},
              timeout=QUERY_HARD_LIMIT_S, connect_wait=4)
    if not rep or "answer" not in rep:
        return "", [], 0.0
    cites = [str(c).replace("\\", "/") for c in rep.get("citations", [])]
    return str(rep["answer"]), cites, float(rep.get("confidence", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    try:
        text, citations, confidence = answer(args.corpus, args.query)
    except Exception as e:  # noqa: BLE001  -- never leave the harness without a file
        print(f"answer failed: {e}", file=sys.stderr)
        text, citations, confidence = "", [], 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")
    tmp = out.with_suffix(".json.tmp")
    # written whole, then renamed: the harness never sees a half-written file
    tmp.write_text(
        json.dumps({"answer": text, "citations": list(citations), "confidence": confidence},
                   ensure_ascii=False),
        encoding="utf-8")
    os.replace(tmp, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
