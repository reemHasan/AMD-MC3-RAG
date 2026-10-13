#!/usr/bin/env python3
"""Ask every question in sample-questions.json to the RUNNING server and grade the answers.

It calls `app.py --query` once per question, as a separate process, exactly like the
harness does, then reads /<output-dir>/<query-id>_output.json and grades it:
  * answer: normalised (uppercase, no whitespace or - . . _) and compared with the
            expected answer or one of its aliases;
  * citations: compared as an EXACT SET;
  * a question scores only if both are right (20 points each).

Usage (server already started and --index already run):
    python tests/ask_sample.py --corpus ./mc3-corpus
    python tests/ask_sample.py --corpus ./mc3-corpus --only 7 8 9
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app" / "app.py"
POINTS = 20


def norm(s: str) -> str:
    return re.sub(r"[\s\-\.·_]", "", (s or "").upper())


def ping(sock_path: str, timeout: float = 5.0) -> dict | None:
    """Ask the server for its status. None if it is not reachable."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall(b'{"op": "ping"}\n')
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8"))
    except (OSError, ValueError):
        return None
    finally:
        s.close()


def main() -> int:
    default_q = HERE / "sample-questions.json"
    if not default_q.exists():
        default_q = Path("sample-questions.json")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="./mc3-corpus", help="corpus root (default ./mc3-corpus)")
    ap.add_argument("--questions", default=str(default_q), help="path to sample-questions.json")
    ap.add_argument("--only", type=int, nargs="*", help="run only these question numbers, e.g. --only 7 9")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-question limit in seconds (default 30)")
    ap.add_argument("--output-dir", default=os.environ.get("MC2_OUTPUT_DIR", "work/output"),
                    help="where answer files go (default $MC2_OUTPUT_DIR or work/output)")
    args = ap.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        print(f"cannot find {qpath}; pass --questions", file=sys.stderr)
        return 2
    queries = json.loads(qpath.read_text(encoding="utf-8"))["queries"]
    if args.only:
        queries = [q for q in queries if q["n"] in set(args.only)]

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MC2_OUTPUT_DIR=str(out_dir))

    # the server must already be up: otherwise app.py would start one and load the model,
    # which takes far longer than the per-question limit
    sock = env.get("MC3_SOCK", "/tmp/mc3_rag.sock")
    status = ping(sock)
    if status is None:
        print(f"No server answering on {sock}.\n"
              f"Start it first:  nohup python app/server.py > work/server.log 2>&1 &\n"
              f"then run:        python app/app.py --index {args.corpus}", file=sys.stderr)
        return 2
    print(f"server up on {sock}: {status.get('chunks', '?')} chunks indexed")
    if not status.get("chunks"):
        print("WARNING: the index is empty. Run  python app/app.py --index <corpus>  first.")
    print(f"output dir: {out_dir}\n")

    results, t_all = [], time.time()
    for q in queries:
        qid = f"query_{q['n']:02d}"
        out_file = out_dir / f"{qid}_output.json"
        out_file.unlink(missing_ok=True)  # never grade a stale file from an earlier run

        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(APP), "--corpus", args.corpus, "--query-id", qid, "--query", q["query"]],
                env=env, capture_output=True, text=True, timeout=args.timeout + 5)
            err = (proc.stderr or "").strip()[-300:] if proc.returncode else ""
        except subprocess.TimeoutExpired:
            err = f"app.py did not finish within {args.timeout + 5:.0f}s"
        dt = time.time() - t0

        answer, cites, problem = "", [], err
        if out_file.exists():
            try:
                data = json.loads(out_file.read_text(encoding="utf-8"))
                if not isinstance(data.get("answer"), str) or not isinstance(data.get("citations"), list):
                    problem = problem or "malformed output (answer/citations missing or wrong type)"
                else:
                    answer = data["answer"]
                    cites = [str(c).replace("\\", "/") for c in data["citations"]]
            except ValueError:
                problem = problem or "output file is not valid JSON"
        else:
            problem = problem or "no output file was written"

        accepted = {norm(x) for x in [q["expected_answer"], *q.get("answer_aliases", [])]}
        ok_answer = not problem and norm(answer) in accepted
        ok_cites = not problem and set(cites) == set(q["expected_citations"])
        ok_time = dt <= args.timeout
        passed = ok_answer and ok_cites and ok_time
        results.append({"n": q["n"], "query": q["query"], "passed": passed, "seconds": round(dt, 1),
                        "answer": answer, "citations": cites, "expected_answer": q["expected_answer"],
                        "expected_citations": q["expected_citations"], "problem": problem})

        flag = "PASS" if passed else "FAIL"
        print(f"Q{q['n']:>2} {flag}  {dt:5.1f}s  answer={answer!r}  cites={cites}")
        if not passed:
            if problem:
                print(f"      problem: {problem}")
            if not ok_answer and not problem:
                print(f"      expected answer   : {q['expected_answer']!r}"
                      + (f"  (aliases: {q['answer_aliases']})" if q.get("answer_aliases") else ""))
            if not ok_cites and not problem:
                missing = sorted(set(q["expected_citations"]) - set(cites))
                extra = sorted(set(cites) - set(q["expected_citations"]))
                print(f"      citations         : missing {missing or 'none'}, extra {extra or 'none'}")
            if not ok_time:
                print(f"      too slow          : {dt:.1f}s > {args.timeout:.0f}s limit")
            if q.get("category"):
                print(f"      category          : {q['category']}")

    total = time.time() - t_all
    n_pass = sum(r["passed"] for r in results)
    slowest = max((r["seconds"] for r in results), default=0)
    print(f"\nSCORE  {n_pass}/{len(results)} questions  =  {n_pass * POINTS}/{len(results) * POINTS} points")
    print(f"TIME   total {total:.0f}s (whole-run limit 600s), slowest question {slowest:.1f}s (limit {args.timeout:.0f}s)")

    report = out_dir / "sample_report.json"
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report written to {report}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
