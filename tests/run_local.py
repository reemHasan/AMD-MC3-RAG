"""Runs the whole contract locally with the mock model: index, 10 questions, grade like the harness."""
import json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; APP = HERE.parent / "app"
Q = json.load(open(sys.argv[1] if len(sys.argv) > 1 else HERE / "sample-questions.json"))["queries"]
work = Path(tempfile.mkdtemp(prefix="mc3test_")); corpus = work / "corpus"
env = dict(os.environ, MC3_SOCK=str(work / "s.sock"), MC3_INDEX_DIR=str(work / "index"), MC2_OUTPUT_DIR=str(work / "out"),
           MC3_SERVER_LOG=str(work / "server.log"), MC3_LLM_FACTORY="mock_llm:MockLLM", MC3_ALLOW_SHUTDOWN="1", PYTHONPATH=str(HERE))
subprocess.run([sys.executable, HERE / "make_corpus.py", corpus], check=True)
norm = lambda s: re.sub(r"[\s\-\.·_]", "", s.upper())
def run(*a, timeout=120): return subprocess.run([sys.executable, APP / "app.py", *a], env=env, capture_output=True, text=True, timeout=timeout)
t = time.time(); r = run("--index", str(corpus), timeout=200)          # spawns the server itself (no CMD here)
print(f"index: rc={r.returncode} {time.time()-t:.1f}s {r.stderr.strip()[:200]}")
score = 0
for q in Q:
    qid = f"query_{q['n']:02d}"; t = time.time()
    r = run("--corpus", str(corpus), "--query-id", qid, "--query", q["query"], timeout=30); dt = time.time() - t
    out = json.loads((work / "out" / f"{qid}_output.json").read_text())
    accepted = {norm(x) for x in [q["expected_answer"], *q.get("answer_aliases", [])]}
    ok_a = norm(out["answer"]) in accepted; ok_c = set(out["citations"]) == set(q["expected_citations"])
    score += ok_a and ok_c
    print(f"Q{q['n']:>2} {'PASS' if ok_a and ok_c else 'FAIL'} {dt:4.1f}s answer={out['answer']!r} cites={out['citations']}" + ("" if ok_a and ok_c else f"   want {q['expected_answer']!r} {q['expected_citations']}"))
print(f"SCORE {score}/{len(Q)}"); print("skipped:", (work/'index'/'skipped.json').read_text().replace("\n", " "))
import socket
c = socket.socket(socket.AF_UNIX); c.connect(env["MC3_SOCK"]); c.sendall(b'{"op":"shutdown"}\n'); c.close()
shutil.rmtree(work, ignore_errors=True)
