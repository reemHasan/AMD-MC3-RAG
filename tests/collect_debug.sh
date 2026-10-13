#!/bin/sh
# One command that gathers everything needed to debug Q5 and Q9 into ONE file.
#   cd /workspace/AMD-MC3-RAG && . ./env.sh && sh tests/collect_debug.sh ./mc3-corpus
# then paste the contents of work/debug.txt.
CORPUS=${1:-./mc3-corpus}
OUT=${DEBUG_OUT:-work/debug.txt}
SLOG=${SERVER_LOG:-work/server.log}
IDX=${MC3_INDEX_DIR:-work/index}
mkdir -p "$(dirname "$OUT")"

echo "running Q5 and Q9 against the server..."
python3 tests/ask_sample.py --corpus "$CORPUS" --only 5 9 > /tmp/ask59.txt 2>&1

{
echo "### 0. files the indexer skipped, and why"; cat "$IDX/skipped.json" 2>&1
echo; echo "### 0b. corpus files with sizes and detected types"; ls -la "$CORPUS"/logs 2>&1; file "$CORPUS"/logs/* 2>&1
echo "first bytes of the log(s):"; for f in "$CORPUS"/logs/*; do head -c 120 "$f" | od -c | head -6; done
echo; echo "### 1. result of Q5 and Q9"; cat /tmp/ask59.txt
echo; echo "### 2. server log: what the model replied and why answers were refused/kept"
grep -E "model reply|REFUSE|ANSWER|bridge|vision skipped|small corpus" "$SLOG" | tail -n 14 | cut -c1-600
echo; echo "### 3. the log file as the model sees it (index chunk)"
python3 - "$IDX" <<'PY'
import json, sys
for line in open(sys.argv[1] + "/chunks.jsonl"):
    c = json.loads(line)
    if c["file"].endswith(".log"):
        print(c["file"], "|", c["loc"], "| status:", c["status"]); print(c["text"][:1500])
PY
echo; echo "### 4. every line mentioning the ticket, across the corpus"
grep -rna "ORR-1847" "$CORPUS" | cut -c1-300
echo; echo "### 5. explain.py for Q9 (answer 4.3.2, cited the csv only)"
python3 tests/explain.py --index-dir "$IDX" --answer 4.3.2 --cited support/bug_database.csv \
  --question "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?" 2>&1 | cut -c1-400
echo; echo "### 6. explain.py for Q5 (expected answer E7731)"
python3 tests/explain.py --index-dir "$IDX" --answer E7731 --cited logs/prod_inference_2026-09-02.log \
  --question "What error code is logged when the thermal throttle engages?" 2>&1 | cut -c1-400
} > "$OUT" 2>&1
echo "done -> $OUT   ($(wc -l < "$OUT") lines). Paste its contents."
