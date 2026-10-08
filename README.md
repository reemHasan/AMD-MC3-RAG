# MC-3: Retrieval-Augmented Generation over a mixed-document corpus

A Docker submission for AMD Mini-Challenge 3. It takes a folder of mixed documents
(PDF, Word, Excel, CSV, text, logs, Python source, PNG/JPG), indexes it once, and then
answers ten questions, each with a **single value** and an **exact set of citations**.

---

## Contents

1. [What the project does](#1-what-the-project-does)
2. [How it differs from a normal RAG project](#2-how-it-differs-from-a-normal-rag-project)
3. [Architecture](#3-architecture)
4. [How it is implemented](#4-how-it-is-implemented)
5. [File layout](#5-file-layout)
6. [Configuration](#6-configuration)
7. [Running it](#7-running-it)
8. [Testing](#8-testing)
9. [Challenge requirements checklist](#9-challenge-requirements-checklist)
10. [Known limitations and next steps](#10-known-limitations-and-next-steps)

---

## 1. What the project does

The harness gives the container a corpus and then calls the same script in two ways:

```bash
# Phase 1: once, before any graded clock (counted in the 10-minute startup budget)
python3 /app/app.py --index /app/corpus

# Phase 2: once per question, ten times, each a NEW process (30 s limit each)
python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
```

and reads back `/app/output/query_01_output.json`:

```json
{"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}
```

A question scores (20 points) **only if the answer and the citation set are both exactly right**.
There is no partial credit. The rules that shape the whole design:

| Rule | Consequence |
|---|---|
| The answer is the **value only** (`94`, not a sentence; `Q3 FY27`, not `Q3`) | The model is prompted for a bare value and the output is cleaned. |
| Citations are an **exact set**, graded by *necessity* (cite a file only if removing it makes the answer impossible) | We cannot dump the retrieval results. Every cited file must be justified. |
| If the corpus has no answer, return `""` and `[]` | Refusal is a first-class outcome, and guessing scores zero. |
| The products are **fictional** | Anything the model "remembers" instead of reads is wrong by construction. |
| The corpus contains an unreadable file, an encrypted PDF, an unknown binary and an empty directory | The indexer must survive all of them. |
| Two questions can only be answered by **looking at an image** | A vision-language model is required, not just OCR-free text search. |
| Superseded documents exist (a *WITHDRAWN* datasheet with an old value) | The current revision must win, and the old file must never be cited. |
| Multi-hop questions need **two files** | The log names a ticket, the bug database has the fix version. Both must be cited. |

---

## 2. How it differs from a normal RAG project

A typical RAG tutorial: split documents into chunks, embed them, store the vectors in
FAISS or Chroma, retrieve the top-k for a question, and let an LLM write a paragraph.
This project keeps the "retrieve, then generate" idea, but almost every part is different
because the grading and the environment are different.

| Aspect | Typical RAG | This project |
|---|---|---|
| **Output** | A free-text answer | A bare value plus an exact citation set, graded automatically |
| **Citations** | "Sources used", usually the retrieved top-k | **Necessity only**: a file is cited if the answer is impossible without it |
| **Failure mode** | A plausible but wrong answer is acceptable | A wrong answer scores zero, so **refusal is a correct output** |
| **Hallucination control** | Prompt wording, hope | A **grounding guard**: the answer must literally appear in the evidence shown, else refuse |
| **Retrieval** | Dense embeddings + vector DB | **BM25** with an identifier-aware tokenizer (no embeddings, no vector DB) |
| **Reasoning** | One retrieve-then-answer pass | A **multi-hop loop**: the model can ask for a follow-up search (e.g. a ticket id found in a log) |
| **Document quality** | All documents are trusted equally | **Superseded/withdrawn** documents are down-ranked, labelled, and never cited |
| **File types** | Mostly text or PDF | PDF, DOCX (with tables), XLSX (every sheet), CSV, TXT/LOG, PY and **images** |
| **Images** | Often ignored or OCR'd once | Transcribed at index time **and** re-read against the actual question at query time |
| **Bad files** | Usually crash or are ignored | Per-file isolation: unreadable, encrypted, corrupt and unknown files are skipped, never fatal |
| **Process model** | One long-running app | Each question is a **new process**, so the model lives in a **resident server** reached through a unix socket |
| **Environment** | Network and any package available | **No network at evaluation**, a fixed ROCm base image, a size cap, per-question and total time limits |

In short: a normal RAG system optimises for *helpful-sounding prose*. This one optimises
for *exactly right or silent*, under tight operational constraints.

---

## 3. Architecture

### The core problem

The harness starts a **new process for every question**. Loading a 16 GB model each time
would exceed the 30 s limit on all ten questions. So the design separates the heavy,
long-lived state from the short-lived command-line calls.

```
┌─────────────────── SERVER process (long-lived, server.py) ───────────────────┐
│ GPU memory (VRAM)  : vision-language model weights (~16 GB, bf16)            │
│ CPU memory (RAM)   : chunks (list of dicts) + BM25 inverted index            │
└──────────────────────────────────────────────────────────────────────────────┘
        ▲                      ▲                       ▲
        │ unix socket          │                       │       (messages only)
  app.py --index         app.py --query          app.py --query ...
  (client, exits)        (client, exits)         (client, exits)

DISK:  /app/index/chunks.jsonl            saved copy of the chunks (for restarts)
       /app/output/<query-id>_output.json answer file written by each client
```

The clients hold **nothing** in memory. They send one JSON message to the server and exit.
That is why the model is loaded once, and why `--index` does not "keep BM25 in memory":
the server builds and keeps it.

### Lifecycle

1. **Container start.** The CMD runs `python3 /app/server.py`. The server takes a file
   lock (so only one server runs), binds the unix socket, then loads the model onto the GPU.
   It binds *before* loading so an early client waits in the queue instead of failing.
2. **`--index`.** The client asks the server to index. The server walks the corpus,
   parses files, transcribes images with the model, builds chunks and BM25, and saves the
   chunks to disk.
3. **`--query` (x10).** The client sends the question. The server searches BM25, asks the
   model, applies the guards, and replies. The client writes the output file.

If the CMD-started server is missing (for example the harness overrides the CMD), the
client starts `server.py` itself as a fallback.

### Question flow

```
question
   │
   ├─► model suggests up to 6 extra search phrases (weight 0.5)
   ├─► BM25 over all chunks  ──► top excerpts (+ best 2 image files)
   │
   │   loop, up to 4 rounds
   │   ┌────────────────────────────────────────────────────────────┐
   │   │ read new images with the vision model (question-aware)     │
   │   │ ask the model: answer | search | none                      │
   │   │   answer ─► guards ─► (answer, citations)                  │
   │   │   search ─► BM25 on the identifier alone ─► add excerpts   │
   │   │   none   ─► widen the search once, else refuse             │
   │   └────────────────────────────────────────────────────────────┘
   ▼
 grounding guard  ─►  necessity guard  ─►  output file
```

---

## 4. How it is implemented

### 4.1 Parsing (`parsers.py`)

Every parser returns chunks of about 1,400 characters shaped `{"text", "loc"}`, or raises.

| Type | Approach |
|---|---|
| **PDF** | `pypdf` text per page. An **encrypted PDF is always skipped**, even if a blank password would open it. A near-empty (scanned) PDF is rendered with `pypdfium2` and transcribed by the model. |
| **DOCX** | The XML is read directly (no extra dependency). Paragraphs, **table rows** (`cell \| cell \| cell`), headers, footers and footnotes. |
| **XLSX / CSV** | **Every sheet** (including hidden ones). Each row becomes `Column: value; Column: value`, and **every chunk repeats the header**, so the model knows what each number means. |
| **TXT / LOG / PY** | Line windows with a 3-line overlap, so a fact on a boundary is not cut. |
| **Images** | Transcribed by the vision model at index time. |
| **Unknown types** | Skipped. |

`file_status()` marks a file `withdrawn` if its **filename** contains *withdrawn / superseded /
obsolete*, or its **opening banner** says so. It deliberately does *not* search the whole
body, so a current datasheet that says "this revision supersedes the withdrawn r1" is not
flagged by mistake.

### 4.2 Indexing (`indexer.py`)

`build_index()` has three phases:

1. **Discover.** `os.walk` collects files. Directories and files without read bits are
   skipped. This also makes local tests match the evaluation, where root has no permission bypass.
2. **Parse.** Non-images first, images last, so a time cutoff only costs image transcriptions
   (images are still read directly at query time). Each file runs inside its own `try/except`
   and a `SIGALRM` timeout (90 s, 240 s for images), so a hostile or pathological file
   costs only itself.
3. **Build.** Chunks are tagged with `file`, `kind`, `status`, `loc` and `text`, and
   wrapped in an `Index`.

### 4.3 Retrieval (`retrieval.py`)

- **Tokenizer.** Identifier-friendly: `TQ-40`, `TQ40` and `tq 40` all match each other;
  `THERM_ALERT#` is searchable; `10,000` becomes `10000`.
- **BM25.** Standard Okapi BM25 (`k1 = 1.4`, `b = 0.75`) over `path + text`, with an
  inverted index. Rare terms such as ticket ids score high, so exact identifiers
  retrieve well. Withdrawn chunks get a 0.4 score multiplier.
- **Persistence.** `Index.save` writes `chunks.jsonl` atomically (temp file then rename).
  `Index.load` reads it back and rebuilds BM25 in memory, so only the chunks are stored.

### 4.4 Model (`llm.py`)

One vision-language model (default **Qwen2.5-VL-7B-Instruct**, bf16) does everything:
image transcription, keyword expansion and answering. It is loaded from a local directory
(`/models/vlm`), placed on `cuda` explicitly, and decodes greedily, so results are deterministic.

> On the ROCm build of PyTorch, `torch.cuda` is simply PyTorch's name for its GPU API; it maps
> to the AMD HIP backend. `torch.cuda.is_available()` is `True` on an AMD GPU, and
> `torch.version` ends in `+rocm...`.

### 4.5 The agent (`agent.py`)

- **Prompt.** The model sees numbered excerpts (`[E1] path | location`), with withdrawn ones
  marked, and must reply with JSON: `status` (`answer`, `search` or `none`), `answer`,
  `search`, `sources`.
- **Multi-hop.** If an excerpt names an identifier (a ticket, error code or part number)
  whose value is not yet in view, the model returns `search`. The system then runs BM25 on
  that identifier **alone**, which is how a row that shares no words with the question can
  still be found. The files that supplied the identifier are recorded as the "chain".
- **Vision at query time.** For image files in the context, the model looks at the real
  picture together with the question, which is more reliable than relying on an earlier
  transcription.
- **Time protection.** A 23 s budget per question; the client has a 28 s hard deadline and
  always writes a valid file.

### 4.6 The two guards (the main difference from a plain RAG pipeline)

**Grounding guard.** After normalisation (uppercase; spaces and `- . · _` removed, the
same as the grader), the answer must appear inside the evidence the model was shown. If
not, it is a guess or a recollection, and the output is a refusal. This is what makes the
"answer exists only in the encrypted file" question come out empty even if the model
confidently produces a number.

**Necessity guard.** The final citations are the files the model named, **intersected with**
the files that either contain the answer or supplied the identifier used in a search
hop. Chain files are added even if the model forgot to cite them. Withdrawn files are
never cited, and if the only support for an answer is a withdrawn file, the output is a
refusal.

The answer is also cleaned: quotes and a trailing full stop are removed, and a number with
a trailing unit (`94 °C`) is reduced to `94`.

### 4.7 Robustness decisions

| Risk | Handling |
|---|---|
| Server not running | The client starts `server.py`; a valid refusal file is written even if everything fails |
| Two servers starting at once | An exclusive `flock` on a lock file; the loser exits quietly |
| Half-written output | Written to a temp file, then renamed atomically |
| Model fails to load | The server stays up and refuses every question, so files are still produced |
| One corrupt or locked file | Per-file `try/except`, per-file timeout, recorded in `skipped.json` |
| Image transcription fails (e.g. out of memory) | The image becomes a filename-only chunk and is read directly at query time |

### 4.8 Why BM25 instead of an embedding model and vector DB

- The questions hinge on exact identifiers (`ORR-1847`, `E7731`, `ORR-FAN-2214-B`), which keyword matching handles well.
- The corpus is small, so a vector database is unnecessary.
- No second model is needed, which saves startup time and VRAM.
- Many embedding and vector-DB packages depend on `torch`. A plain `pip install` of one can **replace the ROCm torch with a CUDA build, which disqualifies the submission**.
- Its weakness is **paraphrase** (different wording gives zero overlap). It is partly offset by model-suggested keywords and the `search` step. See the next-steps section.

---

## 5. File layout

```
Dockerfile
models/vlm/          model weights (you provide them)
app/
  app.py             thin client: --index and --query; always writes a valid output file
  server.py          resident process: model + index behind a unix socket
  agent.py           retrieval loop, multi-hop, vision, grounding and necessity guards
  indexer.py         defensive corpus walker -> Index
  parsers.py         PDF / DOCX / XLSX / CSV / text / image parsers
  retrieval.py       tokenizer, BM25, Index (save / load)
  llm.py             vision-language model wrapper
  requirements.txt   dependencies that do NOT pull in torch
tests/
  make_corpus.py     synthetic corpus, including the hostile files
  mock_llm.py        scripted stand-in model (tests the plumbing, not model quality)
  run_local.py       full contract run, graded like the harness
  sample-questions.json
```

---

## 6. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MC3_MODEL_DIR` | `/models/vlm` | Local directory with the model weights |
| `MC3_INDEX_DIR` | `/app/index` | Where the chunk index is saved |
| `MC2_OUTPUT_DIR` | `/app/output` | Where `<query-id>_output.json` is written |
| `MC3_SOCK` | `/tmp/mc3_rag.sock` | Unix socket path (the lock file is `<path>.lock`) |
| `MC3_SERVER_LOG` | `/tmp/mc3_server.log` | Log file for a server started by the client fallback |
| `MC3_IMAGE_CUTOFF_S` | `420` | Seconds after server start when image transcription at index time stops |
| `MC3_CORPUS` | `/app/corpus` | Default corpus root for the server |
| `MC3_LLM_FACTORY` | unset | `module:Class`, swaps in a stand-in model (tests only) |
| `MC3_ALLOW_SHUTDOWN` | unset | Enables a `shutdown` socket op (tests only) |

Tunable constants: `MAX_ROUNDS`, `MAX_VISION_CALLS` and the prompts in `agent.py`; chunk size
in `parsers.py`; `k1`, `b` and the withdrawn multiplier in `retrieval.py`.

---

## 7. Running it

### 7.1 On a cloud JupyterLab with an AMD GPU

Use a **Terminal** for the long-running parts.

```bash
# 1. Confirm the ROCm build and the GPU
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"

# 2. Install dependencies WITHOUT touching torch
pip install -r app/requirements.txt
pip install --no-deps -U transformers tokenizers safetensors huggingface_hub
python -c "import torch; print(torch.__version__)"      # must still contain 'rocm'

# 3. Download the weights first (~16 GB). Do NOT have HF_HUB_OFFLINE=1 set yet.
#(newer huggingface_hub ships `hf download ...`; `huggingface-cli` no longer exists)
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='models/vlm')"

# 4. Point the project at local paths
export MC3_MODEL_DIR=$PWD/models/vlm MC3_INDEX_DIR=$PWD/work/index \
       MC2_OUTPUT_DIR=$PWD/work/output MC3_SOCK=/tmp/mc3_rag.sock HF_HUB_OFFLINE=1
mkdir -p work

# 5. Start the server and wait for "ready: ... (model load took Ns)"
nohup python app/server.py > work/server.log 2>&1 &
tail -f work/server.log

# 6. Index, then ask
time python app/app.py --index ./sample-corpus
time python app/app.py --corpus ./sample-corpus --query-id query_01 \
     --query "What is the maximum junction temperature of the TQ-40?"
cat work/output/query_01_output.json

# 7. Watch VRAM in a second terminal (must stay between 1 and 48 GiB)
watch -n1 'amd-smi metric --mem | grep USED_VRAM'
```

### 7.2 As the Docker submission

```bash
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='models/vlm')"
docker build -t mc3:v1 .
docker run --rm mc3:v1 python3 -c "import torch; print(torch.__version__)"   # +rocm
python3 selfcheck.py mc3:v1 ./sample-corpus

# hostile-file check: no network, root without DAC_OVERRIDE, like the evaluation
docker run -d --network none --cap-drop DAC_OVERRIDE \
       --device /dev/kfd --device /dev/dri --name t mc3:v1
docker cp ./sample-corpus/. t:/app/corpus
docker exec t python3 /app/app.py --index /app/corpus
```

Ship the weights **inside** the image. Your MC-3 documents say the container has no
network access while graded, so do not rely on downloading at run time.

---

## 8. Testing

```bash
pip install pypdf openpyxl pillow reportlab python-docx
python3 tests/run_local.py
```

This builds a synthetic corpus (current and withdrawn datasheets, DOCX with a table, XLSX
with two sheets, a CSV whose fixing row shares no words with the question, a log, a Python
file, two images, an encrypted PDF, an unknown `.dat`, a `chmod 000` file and an empty
directory), runs `--index` and all ten sample questions through the real client, server
and agent, and grades the output the way the harness does.

**What this proves:** the plumbing, parsers, multi-hop chain, refusal path, citation
filtering, the hostile-file handling, and that a valid file is always written in time.

**What it does not prove:** model quality. It uses a scripted stand-in for the model.
Whether the real model reads the pinout diagram, or picks the right row among near-misses,
must be measured on your hardware.

---

## 9. Challenge requirements checklist

| Requirement | Where it is addressed | Verified |
|---|---|---|
| Built on the mandated ROCm base image (layer identity) | `FROM` line identical to the starter; single stage, no squash | Yes (static) |
| Torch stays the ROCm build | `requirements.txt` has no torch dependency; `transformers` installed `--no-deps`; the build asserts `'rocm' in torch.__version__` | Resolution checked; build **not run** |
| Image under 60 GiB uncompressed | About 29 GiB base + about 16 GiB weights | Estimate only |
| `/app/app.py`, `/app/requirements.txt`, `/models` | Dockerfile `COPY` layout | Yes (static) |
| Startup under 10 min (load + index) | Model loads once in the server; images transcribed after the cutoff are skipped at index time | **Measure on hardware** |
| 30 s per question | 23 s agent budget, 28 s client deadline, refusal file as the fallback | Timeout paths tested |
| Output at `/app/output/<query-id>_output.json` | `app.py` | Yes |
| Both fields always present | Refusal file written on any failure | Yes |
| Unreadable file, encrypted PDF, unknown type, empty dir | Per-file isolation and permission checks | Yes, with and without `DAC_OVERRIDE` |
| No network at evaluation | No network calls in the code; offline flags set | Yes (static) |
| Uses the GPU | Model placed on `cuda` explicitly | **Measure VRAM on hardware** |
| No hardcoded sample answers | Sample values appear only in `tests/` | Yes (grep) |

---

## 10. Known limitations and next steps

- **Not run on a GPU here.** Model loading on ROCm, real answer quality, and real timings
  need to be verified on your hardware, as does compatibility between your chosen
  `transformers` version, model and Python 3.14.
- **BM25 does not match paraphrases.** If the question and the document use different
  words, retrieval can miss. Keyword expansion and the `search` step reduce this but do not
  remove it. A hybrid retriever (a small embedding model loaded with plain `transformers`,
  a dot-product score, merged with BM25 by reciprocal rank fusion) is the natural upgrade,
  and should be measured against the real sample corpus before adopting it.
- **Image reading depends on the model.** Pin-diagram and label questions are the weakest
  point; `VISION_PROMPT` in `agent.py` is the first thing to tune.
- **PDF tables** are read as text lines with `pypdf`, which can misalign columns. If a
  datasheet table is answered wrongly, try layout-preserving extraction.
- **The grounding guard can over-refuse** if the correct answer is computed rather than
  quoted. The sample questions are all lookups, so this trade-off favours safety.
- **Startup fallback.** If the harness does not honour the image CMD, the first `--index`
  waits up to 20 s before starting the server itself.
