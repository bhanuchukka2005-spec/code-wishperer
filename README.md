# CodeWhisperer Local

AI that reads your entire codebase and answers questions with exact file + line citations.

Built with **CodeBERT embeddings**, **ChromaDB** vector store, **llama-3.3-70b** via Groq API, and a cross-encoder re-ranker.

---

## What it does

- Clone any public GitHub repo or point it at a local folder
- Parses every file into semantic chunks (AST-aware for Python, JS, TS, Java, Go, Rust)
- Embeds and stores chunks in a local vector database
- Answers natural language questions with citations to exact files and line numbers
- Impact Analyzer — type a filename or class name, see every file that will break if you delete it
- Eval tab — benchmark retrieval accuracy with your own ground-truth pairs

---

## Stack

| Component | Technology |
|-----------|-----------|
| Code embeddings | `st-codesearch-distilroberta-base` (local, GPU) |
| Vector database | ChromaDB (local, persistent) |
| Retrieval | HyDE dual-query + cross-encoder re-ranking |
| LLM | `llama-3.3-70b-versatile` via Groq API |
| Chunking | Python AST · Tree-sitter (JS/TS/Java/Go/Rust) · sliding window fallback |
| UI | Gradio |

---

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/your-username/codewhisperer-local
cd codewhisperer-local
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install gradio chromadb sentence-transformers gitpython torch requests
pip install "sentence-transformers[cross-encoder]"
pip install tree-sitter-languages
```

> `tree-sitter-languages` is optional. If it fails to install, the app falls back to sliding-window chunking automatically.

### 4. Get a Groq API key

Sign up free at https://console.groq.com → API Keys → Create key.

Paste it into `app.py` line 20:
```python
GROQ_API_KEY = "gsk_your_key_here"
```

### 5. Run

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

---

## How to use

### Load a codebase

**GitHub URL tab** — paste any public repo URL and click **Clone & Ingest**:
```
https://github.com/pallets/flask
https://github.com/tiangolo/fastapi
https://github.com/psf/requests
```

**Local Folder tab** — paste the full path to a folder on your machine:
```
C:\Users\you\projects\myapp        (Windows)
/home/you/projects/myapp           (Linux/Mac)
```

### Ask questions

Switch to **Ask the Codebase** and type anything:

- `What does the authenticate() function do?`
- `Where are all the API routes defined?`
- `What breaks if I delete the UserModel class?`
- `Write a new endpoint that follows the pattern in this codebase`
- `Find missing error handling in the database layer`

### Impact Analyzer

Switch to **Impact Analyzer**, type a filename or class name:
```
user.py
UserModel
authenticate
utils.js
```
It builds a dependency graph and shows every file that imports from the target.

### Eval / Benchmark

Switch to **Eval / Benchmark**, fill in query → expected file pairs, click **Run Eval**.
Reports retrieval accuracy as a percentage — useful for comparing changes to the pipeline.

---

## Demo questions (rehearse these)

**1. Architecture overview**
> "What are all the main functions and classes?"

**2. Impact analysis**
> "What breaks if I delete the Request class?"

Visually impressive — lists every affected file with exact paths.

**3. Code generation**
> "Write a new middleware function that follows the exact pattern already used in this codebase."

Generates code that matches the repo's actual style.

---

## Best repos to demo

| Repo | Why |
|------|-----|
| `https://github.com/pallets/flask` | Clean Python, well-known, faculty recognises it |
| `https://github.com/tiangolo/fastapi` | Modern, well-structured, great for Q&A |
| `https://github.com/psf/requests` | Simple enough to demo clearly |

---

## Troubleshooting

**Slow on CPU** — embedding takes ~30s, answers ~5s. Normal without a GPU.

**ChromaDB error on restart** — delete the `codewhisperer_db/` folder and restart.

**`tree-sitter-languages` install fails** — skip it. The app works without it, using sliding-window chunking for non-Python files.

**Groq rate limit** — free tier allows ~30 requests/minute. Plenty for demos.

**Large repos (5000+ files)** — ingestion may take 3–5 minutes on CPU. Progress bar shows status.

---

## Architecture

```
GitHub URL / Local Folder
        │
        ▼
   Git Clone
        │
        ▼
   File Parser
   ├── Python  → AST (functions, classes)
   ├── JS/TS   → Tree-sitter (functions, classes)
   ├── Java/Go → Tree-sitter
   └── Other   → Sliding window (40-line chunks)
        │
        ▼
  CodeBERT Embedder  ──→  ChromaDB (local)
        │
   User Query
        │
        ▼
  HyDE Dual-Query Retrieval
  (raw query + "code implementation of: {query}")
        │
        ▼
  Cross-Encoder Re-Ranker
  (ms-marco-MiniLM-L-6-v2)
        │
        ▼
  Top-6 Chunks + Context
        │
        ▼
  Groq API (llama-3.3-70b)
        │
        ▼
  Streamed Answer + Citations
```

---

## Project structure

```
codewhisperer-local/
├── app.py
├── requirements.txt
├── README.md
├── .env              ← exists locally, never pushed
├── .gitignore        ← pushed, blocks .env and .gradio
└── codewhisperer_db/ ← exists locally, never pushed
```

---

## Requirements

```
gradio>=4.0
sentence-transformers>=2.2
chromadb>=1.0
transformers>=4.38
accelerate
gitpython
requests
torch
scipy
python-dotenv
```