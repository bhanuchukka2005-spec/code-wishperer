"""
CodeWhisperer v3 — AI codebase intelligence
Dark terminal UI. Groq LLM. Full RAG pipeline.
"""

import os, re, time, json, subprocess, tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import gradio as gr
import requests as req

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
DB_PATH      = Path("codewhisperer_db")
DB_PATH.mkdir(exist_ok=True)

print(f"[CodeWhisperer] Device: {DEVICE.upper()}")

# ── Globals ───────────────────────────────────────────────────────────────────
_embedder   = None
_reranker   = None
_chroma     = None
_collection = None
_repo_root  = None
_dep_graph  = {}
_file_tree  = []

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".jsx", ".tsx", ".vue", ".html", ".css", ".sh", ".md"
}

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("[Embedder] Loading...")
        _embedder = SentenceTransformer(
            "flax-sentence-embeddings/st-codesearch-distilroberta-base", device=DEVICE)
    return _embedder

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("[Reranker] Loading...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)
    return _reranker

def get_collection():
    global _chroma, _collection
    if _collection is None:
        import chromadb
        _chroma     = chromadb.PersistentClient(path=str(DB_PATH))
        _collection = _chroma.get_or_create_collection(
            "codewhisperer", metadata={"hnsw:space": "cosine"})
    return _collection

def reset_collection():
    global _collection, _chroma
    import chromadb
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        _chroma.delete_collection("codewhisperer")
    except Exception:
        pass
    _collection = _chroma.get_or_create_collection(
        "codewhisperer", metadata={"hnsw:space": "cosine"})

# ══════════════════════════════════════════════════════════════════════════════
#  GROQ STREAMING
# ══════════════════════════════════════════════════════════════════════════════

def call_groq_stream(system: str, user: str):
    if not GROQ_API_KEY:
        yield "⚠️ GROQ_API_KEY not set. Add it to your .env file."
        return
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        "max_tokens": 1500, "temperature": 0.2, "stream": True,
    }
    try:
        with req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, stream=True, timeout=60
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line: continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]": break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta: yield delta
                    except Exception:
                        continue
    except Exception as e:
        yield f"⚠️ Groq API error: {str(e)}"

# ══════════════════════════════════════════════════════════════════════════════
#  PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _try_treesitter_chunks(source, filepath, lang):
    try:
        from tree_sitter_languages import get_parser
    except ImportError:
        return None
    lang_map = {
        ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust",
        ".cpp": "cpp", ".c": "c", ".h": "c",
        ".rb": "ruby", ".kt": "kotlin",
    }
    ts_lang = lang_map.get(lang)
    if not ts_lang: return None
    try:
        parser = get_parser(ts_lang)
        tree   = parser.parse(bytes(source, "utf-8"))
        lines  = source.splitlines()
        chunks = []
        DEF_TYPES = {
            "function_declaration", "function_definition",
            "method_definition", "method_declaration",
            "class_declaration", "class_definition",
            "arrow_function", "decorated_definition",
        }
        def walk(node):
            if node.type in DEF_TYPES:
                start = node.start_point[0]
                end   = node.end_point[0]
                code  = "\n".join(lines[start:end + 1])
                name  = "anonymous"
                for child in node.children:
                    if child.type in ("identifier", "name"):
                        name = child.text.decode("utf-8"); break
                chunks.append({
                    "text": f"# {node.type}: {name}\n{code}", "name": name,
                    "kind": node.type, "file": filepath,
                    "line_start": start + 1, "line_end": end + 1, "docstring": ""
                })
            for child in node.children: walk(child)
        walk(tree.root_node)
        return chunks if chunks else None
    except Exception:
        return None

def parse_python_chunks(source, filepath):
    import ast as pyast
    chunks = []
    try:
        tree = pyast.parse(source)
        for node in pyast.walk(tree):
            if isinstance(node, (pyast.FunctionDef, pyast.AsyncFunctionDef, pyast.ClassDef)):
                start = node.lineno
                end   = getattr(node, "end_lineno", start + 10)
                code  = "\n".join(source.splitlines()[start - 1: end])
                kind  = "class" if isinstance(node, pyast.ClassDef) else "function"
                chunks.append({
                    "text": f"# {kind}: {node.name}\n{code}", "name": node.name,
                    "kind": kind, "file": filepath, "line_start": start,
                    "line_end": end, "docstring": (pyast.get_docstring(node) or "")[:200]
                })
    except Exception:
        pass
    return chunks

def parse_generic_chunks(source, filepath, chunk_size=40):
    lines, chunks, step = source.splitlines(), [], chunk_size // 2
    for i in range(0, max(1, len(lines) - chunk_size + 1), step):
        block = lines[i: i + chunk_size]
        text  = "\n".join(block).strip()
        if len(text) > 30:
            chunks.append({
                "text": text, "name": f"block_{i}", "kind": "code_block",
                "file": filepath, "line_start": i + 1,
                "line_end": i + len(block), "docstring": ""
            })
    return chunks

def parse_file(filepath, repo_root):
    rel = os.path.relpath(filepath, repo_root)
    try:
        source = open(filepath, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return []
    if not source.strip(): return []
    ext = Path(filepath).suffix.lower()
    if ext == ".py":
        chunks = parse_python_chunks(source, rel)
        return chunks if chunks else parse_generic_chunks(source, rel)
    ts = _try_treesitter_chunks(source, rel, ext)
    return ts if ts else parse_generic_chunks(source, rel)

# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_dependency_graph(repo_root):
    import ast as pyast
    graph = {}
    for pyfile in Path(repo_root).rglob("*.py"):
        rel = str(pyfile.relative_to(repo_root))
        try:
            src  = pyfile.read_text(encoding="utf-8", errors="ignore")
            tree = pyast.parse(src)
        except Exception:
            continue
        imports = []
        for node in pyast.walk(tree):
            if isinstance(node, pyast.Import):
                for alias in node.names: imports.append(alias.name)
            elif isinstance(node, pyast.ImportFrom):
                if node.module: imports.append(node.module)
        graph[rel] = imports
    JS_IMPORT = re.compile(
        r"""(?:import\s+.*?\s+from\s+['"]([^'"]+)['"])|(?:require\s*\(\s*['"]([^'"]+)['"]\s*\))""",
        re.MULTILINE)
    for ext in ("*.js", "*.ts", "*.jsx", "*.tsx"):
        for jsfile in Path(repo_root).rglob(ext):
            if "node_modules" in str(jsfile): continue
            rel = str(jsfile.relative_to(repo_root))
            try:
                src = jsfile.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            graph[rel] = [
                m.group(1) or m.group(2)
                for m in JS_IMPORT.finditer(src)
                if m.group(1) or m.group(2)
            ]
    return graph

def find_dependents(graph, target_file):
    mod       = target_file.replace(os.sep, ".").replace("/", ".").replace(".py", "")
    mod_short = mod.split(".")[-1]
    stem      = Path(target_file).stem
    result    = []
    for f, imports in graph.items():
        for imp in imports:
            if mod_short in imp or mod in imp or stem in imp:
                result.append(f); break
    return list(set(result))

# ══════════════════════════════════════════════════════════════════════════════
#  INGEST
# ══════════════════════════════════════════════════════════════════════════════

def ingest_repo(repo_path, progress=gr.Progress()):
    global _repo_root, _dep_graph, _file_tree
    _repo_root = repo_path
    col        = get_collection()
    embedder   = get_embedder()

    all_files = [
        str(p) for p in Path(repo_path).rglob("*")
        if p.is_file() and p.suffix.lower() in CODE_EXTENSIONS
        and ".git" not in str(p) and "node_modules" not in str(p)
        and "__pycache__" not in str(p)
    ]

    if not all_files:
        return "No supported code files found.", "", "No codebase loaded"

    _file_tree = [os.path.relpath(f, repo_path) for f in all_files]
    progress(0, desc="Building dependency graph...")
    _dep_graph = build_dependency_graph(repo_path)

    all_chunks = []
    for i, fp in enumerate(all_files):
        all_chunks.extend(parse_file(fp, repo_path))
        if i % 20 == 0:
            progress(0.1 + 0.4 * i / len(all_files),
                     desc=f"Parsing {i}/{len(all_files)} files...")

    if not all_chunks:
        return "Parsing produced no chunks.", "", "Parse error"

    progress(0.5, desc=f"Embedding {len(all_chunks)} chunks...")
    texts   = [c["text"][:1500] for c in all_chunks]
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=False).tolist()

    progress(0.85, desc="Storing in ChromaDB...")
    ts    = str(int(time.time() * 1000))
    ids   = [f"{ts}_{i}" for i in range(len(all_chunks))]
    metas = [{
        "name": c["name"], "kind": c["kind"], "file": c["file"],
        "line_start": c["line_start"], "line_end": c["line_end"],
        "docstring": c["docstring"]
    } for c in all_chunks]

    for b in range(0, len(all_chunks), 2000):
        col.add(
            documents=texts[b:b+2000], embeddings=vectors[b:b+2000],
            ids=ids[b:b+2000], metadatas=metas[b:b+2000]
        )

    langs = sorted({Path(f).suffix for f in all_files})
    progress(1.0, desc="Done!")

    stats = (
        f"### ✅ Codebase loaded\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Files | **{len(all_files)}** |\n"
        f"| Chunks | **{len(all_chunks)}** |\n"
        f"| Languages | {' '.join(f'`{l}`' for l in langs)} |\n"
        f"| Dep graph | **{len(_dep_graph)}** files mapped |"
    )
    file_md = "\n".join(f"- `{f}`" for f in _file_tree[:60])
    if len(_file_tree) > 60:
        file_md += f"\n- *...and {len(_file_tree)-60} more*"
    status = f"Loaded: {len(all_files)} files · {len(all_chunks)} chunks"

    return stats, file_md, status

def clone_and_ingest(github_url, progress=gr.Progress()):
    global _repo_root
    if not github_url.strip():
        return "Please enter a GitHub URL.", "", "No codebase loaded"
    progress(0, desc="Cloning repo...")
    clone_dir = Path(tempfile.mkdtemp()) / "repo"
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", github_url.strip(), str(clone_dir)],
            check=True, capture_output=True, timeout=180
        )
    except subprocess.CalledProcessError as e:
        return f"Git clone failed:\n```\n{e.stderr.decode()[:400]}\n```", "", "Clone failed"
    except FileNotFoundError:
        return "Git not found. Install from https://git-scm.com", "", "Git missing"
    _repo_root = str(clone_dir)
    reset_collection()
    return ingest_repo(str(clone_dir), progress)

def ingest_local(folder_path, progress=gr.Progress()):
    global _repo_root
    if not folder_path or not Path(folder_path).exists():
        return "Folder not found. Check the path.", "", "Not found"
    _repo_root = folder_path
    reset_collection()
    return ingest_repo(folder_path, progress)

# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query, n=10):
    col   = get_collection()
    total = col.count()
    if total == 0: return [], []
    embedder = get_embedder()
    vecs     = embedder.encode([query, f"code implementation of: {query}"]).tolist()
    seen     = {}
    for vec in vecs:
        res = col.query(query_embeddings=[vec], n_results=min(n, total))
        for doc, meta, cid in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
            if cid not in seen: seen[cid] = (doc, meta)
    if not seen: return [], []
    candidates = list(seen.values())
    docs  = [c[0] for c in candidates]
    metas = [c[1] for c in candidates]
    try:
        reranker = get_reranker()
        scores   = reranker.predict([(query, d[:512]) for d in docs])
        ranked   = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)[:6]
        return [r[1] for r in ranked], [r[2] for r in ranked]
    except Exception:
        return docs[:6], metas[:6]

def format_context(docs, metas):
    parts = []
    for doc, m in zip(docs, metas):
        parts.append(
            f"# File: {m['file']}  Lines {m['line_start']}–{m['line_end']}  "
            f"({m['kind']}: {m['name']})\n{doc[:800]}"
        )
    return "\n\n---\n\n".join(parts)

def format_sources(metas):
    seen = []
    for m in metas:
        e = f"`{m['file']}` line {m['line_start']} — *{m['kind']}* `{m['name']}`"
        if e not in seen: seen.append(e)
    return "\n".join(f"- {e}" for e in seen)

# ══════════════════════════════════════════════════════════════════════════════
#  ANSWER
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are CodeWhisperer, an expert AI code intelligence assistant.
- Answer using ONLY the provided code context
- Always cite exact file names and line numbers
- Format code with markdown code blocks with language tags
- Be precise and technical
- If context is insufficient, say exactly what is missing
- Use headers to structure multi-part answers"""

def answer(query, history):
    if get_collection().count() == 0:
        yield history + [[query, "⚠️ No codebase loaded. Go to **Load Codebase** tab first."]]
        return

    impact_match = re.search(
        r"(what.*(break|happen|affect).*delete|delete.*what|impact of (deleting|removing))\s+(.+)",
        query, re.IGNORECASE)
    if impact_match and _dep_graph:
        target  = impact_match.group(4).strip().strip("`'\"")
        matches = [f for f in _dep_graph if target.lower() in f.lower()]
        if matches:
            dependents = find_dependents(_dep_graph, matches[0])
            if dependents:
                dep_list = "\n".join(f"- `{d}`" for d in dependents[:20])
                yield history + [[query,
                    f"### Impact Analysis: `{matches[0]}`\n\n"
                    f"Deleting this will break **{len(dependents)} file(s)**:\n\n"
                    f"{dep_list}\n\n"
                    f"> These files have direct import dependencies on `{matches[0]}`"
                ]]
                return

    docs, metas = retrieve(query)
    if not docs:
        yield history + [[query, "⚠️ No relevant code found. Ingest a codebase first."]]
        return

    context  = format_context(docs, metas)
    sources  = format_sources(metas)
    user_msg = f"### Code Context\n{context}\n\n### Question\n{query}"

    collected   = ""
    new_history = history + [[query, ""]]
    for chunk in call_groq_stream(SYSTEM_PROMPT, user_msg):
        collected          += chunk
        new_history[-1][1]  = collected
        yield new_history

    new_history[-1][1] = collected + f"\n\n---\n**📁 Sources:**\n{sources}"
    yield new_history

# ══════════════════════════════════════════════════════════════════════════════
#  CODE REVIEW
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_PROMPTS = {
    "Security":     "Review for security vulnerabilities: SQL injection, XSS, hardcoded secrets, improper auth, CSRF, insecure deserialization. Show severity and fix for each.",
    "Performance":  "Review for performance issues: N+1 queries, missing indexes, memory leaks, blocking I/O, unnecessary loops, missing caching. Show impact and fix.",
    "Code Quality": "Review for quality: naming, SOLID violations, DRY, missing error handling, deep nesting, magic numbers, missing types. Show fix for each issue.",
    "Bugs":         "Find all bugs, edge cases, and runtime errors. Show the exact condition that triggers each bug and the correct fix.",
}

def review_code(code_input, focus):
    if not code_input.strip():
        yield "Paste some code to review."
        return
    prompt_detail = REVIEW_PROMPTS.get(focus, REVIEW_PROMPTS["Code Quality"])
    context = ""
    if get_collection().count() > 0:
        docs, metas = retrieve("patterns conventions style error handling", n=4)
        context = format_context(docs, metas)

    system = f"""You are an expert code reviewer.
{prompt_detail}

Format:
## Issues Found
**[SEVERITY: HIGH/MED/LOW]** — description
- Line: X
- Problem: what goes wrong
- Fix: concrete corrected code snippet

## Summary
Overall quality score (1-10) and top recommendations.

{'Compare against the codebase patterns provided.' if context else ''}"""

    user = (
        f"{'### Codebase patterns:\n' + context[:1500] + chr(10)*2 if context else ''}"
        f"### Code to review:\n```\n{code_input}\n```"
    )
    collected = ""
    for chunk in call_groq_stream(system, user):
        collected += chunk
        yield collected

# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE CODE
# ══════════════════════════════════════════════════════════════════════════════

def generate_code(description, language):
    if not description.strip():
        yield "Describe what you want to generate."
        return
    context = ""
    if get_collection().count() > 0:
        docs, metas = retrieve(description, n=5)
        context = format_context(docs, metas)

    system = """You are an expert software engineer.
Generate clean, production-ready code that:
- Follows patterns from the codebase context if provided
- Includes proper error handling
- Has comments explaining non-obvious logic
- Matches the style of existing code"""

    user = (
        f"{'### Existing codebase patterns:\n' + context[:2000] + chr(10)*2 if context else ''}"
        f"### Generate {language} code for:\n{description}\n\n"
        f"Return ONLY the code. No preamble or explanation."
    )
    collected = ""
    for chunk in call_groq_stream(system, user):
        collected += chunk
        yield collected

# ══════════════════════════════════════════════════════════════════════════════
#  EXPLAIN FILE
# ══════════════════════════════════════════════════════════════════════════════

def explain_file(file_path_input):
    if not file_path_input.strip():
        yield "Enter a file path or name."
        return
    if get_collection().count() == 0:
        yield "⚠️ No codebase loaded."
        return
    docs, metas = retrieve(f"contents of file {file_path_input}", n=8)
    file_docs   = [(d, m) for d, m in zip(docs, metas)
                   if file_path_input.lower() in m["file"].lower()]
    if not file_docs:
        file_docs = list(zip(docs, metas))
    if not file_docs:
        yield f"Could not find `{file_path_input}` in the codebase."
        return

    context = "\n\n---\n\n".join(
        f"# {m['file']} lines {m['line_start']}-{m['line_end']} ({m['kind']}: {m['name']})\n{d[:600]}"
        for d, m in file_docs[:6]
    )
    system = """You are a code documentation expert. Explain the file clearly with:
## Purpose
What this file does and why it exists.
## Key Components
Each major function/class with a one-line description.
## How It Fits
How this file connects to the rest of the codebase.
## Entry Points
Where execution starts or how this module is used by others."""

    collected = ""
    for chunk in call_groq_stream(system, f"### File context:\n{context}\n\nExplain this file in detail."):
        collected += chunk
        yield collected

# ══════════════════════════════════════════════════════════════════════════════
#  EVAL
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(pairs_df):
    if get_collection().count() == 0:
        return "No codebase loaded yet."
    rows    = pairs_df.values.tolist()
    correct = 0
    lines   = ["| Query | Expected | Top Retrieved | Pass |",
               "|-------|----------|---------------|------|"]
    for row in rows:
        if len(row) < 2 or not row[0] or not row[1]: continue
        query, expected = str(row[0]), str(row[1]).lower()
        _, metas = retrieve(query, n=6)
        top_file = metas[0]["file"] if metas else "—"
        passed   = "✅" if expected in top_file.lower() else "❌"
        if expected in top_file.lower(): correct += 1
        lines.append(f"| {query[:45]} | `{expected}` | `{top_file}` | {passed} |")
    total = len([r for r in rows if len(r) >= 2 and r[0] and r[1]])
    pct   = 100 * correct // max(total, 1)
    lines.append(f"\n### Score: {correct}/{total} — **{pct}% file citation accuracy**")
    if pct >= 80:   lines.append("> 🟢 Good retrieval quality")
    elif pct >= 60: lines.append("> 🟡 Moderate — consider larger chunk overlap")
    else:           lines.append("> 🔴 Poor — check chunking and embedding model")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  — forces dark everywhere, overrides Gradio internals
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ══ OVERRIDE GRADIO CSS VARIABLES — kills light/dark toggle effect ══ */
:root,
.light, .dark,
[data-theme="light"], [data-theme="dark"],
[class*="theme"] {
    --color-accent: #00ff9c !important;
    --color-accent-soft: #00ff9c20 !important;
    --background-fill-primary: #0a0a0a !important;
    --background-fill-secondary: #0d1117 !important;
    --background-fill-tertiary: #111827 !important;
    --border-color-primary: #1e293b !important;
    --border-color-accent: #00ff9c40 !important;
    --color-text-body: #e2e8f0 !important;
    --color-text-label: #475569 !important;
    --color-text-subdued: #64748b !important;
    --button-primary-background-fill: #00ff9c !important;
    --button-primary-text-color: #0a0a0a !important;
    --button-secondary-background-fill: #0d1117 !important;
    --button-secondary-text-color: #64748b !important;
    --input-background-fill: #0a0a0a !important;
    --input-border-color: #1e293b !important;
    --block-background-fill: #0d1117 !important;
    --block-border-color: #1e293b !important;
    --block-label-background-fill: #0d1117 !important;
    --chatbot-background: #0a0a0a !important;
    --panel-background-fill: #0d1117 !important;
    --table-row-focus: #1e293b !important;
    --checkbox-background-color: #0a0a0a !important;
    --slider-color: #00ff9c !important;
    --loader-color: #00ff9c !important;
    --shadow-drop: none !important;
    --shadow-spread: none !important;
    --radius-sm: 4px !important;
    --radius-md: 6px !important;
    --radius-lg: 8px !important;
}

/* ── Base — full width, no shifts ── */
*, *::before, *::after { box-sizing: border-box; }

html, body {
    background: #0a0a0a !important;
    color: #e2e8f0 !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
    width: 100% !important;
}

/* CRITICAL: lock container to full width — no tab-triggered resizing */
.gradio-container,
.gradio-container > div,
.main,
.wrap,
.app {
    background: #0a0a0a !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* All tab panels same fixed width */
.tabitem, .tab-content, [role="tabpanel"] {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    background: #0a0a0a !important;
}

/* Inner content padding */
.tabitem > .gap,
.tabitem > div > .gap {
    padding: 16px 24px !important;
}

/* ── Header ── */
.cw-header {
    background: linear-gradient(135deg, #0a0a0a 0%, #0f172a 100%) !important;
    border-bottom: 1px solid #00ff9c20 !important;
    padding: 28px 36px 20px !important;
}
.cw-logo {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #00ff9c !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.5px !important;
    text-shadow: 0 0 24px #00ff9c33 !important;
}
.cw-sub {
    color: #475569 !important;
    font-size: 0.82rem !important;
    margin-top: 4px !important;
    font-family: 'Inter', sans-serif !important;
}
.cw-pill {
    display: inline-block !important;
    background: #00ff9c0d !important;
    border: 1px solid #00ff9c25 !important;
    color: #00ff9c !important;
    font-size: 0.68rem !important;
    padding: 2px 9px !important;
    border-radius: 99px !important;
    margin: 8px 4px 0 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Status bar ── */
.cw-statusbar {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    padding: 7px 16px !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #475569 !important;
    margin: 8px 0 !important;
}

/* ── Tabs ── */
.tabs > .tab-nav,
div.tab-nav {
    background: #0a0a0a !important;
    border-bottom: 1px solid #1e293b !important;
    padding: 0 8px !important;
}
.tab-nav > button,
button.selected {
    background: transparent !important;
    color: #475569 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 10px 18px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: color 0.15s, border-color 0.15s !important;
    margin: 0 !important;
}
.tab-nav > button.selected,
button[aria-selected="true"] {
    color: #00ff9c !important;
    border-bottom: 2px solid #00ff9c !important;
    background: #00ff9c08 !important;
}
.tab-nav > button:hover {
    color: #94a3b8 !important;
    background: #ffffff05 !important;
}

/* ── Panels / blocks ── */
.gr-panel,
.gr-box,
.block,
.panel,
div.form,
fieldset,
.gr-form,
.gap,
.contain {
    background: #0d1117 !important;
    border-color: #1e293b !important;
    border-radius: 8px !important;
}

/* ── Inputs ── */
input[type="text"],
input[type="email"],
input[type="password"],
textarea,
.gr-textbox textarea,
.gr-textbox input,
.scroll-hide {
    background: #0a0a0a !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    outline: none !important;
    transition: border-color 0.15s !important;
}
input:focus, textarea:focus {
    border-color: #00ff9c40 !important;
    box-shadow: 0 0 0 2px #00ff9c10 !important;
}
::placeholder {
    color: #334155 !important;
}

/* ── Primary button (Clone & Ingest, Ask, etc) ── */
button.primary,
.gr-button-primary,
button[variant="primary"],
.primary {
    background: #00ff9c !important;
    color: #0a0a0a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    letter-spacing: 0.02em !important;
}
button.primary:hover,
.gr-button-primary:hover {
    background: #00e58a !important;
    box-shadow: 0 0 18px #00ff9c30 !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary button ── */
button.secondary,
.gr-button-secondary,
button[variant="secondary"],
.secondary {
    background: #0d1117 !important;
    color: #64748b !important;
    border: 1px solid #1e293b !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 6px !important;
    padding: 9px 16px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
button.secondary:hover {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}

/* ── Quick-action sidebar buttons ── */
.quick-btn button,
button.quick-btn {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    color: #475569 !important;
    font-size: 0.77rem !important;
    text-align: left !important;
    padding: 8px 12px !important;
    border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    width: 100% !important;
    margin: 3px 0 !important;
    transition: all 0.12s !important;
    cursor: pointer !important;
}
.quick-btn button:hover {
    border-color: #00ff9c30 !important;
    color: #00ff9c !important;
    background: #00ff9c08 !important;
}

/* ── Review focus buttons ── */
.review-focus button {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    color: #64748b !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 8px 14px !important;
    border-radius: 5px !important;
    margin: 4px !important;
    cursor: pointer !important;
    transition: all 0.12s !important;
}
.review-focus button:hover,
.review-focus button.selected {
    background: #00ff9c12 !important;
    border-color: #00ff9c40 !important;
    color: #00ff9c !important;
}

/* ── Chatbot ── */
.gr-chatbot,
.chatbot,
[class*="chatbot"] {
    background: #0a0a0a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
}
.message-wrap .message.user > div,
[class*="user"] > [class*="message"] {
    background: #0f2444 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px 10px 2px 10px !important;
    color: #93c5fd !important;
    font-family: 'Inter', sans-serif !important;
}
.message-wrap .message.bot > div,
[class*="bot"] > [class*="message"] {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    border-radius: 2px 10px 10px 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Code blocks inside chat ── */
.message pre, .prose pre {
    background: #0a0a0a !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    padding: 12px !important;
}
.message code, .prose code {
    background: #1e293b !important;
    color: #00ff9c !important;
    border-radius: 3px !important;
    padding: 1px 5px !important;
    font-size: 0.82em !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Markdown ── */
.gr-markdown, .prose, [class*="markdown"] {
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.65 !important;
}
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3,
.prose h1, .prose h2, .prose h3 {
    color: #00ff9c !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}
.gr-markdown a { color: #38bdf8 !important; }
.gr-markdown code, .prose code {
    background: #1e293b !important;
    color: #00ff9c !important;
    border-radius: 3px !important;
    padding: 1px 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.gr-markdown table { border-collapse: collapse !important; width: 100% !important; }
.gr-markdown th {
    background: #0d1117 !important;
    color: #00ff9c !important;
    border: 1px solid #1e293b !important;
    padding: 8px 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.gr-markdown td {
    border: 1px solid #1e293b !important;
    padding: 7px 12px !important;
    color: #94a3b8 !important;
}

/* ── Labels ── */
label, .gr-label, span.svelte-1gfkn6j {
    color: #475569 !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Dataframe ── */
table { background: #0a0a0a !important; }
th { background: #0d1117 !important; color: #00ff9c !important; border-color: #1e293b !important; }
td { color: #94a3b8 !important; border-color: #1e293b !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; }

/* ── Footer / Gradio branding ── */
footer, .footer, [class*="footer"] {
    background: #0a0a0a !important;
    border-top: 1px solid #1e293b !important;
    color: #1e293b !important;
}
footer a, footer svg { opacity: 0.15 !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Radio/Checkbox ── */
input[type="radio"], input[type="checkbox"] {
    accent-color: #00ff9c !important;
}

/* ── Code output ── */
.gr-code, .code-wrap, [class*="code"] {
    background: #0a0a0a !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Dropdown ── */
select, .gr-dropdown select, ul.options {
    background: #0a0a0a !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
ul.options li:hover { background: #1e293b !important; }

/* ── Progress bar ── */
.progress-bar { background: #00ff9c !important; }
.progress-bar-wrap { background: #1e293b !important; }

/* ── Section dividers ── */
.cw-section {
    border-top: 1px solid #1e293b !important;
    margin: 16px 0 !important;
    padding-top: 16px !important;
}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="CodeWhisperer", css=CSS, theme=gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.slate,
)) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="cw-header">
        <div class="cw-logo">⚡ CodeWhisperer</div>
        <div class="cw-sub">AI codebase intelligence — ask anything, understand everything</div>
        <div>
            <span class="cw-pill">CodeBERT embeddings</span>
            <span class="cw-pill">HyDE retrieval</span>
            <span class="cw-pill">Cross-encoder reranking</span>
            <span class="cw-pill">llama-3.3-70b</span>
            <span class="cw-pill">{DEVICE.upper()}</span>
        </div>
    </div>
    """)

    status_bar = gr.Markdown(
        "*No codebase loaded — go to Load Codebase to get started*",
        elem_classes=["cw-statusbar"]
    )

    with gr.Tabs():

        # ── LOAD ──────────────────────────────────────────────────────────────
        with gr.Tab("📁 Load Codebase"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Tab("GitHub URL"):
                        gh_url  = gr.Textbox(
                            label="Repository URL",
                            placeholder="https://github.com/pallets/flask",
                            lines=1)
                        gh_btn  = gr.Button("Clone & Ingest", variant="primary")
                    with gr.Tab("Local Folder"):
                        local_path = gr.Textbox(
                            label="Folder Path",
                            placeholder="C:\\projects\\myapp  or  /home/user/myapp",
                            lines=1)
                        local_btn  = gr.Button("Ingest Folder", variant="primary")
                    ingest_status = gr.Markdown("*Waiting for codebase...*")

                with gr.Column(scale=2):
                    gr.Markdown("**📂 Files loaded**")
                    file_list_md = gr.Markdown("*None*")

            gr.HTML('<div class="cw-section"></div>')
            gr.Markdown("**💡 Suggested repos**")
            with gr.Row():
                gr.Markdown("`https://github.com/pallets/flask` — clean Python, well-known")
                gr.Markdown("`https://github.com/tiangolo/fastapi` — modern API framework")
                gr.Markdown("`https://github.com/psf/requests` — simple, widely recognised")

            gh_btn.click(clone_and_ingest, inputs=gh_url,
                         outputs=[ingest_status, file_list_md, status_bar])
            local_btn.click(ingest_local, inputs=local_path,
                            outputs=[ingest_status, file_list_md, status_bar])

        # ── ASK ───────────────────────────────────────────────────────────────
        with gr.Tab("💬 Ask"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        height=500, label="", show_label=False, allow_tags=False)
                    with gr.Row():
                        question = gr.Textbox(
                            label="Question",
                            placeholder="Ask anything about the codebase...",
                            lines=2, scale=5)
                        with gr.Column(scale=1, min_width=120):
                            ask_btn   = gr.Button("Ask ↵", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")

                with gr.Column(scale=1, min_width=210):
                    gr.Markdown("**Quick questions**")
                    q1 = gr.Button("→ Main functions & classes",  elem_classes=["quick-btn"])
                    q2 = gr.Button("→ Overall architecture",       elem_classes=["quick-btn"])
                    q3 = gr.Button("→ Authentication flow",        elem_classes=["quick-btn"])
                    q4 = gr.Button("→ API endpoints & routes",     elem_classes=["quick-btn"])
                    q5 = gr.Button("→ Find bugs & risks",          elem_classes=["quick-btn"])
                    q6 = gr.Button("→ Database / models layer",    elem_classes=["quick-btn"])
                    q7 = gr.Button("→ Error handling patterns",    elem_classes=["quick-btn"])
                    q8 = gr.Button("→ Write new endpoint",         elem_classes=["quick-btn"])

            ask_btn.click(answer, [question, chatbot], chatbot)
            question.submit(answer, [question, chatbot], chatbot)
            clear_btn.click(lambda: [], outputs=chatbot)

            def q1_fn(h): yield from answer("What are all the main functions and classes?", h)
            def q2_fn(h): yield from answer("Explain the overall architecture and folder structure in detail.", h)
            def q3_fn(h): yield from answer("Where is authentication or login handled? Show all relevant code.", h)
            def q4_fn(h): yield from answer("What are all the API endpoints or routes? List them all.", h)
            def q5_fn(h): yield from answer("Find potential bugs, security issues, and missing error handling.", h)
            def q6_fn(h): yield from answer("Show the database models and data layer. How is data stored?", h)
            def q7_fn(h): yield from answer("What error handling patterns are used across this codebase?", h)
            def q8_fn(h): yield from answer("Write a new API endpoint following the exact pattern used here.", h)

            q1.click(q1_fn, chatbot, chatbot)
            q2.click(q2_fn, chatbot, chatbot)
            q3.click(q3_fn, chatbot, chatbot)
            q4.click(q4_fn, chatbot, chatbot)
            q5.click(q5_fn, chatbot, chatbot)
            q6.click(q6_fn, chatbot, chatbot)
            q7.click(q7_fn, chatbot, chatbot)
            q8.click(q8_fn, chatbot, chatbot)

        # ── IMPACT ANALYZER ───────────────────────────────────────────────────
        with gr.Tab("🔗 Impact Analyzer"):
            gr.Markdown(
                "**What breaks if I delete or change X?**  "
                "Traces all import dependencies across Python and JS/TS.")
            with gr.Row():
                impact_input = gr.Textbox(
                    label="File or symbol",
                    placeholder="user.py  /  UserModel  /  authenticate  /  utils.js",
                    scale=5)
                impact_btn = gr.Button("Analyze", variant="primary", scale=1)
            impact_output = gr.Markdown()

            def analyze_impact(target):
                if not _dep_graph:
                    return "⚠️ No codebase loaded."
                matches = [f for f in _dep_graph if target.lower() in f.lower()]
                if not matches:
                    _, metas = retrieve(f"definition of {target}", n=3)
                    matches  = list({m["file"] for m in metas})
                    if not matches:
                        return f"Could not find `{target}` in the loaded codebase."
                lines = []
                for m in matches:
                    deps = find_dependents(_dep_graph, m)
                    lines.append(f"### `{m}`")
                    if deps:
                        lines.append(f"**{len(deps)} file(s) will break if deleted:**\n")
                        lines.extend(f"- `{d}`" for d in sorted(deps)[:25])
                        if len(deps) > 25:
                            lines.append(f"- *...and {len(deps)-25} more*")
                    else:
                        lines.append("✅ No other files import this — safe to modify.")
                    lines.append("")
                return "\n".join(lines)

            impact_btn.click(analyze_impact, inputs=impact_input, outputs=impact_output)

        # ── CODE REVIEW ───────────────────────────────────────────────────────
        with gr.Tab("🔍 Code Review"):
            gr.Markdown("**Paste any code. Get an AI review matched to your codebase patterns.**")
            with gr.Row():
                with gr.Column(scale=3):
                    review_input = gr.Textbox(
                        label="Code to review",
                        placeholder="Paste your code here...",
                        lines=16)
                with gr.Column(scale=1, min_width=180):
                    gr.Markdown("**Review focus**")
                    # Four explicit buttons — no radio, no dropdown
                    rev_security = gr.Button("🔒 Security",     variant="secondary")
                    rev_perf     = gr.Button("⚡ Performance",   variant="secondary")
                    rev_quality  = gr.Button("✨ Code Quality",  variant="secondary")
                    rev_bugs     = gr.Button("🐛 Bugs",          variant="secondary")
                    gr.Markdown("*Click a focus then Review*", )
                    review_btn   = gr.Button("Review Code", variant="primary")

            review_output = gr.Markdown()

            # State to track selected focus
            focus_state = gr.State("Code Quality")

            def set_security():  return "Security"
            def set_perf():      return "Performance"
            def set_quality():   return "Code Quality"
            def set_bugs():      return "Bugs"

            rev_security.click(set_security, outputs=focus_state)
            rev_perf.click(set_perf,         outputs=focus_state)
            rev_quality.click(set_quality,   outputs=focus_state)
            rev_bugs.click(set_bugs,         outputs=focus_state)

            review_btn.click(review_code, inputs=[review_input, focus_state], outputs=review_output)

        # ── GENERATE ──────────────────────────────────────────────────────────
        with gr.Tab("⚙️ Generate"):
            gr.Markdown("**Generate code that matches your codebase's exact style and patterns.**")
            with gr.Row():
                gen_desc = gr.Textbox(
                    label="What to generate",
                    placeholder="A middleware that logs all API requests with timestamp and user ID...",
                    lines=4, scale=3)
                with gr.Column(scale=1, min_width=160):
                    gen_lang = gr.Dropdown(
                        ["Auto-detect", "Python", "JavaScript",
                         "TypeScript", "Java", "Go", "Rust"],
                        label="Language", value="Auto-detect")
                    gen_btn = gr.Button("Generate", variant="primary")
            gen_output = gr.Code(label="Generated code", language="python", lines=22)
            gen_btn.click(generate_code, inputs=[gen_desc, gen_lang], outputs=gen_output)

        # ── EXPLAIN FILE ──────────────────────────────────────────────────────
        with gr.Tab("📄 Explain File"):
            gr.Markdown("**Get a full explanation of any file in the loaded codebase.**")
            with gr.Row():
                explain_input = gr.Textbox(
                    label="File name or path",
                    placeholder="app.py  /  src/auth/middleware.ts  /  models/user",
                    scale=5)
                explain_btn = gr.Button("Explain", variant="primary", scale=1)
            explain_output = gr.Markdown()
            explain_btn.click(explain_file, inputs=explain_input, outputs=explain_output)

        # ── EVAL ──────────────────────────────────────────────────────────────
        with gr.Tab("📊 Eval"):
            gr.Markdown(
                "**Benchmark retrieval accuracy.**  "
                "Add query → expected file pairs, run, see % correct.")
            eval_pairs = gr.Dataframe(
                headers=["Query", "Expected file (substring)"],
                datatype=["str", "str"], row_count=6, col_count=(2, "fixed"),
                label="Ground-truth pairs",
                value=[
                    ["What does the Blueprint class do?",    "blueprints"],
                    ["Where are routes defined?",            "routing"],
                    ["How does the app context work?",       "ctx"],
                    ["What does the Request class contain?", "wrappers"],
                    ["Where is CLI handling?",               "cli"],
                    ["How is templating handled?",           "templating"],
                ])
            eval_btn    = gr.Button("Run Benchmark", variant="primary")
            eval_output = gr.Markdown()
            eval_btn.click(run_eval, inputs=eval_pairs, outputs=eval_output)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="text-align:center;padding:14px;border-top:1px solid #1e293b;
                color:#1e293b;font-size:0.72rem;font-family:'JetBrains Mono',monospace;">
        CodeWhisperer v3 &nbsp;·&nbsp; {DEVICE.upper()}
        &nbsp;·&nbsp; llama-3.3-70b via Groq
        &nbsp;·&nbsp; CodeBERT + cross-encoder reranking
    </div>
    """)

demo.launch(share=True, server_name="0.0.0.0")
