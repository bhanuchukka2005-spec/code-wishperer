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
if not GROQ_API_KEY:
    print("[WARNING] GROQ_API_KEY not set.")

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
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
        "max_tokens": 1500, "temperature": 0.2, "stream": True,
    }
    try:
        with req.post("https://api.groq.com/openai/v1/chat/completions",
                      headers=headers, json=payload, stream=True, timeout=60) as resp:
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
                chunks.append({"text": f"# {node.type}: {name}\n{code}", "name": name,
                                "kind": node.type, "file": filepath,
                                "line_start": start + 1, "line_end": end + 1, "docstring": ""})
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
                chunks.append({"text": f"# {kind}: {node.name}\n{code}", "name": node.name,
                                "kind": kind, "file": filepath, "line_start": start,
                                "line_end": end, "docstring": (pyast.get_docstring(node) or "")[:200]})
    except Exception:
        pass
    return chunks

def parse_generic_chunks(source, filepath, chunk_size=40):
    lines, chunks, step = source.splitlines(), [], chunk_size // 2
    for i in range(0, max(1, len(lines) - chunk_size + 1), step):
        block = lines[i: i + chunk_size]
        text  = "\n".join(block).strip()
        if len(text) > 30:
            chunks.append({"text": text, "name": f"block_{i}", "kind": "code_block",
                            "file": filepath, "line_start": i + 1,
                            "line_end": i + len(block), "docstring": ""})
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
            graph[rel] = [m.group(1) or m.group(2) for m in JS_IMPORT.finditer(src) if m.group(1) or m.group(2)]
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

    all_files = [str(p) for p in Path(repo_path).rglob("*")
                 if p.is_file() and p.suffix.lower() in CODE_EXTENSIONS
                 and ".git" not in str(p) and "node_modules" not in str(p)
                 and "__pycache__" not in str(p)]

    if not all_files:
        return "No supported code files found.", "", "No codebase loaded"

    _file_tree = [os.path.relpath(f, repo_path) for f in all_files]
    progress(0, desc="Building dependency graph...")
    _dep_graph = build_dependency_graph(repo_path)

    all_chunks = []
    for i, fp in enumerate(all_files):
        all_chunks.extend(parse_file(fp, repo_path))
        if i % 20 == 0:
            progress(0.1 + 0.4 * i / len(all_files), desc=f"Parsing {i}/{len(all_files)} files...")

    if not all_chunks:
        return "Parsing produced no chunks.", "", "Parse error"

    progress(0.5, desc=f"Embedding {len(all_chunks)} chunks...")
    texts   = [c["text"][:1500] for c in all_chunks]
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=False).tolist()

    progress(0.85, desc="Storing in ChromaDB...")
    ts    = str(int(time.time() * 1000))
    ids   = [f"{ts}_{i}" for i in range(len(all_chunks))]
    metas = [{"name": c["name"], "kind": c["kind"], "file": c["file"],
               "line_start": c["line_start"], "line_end": c["line_end"],
               "docstring": c["docstring"]} for c in all_chunks]

    for b in range(0, len(all_chunks), 2000):
        col.add(documents=texts[b:b+2000], embeddings=vectors[b:b+2000],
                ids=ids[b:b+2000], metadatas=metas[b:b+2000])

    langs = sorted({Path(f).suffix for f in all_files})
    progress(1.0, desc="Done!")

    stats = (f"### ✅ Codebase loaded\n\n| Metric | Value |\n|--------|-------|\n"
             f"| Files | **{len(all_files)}** |\n| Chunks | **{len(all_chunks)}** |\n"
             f"| Languages | {' '.join(f'`{l}`' for l in langs)} |\n"
             f"| Dep graph | **{len(_dep_graph)}** files mapped |")
    file_md = "\n".join(f"- `{f}`" for f in _file_tree[:60])
    if len(_file_tree) > 60: file_md += f"\n- *...and {len(_file_tree)-60} more*"
    return stats, file_md, f"Loaded: {len(all_files)} files · {len(all_chunks)} chunks"

def clone_and_ingest(github_url, progress=gr.Progress()):
    global _repo_root
    if not github_url.strip():
        return "Please enter a GitHub URL.", "", "No codebase loaded"
    progress(0, desc="Cloning repo...")
    clone_dir = Path(tempfile.mkdtemp()) / "repo"
    try:
        subprocess.run(["git", "clone", "--depth=1", github_url.strip(), str(clone_dir)],
                       check=True, capture_output=True, timeout=180)
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
        parts.append(f"# File: {m['file']}  Lines {m['line_start']}–{m['line_end']}  ({m['kind']}: {m['name']})\n{doc[:800]}")
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
                    f"Deleting this will break **{len(dependents)} file(s)**:\n\n{dep_list}\n\n"
                    f"> These files have direct import dependencies on `{matches[0]}`"]]
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
    "Security":     "Review for security vulnerabilities: SQL injection, XSS, hardcoded secrets, improper auth, CSRF. Show severity and fix for each.",
    "Performance":  "Review for performance issues: N+1 queries, memory leaks, blocking I/O, missing caching. Show impact and fix.",
    "Code Quality": "Review for quality: naming, SOLID violations, DRY, missing error handling, deep nesting. Show fix for each issue.",
    "Bugs":         "Find all bugs, edge cases, and runtime errors. Show exact condition that triggers each and the correct fix.",
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

    user = (f"{'### Codebase patterns:\n' + context[:1500] + chr(10)*2 if context else ''}"
            f"### Code to review:\n```\n{code_input}\n```")
    collected = ""
    for chunk in call_groq_stream(system, user):
        collected += chunk
        yield collected

# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE + EXPLAIN FILE + EVAL
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
Generate clean, production-ready code that follows patterns from the codebase context if provided,
includes proper error handling, and matches the style of existing code."""
    user = (f"{'### Existing codebase patterns:\n' + context[:2000] + chr(10)*2 if context else ''}"
            f"### Generate {language} code for:\n{description}\n\nReturn ONLY the code.")
    collected = ""
    for chunk in call_groq_stream(system, user):
        collected += chunk
        yield collected

def explain_file(file_path_input):
    if not file_path_input.strip():
        yield "Enter a file path or name."
        return
    if get_collection().count() == 0:
        yield "⚠️ No codebase loaded."
        return
    docs, metas = retrieve(f"contents of file {file_path_input}", n=8)
    file_docs   = [(d, m) for d, m in zip(docs, metas) if file_path_input.lower() in m["file"].lower()]
    if not file_docs: file_docs = list(zip(docs, metas))
    if not file_docs:
        yield f"Could not find `{file_path_input}` in the codebase."
        return
    context = "\n\n---\n\n".join(
        f"# {m['file']} lines {m['line_start']}-{m['line_end']} ({m['kind']}: {m['name']})\n{d[:600]}"
        for d, m in file_docs[:6])
    system = """You are a code documentation expert. Explain the file with:
## Purpose — what this file does and why it exists.
## Key Components — each major function/class with a one-line description.
## How It Fits — how this connects to the rest of the codebase.
## Entry Points — where execution starts or how this module is used."""
    collected = ""
    for chunk in call_groq_stream(system, f"### File context:\n{context}\n\nExplain in detail."):
        collected += chunk
        yield collected

def auto_generate_eval_pairs():
    """
    Build ground-truth eval pairs from the actual loaded codebase.
    Samples real functions/classes and generates sensible queries for them.
    """
    if get_collection().count() == 0:
        return [["No codebase loaded — ingest a repo first", ""]] * 6

    col    = get_collection()
    total  = col.count()
    sample = col.get(limit=min(total, 300))
    metas  = sample["metadatas"]

    # Prefer named functions and classes over generic blocks
    named = [m for m in metas if m.get("kind") in (
        "function", "class", "function_declaration", "function_definition",
        "method_definition", "class_declaration", "class_definition")
        and m.get("name") and m["name"] not in ("anonymous", "__init__")
        and len(m.get("name", "")) > 2]

    if not named:
        named = [m for m in metas if m.get("file")]

    # One entry per file for diversity
    seen_files, picked = set(), []
    for m in named:
        f = m.get("file", "")
        if f not in seen_files:
            picked.append(m)
            seen_files.add(f)
        if len(picked) >= 8:
            break
    if len(picked) < 4:
        picked = (named + metas)[:8]

    def make_query(m):
        name = m.get("name", "unknown")
        kind = m.get("kind", "")
        if "class" in kind.lower():
            return f"What does the {name} class do?"
        elif "function" in kind.lower() or kind == "function":
            return f"What does the {name} function do and where is it defined?"
        else:
            return f"Where is {name} defined and what does it do?"

    rows = [[make_query(m), Path(m.get("file", "")).stem] for m in picked[:8]]
    while len(rows) < 6:
        rows.append(["", ""])
    return rows


def run_eval(pairs_df):
    if get_collection().count() == 0:
        return "No codebase loaded yet."
    rows = pairs_df.values.tolist()
    rows = [r for r in rows if len(r) >= 2 and str(r[0]).strip() and str(r[1]).strip()]

    if not rows:
        return "⚠️ No pairs to evaluate. Click **Auto-generate pairs** first, or add your own rows."

    correct = 0
    lines   = ["| Query | Expected file | Top retrieved | Pass |",
               "|-------|--------------|---------------|------|"]
    for row in rows:
        query, expected = str(row[0]).strip(), str(row[1]).strip().lower()
        _, metas = retrieve(query, n=6)
        top_file = metas[0]["file"] if metas else "—"
        passed   = "✅" if expected in top_file.lower() else "❌"
        if expected in top_file.lower(): correct += 1
        lines.append(f"| {query[:45]} | `{expected}` | `{top_file}` | {passed} |")

    total = len(rows)
    pct   = 100 * correct // max(total, 1)
    lines.append(f"\n### Score: {correct}/{total} — **{pct}% file citation accuracy**")
    if pct >= 80:   lines.append("> 🟢 Good retrieval quality")
    elif pct >= 60: lines.append("> 🟡 Moderate — try increasing chunk overlap")
    else:           lines.append("> 🔴 Poor — check that the correct codebase is loaded")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

@keyframes float-3d{0%,100%{transform:translateY(0) rotateX(0deg)}50%{transform:translateY(-8px) rotateX(1.5deg)}}
@keyframes pulse-neon{0%,100%{box-shadow:0 0 5px #00ff9c10,0 0 20px #00ff9c05;border-color:#00ff9c20}50%{box-shadow:0 0 15px #00ff9c25,0 0 40px #00ff9c10;border-color:#00ff9c40}}
@keyframes glitch{0%,88%,100%{text-shadow:0 0 20px #00ff9c40,0 0 40px #00ff9c15;transform:translate(0)}90%{text-shadow:-3px 0 #ff006e,3px 0 #00f0ff;transform:translate(-2px,1px)}93%{text-shadow:3px 0 #ff006e,-3px 0 #00f0ff;transform:translate(2px,-1px)}96%{text-shadow:-1px 0 #ff006e,1px 0 #00f0ff;transform:translate(0)}}
@keyframes fadeInUp{from{opacity:0;transform:translateY(25px) scale(0.97)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes shimmer{0%{background-position:-200% center}100%{background-position:200% center}}
@keyframes scanLine{0%{top:-10%}100%{top:110%}}
@keyframes breathe{0%,100%{opacity:0.3;transform:scale(1)}50%{opacity:0.7;transform:scale(1.08)}}
@keyframes pill-shimmer{0%{background-position:-100% 0}100%{background-position:200% 0}}

/* ══ OVERRIDE ALL GRADIO CSS VARIABLES — theme toggle does nothing ══ */
:root,.light,.dark,[data-theme],[data-theme="light"],[data-theme="dark"],[class*="theme"]{
    --color-accent:#00ff9c!important;
    --background-fill-primary:#050508!important;
    --background-fill-secondary:#0a0e1a!important;
    --background-fill-tertiary:#111827!important;
    --border-color-primary:#1a1f35!important;
    --color-text-body:#e2e8f0!important;
    --color-text-label:#475569!important;
    --button-primary-background-fill:#00ff9c!important;
    --button-primary-text-color:#050508!important;
    --button-secondary-background-fill:#0a0e1a!important;
    --button-secondary-text-color:#64748b!important;
    --input-background-fill:#050508!important;
    --input-border-color:#1a1f35!important;
    --block-background-fill:#0a0e1a!important;
    --block-border-color:#1a1f35!important;
    --panel-background-fill:#0a0e1a!important;
    --chatbot-background:#050508!important;
    --shadow-drop:none!important;
    --radius-sm:6px!important;--radius-md:10px!important;--radius-lg:14px!important;
}

/* ══ Base — locked full width ══ */
*,*::before,*::after{box-sizing:border-box}
html,body{background:#050508!important;color:#e2e8f0!important;margin:0!important;padding:0!important;width:100%!important;overflow-x:hidden!important}
.gradio-container,.gradio-container>div,.main,.wrap,.app{
    background:#050508!important;
    width:100%!important;max-width:100%!important;min-width:100%!important;
    margin:0!important;padding:0!important;
}
/* CRITICAL: every tab panel same size — prevents resize on tab switch */
.tabitem,.tab-content,[role="tabpanel"]{
    width:100%!important;min-width:100%!important;max-width:100%!important;
    background:#050508!important;
    animation:fadeInUp 0.4s cubic-bezier(0.16,1,0.3,1) both!important;
}
.tabitem>.gap,.tabitem>div>.gap{padding:20px 28px!important}

/* ══ Header ══ */
.cw-header{position:relative!important;background:linear-gradient(160deg,#050508 0%,#0a1628 40%,#100a22 70%,#050508 100%)!important;border-bottom:1px solid #00ff9c25!important;padding:36px 40px 28px!important;overflow:hidden!important}
.cw-header::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#00ff9c,transparent);opacity:0.4;animation:scanLine 4s linear infinite;pointer-events:none}
.cw-header::after{content:'';position:absolute;top:-40%;right:-5%;width:350px;height:350px;background:radial-gradient(circle,#00ff9c08 0%,transparent 70%);border-radius:50%;animation:breathe 6s ease-in-out infinite;pointer-events:none}
.cw-logo{font-size:2rem!important;font-weight:700!important;color:#00ff9c!important;font-family:'JetBrains Mono',monospace!important;letter-spacing:-0.5px!important;animation:glitch 8s ease-in-out infinite,float-3d 6s ease-in-out infinite!important;display:inline-block!important;position:relative;z-index:1;text-shadow:0 0 20px #00ff9c40}
.cw-sub{color:#475569!important;font-size:0.85rem!important;margin-top:6px!important;font-family:'Inter',sans-serif!important;position:relative;z-index:1}
.cw-pill{display:inline-block!important;background:linear-gradient(90deg,#00ff9c08,#00ff9c18,#00ff9c08)!important;background-size:200% 100%!important;border:1px solid #00ff9c25!important;color:#00ff9c90!important;font-size:0.7rem!important;padding:3px 12px!important;border-radius:99px!important;margin:10px 5px 0 0!important;font-family:'JetBrains Mono',monospace!important;animation:pill-shimmer 3s ease-in-out infinite!important;transition:all 0.3s!important;position:relative;z-index:1}
.cw-pill:hover{border-color:#00ff9c!important;color:#00ff9c!important;box-shadow:0 0 15px #00ff9c20!important;transform:translateY(-2px)!important}

/* ══ Status bar ══ */
.cw-statusbar{background:rgba(10,14,26,0.7)!important;backdrop-filter:blur(12px)!important;border:1px solid #1a1f35!important;border-radius:10px!important;padding:9px 18px!important;font-size:0.76rem!important;font-family:'JetBrains Mono',monospace!important;color:#475569!important;margin:10px 24px!important;animation:pulse-neon 4s ease-in-out infinite!important}

/* ══ Tabs ══ */
.tabs>.tab-nav,div.tab-nav{background:#050508!important;border-bottom:1px solid #1a1f35!important;padding:0 12px!important}
.tab-nav>button{background:transparent!important;color:#475569!important;font-family:'JetBrains Mono',monospace!important;font-size:0.82rem!important;padding:12px 20px!important;border:none!important;border-bottom:2px solid transparent!important;border-radius:0!important;transition:all 0.3s!important;margin:0!important}
.tab-nav>button.selected,button[aria-selected="true"]{color:#00ff9c!important;border-bottom:2px solid #00ff9c!important;background:linear-gradient(to top,#00ff9c08,transparent)!important;text-shadow:0 0 15px #00ff9c30!important}
.tab-nav>button:hover{color:#94a3b8!important;background:#ffffff05!important}

/* ══ Panels ══ */
.gr-panel,.gr-box,.block,.panel,div.form,fieldset,.gr-form,.gap,.contain{background:rgba(10,14,26,0.7)!important;backdrop-filter:blur(12px)!important;border:1px solid #1a1f35!important;border-radius:12px!important}

/* ══ Inputs ══ */
input[type="text"],input[type="email"],input[type="password"],textarea,.gr-textbox textarea,.gr-textbox input{background:#050508!important;border:1px solid #1a1f35!important;color:#e2e8f0!important;font-family:'JetBrains Mono',monospace!important;font-size:0.85rem!important;border-radius:8px!important;padding:11px 15px!important;outline:none!important;transition:all 0.3s!important}
input:focus,textarea:focus{border-color:#00ff9c50!important;box-shadow:0 0 0 3px #00ff9c10,0 0 20px #00ff9c08!important}
::placeholder{color:#334155!important}

/* ══ Primary button ══ */
button.primary,.gr-button-primary,button[variant="primary"]{background:linear-gradient(135deg,#00ff9c,#00e68a)!important;color:#050508!important;font-family:'JetBrains Mono',monospace!important;font-weight:700!important;font-size:0.85rem!important;border:none!important;border-radius:8px!important;padding:11px 22px!important;cursor:pointer!important;transition:all 0.3s!important;box-shadow:0 2px 10px #00ff9c20!important}
button.primary:hover,.gr-button-primary:hover{box-shadow:0 4px 25px #00ff9c35!important;transform:translateY(-2px) scale(1.02)!important}
button.primary:active{transform:translateY(0) scale(0.98)!important}

/* ══ Secondary button ══ */
button.secondary,.gr-button-secondary,button[variant="secondary"]{background:rgba(10,14,26,0.8)!important;color:#64748b!important;border:1px solid #1a1f35!important;font-family:'JetBrains Mono',monospace!important;font-size:0.82rem!important;border-radius:8px!important;padding:10px 18px!important;cursor:pointer!important;transition:all 0.3s!important}
button.secondary:hover{background:#1a1f35!important;color:#e2e8f0!important;border-color:#00ff9c30!important;transform:translateY(-1px)!important}

/* ══ Quick buttons ══ */
.quick-btn button{background:rgba(10,14,26,0.6)!important;border:1px solid #1a1f35!important;color:#475569!important;font-size:0.77rem!important;text-align:left!important;padding:9px 13px!important;border-radius:8px!important;font-family:'JetBrains Mono',monospace!important;width:100%!important;margin:3px 0!important;transition:all 0.3s!important;cursor:pointer!important}
.quick-btn button:hover{border-color:#00ff9c35!important;color:#00ff9c!important;background:rgba(0,255,156,0.05)!important;transform:translateX(4px)!important}

/* ══ Chatbot ══ */
.gr-chatbot,.chatbot{background:#050508!important;border:1px solid #1a1f35!important;border-radius:12px!important}
.message-wrap .message.user>div{background:linear-gradient(135deg,#0f2444,#0a1e3d)!important;border:1px solid #1e3a5f80!important;border-radius:12px 12px 3px 12px!important;color:#93c5fd!important}
.message-wrap .message.bot>div{background:rgba(10,14,26,0.8)!important;border:1px solid #1a1f35!important;border-radius:3px 12px 12px 12px!important;color:#e2e8f0!important;backdrop-filter:blur(8px)!important}
.message pre,.prose pre{background:#050508!important;border:1px solid #1a1f35!important;border-radius:8px!important;padding:14px!important}
.message code,.prose code{background:#1a1f35!important;color:#00ff9c!important;border-radius:4px!important;padding:2px 6px!important;font-family:'JetBrains Mono',monospace!important}

/* ══ Markdown ══ */
.gr-markdown,.prose{color:#94a3b8!important;font-family:'Inter',sans-serif!important;line-height:1.7!important}
.gr-markdown h1,.gr-markdown h2,.gr-markdown h3{color:#00ff9c!important;font-family:'JetBrains Mono',monospace!important;font-weight:600!important}
.gr-markdown a{color:#38bdf8!important}
.gr-markdown code,.prose code{background:#1a1f35!important;color:#00ff9c!important;border-radius:4px!important;padding:2px 6px!important;font-family:'JetBrains Mono',monospace!important}
.gr-markdown table{border-collapse:collapse!important;width:100%!important}
.gr-markdown th{background:#0a0e1a!important;color:#00ff9c!important;border:1px solid #1a1f35!important;padding:9px 13px!important;font-family:'JetBrains Mono',monospace!important}
.gr-markdown td{border:1px solid #1a1f35!important;padding:8px 13px!important;color:#94a3b8!important}

/* ══ Labels ══ */
label,.gr-label{color:#475569!important;font-size:0.75rem!important;font-family:'JetBrains Mono',monospace!important;text-transform:uppercase!important;letter-spacing:0.06em!important}

/* ══ Table ══ */
table{background:#050508!important}
th{background:#0a0e1a!important;color:#00ff9c!important;border-color:#1a1f35!important}
td{color:#94a3b8!important;border-color:#1a1f35!important;font-family:'JetBrains Mono',monospace!important;font-size:0.8rem!important}

/* ══ Dropdown ══ */
select{background:#050508!important;border:1px solid #1a1f35!important;color:#e2e8f0!important;border-radius:8px!important;font-family:'JetBrains Mono',monospace!important}

/* ══ Code output ══ */
.gr-code{background:#050508!important;border:1px solid #1a1f35!important;border-radius:8px!important;color:#e2e8f0!important;font-family:'JetBrains Mono',monospace!important}

/* ══ Progress bar ══ */
.progress-bar{background:linear-gradient(90deg,#00ff9c,#00f0ff,#00ff9c)!important;background-size:200% 100%!important;animation:shimmer 2s linear infinite!important}
.progress-bar-wrap{background:#1a1f35!important;border-radius:99px!important;overflow:hidden!important}

/* ══ Footer ══ */
footer,.footer{background:#050508!important;border-top:1px solid #1a1f35!important}
footer a,footer svg{opacity:0.15!important}

/* ══ Scrollbars ══ */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#050508}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,#1a1f35,#00ff9c30);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,#1a1f35,#00ff9c60)}

/* ══ Particle canvas ══ */
#cw-particles{position:fixed!important;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;opacity:0.5}
.gradio-container{position:relative;z-index:1}
#cw-mouse-glow{position:fixed;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,#00ff9c08 0%,transparent 70%);pointer-events:none;z-index:0;transform:translate(-50%,-50%);transition:left 0.08s,top 0.08s;opacity:0}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  UI  — css and theme go in gr.Blocks(), NOT launch()
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="CodeWhisperer",
    css=CSS,                          # ← correct: css here
    theme=gr.themes.Base(             # ← correct: theme here
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
    )
) as demo:

    # ── Header + particles ────────────────────────────────────────────────────
    gr.HTML(f"""
    <canvas id="cw-particles"></canvas>
    <div id="cw-mouse-glow"></div>
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
    <script>
    (function(){{
        const c=document.getElementById('cw-particles');
        if(!c)return;
        const ctx=c.getContext('2d');
        let w,h;
        function resize(){{w=c.width=window.innerWidth;h=c.height=window.innerHeight}}
        resize();window.addEventListener('resize',resize);
        const pts=[];
        for(let i=0;i<60;i++)pts.push({{x:Math.random()*w,y:Math.random()*h,vx:(Math.random()-0.5)*0.4,vy:(Math.random()-0.5)*0.4,r:Math.random()*1.5+0.5}});
        function draw(){{
            ctx.clearRect(0,0,w,h);
            for(let i=0;i<pts.length;i++){{
                const p=pts[i];p.x+=p.vx;p.y+=p.vy;
                if(p.x<0||p.x>w)p.vx*=-1;if(p.y<0||p.y>h)p.vy*=-1;
                ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
                ctx.fillStyle='rgba(0,255,156,0.25)';ctx.fill();
                for(let j=i+1;j<pts.length;j++){{
                    const q=pts[j],dx=p.x-q.x,dy=p.y-q.y,d=Math.sqrt(dx*dx+dy*dy);
                    if(d<120){{ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);
                        ctx.strokeStyle='rgba(0,255,156,'+(0.08*(1-d/120))+')';ctx.lineWidth=0.5;ctx.stroke();}}
                }}
            }}
            requestAnimationFrame(draw);
        }}
        draw();
        const glow=document.getElementById('cw-mouse-glow');
        if(glow){{
            document.addEventListener('mousemove',function(e){{glow.style.left=e.clientX+'px';glow.style.top=e.clientY+'px';glow.style.opacity='1'}});
            document.addEventListener('mouseleave',function(){{glow.style.opacity='0'}});
        }}
    }})();
    </script>
    """)

    status_bar = gr.Markdown(
        "*No codebase loaded — go to Load Codebase to get started*",
        elem_classes=["cw-statusbar"])

    with gr.Tabs():

        # ── LOAD ──────────────────────────────────────────────────────────────
        with gr.Tab("📁 Load Codebase"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Tab("GitHub URL"):
                        gh_url  = gr.Textbox(label="Repository URL",
                                             placeholder="https://github.com/pallets/flask", lines=1)
                        gh_btn  = gr.Button("Clone & Ingest", variant="primary")
                    with gr.Tab("Local Folder"):
                        local_path = gr.Textbox(label="Folder Path",
                                                placeholder="C:\\projects\\myapp  or  /home/user/myapp", lines=1)
                        local_btn  = gr.Button("Ingest Folder", variant="primary")
                    ingest_status = gr.Markdown("*Waiting for codebase...*")
                with gr.Column(scale=2):
                    gr.Markdown("**📂 Files loaded**")
                    file_list_md = gr.Markdown("*None*")

            gr.HTML('<div style="border-top:1px solid #1a1f3580;margin:18px 0;padding-top:18px"></div>')
            gr.Markdown("**💡 Suggested repos**")
            with gr.Row():
                gr.Markdown("`https://github.com/pallets/flask` — clean Python, well-known")
                gr.Markdown("`https://github.com/tiangolo/fastapi` — modern API framework")
                gr.Markdown("`https://github.com/psf/requests` — simple, widely recognised")

            gh_btn.click(clone_and_ingest, inputs=gh_url, outputs=[ingest_status, file_list_md, status_bar])
            local_btn.click(ingest_local, inputs=local_path, outputs=[ingest_status, file_list_md, status_bar])

        # ── ASK ───────────────────────────────────────────────────────────────
        with gr.Tab("💬 Ask"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(height=500, label="", show_label=False, allow_tags=False)
                    with gr.Row():
                        question = gr.Textbox(label="Question",
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

        # ── IMPACT ────────────────────────────────────────────────────────────
        with gr.Tab("🔗 Impact Analyzer"):
            gr.Markdown("**What breaks if I delete or change X?** Traces all import dependencies.")
            with gr.Row():
                impact_input = gr.Textbox(label="File or symbol",
                                          placeholder="user.py  /  UserModel  /  authenticate  /  utils.js",
                                          scale=5)
                impact_btn = gr.Button("Analyze", variant="primary", scale=1)
            impact_output = gr.Markdown()

            def analyze_impact(target):
                if not _dep_graph: return "⚠️ No codebase loaded."
                matches = [f for f in _dep_graph if target.lower() in f.lower()]
                if not matches:
                    _, metas = retrieve(f"definition of {target}", n=3)
                    matches  = list({m["file"] for m in metas})
                    if not matches: return f"Could not find `{target}` in the loaded codebase."
                lines = []
                for m in matches:
                    deps = find_dependents(_dep_graph, m)
                    lines.append(f"### `{m}`")
                    if deps:
                        lines.append(f"**{len(deps)} file(s) will break if deleted:**\n")
                        lines.extend(f"- `{d}`" for d in sorted(deps)[:25])
                        if len(deps) > 25: lines.append(f"- *...and {len(deps)-25} more*")
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
                    review_input = gr.Textbox(label="Code to review",
                                              placeholder="Paste your code here...", lines=16)
                with gr.Column(scale=1, min_width=180):
                    gr.Markdown("**Review focus**")
                    rev_security = gr.Button("🔒 Security",    variant="secondary")
                    rev_perf     = gr.Button("⚡ Performance",  variant="secondary")
                    rev_quality  = gr.Button("✨ Code Quality", variant="secondary")
                    rev_bugs     = gr.Button("🐛 Bugs",         variant="secondary")
                    gr.Markdown("*Click a focus, then Review*")
                    review_btn   = gr.Button("Review Code", variant="primary")

            review_output = gr.Markdown()

            # ── State defined BEFORE wiring ──
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
                gen_desc = gr.Textbox(label="What to generate",
                                      placeholder="A middleware that logs all API requests...",
                                      lines=4, scale=3)
                with gr.Column(scale=1, min_width=160):
                    gen_lang = gr.Dropdown(
                        ["Auto-detect", "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust"],
                        label="Language", value="Auto-detect")
                    gen_btn = gr.Button("Generate", variant="primary")
            gen_output = gr.Code(label="Generated code", language="python", lines=22)
            gen_btn.click(generate_code, inputs=[gen_desc, gen_lang], outputs=gen_output)

        # ── EXPLAIN FILE ──────────────────────────────────────────────────────
        with gr.Tab("📄 Explain File"):
            gr.Markdown("**Get a full explanation of any file in the loaded codebase.**")
            with gr.Row():
                explain_input = gr.Textbox(label="File name or path",
                                           placeholder="app.py  /  src/auth/middleware.ts",
                                           scale=5)
                explain_btn = gr.Button("Explain", variant="primary", scale=1)
            explain_output = gr.Markdown()
            explain_btn.click(explain_file, inputs=explain_input, outputs=explain_output)

        # ── EVAL ──────────────────────────────────────────────────────────────
        with gr.Tab("📊 Eval"):
            gr.Markdown("""**Benchmark retrieval accuracy against the loaded codebase.**
Click **Auto-generate pairs** to create test cases from your actual loaded repo,
or manually edit the table. Then click **Run Benchmark**.""")

            with gr.Row():
                autogen_btn = gr.Button("⚡ Auto-generate pairs", variant="secondary")
                eval_btn    = gr.Button("Run Benchmark", variant="primary")

            eval_pairs = gr.Dataframe(
                headers=["Query", "Expected file (stem)"],
                datatype=["str", "str"],
                row_count=8,
                col_count=(2, "fixed"),
                label="Ground-truth pairs — edit or auto-generate",
                value=[["Load a codebase first, then click Auto-generate pairs", ""]] * 6
            )
            eval_output = gr.Markdown()

            autogen_btn.click(auto_generate_eval_pairs, outputs=eval_pairs)
            eval_btn.click(run_eval, inputs=eval_pairs, outputs=eval_output)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="text-align:center;padding:18px;border-top:1px solid #1a1f35;
                background:linear-gradient(180deg,#050508,#0a0e1a);
                color:#334155;font-size:0.72rem;font-family:'JetBrains Mono',monospace;position:relative;">
        <div style="position:absolute;top:0;left:0;right:0;height:1px;
                    background:linear-gradient(90deg,transparent,#00ff9c30,transparent);"></div>
        CodeWhisperer v3
        <span style="color:#00ff9c40;margin:0 8px">◆</span> {DEVICE.upper()}
        <span style="color:#00ff9c40;margin:0 8px">◆</span> llama-3.3-70b via Groq
        <span style="color:#00ff9c40;margin:0 8px">◆</span> CodeBERT + cross-encoder reranking
    </div>
    """)

# ── Launch — NO css or theme here ────────────────────────────────────────────
demo.launch(share=True, server_name="0.0.0.0")
