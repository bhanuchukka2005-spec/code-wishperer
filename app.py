"""
CodeWhisperer Local — AI codebase assistant v2
- Groq API (llama-3.3-70b) — no VRAM for inference
- HyDE dual-query retrieval
- Cross-encoder re-ranking
- Tree-sitter chunking for JS/TS/Java/Go/Rust
- JS/TS dependency graph
- Streaming responses
- reset_collection() bug fixed
- All Gradio deprecation warnings fixed
"""

import os, re, time, json, subprocess, tempfile
from pathlib import Path

import torch
import gradio as gr
import requests as req
from dotenv import load_dotenv
load_dotenv()
# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # set via env or paste key here
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
        print("[Embedder] Loading st-codesearch-distilroberta-base...")
        _embedder = SentenceTransformer(
            "flax-sentence-embeddings/st-codesearch-distilroberta-base", device=DEVICE)
    return _embedder

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("[Reranker] Loading cross-encoder/ms-marco-MiniLM-L-6-v2...")
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
        yield "Error: GROQ_API_KEY not set."
        return
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
        "max_tokens": 1024, "temperature": 0.2, "stream": True,
    }
    with req.post("https://api.groq.com/openai/v1/chat/completions",
                  headers=headers, json=payload, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue

# ══════════════════════════════════════════════════════════════════════════════
#  PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _try_treesitter_chunks(source: str, filepath: str, lang: str):
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
    if not ts_lang:
        return None
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
                        name = child.text.decode("utf-8")
                        break
                chunks.append({"text": f"# {node.type}: {name}\n{code}", "name": name,
                                "kind": node.type, "file": filepath,
                                "line_start": start + 1, "line_end": end + 1, "docstring": ""})
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return chunks if chunks else None
    except Exception:
        return None

def parse_python_chunks(source: str, filepath: str):
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

def parse_generic_chunks(source: str, filepath: str, chunk_size: int = 40):
    lines, chunks, step = source.splitlines(), [], chunk_size // 2
    for i in range(0, max(1, len(lines) - chunk_size + 1), step):
        block = lines[i: i + chunk_size]
        text  = "\n".join(block).strip()
        if len(text) > 30:
            chunks.append({"text": text, "name": f"block_{i}", "kind": "code_block",
                            "file": filepath, "line_start": i + 1,
                            "line_end": i + len(block), "docstring": ""})
    return chunks

def parse_file(filepath: str, repo_root: str):
    rel = os.path.relpath(filepath, repo_root)
    try:
        source = open(filepath, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return []
    if not source.strip():
        return []
    ext = Path(filepath).suffix.lower()
    if ext == ".py":
        chunks = parse_python_chunks(source, rel)
        return chunks if chunks else parse_generic_chunks(source, rel)
    ts = _try_treesitter_chunks(source, rel, ext)
    return ts if ts else parse_generic_chunks(source, rel)

# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_dependency_graph(repo_root: str):
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

def find_dependents(graph: dict, target_file: str):
    mod       = target_file.replace(os.sep, ".").replace("/", ".").replace(".py", "")
    mod_short = mod.split(".")[-1]
    stem      = Path(target_file).stem
    result    = []
    for f, imports in graph.items():
        for imp in imports:
            if mod_short in imp or mod in imp or stem in imp:
                result.append(f)
                break
    return list(set(result))

# ══════════════════════════════════════════════════════════════════════════════
#  INGEST
# ══════════════════════════════════════════════════════════════════════════════

def ingest_repo(repo_path: str, progress=gr.Progress()):
    global _repo_root, _dep_graph, _file_tree
    _repo_root = repo_path
    col        = get_collection()
    embedder   = get_embedder()

    all_files = [str(p) for p in Path(repo_path).rglob("*")
                 if p.is_file() and p.suffix.lower() in CODE_EXTENSIONS
                 and ".git" not in str(p) and "node_modules" not in str(p)
                 and "__pycache__" not in str(p)]

    if not all_files:
        return "No supported code files found.", []

    _file_tree = [os.path.relpath(f, repo_path) for f in all_files]
    progress(0, desc="Building dependency graph...")
    _dep_graph = build_dependency_graph(repo_path)

    all_chunks = []
    progress(0.1, desc="Parsing files...")
    for i, fp in enumerate(all_files):
        all_chunks.extend(parse_file(fp, repo_path))
        if i % 20 == 0:
            progress(0.1 + 0.4 * i / len(all_files),
                     desc=f"Parsing {os.path.relpath(fp, repo_path)}")

    if not all_chunks:
        return "Parsing produced no chunks.", []

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

    progress(1.0, desc="Done!")
    return (f"Ingested **{len(all_files)} files** → **{len(all_chunks)} chunks**.\n\n"
            f"Languages: {', '.join(sorted({Path(f).suffix for f in all_files}))}\n\n"
            f"Dependency graph: **{len(_dep_graph)} files** mapped."), [[f] for f in _file_tree]

def clone_and_ingest(github_url: str, progress=gr.Progress()):
    global _repo_root
    progress(0, desc="Cloning repo...")
    clone_dir = Path(tempfile.mkdtemp()) / "repo"
    try:
        subprocess.run(["git", "clone", "--depth=1", github_url, str(clone_dir)],
                       check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        return f"Git clone failed: {e.stderr.decode()[:300]}", []
    except FileNotFoundError:
        return "Git not found.", []
    _repo_root = str(clone_dir)
    reset_collection()
    return ingest_repo(str(clone_dir), progress)

def ingest_local(folder_path: str, progress=gr.Progress()):
    global _repo_root
    if not folder_path or not Path(folder_path).exists():
        return "Folder not found.", []
    _repo_root = folder_path
    reset_collection()
    return ingest_repo(folder_path, progress)

# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, n: int = 10):
    col   = get_collection()
    total = col.count()
    if total == 0:
        return [], []
    embedder = get_embedder()
    vecs     = embedder.encode([query, f"code implementation of: {query}"]).tolist()
    seen     = {}
    for vec in vecs:
        res = col.query(query_embeddings=[vec], n_results=min(n, total))
        for doc, meta, cid in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
            if cid not in seen:
                seen[cid] = (doc, meta)
    if not seen:
        return [], []
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
    return "\n".join(f"- {s}" for s in seen)

# ══════════════════════════════════════════════════════════════════════════════
#  ANSWER  — tuple format (works with all Gradio versions)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are CodeWhisperer, an expert AI code assistant. "
    "Answer using ONLY the provided code context. "
    "Always cite exact file names and line numbers. "
    "Be precise and concise. If context is insufficient, say so."
)

def answer(query: str, history: list):
    if get_collection().count() == 0:
        yield history + [[query, "No codebase loaded yet. Please ingest a repo first."]]
        return

    # Impact analysis shortcut
    impact_match = re.search(
        r"(what.*(break|happen|affect).*delete|delete.*what|impact of (deleting|removing))\s+(.+)",
        query, re.IGNORECASE)
    if impact_match and _dep_graph:
        target  = impact_match.group(4).strip().strip("`'\"")
        matches = [f for f in _dep_graph if target.lower() in f.lower()]
        if matches:
            dependents = find_dependents(_dep_graph, matches[0])
            if dependents:
                dep_list = "\n".join(f"- `{d}`" for d in dependents[:15])
                yield history + [[query,
                    f"Deleting **`{matches[0]}`** will directly affect **{len(dependents)} file(s)**:\n\n"
                    f"{dep_list}\n\nThese files import from `{matches[0]}` and will break."]]
                return

    docs, metas = retrieve(query)
    context     = format_context(docs, metas)
    sources     = format_sources(metas)
    user_msg    = f"### Code Context\n{context}\n\n### Question\n{query}"

    collected   = ""
    new_history = history + [[query, ""]]
    for chunk in call_groq_stream(SYSTEM_PROMPT, user_msg):
        collected          += chunk
        new_history[-1][1]  = collected
        yield new_history

    new_history[-1][1] = collected + f"\n\n**Sources:**\n{sources}"
    yield new_history

# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="CodeWhisperer Local", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🔍 CodeWhisperer Local
**AI that understands your entire codebase.** Ask questions, find bugs, generate code — with exact file & line citations.
Powered by: `st-codesearch-distilroberta` embeddings · `ChromaDB` · `llama-3.3-70b` via Groq
""")

    with gr.Tabs():

        with gr.Tab("Load Codebase"):
            gr.Markdown("### Step 1 — Load a codebase")
            with gr.Tab("GitHub URL"):
                gh_url    = gr.Textbox(label="GitHub repo URL", placeholder="https://github.com/pallets/flask")
                gh_btn    = gr.Button("Clone & Ingest", variant="primary")
                gh_status = gr.Markdown("Waiting...")
                gh_tree   = gr.Dataframe(headers=["Files ingested"], label="File tree", interactive=False, wrap=True)
                gh_btn.click(clone_and_ingest, inputs=gh_url, outputs=[gh_status, gh_tree])
            with gr.Tab("Local Folder"):
                local_path   = gr.Textbox(label="Absolute path to folder", placeholder="/content/myproject")
                local_btn    = gr.Button("Ingest Folder", variant="primary")
                local_status = gr.Markdown("Waiting...")
                local_tree   = gr.Dataframe(headers=["Files ingested"], label="File tree", interactive=False, wrap=True)
                local_btn.click(ingest_local, inputs=local_path, outputs=[local_status, local_tree])

        with gr.Tab("Ask the Codebase"):
            gr.Markdown("### Step 2 — Ask anything about the loaded code")
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot  = gr.Chatbot(height=500, label="CodeWhisperer Chat", allow_tags=False)
                    question = gr.Textbox(label="Your question",
                                         placeholder="What does authenticate() do?  /  What breaks if I delete UserModel?")
                    with gr.Row():
                        ask_btn   = gr.Button("Ask", variant="primary")
                        clear_btn = gr.Button("Clear")
                with gr.Column(scale=1):
                    gr.Markdown("**Try these questions:**")
                    q1 = gr.Button("What are all the main functions?")
                    q2 = gr.Button("Explain the overall architecture")
                    q3 = gr.Button("Where is authentication handled?")
                    q4 = gr.Button("What are the API endpoints?")
                    q5 = gr.Button("Find potential bugs")
                    q6 = gr.Button("Write a new endpoint")

            ask_btn.click(answer, [question, chatbot], chatbot)
            question.submit(answer, [question, chatbot], chatbot)
            clear_btn.click(lambda: [], outputs=chatbot)

            def q1_fn(h): yield from answer("What are all the main functions and classes?", h)
            def q2_fn(h): yield from answer("Explain the overall architecture and folder structure.", h)
            def q3_fn(h): yield from answer("Where is authentication or login handled? Show relevant code.", h)
            def q4_fn(h): yield from answer("What are all the API endpoints or routes defined?", h)
            def q5_fn(h): yield from answer("Find potential bugs, missing error handling, or risky code patterns.", h)
            def q6_fn(h): yield from answer("Write a new API endpoint following the exact pattern used in this codebase.", h)

            q1.click(q1_fn, chatbot, chatbot)
            q2.click(q2_fn, chatbot, chatbot)
            q3.click(q3_fn, chatbot, chatbot)
            q4.click(q4_fn, chatbot, chatbot)
            q5.click(q5_fn, chatbot, chatbot)
            q6.click(q6_fn, chatbot, chatbot)

        with gr.Tab("Impact Analyzer"):
            gr.Markdown("### What breaks if I delete or change X?\nWorks for Python and JS/TS imports.")
            impact_input  = gr.Textbox(label="File or symbol name",
                                       placeholder="user.py  or  UserModel  or  utils.js")
            impact_btn    = gr.Button("Analyze Impact", variant="primary")
            impact_output = gr.Markdown()

            def analyze_impact(target):
                if not _dep_graph:
                    return "No codebase loaded yet."
                matches = [f for f in _dep_graph if target.lower() in f.lower()]
                if not matches:
                    _, metas = retrieve(f"definition of {target}", n=3)
                    matches  = list({m["file"] for m in metas})
                    if not matches:
                        return f"Could not find `{target}` in the codebase."
                lines = []
                for m in matches:
                    deps = find_dependents(_dep_graph, m)
                    lines.append(f"### `{m}`")
                    if deps:
                        lines.append(f"**{len(deps)} file(s) depend on this:**")
                        lines.extend(f"- `{d}`" for d in deps[:20])
                    else:
                        lines.append("No other files import this directly.")
                    lines.append("")
                return "\n".join(lines)

            impact_btn.click(analyze_impact, inputs=impact_input, outputs=impact_output)

        with gr.Tab("Eval / Benchmark"):
            gr.Markdown("### Retrieval Benchmark\nTests whether the correct file is retrieved for known queries.")
            eval_pairs = gr.Dataframe(
                headers=["Query", "Expected file (substring)"],
                datatype=["str", "str"], row_count=5, col_count=(2, "fixed"),
                label="Ground-truth pairs",
                value=[["What does the Blueprint class do?", "blueprints"],
                       ["Where are routes defined?",         "routing"],
                       ["How does the app context work?",    "ctx"],
                       ["What does the Request class do?",   "wrappers"],
                       ["Where is CLI handling?",            "cli"]])
            eval_btn    = gr.Button("Run Eval", variant="primary")
            eval_output = gr.Markdown()

            def run_eval(pairs_df):
                if get_collection().count() == 0:
                    return "No codebase loaded yet."
                rows    = pairs_df.values.tolist()
                correct = 0
                lines   = ["| Query | Expected | Top file retrieved | Pass |",
                           "|-------|----------|--------------------|------|"]
                for row in rows:
                    if len(row) < 2 or not row[0] or not row[1]: continue
                    query, expected = str(row[0]), str(row[1]).lower()
                    _, metas = retrieve(query, n=6)
                    top_file = metas[0]["file"] if metas else "—"
                    passed   = "✅" if expected in top_file.lower() else "❌"
                    if expected in top_file.lower(): correct += 1
                    lines.append(f"| {query[:50]} | `{expected}` | `{top_file}` | {passed} |")
                total = len([r for r in rows if len(r) >= 2 and r[0] and r[1]])
                lines.append(f"\n**Score: {correct}/{total} ({100*correct//max(total,1)}%)**")
                return "\n".join(lines)

            eval_btn.click(run_eval, inputs=eval_pairs, outputs=eval_output)

    gr.Markdown(f"> Running on **{DEVICE.upper()}** · LLM: **{GROQ_MODEL}** via Groq · Embeddings: local · DB: ChromaDB")

demo.launch(share=True, server_name="0.0.0.0")