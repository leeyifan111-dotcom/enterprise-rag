"""
backend.py — FastAPI 后端

POST /chat             → 问答
POST /upload           → 上传文件到指定分类
GET  /categories       → 列出所有分类
GET  /stats            → 各分类索引统计

── 会话管理 ──
GET    /sessions              → 列出所有会话
POST   /sessions              → 创建新会话
GET    /sessions/{id}         → 获取单个会话
PUT    /sessions/{id}         → 更新会话名称/config
DELETE /sessions/{id}         → 删除会话
POST   /sessions/{id}/clear   → 清空消息
DELETE /sessions/{id}/messages/{idx} → 删除指定消息

── 文档管理 ──
GET    /documents/{category}              → 列出文件
GET    /documents/{category}/{filename}   → 读取文件内容
PUT    /documents/{category}/{filename}   → 保存/编辑文件 + 重建索引
DELETE /documents/{category}/{filename}   → 删除文件 + 清理索引
POST   /documents/reindex                 → 全量重建索引
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core import (
    ask, build_index, classify, delete_doc_chunks, remove_category_index,
    CATEGORIES, CATEGORY_MAP, CHROMA_DIR, RAGConfig, replace, _config,
)
import os
import json
import uuid
import shutil
from datetime import datetime, timezone

app = FastAPI(title="Enterprise RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)


# ── 辅助函数 ──────────────────────────────────────────────

def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _load_session(session_id: str) -> dict | None:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_session(session: dict):
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(_session_path(session["id"]), "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


# ── 请求模型 ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    category: str | None = None
    search_top_k: int | None = None
    max_turns: int | None = None


class SessionCreate(BaseModel):
    name: str = "新会话"


class SessionUpdate(BaseModel):
    name: str | None = None
    system_prompt: str | None = None
    search_top_k: int | None = None
    max_turns: int | None = None


class DocSaveRequest(BaseModel):
    content: str


# ── 问答 ──────────────────────────────────────────────────

@app.post("/chat")
def chat(req: ChatRequest):
    # 加载会话配置
    session = None
    if req.session_id:
        session = _load_session(req.session_id)

    # 构建 config 覆盖
    overrides = {}
    if req.search_top_k is not None:
        overrides["search_top_k"] = req.search_top_k
    elif session and session.get("config", {}).get("search_top_k"):
        overrides["search_top_k"] = session["config"]["search_top_k"]
    if req.max_turns is not None:
        overrides["agent_max_turns"] = req.max_turns
    elif session and session.get("config", {}).get("max_turns"):
        overrides["agent_max_turns"] = session["config"]["max_turns"]
    cfg = replace(_config, **overrides) if overrides else _config

    # 自定义 system_prompt
    system_prompt = None
    if session and session.get("config", {}).get("system_prompt"):
        system_prompt = session["config"]["system_prompt"]

    result = ask(req.query, req.category, config=cfg, system_prompt=system_prompt)

    # 保存消息到会话
    if session:
        now = datetime.now(timezone.utc).isoformat()
        session["messages"].append({"role": "user", "content": req.query, "timestamp": now})
        session["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "category": result.get("category", "-"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        _save_session(session)

    return {
        **result,
        "session_id": req.session_id,
    }


# ── 上传 ──────────────────────────────────────────────────

@app.post("/upload")
def upload(file: UploadFile = File(...), category: str = Form(...),
           chunk_size: int | None = Form(None), chunk_overlap: int | None = Form(None)):
    if category not in CATEGORIES:
        return {"error": f"分类 {category} 不存在，可选: {list(CATEGORY_MAP.values())}"}

    cat_dir = os.path.join("knowledge", CATEGORY_MAP[category])
    os.makedirs(cat_dir, exist_ok=True)

    filepath = os.path.join(cat_dir, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    stats = build_index("knowledge", category, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return {"status": "ok", "file": file.filename, "category": CATEGORY_MAP[category], "chunks": stats.get(category, 0)}


# ── 分类 / 统计 ───────────────────────────────────────────

@app.get("/categories")
def list_categories():
    return {"categories": list(CATEGORY_MAP.values())}


@app.get("/stats")
def stats():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    result = {}
    for cat in CATEGORIES:
        try:
            col = client.get_collection(f"rag_{cat}")
            result[CATEGORY_MAP[cat]] = col.count()
        except Exception:
            result[CATEGORY_MAP[cat]] = 0
    return result


# ═══════════════════════════════════════════════════════════
# 会话管理
# ═══════════════════════════════════════════════════════════

@app.get("/sessions")
def list_sessions():
    sessions = []
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(SESSIONS_DIR, fname), "r", encoding="utf-8") as f:
            s = json.load(f)
        sessions.append({
            "id": s["id"],
            "name": s["name"],
            "created_at": s.get("created_at", ""),
            "updated_at": s.get("updated_at", ""),
            "msg_count": len(s.get("messages", [])),
        })
    return {"sessions": sessions}


@app.post("/sessions")
def create_session(req: SessionCreate):
    session = {
        "id": uuid.uuid4().hex[:12],
        "name": req.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "system_prompt": "",
            "search_top_k": 5,
            "max_turns": 6,
        },
        "messages": [],
    }
    _save_session(session)
    return session


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = _load_session(session_id)
    if not session:
        return {"error": "会话不存在"}, 404
    return session


@app.put("/sessions/{session_id}")
def update_session(session_id: str, req: SessionUpdate):
    session = _load_session(session_id)
    if not session:
        return {"error": "会话不存在"}, 404
    if req.name is not None:
        session["name"] = req.name
    config = session.setdefault("config", {})
    if req.system_prompt is not None:
        config["system_prompt"] = req.system_prompt
    if req.search_top_k is not None:
        config["search_top_k"] = req.search_top_k
    if req.max_turns is not None:
        config["max_turns"] = req.max_turns
    _save_session(session)
    return session


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted"}
    return {"error": "会话不存在"}, 404


@app.post("/sessions/{session_id}/clear")
def clear_session_messages(session_id: str):
    session = _load_session(session_id)
    if not session:
        return {"error": "会话不存在"}, 404
    session["messages"] = []
    _save_session(session)
    return {"status": "cleared"}


@app.delete("/sessions/{session_id}/messages/{idx}")
def delete_session_message(session_id: str, idx: int):
    session = _load_session(session_id)
    if not session:
        return {"error": "会话不存在"}, 404
    if 0 <= idx < len(session["messages"]):
        session["messages"].pop(idx)
        _save_session(session)
        return {"status": "deleted"}
    return {"error": "下标越界"}, 400


# ═══════════════════════════════════════════════════════════
# 文档管理
# ═══════════════════════════════════════════════════════════

@app.get("/documents/{category}")
def list_documents(category: str):
    """列出该分类下所有文件（名称、大小、行数）"""
    if category not in CATEGORIES:
        return {"error": f"分类 {category} 不存在"}, 404

    cat_dir = os.path.join("knowledge", CATEGORY_MAP[category])
    if not os.path.isdir(cat_dir):
        return {"category": CATEGORY_MAP[category], "files": []}

    files = []
    for fname in sorted(os.listdir(cat_dir)):
        if not fname.endswith((".md", ".txt")):
            continue
        fpath = os.path.join(cat_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        size = os.path.getsize(fpath)
        files.append({"name": fname, "size": size, "lines": lines})

    return {"category": CATEGORY_MAP[category], "key": category, "files": files}


@app.get("/documents/{category}/{filename}")
def read_document(category: str, filename: str):
    """读取文件内容"""
    if category not in CATEGORIES:
        return {"error": f"分类 {category} 不存在"}, 404

    cat_dir = os.path.join("knowledge", CATEGORY_MAP[category])
    fpath = os.path.join(cat_dir, filename)
    if not os.path.exists(fpath):
        return {"error": "文件不存在"}, 404

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    return {"category": CATEGORY_MAP[category], "filename": filename, "content": content, "size": len(content)}


@app.put("/documents/{category}/{filename}")
def save_document(category: str, filename: str, req: DocSaveRequest):
    """保存/编辑文档 → 删除旧 chunks → 重建该分类索引"""
    if category not in CATEGORIES:
        return {"error": f"分类 {category} 不存在"}, 404

    cat_dir = os.path.join("knowledge", CATEGORY_MAP[category])
    os.makedirs(cat_dir, exist_ok=True)
    fpath = os.path.join(cat_dir, filename)

    # 判断是否已存在（用于确定 source path）
    source_path = f"{CATEGORY_MAP[category]}/{filename}"

    # 如果已存在，先清理旧 chunks
    if os.path.exists(fpath):
        delete_doc_chunks(source_path, category)

    # 写文件
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(req.content)

    # 重建索引
    stats = build_index("knowledge", category)
    return {
        "status": "ok",
        "filename": filename,
        "category": CATEGORY_MAP[category],
        "size": len(req.content),
        "chunks": stats.get(category, 0),
    }


@app.delete("/documents/{category}/{filename}")
def delete_document(category: str, filename: str):
    """删除文档 → 清理 ChromaDB chunks"""
    if category not in CATEGORIES:
        return {"error": f"分类 {category} 不存在"}, 404

    cat_dir = os.path.join("knowledge", CATEGORY_MAP[category])
    fpath = os.path.join(cat_dir, filename)
    if not os.path.exists(fpath):
        return {"error": "文件不存在"}, 404

    source_path = f"{CATEGORY_MAP[category]}/{filename}"
    deleted_chunks = delete_doc_chunks(source_path, category)
    os.remove(fpath)

    return {"status": "deleted", "filename": filename, "deleted_chunks": deleted_chunks}


@app.post("/documents/reindex")
def reindex_all():
    """全量重建所有分类索引"""
    stats = build_index("knowledge")
    return {"status": "ok", "stats": {CATEGORY_MAP.get(c, c): n for c, n in stats.items()}}


# ── 启动 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
