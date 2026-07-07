"""
backend.py — FastAPI 后端

POST /chat     → 问答
POST /upload   → 上传文件到指定分类
GET  /categories → 列出所有分类
GET  /stats    → 各分类索引统计
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from core import ask, build_index, classify, CATEGORIES, CHROMA_DIR
import os
import shutil

app = FastAPI(title="Enterprise RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
def chat(query: str, category: str = None):
    result = ask(query, category)
    return result


@app.post("/upload")
def upload(file: UploadFile = File(...), category: str = Form(...)):
    if category not in CATEGORIES:
        return {"error": f"分类 {category} 不存在，可选: {CATEGORIES}"}

    cat_dir = os.path.join("knowledge", category)
    os.makedirs(cat_dir, exist_ok=True)

    filepath = os.path.join(cat_dir, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    stats = build_index("knowledge", category)
    return {"status": "ok", "file": file.filename, "category": category, "chunks": stats.get(category, 0)}


@app.get("/categories")
def list_categories():
    return {"categories": CATEGORIES}


@app.get("/stats")
def stats():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    result = {}
    for cat in CATEGORIES:
        try:
            col = client.get_collection(f"rag_{cat}")
            result[cat] = col.count()
        except Exception:
            result[cat] = 0
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
