"""
core.py — RAG 内核，复用 tiny-rag 的检索 + 评估逻辑

四大函数：
  build_index(docs, category)  → 为指定分类建向量索引
  search(query, category)      → 在指定分类中混合检索
  ask(query, category)         → 检索 + LLM 生成
  classify(query)              → 自动判断问题属于哪个分类
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from rank_bm25 import BM25Okapi
import jieba

load_dotenv()

# 硅基流动（bge-m3 embedding）
embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

# DeepSeek（LLM 生成）
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

CHROMA_DIR = "chroma_store"
RRF_K = 60

_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# 预定义知识分类
CATEGORIES = ["技术文档", "规章制度", "产品手册", "培训资料", "FAQ"]


# ── 自动分类 ──────────────────────────────────────────────

def classify(query: str) -> str:
    """判断用户问题属于哪个知识分类"""
    cats = "\n".join(f"- {c}" for c in CATEGORIES)
    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": f"以下问题属于哪个分类？只输出分类名。\n\n分类：\n{cats}\n\n问题：{query}",
        }],
    )
    result = resp.choices[0].message.content.strip()
    # 兜底：如果 LLM 输出不在列表，回退到 FAQ
    return result if result in CATEGORIES else "FAQ"


# ── 建索引 ────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """固定长度分块 + 句子边界保护"""
    sentences = text.replace("\n", " ").split("。")
    chunks, current = [], ""
    for s in sentences:
        if not s.strip():
            continue
        seg = s + "。"
        if len(current) + len(seg) <= chunk_size:
            current += seg
        else:
            if current.strip():
                chunks.append(current.strip())
            current = current[-overlap:] + seg if current else seg
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_index(docs_dir: str = "knowledge", category: str = None) -> dict:
    """扫描 knowledge 目录，为每个分类建 Chroma collection"""
    result = {}
    categories = [category] if category else CATEGORIES

    for cat in categories:
        cat_dir = os.path.join(docs_dir, cat)
        if not os.path.isdir(cat_dir):
            continue

        # 读取该分类下所有文件
        docs = []
        for fname in os.listdir(cat_dir):
            fpath = os.path.join(cat_dir, fname)
            if not fname.endswith((".md", ".txt")):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                docs.append({"text": f.read(), "source": f"{cat}/{fname}"})

        if not docs:
            result[cat] = 0
            continue

        # 分块
        all_chunks = []
        for doc in docs:
            for chunk in _chunk_text(doc["text"]):
                all_chunks.append({"text": chunk, "source": doc["source"]})

        # 批量 embedding
        texts = [c["text"] for c in all_chunks]
        vectors = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = embed_client.embeddings.create(model="BAAI/bge-m3", input=batch)
            vectors.extend([d.embedding for d in resp.data])

        # 存入 Chroma collection（每个分类一个 collection）
        collection = _chroma_client.get_or_create_collection(f"rag_{cat}")
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        ids = [str(i) for i in range(len(all_chunks))]
        metadatas = [{"source": c["source"]} for c in all_chunks]
        collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        result[cat] = collection.count()

    return result


# ── 检索 ────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    resp = embed_client.embeddings.create(model="BAAI/bge-m3", input=[text])
    return resp.data[0].embedding


def _bm25_search(query: str, chunks: list[dict], top_k: int = 20) -> list[tuple[int, float]]:
    if not chunks:
        return []
    corpus = [c["text"] for c in chunks]
    tokenized = [list(jieba.cut(t)) for t in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(list(jieba.cut(query)))
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    return ranked[:top_k]


def search(query: str, category: str, top_k: int = 5) -> list[dict]:
    """在指定分类中混合检索（向量 + BM25 + RRF）"""
    collection = _chroma_client.get_or_create_collection(f"rag_{category}")
    all_data = collection.get()
    chunks = [
        {"text": doc, "source": meta["source"] if meta else "?"}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    if not chunks:
        return []

    # 向量检索
    qv = _embed(query)
    dense_result = collection.query(query_embeddings=[qv], n_results=min(20, len(chunks)))
    dense = [(int(i), d) for i, d in zip(dense_result["ids"][0], dense_result["distances"][0])]

    # BM25 检索
    sparse = _bm25_search(query, chunks, top_k=20)

    # RRF 融合
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense):
        scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)
    for rank, (idx, _) in enumerate(sparse):
        scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)
    merged = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    return [
        {"text": chunks[idx]["text"], "source": chunks[idx]["source"], "score": round(sc, 3)}
        for idx, sc in merged
    ]


# ── 问答 ────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """你是企业知识库助手。只能基于参考资料回答。
不知道就说"参考资料中未包含相关内容"。标注来源。"""


def ask(query: str, category: str = None) -> dict:
    """完整 RAG 流程：分类 → 检索 → 生成"""
    if category is None:
        category = classify(query)

    chunks = search(query, category)
    if not chunks:
        return {
            "answer": "未找到相关知识。请确认知识库已导入相关资料。",
            "category": category,
            "sources": [],
        }

    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks[:5]
    )
    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "category": category,
        "sources": list({c["source"] for c in chunks}),
    }
