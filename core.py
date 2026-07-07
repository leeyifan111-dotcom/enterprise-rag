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
from dataclasses import dataclass, replace
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

_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# 预定义知识分类（key=英文标识，value=中文显示名）
CATEGORY_MAP = {
    "tech_doc": "技术文档",
    "policy": "规章制度",
    "product": "产品手册",
    "training": "培训资料",
    "faq": "FAQ",
}
CATEGORIES = list(CATEGORY_MAP.keys())

# 中文名→key 反查
CATEGORY_NAMES = {v: k for k, v in CATEGORY_MAP.items()}


# ── 可配置参数 ──────────────────────────────────────────────

@dataclass
class RAGConfig:
    """RAG 所有可调参数，支持环境变量 + 运行时覆盖"""
    chunk_size: int = 1000
    chunk_overlap: int = 150
    search_top_k: int = 5
    tool_top_k: int = 3
    agent_max_turns: int = 6
    embedding_batch_size: int = 32
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "deepseek-chat"
    rrf_k: int = 60

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", 150)),
            search_top_k=int(os.getenv("RAG_SEARCH_TOP_K", 5)),
            tool_top_k=int(os.getenv("RAG_TOOL_TOP_K", 3)),
            agent_max_turns=int(os.getenv("RAG_AGENT_MAX_TURNS", 6)),
            embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", 32)),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
            llm_model=os.getenv("RAG_LLM_MODEL", "deepseek-chat"),
            rrf_k=int(os.getenv("RAG_RRF_K", 60)),
        )


# 全局默认配置实例
_config = RAGConfig.from_env()


def _cat_dir(cat: str) -> str:
    """分类名 → knowledge/ 子目录名（中文）"""
    return CATEGORY_MAP.get(cat, cat)


def _cat_collection(cat: str) -> str:
    """分类名 → Chroma collection 名（英文）"""
    return f"rag_{cat}"


# ── 自动分类 ──────────────────────────────────────────────

def classify(query: str, config: RAGConfig | None = None) -> str:
    """判断用户问题属于哪个知识分类，返回英文 key"""
    cfg = config or _config
    cats = "\n".join(f"- {v}" for v in CATEGORY_MAP.values())
    resp = llm_client.chat.completions.create(
        model=cfg.llm_model,
        messages=[{
            "role": "user",
            "content": f"以下问题属于哪个分类？只输出分类名。\n\n分类：\n{cats}\n\n问题：{query}",
        }],
    )
    result = resp.choices[0].message.content.strip()
    # 中文名→英文 key 反查，失败回退 faq
    return CATEGORY_NAMES.get(result, "faq")


# ── 建索引 ────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None, config: RAGConfig | None = None) -> list[str]:
    """固定长度分块 + 句子边界保护"""
    cfg = config or _config
    size = chunk_size if chunk_size is not None else cfg.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap
    sentences = text.replace("\n", " ").split("。")
    chunks, current = [], ""
    for s in sentences:
        if not s.strip():
            continue
        seg = s + "。"
        if len(current) + len(seg) <= size:
            current += seg
        else:
            if current.strip():
                chunks.append(current.strip())
            current = current[-overlap:] + seg if current else seg
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_index(docs_dir: str = "knowledge", category: str = None, chunk_size: int | None = None, chunk_overlap: int | None = None, config: RAGConfig | None = None) -> dict:
    """扫描 knowledge 目录，为每个分类建 Chroma collection"""
    cfg = config or _config
    result = {}
    categories = [category] if category else CATEGORIES

    for cat in categories:
        cat_dir = os.path.join(docs_dir, _cat_dir(cat))
        if not os.path.isdir(cat_dir):
            continue

        # 读取该分类下所有文件
        docs = []
        for fname in os.listdir(cat_dir):
            fpath = os.path.join(cat_dir, fname)
            if not fname.endswith((".md", ".txt")):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                docs.append({"text": f.read(), "source": f"{_cat_dir(cat)}/{fname}"})

        if not docs:
            result[cat] = 0
            continue

        # 分块
        all_chunks = []
        for doc in docs:
            for chunk in _chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap, config=cfg):
                all_chunks.append({"text": chunk, "source": doc["source"]})

        # 批量 embedding
        texts = [c["text"] for c in all_chunks]
        vectors = []
        batch_size = cfg.embedding_batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = embed_client.embeddings.create(model=cfg.embedding_model, input=batch)
            vectors.extend([d.embedding for d in resp.data])

        # 存入 Chroma collection（英文名）
        collection = _chroma_client.get_or_create_collection(_cat_collection(cat))
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        ids = [str(i) for i in range(len(all_chunks))]
        metadatas = [{"source": c["source"]} for c in all_chunks]
        collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        result[cat] = collection.count()

    return result


# ── 检索 ────────────────────────────────────────────────────

def _embed(text: str, config: RAGConfig | None = None) -> list[float]:
    cfg = config or _config
    resp = embed_client.embeddings.create(model=cfg.embedding_model, input=[text])
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


def search(query: str, category: str, top_k: int | None = None, config: RAGConfig | None = None) -> list[dict]:
    """在指定分类中混合检索（向量 + BM25 + RRF）"""
    cfg = config or _config
    k = top_k if top_k is not None else cfg.search_top_k
    collection = _chroma_client.get_or_create_collection(_cat_collection(category))
    all_data = collection.get()
    chunks = [
        {"text": doc, "source": meta["source"] if meta else "?"}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    if not chunks:
        return []

    # 向量检索
    qv = _embed(query, config=cfg)
    dense_result = collection.query(query_embeddings=[qv], n_results=min(20, len(chunks)))
    dense = [(int(i), d) for i, d in zip(dense_result["ids"][0], dense_result["distances"][0])]

    # BM25 检索
    sparse = _bm25_search(query, chunks, top_k=20)

    # RRF 融合
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense):
        scores[idx] = scores.get(idx, 0) + 1 / (cfg.rrf_k + rank + 1)
    for rank, (idx, _) in enumerate(sparse):
        scores[idx] = scores.get(idx, 0) + 1 / (cfg.rrf_k + rank + 1)
    merged = sorted(scores.items(), key=lambda x: -x[1])[:k]

    return [
        {"text": chunks[idx]["text"], "source": chunks[idx]["source"], "score": round(sc, 3)}
        for idx, sc in merged
    ]


# ── 问答 ────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """# 身份
你是企业知识库 AI 助手，服务于公司内部员工。你在一个 Agent 循环里工作——
你可以多次调用 search_knowledge 工具来查询不同分类的知识库，直到获得足够的信息后再回答。

# 核心工作流程（Chain of Thought）
1. 分析问题：这个问题涉及几个子问题？分别属于哪个分类？
2. 逐个解决：对每个子问题，调用 search_knowledge 检索对应分类
3. 评估完整性：检索到的资料够不够回答？不够就调整 query 或换分类再搜
4. 综合回答：汇总所有子问题的结果，给出最终答案

# 查询拆分原则
- 复杂问题拆成多个简单子问题，逐个 query 检索
- 例："年假制度和报销流程" → 拆为"年假制度"和"报销流程"两次检索
- 每个子问题可以根据其内容指定不同分类（技术文档/规章制度/产品手册/培训资料/FAQ）

# 工具使用规则
- 需要知识库信息时，调用 search_knowledge 工具
- 每次调用指定 category（分类）和 query（查询内容）
- 检索不到相关信息时，换个 query 角度或换分类再试，最多试 2 次
- 同一 query 在同一分类中搜 2 次仍无结果 → 放弃该子问题，告知用户

# 不确定性处理
- 资料中没有的信息，明确说"公司现有资料中未包含此信息，建议咨询 HR/行政部门"
- 禁止推测公司政策、编造数字或日期

# 输出格式
- 先直接回答问题，再列出依据和来源
- 计算类问题：显式写出计算过程 → 结果 → 依据条款"""


# ── Agent 工具定义 ─────────────────────────────────────────

_tool_schema = [{
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "搜索企业知识库。指定分类和查询内容，返回相关文档片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "知识分类: tech_doc/policy/product/training/faq"},
                "query": {"type": "string", "description": "要查询的具体问题"},
            },
            "required": ["category", "query"],
        },
    },
}]


def _search_tool(category: str, query: str, config: RAGConfig | None = None) -> str:
    """工具函数：检索并返回格式化结果"""
    cfg = config or _config
    chunks = search(query, category, top_k=cfg.tool_top_k, config=cfg)
    if not chunks:
        return f"（在 {category} 中未找到相关内容）"
    return "\n\n".join(
        f"[来源: {c['source']}]\n{c['text']}" for c in chunks
    )


# ── Agentic RAG 问答 ─────────────────────────────────────

def ask(query: str, category: str = None, max_turns: int | None = None, config: RAGConfig | None = None, system_prompt: str | None = None) -> dict:
    """Agentic RAG：LLM 在循环中自主检索，含 CoT + query 拆分

    - CoT: 每步先 thought 再决定是否调工具
    - 查询拆分: LLM 可将复杂问题拆成多个子问题，多次调 search_knowledge
    - 循环控制: max_turns 上限，重复决策检测
    - system_prompt: 自定义系统提示词，不传则用默认 RAG_SYSTEM_PROMPT
    """
    cfg = config or _config
    turns = max_turns if max_turns is not None else cfg.agent_max_turns
    prompt = system_prompt or RAG_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    tools = _tool_schema
    seen_sigs = set()
    all_sources = set()

    for turn in range(turns):
        resp = llm_client.chat.completions.create(
            model=cfg.llm_model,
            messages=messages,
            tools=tools,
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # CoT 思考过程
        if msg.content:
            print(f"  [Turn {turn}] {msg.content[:120]}")

        # 不再要工具 → 给出最终答案
        if not msg.tool_calls:
            return {
                "answer": msg.content or "",
                "category": category or classify(query, config=cfg),
                "sources": list(all_sources),
            }

        # 循环检测：同一轮决策是否重复
        sig = "|".join(sorted(
            f"{c.function.name}({c.function.arguments})" for c in msg.tool_calls
        ))
        if sig in seen_sigs:
            print(f"  Loop detected, forcing stop")
            break
        seen_sigs.add(sig)

        # 逐个执行工具调用
        for call in msg.tool_calls:
            if call.function.name != "search_knowledge":
                continue
            args = json.loads(call.function.arguments)
            result = _search_tool(
                category=args.get("category", "faq"),
                query=args.get("query", query),
                config=cfg,
            )
            all_sources.update(
                line.split("]")[0].replace("[来源: ", "")
                for line in result.split("\n") if line.startswith("[来源:")
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    # 达到 max_turns，让 LLM 强行总结
    final = llm_client.chat.completions.create(
        model=cfg.llm_model,
        messages=messages + [{"role": "user", "content": "请基于已获取的资料，给出当前能得出的最佳答案。"}],
    )
    return {
        "answer": final.choices[0].message.content,
        "category": category or classify(query, config=cfg),
        "sources": list(all_sources),
    }


# ── 文档管理 ──────────────────────────────────────────────

def delete_doc_chunks(source_path: str, category: str) -> int:
    """删除指定文档在 ChromaDB 中的所有 chunk。source_path 如 规章制度/考勤制度.md"""
    try:
        collection = _chroma_client.get_collection(_cat_collection(category))
    except Exception:
        return 0
    all_data = collection.get()
    if not all_data["ids"]:
        return 0
    ids_to_delete = [
        id_ for id_, meta in zip(all_data["ids"], all_data["metadatas"])
        if meta and meta.get("source") == source_path
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def remove_category_index(category: str):
    """删除整个分类的 Chroma collection"""
    try:
        _chroma_client.delete_collection(_cat_collection(category))
    except Exception:
        pass
