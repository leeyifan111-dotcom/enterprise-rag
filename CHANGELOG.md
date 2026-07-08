# 更新日志

## 2026-07-08（后续迭代）

### 🔧 优化

- **并行检索**：system prompt 引导 LLM 对互不依赖的子问题并行调用 search_knowledge，减少轮次
- **max_turns 降为 4**：并行后 Agent 循环需求减少，6→4 降低响应延迟

### 🐛 修复

- **切换会话时配置面板不更新**：widget key 绑定 session_id，切换会话自动重建新鲜 widget（Streamlit key 缓存机制导致 value= 被忽略）
- **会话配置保存时 retrieval_on 丢失**：`update_session` 端点漏处理 retrieval_on 字段，保存后被丢弃

---

## 2026-07-08（当日主要提交）

### ✨ 新增

- **通用工具系统**：Agent 支持注册自定义工具（`register_tool()`），内置 `search_knowledge` + `calculator`，工具派发改为通用 `_dispatch_tool()`
- **前端标签页布局**：主区域拆为「💬 对话」和「📄 文档管理」两个独立标签页，编辑文档不再干扰聊天界面
- **检索开关**：会话配置新增 toggle「启用知识库检索」，关闭后 LLM 直接回答不搜知识库
- **停止回答按钮**：生成答案期间显示 🛑 按钮，点击后丢弃返回结果，用户问题保留不丢失
- **单分类重建索引**：文档管理面板支持按单个分类重建索引，不再只能全量重建

### 🐛 修复

- **多轮对话上下文丢失**：`ask()` 新增 `history` 参数，每次请求注入会话历史消息，第二轮不再遗忘上下文
- **prompt 空串回退**：`system_prompt` 为空字符串或纯空白时显式回退到默认 `RAG_SYSTEM_PROMPT`
- **空 collection 自动清理**：删除分类下最后一个文档后自动清理对应 Chroma collection

---

## 2026-07-07（另一处提交同步）

### ✨ 新增

- **多会话管理系统**：新建/重命名/删除会话，JSON 持久化，切换会话自动加载历史消息和配置
- **文档在线管理**：编辑器 + 保存 + 删除 + 自动重建索引
- **会话配置系统**：每个会话可独立配置 system_prompt / search_top_k / max_turns，空时回退默认值

### 🔧 改进

- **RAGConfig 参数化**：所有可调参数统一为 dataclass，支持环境变量覆盖
- **chunk 参数调大**：chunk_size 500→1000，overlap 50→150（中文字符）

---

## 2026-07-07（会话启动时）

### ✨ 新增

- **Agentic RAG**：`ask()` 从单向 pipeline 改为 ReAct Agent 循环
- **CoT + Query 拆分**：LLM 每步先 thought 再决定 action，复杂问题自动拆分为多个子问题分别检索
- **前端聊天界面**：Streamlit 聊天面板 + 上传面板

### 🐛 修复

- **Chroma 中文名兼容**：collection 名从中文改为英文 key（tech_doc/policy/product/training/faq），前端做中文映射

---

## 2026-07-07（项目启动）

### ✨ 新增

- **项目骨架**：core.py + backend.py + frontend.py 三件套
- **五分类知识库**：技术文档/规章制度/产品手册/培训资料/FAQ
- **混合检索**：bge-m3 向量 + BM25 + RRF 融合
- **FastAPI 后端**：POST /chat /upload，GET /categories /stats
- **Streamlit 前端**：聊天面板 + 上传面板
