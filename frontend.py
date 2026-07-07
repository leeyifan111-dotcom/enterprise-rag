"""
frontend.py — Streamlit 前端

侧边栏：会话列表 → 会话配置 → 文档管理 → 上传文件
主区域：聊天面板
"""

import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="企业知识库 AI", layout="wide")
st.title("企业知识库 AI 助手")

# ── 初始化 session_state ────────────────────────────────

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "session_config" not in st.session_state:
    st.session_state.session_config = {"system_prompt": "", "search_top_k": 5, "max_turns": 6}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "default_prompt" not in st.session_state:
    try:
        resp = requests.get(f"{API}/default-prompt", timeout=3)
        st.session_state.default_prompt = resp.json()["system_prompt"]
    except Exception:
        st.session_state.default_prompt = ""
if "doc_category" not in st.session_state:
    st.session_state.doc_category = "tech_doc"
if "edit_file" not in st.session_state:
    st.session_state.edit_file = None
if "edit_content" not in st.session_state:
    st.session_state.edit_content = ""

# 中文名↔英文 key 映射
CN_TO_KEY = {"技术文档": "tech_doc", "规章制度": "policy", "产品手册": "product", "培训资料": "training", "FAQ": "faq"}
KEY_TO_CN = {v: k for k, v in CN_TO_KEY.items()}


def load_sessions():
    """刷新会话列表"""
    try:
        resp = requests.get(f"{API}/sessions", timeout=3)
        st.session_state.sessions = resp.json()["sessions"]
    except Exception:
        pass


def load_session(session_id):
    """加载指定会话的消息和配置"""
    try:
        resp = requests.get(f"{API}/sessions/{session_id}", timeout=3)
        if resp.status_code == 200:
            s = resp.json()
            st.session_state.messages = s.get("messages", [])
            st.session_state.session_config = s.get("config", {"system_prompt": "", "search_top_k": 5, "max_turns": 6})
    except Exception:
        pass


def switch_session(session_id):
    """切换到指定会话"""
    st.session_state.current_session_id = session_id
    load_session(session_id)


# ═══════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    # ── 会话列表 ──────────────────────────────────────────
    st.subheader("会话列表")

    col1, col2 = st.columns([3, 1])
    with col1:
        new_name = st.text_input("新会话名称", "新会话", key="new_session_name", label_visibility="collapsed")
    with col2:
        if st.button("➕ 新建", use_container_width=True):
            try:
                resp = requests.post(f"{API}/sessions", json={"name": new_name}, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    switch_session(data["id"])
                    load_sessions()
                    st.rerun()
            except Exception:
                st.error("后端未启动")

    load_sessions()

    for s in st.session_state.sessions:
        selected = s["id"] == st.session_state.current_session_id
        cols = st.columns([5, 1, 1])
        with cols[0]:
            label = f"{'●' if selected else '○'} {s['name']}"
            if st.button(label, key=f"sel_{s['id']}", use_container_width=True):
                switch_session(s["id"])
                st.rerun()
        with cols[1]:
            if st.button("✏️", key=f"ren_{s['id']}", help="重命名"):
                st.session_state[f"renaming_{s['id']}"] = True
        with cols[2]:
            if st.button("🗑️", key=f"del_{s['id']}", help="删除会话"):
                requests.delete(f"{API}/sessions/{s['id']}", timeout=3)
                if st.session_state.current_session_id == s["id"]:
                    st.session_state.current_session_id = None
                    st.session_state.messages = []
                load_sessions()
                st.rerun()

        # 内联重命名
        if st.session_state.get(f"renaming_{s['id']}"):
            new_name_input = st.text_input("新名称", s["name"], key=f"rename_input_{s['id']}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("保存", key=f"save_rename_{s['id']}"):
                    requests.put(f"{API}/sessions/{s['id']}", json={"name": new_name_input}, timeout=3)
                    del st.session_state[f"renaming_{s['id']}"]
                    load_sessions()
                    st.rerun()
            with c2:
                if st.button("取消", key=f"cancel_rename_{s['id']}"):
                    del st.session_state[f"renaming_{s['id']}"]
                    st.rerun()

    st.divider()

    # ── 当前会话配置 ──────────────────────────────────────
    if st.session_state.current_session_id:
        with st.expander("当前会话配置", expanded=False):
            sid = st.session_state.current_session_id

            # 空则展示默认 prompt
            current_prompt = st.session_state.session_config.get("system_prompt", "")
            display_prompt = current_prompt if current_prompt else st.session_state.default_prompt

            prompt_val = st.text_area(
                "系统提示词",
                value=display_prompt,
                height=200,
                key="cfg_prompt",
                help="自定义 AI 助手的角色和行为规则",
            )

            if st.button("重置为默认提示词", key="reset_prompt"):
                st.session_state.session_config["system_prompt"] = st.session_state.default_prompt
                st.rerun()

            topk_val = st.slider(
                "搜索 Top-K", 1, 20,
                st.session_state.session_config.get("search_top_k", 5),
                key="cfg_topk",
            )

            turns_val = st.slider(
                "Agent 最大轮次", 1, 10,
                st.session_state.session_config.get("max_turns", 6),
                key="cfg_turns",
            )

            if st.button("保存配置", key="save_config"):
                requests.put(f"{API}/sessions/{sid}", json={
                    "system_prompt": prompt_val,
                    "search_top_k": topk_val,
                    "max_turns": turns_val,
                }, timeout=3)
                st.session_state.session_config = {"system_prompt": prompt_val, "search_top_k": topk_val, "max_turns": turns_val}
                load_sessions()
                st.success("已保存")

            # 清空消息
            if st.button("清空当前会话消息", key="clear_msgs"):
                requests.post(f"{API}/sessions/{sid}/clear", timeout=3)
                st.session_state.messages = []
                st.rerun()

    st.divider()

    # ── 文档管理 ──────────────────────────────────────────
    with st.expander("文档管理", expanded=False):
        cat_cn = st.selectbox("分类", list(CN_TO_KEY.keys()), key="doc_cat_select")
        cat_key = CN_TO_KEY[cat_cn]
        st.session_state.doc_category = cat_key

        if st.button("🔄 刷新文件列表", key="refresh_docs"):
            st.rerun()

        try:
            resp = requests.get(f"{API}/documents/{cat_key}", timeout=3)
            files = resp.json().get("files", [])
        except Exception:
            files = []

        if not files:
            st.caption("该分类暂无文件")
        else:
            for f in files:
                cols = st.columns([5, 1, 1])
                with cols[0]:
                    st.text(f"📄 {f['name']} ({f['lines']}行)")
                with cols[1]:
                    if st.button("✏️", key=f"edit_{cat_key}_{f['name']}", help="编辑"):
                        resp = requests.get(f"{API}/documents/{cat_key}/{f['name']}", timeout=3)
                        if resp.status_code == 200:
                            st.session_state.edit_file = {"category": cat_key, "filename": f["name"]}
                            st.session_state.edit_content = resp.json()["content"]
                            st.rerun()
                with cols[2]:
                    if st.button("🗑️", key=f"del_doc_{cat_key}_{f['name']}", help="删除"):
                        requests.delete(f"{API}/documents/{cat_key}/{f['name']}", timeout=3)
                        st.success(f"已删除 {f['name']}")
                        st.rerun()

        if st.button("🔄 重建全部索引", use_container_width=True, key="reindex_all"):
            with st.spinner("重建索引中..."):
                resp = requests.post(f"{API}/documents/reindex", timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"索引重建完成: {data['stats']}")
                else:
                    st.error("重建失败")

    st.divider()

    # ── 上传文件 ──────────────────────────────────────────
    with st.expander("上传文件", expanded=False):
        up_cat_cn = st.selectbox("选择分类", list(CN_TO_KEY.keys()), key="upload_cat")
        up_cat_key = CN_TO_KEY[up_cat_cn]

        chunk_size = st.number_input("分块大小", 200, 3000, 1000, step=100, key="up_chunk_size")
        chunk_overlap = st.number_input("重叠长度", 0, 500, 150, step=50, key="up_chunk_overlap")

        uploaded_file = st.file_uploader("选择文件", type=["md", "txt"], key="file_uploader")
        if uploaded_file and st.button("上传并建索引", key="upload_btn"):
            with st.spinner("上传中..."):
                resp = requests.post(
                    f"{API}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    data={
                        "category": up_cat_key,
                        "chunk_size": str(chunk_size),
                        "chunk_overlap": str(chunk_overlap),
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"已导入 {data['file']} → {data['chunks']} 个 chunks")
                else:
                    st.error("上传失败")

    st.divider()

    # ── 索引统计 ──────────────────────────────────────────
    try:
        st.caption("索引统计")
        stats = requests.get(f"{API}/stats", timeout=3).json()
        for cat, count in stats.items():
            st.text(f"{cat}: {count} chunks")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
# 编辑文档弹窗
# ═══════════════════════════════════════════════════════════

if st.session_state.edit_file:
    ef = st.session_state.edit_file
    st.markdown("---")
    st.subheader(f"编辑文档: {ef['filename']} ({ef['category']})")

    new_content = st.text_area("文档内容", st.session_state.edit_content, height=400, key="edit_textarea")

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("💾 保存", key="save_doc"):
            with st.spinner("保存中..."):
                resp = requests.put(
                    f"{API}/documents/{ef['category']}/{ef['filename']}",
                    json={"content": new_content},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"已保存 → {data['chunks']} chunks 已重建索引")
                    st.session_state.edit_file = None
                    st.session_state.edit_content = ""
                    st.rerun()
                else:
                    st.error(f"保存失败: {resp.json()}")
    with c2:
        if st.button("取消", key="cancel_edit"):
            st.session_state.edit_file = None
            st.session_state.edit_content = ""
            st.rerun()

    st.markdown("---")


# ═══════════════════════════════════════════════════════════
# 主区域：聊天
# ═══════════════════════════════════════════════════════════

# 当前会话提示
if not st.session_state.current_session_id:
    st.info("请先在左侧创建一个会话，或选择已有会话")
else:
    # 历史消息
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("来源: " + ", ".join(msg["sources"]))
            if msg.get("category") and msg["category"] != "-":
                st.caption(f"分类: {msg['category']}")
            # 删除按钮
            if st.button("✕", key=f"del_msg_{i}", help="删除此消息"):
                sid = st.session_state.current_session_id
                requests.delete(f"{API}/sessions/{sid}/messages/{i}", timeout=3)
                del st.session_state.messages[i]
                st.rerun()

    # 输入框
    if prompt := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    body = {
                        "query": prompt,
                        "session_id": st.session_state.current_session_id,
                        "search_top_k": st.session_state.session_config.get("search_top_k", 5),
                        "max_turns": st.session_state.session_config.get("max_turns", 6),
                    }
                    resp = requests.post(f"{API}/chat", json=body, timeout=60)
                    data = resp.json()
                except Exception:
                    data = {"answer": "后端未启动，请先运行 python backend.py", "category": "-", "sources": []}

            answer = data.get("answer", f"后端错误: {data}")
            sources = data.get("sources", [])
            category = data.get("category", "-")

            st.markdown(answer)
            if sources:
                st.caption(f"来源: {', '.join(sources)} | 分类: {category}")

        # 重新从服务器加载最新消息（因为 ask 已经把消息存入 session 了）
        load_session(st.session_state.current_session_id)
