"""
frontend.py — Streamlit 前端

左侧：聊天面板（输入问题 → 显示答案 + 来源）
右侧：上传面板（选文件 + 分类 → 后台入库）
"""

import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="企业知识库 AI", layout="wide")
st.title("企业知识库 AI 助手")

# ── 侧边栏：上传 ────────────────────────────────────

with st.sidebar:
    st.header("知识库管理")

    # 获取分类列表
    try:
        cats = requests.get(f"{API}/categories", timeout=3).json()["categories"]
    except Exception:
        cats = ["技术文档", "规章制度", "产品手册", "培训资料", "FAQ"]

    category = st.selectbox("选择分类", cats)

    uploaded_file = st.file_uploader("上传文件", type=["md", "txt", "pdf"])
    if uploaded_file and st.button("上传并建索引"):
        with st.spinner("上传中..."):
            resp = requests.post(
                f"{API}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                data={"category": category},
            )
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"已导入 {data['file']} → {data['chunks']} 个 chunks")
            else:
                st.error("上传失败")

    st.divider()

    # 索引统计
    try:
        st.caption("索引统计")
        stats = requests.get(f"{API}/stats", timeout=3).json()
        for cat, count in stats.items():
            st.text(f"{cat}: {count} chunks")
    except Exception:
        pass

# ── 主区域：聊天 ────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# 历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("来源: " + ", ".join(msg["sources"]))

# 输入框
if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                resp = requests.get(f"{API}/chat", params={"query": prompt}, timeout=30)
                data = resp.json()
            except Exception:
                data = {"answer": "后端未启动，请先运行 python backend.py", "category": "-", "sources": []}

        st.markdown(data["answer"])
        if data["sources"]:
            st.caption(f"来源: {', '.join(data['sources'])} | 分类: {data['category']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": data["answer"],
        "sources": data.get("sources", []),
    })
