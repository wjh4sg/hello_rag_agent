from __future__ import annotations

import streamlit as st

from hello_rag_agent import get_service


@st.cache_resource
def load_service():
    return get_service()


service = load_service()

st.set_page_config(page_title="Hello Agents RAG", page_icon="AI", layout="wide")
st.title("Hello Agents RAG")
st.caption("基于 hello_agents 的精简知识库问答示例。")

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "streamlit_demo_user"

if "session_id" not in st.session_state:
    st.session_state["session_id"] = service.create_session(user_id=st.session_state["user_id"])

session_id = st.session_state["session_id"]
user_id = st.session_state["user_id"]
stats = service.knowledge_stats()
history = service.get_history(session_id)

with st.sidebar:
    st.subheader("会话")
    st.code(session_id)
    st.caption(f"user_id: `{user_id}`")
    if st.button("重置会话", use_container_width=True):
        service.reset_session(session_id)
        st.rerun()

    st.subheader("知识库")
    st.write(f"文档数: {stats['document_count']}")
    st.write(f"分片数: {stats['chunk_count']}")
    st.write(f"数据目录: `{stats['data_dir']}`")

    if stats["skipped_files"]:
        st.subheader("跳过文件")
        for item in stats["skipped_files"]:
            st.write(f"- {item}")

for message in history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("请输入你的问题")

if prompt:
    try:
        with st.spinner("正在生成回答..."):
            service.ask(prompt, session_id=session_id, user_id=user_id)
    except Exception as exc:
        st.error(str(exc))
    st.rerun()
