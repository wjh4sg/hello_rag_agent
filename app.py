from __future__ import annotations

import streamlit as st

from hello_rag_agent import get_service


@st.cache_resource
def load_service():
    return get_service()


def render_assistant_message(content: str) -> None:
    st.markdown(content)


def stream_assistant_message(stream) -> str:
    chunks: list[str] = []
    placeholder = st.empty()
    for chunk in stream:
        chunks.append(str(chunk))
        placeholder.markdown("".join(chunks) + "▌")
    final_answer = "".join(chunks).strip()
    placeholder.markdown(final_answer)
    return final_answer


service = load_service()

st.set_page_config(page_title="Hello Agents RAG", page_icon="AI", layout="wide")

st.title("Hello Agents RAG")
st.caption("基于 hello_agents 的轻量知识库问答演示，支持记忆增强和流式回答。")

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

    skipped_files = stats.get("skipped_files", [])
    if skipped_files:
        st.subheader("跳过文件")
        for item in skipped_files:
            st.write(f"- {item}")

for message in history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message["content"])
        else:
            st.write(message["content"])

prompt = st.chat_input("请输入你的问题")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    try:
        with st.chat_message("assistant"):
            stream, resolved_session_id = service.stream_ask(
                prompt,
                session_id=session_id,
                user_id=user_id,
            )
            streamed = stream_assistant_message(stream)
        st.session_state["session_id"] = resolved_session_id
    except Exception as exc:
        st.error(str(exc))
    else:
        st.session_state["last_streamed_answer"] = streamed
        st.rerun()
