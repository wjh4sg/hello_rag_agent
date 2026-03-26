from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
from threading import RLock
from typing import Dict
from uuid import uuid4

from hello_agents import ReActAgent, ToolRegistry
from hello_agents.context import ContextBuilder, ContextConfig, ContextPacket
from hello_agents.core.message import Message

from hello_rag_agent.config import AppSettings, load_settings
from hello_rag_agent.knowledge_base import KnowledgeBase
from hello_rag_agent.llm import SafeHelloAgentsLLM
from hello_rag_agent.tools import MemoryTool, RAGTool


@dataclass
class SessionState:
    session_id: str
    agent: ReActAgent
    memory_tool: MemoryTool
    rag_tool: RAGTool
    history: list[Message] = field(default_factory=list)
    lock: RLock = field(default_factory=RLock)


class HelloRagAgentService:
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or load_settings()
        self.knowledge_base = KnowledgeBase(
            self.settings.knowledge_base,
            api_key=self.settings.resolve_api_key(),
            base_url=self.settings.llm.base_url,
        )
        self._sessions: Dict[str, SessionState] = {}
        self._lock = RLock()
        self._system_prompt = Path(self.settings.prompt_path).read_text(encoding="utf-8")
        self._llm: SafeHelloAgentsLLM | None = None
        self._context_builder = ContextBuilder(
            config=ContextConfig(
                max_tokens=8000,
                reserve_ratio=0.15,
                min_relevance=0.0,
                enable_compression=True,
            )
        )

    def create_session(self) -> str:
        return self._get_or_create_session().session_id

    def ask(self, query: str, session_id: str | None = None) -> tuple[str, str]:
        question = query.strip()
        if not question:
            raise ValueError("Query cannot be empty.")

        session = self._get_or_create_session(session_id)
        with session.lock:
            session.agent.clear_history()
            prompt = self._build_context(query=question, session=session)

            with contextlib.redirect_stdout(io.StringIO()):
                answer = session.agent.run(prompt)

            answer = self._normalize_answer(answer)
            if self._needs_fallback(answer):
                answer = self._answer_with_retrieval(query=question, session=session)
            answer = self._normalize_answer(answer)

            user_message = Message(content=question, role="user")
            assistant_message = Message(content=answer, role="assistant")
            session.history.extend([user_message, assistant_message])
            session.memory_tool.remember_message(user_message)
            session.memory_tool.remember_message(assistant_message)

            return answer, session.session_id

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [self._serialize_message(item) for item in session.history]

    def reset_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        with session.lock:
            session.history.clear()
            session.memory_tool.clear_all()
            session.agent.clear_history()

    def knowledge_stats(self) -> dict[str, object]:
        return self.knowledge_base.stats()

    def _get_or_create_session(self, session_id: str | None = None) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]

            new_session_id = session_id or uuid4().hex[:8]
            memory_tool = MemoryTool(
                session_id=new_session_id,
                default_top_k=max(self.settings.agent.history_turns // 2, 2),
            )
            rag_tool = RAGTool(
                knowledge_base=self.knowledge_base,
                llm=self._get_llm(),
                system_prompt=self._system_prompt,
                default_top_k=self.settings.knowledge_base.top_k,
                default_search_strategy=self.settings.knowledge_base.retrieval_mode,
            )
            session = SessionState(
                session_id=new_session_id,
                agent=self._build_agent(memory_tool=memory_tool, rag_tool=rag_tool),
                memory_tool=memory_tool,
                rag_tool=rag_tool,
            )
            self._sessions[new_session_id] = session
            return session

    def _build_agent(self, memory_tool: MemoryTool, rag_tool: RAGTool) -> ReActAgent:
        llm = self._get_llm()
        registry = ToolRegistry()
        with contextlib.redirect_stdout(io.StringIO()):
            registry.register_tool(memory_tool)
            registry.register_tool(rag_tool)
            return ReActAgent(
                name=self.settings.agent.name,
                llm=llm,
                tool_registry=registry,
                system_prompt=self._system_prompt,
                max_steps=self.settings.agent.max_steps,
            )

    def _build_context(self, query: str, session: SessionState) -> str:
        recent_history = session.history[-self.settings.agent.history_turns * 2 :]
        additional_packets = self._build_additional_packets(query=query, session=session)
        return self._context_builder.build(
            user_query=query,
            conversation_history=recent_history,
            additional_packets=additional_packets,
        )

    def _build_additional_packets(self, query: str, session: SessionState) -> list[ContextPacket]:
        packets: list[ContextPacket] = []

        memory_packet = session.memory_tool.build_context_packet(query=query)
        if memory_packet is not None:
            packets.append(memory_packet)

        packets.append(
            ContextPacket(
                content=(
                    "Available tools:\n"
                    "- rag_tool: use action='ask' to answer from the local knowledge base, "
                    "action='search' to inspect evidence, and action='stats' to inspect the index.\n"
                    "- memory_tool: use action='search' to retrieve earlier conversation details, "
                    "action='summary' for a recap, and action='stats' to inspect memory state."
                ),
                metadata={"type": "task_state"},
            )
        )
        return packets

    def _get_llm(self) -> SafeHelloAgentsLLM:
        if self._llm is None:
            self._llm = SafeHelloAgentsLLM(
                model=self.settings.llm.model,
                api_key=self.settings.resolve_api_key(),
                base_url=self.settings.llm.base_url,
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_tokens,
            )
        return self._llm

    def _answer_with_retrieval(self, query: str, session: SessionState) -> str:
        history = [
            self._serialize_message(item)
            for item in session.history[-self.settings.agent.history_turns * 2 :]
        ]
        response = session.rag_tool.run(
            {
                "action": "ask",
                "query": query,
                "top_k": self.settings.knowledge_base.top_k,
                "search_strategy": self.settings.knowledge_base.retrieval_mode,
                "history": history,
            }
        )
        answer = response.text.strip()
        return answer or "I could not generate a final answer."

    @staticmethod
    def _needs_fallback(answer: str) -> bool:
        normalized = answer.strip()
        if not normalized:
            return True

        lowered = normalized.lower()
        fallback_markers = (
            "thought:",
            "action:",
            "observation:",
            "rag_tool",
            "memory_tool",
            "search_knowledge",
            "search_memory",
            "please wait",
            "let me",
            "need to search",
            "need to retrieve",
            "need to look up",
            "i need more detail",
        )
        clarification_markers = (
            "我需要先了解具体的问题",
            "我需要先了解您的具体问题",
            "才能提供准确的回答",
            "请明确您想了解",
            "请明确具体内容",
            "请提供更具体的问题",
            "请补充更多信息",
            "请告知您需要咨询的具体内容",
            "需要明确问题主题",
        )
        timeout_markers = (
            "max step",
            "step limit",
            "鏃犳硶鍦ㄩ檺瀹氭鏁板唴",
            "闄愬畾姝ユ暟",
        )
        return (
            any(marker in lowered for marker in fallback_markers)
            or any(marker in normalized for marker in clarification_markers)
            or any(marker in normalized for marker in timeout_markers)
        )

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        normalized = answer.strip()
        if not normalized:
            return normalized

        finish_index = normalized.rfind("Finish")
        if finish_index > 0:
            finish_segment = normalized[finish_index:].strip()
            json_start = finish_segment.find("{")
            if json_start != -1:
                try:
                    payload = json.loads(finish_segment[json_start:])
                    if isinstance(payload, dict):
                        final_answer = payload.get("answer")
                        if isinstance(final_answer, str) and final_answer.strip():
                            return final_answer.strip()
                except json.JSONDecodeError:
                    pass

            if finish_segment.startswith("Finish:"):
                candidate = finish_segment[len("Finish:") :].strip()
                if candidate:
                    return candidate

        if normalized.endswith("Finish"):
            normalized = normalized[: -len("Finish")].rstrip()

        if normalized.startswith("Finish"):
            json_start = normalized.find("{")
            if json_start != -1:
                try:
                    payload = json.loads(normalized[json_start:])
                    if isinstance(payload, dict):
                        final_answer = payload.get("answer")
                        if isinstance(final_answer, str) and final_answer.strip():
                            return final_answer.strip()
                except json.JSONDecodeError:
                    pass
            return normalized.removeprefix("Finish").strip()

        return normalized

    @staticmethod
    def _serialize_message(message: Message) -> dict[str, str]:
        return {
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp.isoformat() if message.timestamp else "",
        }


_SERVICE: HelloRagAgentService | None = None


def get_service() -> HelloRagAgentService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = HelloRagAgentService()
    return _SERVICE
