from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import re
from threading import RLock
from typing import Dict
from uuid import uuid4

from hello_agents import ReActAgent, ToolRegistry
from hello_agents.context import ContextBuilder, ContextConfig, ContextPacket
from hello_agents.core.message import Message

from hello_rag_agent.config import AppSettings, load_settings
from hello_rag_agent.knowledge_base import KnowledgeBase
from hello_rag_agent.llm import SafeHelloAgentsLLM
from hello_rag_agent.memory_manager import MemoryManager
from hello_rag_agent.memory_store import SQLiteMemoryStore
from hello_rag_agent.tools import MemoryTool, RAGTool

MEMORY_QUERY_PATTERNS = (
    "你还记得",
    "还记得吗",
    "记不记得",
    "我刚才说",
    "我刚刚说",
    "我之前说",
    "我之前提到",
    "我上次说",
    "回忆一下",
    "总结一下我",
    "当前会话",
    "我的户型",
    "我的使用诉求",
    "我家地面",
    "我最在意",
    "我最近做过什么维护",
)

EXPLICIT_MEMORY_PATTERNS = (
    "你还记得",
    "还记得吗",
    "记不记得",
    "我刚才说",
    "我刚刚说",
    "我之前说",
    "我之前提到",
    "我上次说",
    "回忆一下",
    "总结一下我",
    "当前会话",
    "我最近做过什么维护",
)

FIRST_PERSON_PATTERNS = (
    "我家",
    "我更在意",
    "我不希望",
    "结合我",
    "根据我",
    "适合我",
    "推荐我",
    "我这种情况",
    "我的户型",
    "我的使用诉求",
    "按我的情况",
)

KNOWLEDGE_QUERY_PATTERNS = (
    "推荐",
    "适合",
    "怎么选",
    "哪些参数",
    "是不是",
    "为什么",
    "多久",
    "如何",
    "怎么",
    "排查",
    "原因",
    "维护",
    "保养",
    "回充",
    "连不上",
    "出水",
    "吸力",
    "滤网",
    "滚刷",
    "边刷",
)

GREETING_PATTERNS = (
    "你好",
    "hi",
    "hello",
    "在吗",
    "谢谢",
    "多谢",
)

LOW_SIGNAL_ASSISTANT_PATTERNS = (
    "收到，我会记住",
    "我会记住这条信息",
    "好的，我记住了",
    "已记录",
)

INSUFFICIENT_ANSWER = "根据当前知识库无法确定。"

QUERY_EXPANSION_RULES = (
    {
        "match": ("app", "连不上"),
        "queries": (
            "APP 无法连接 2.4G WiFi 绑定 重启 网络",
            "APP 无法连接 机器人 网络 重启 更新",
        ),
        "terms": ("APP", "WiFi", "2.4G", "网络", "重启", "绑定", "更新"),
        "intent": "troubleshooting",
    },
    {
        "match": ("大户型",),
        "queries": (
            "大户型 选购 电池 续航 集尘 断点续扫 水箱",
            "大户型 参数 电池 集尘 自动洗拖布",
        ),
        "terms": ("大户型", "电池", "续航", "集尘", "断点续扫", "水箱"),
        "intent": "selection",
    },
    {
        "match": ("出水",),
        "queries": (
            "出水异常 水箱 出水管 传感器 水位 拖布",
            "拖地 出水异常 水箱 传感器 拖布 电机",
        ),
        "terms": ("出水", "水箱", "出水管", "传感器", "水位", "拖布", "电机"),
        "intent": "troubleshooting",
    },
    {
        "match": ("回充",),
        "queries": (
            "回充失败 充电座 障碍物 传感器 重启",
        ),
        "terms": ("回充", "充电座", "障碍物", "传感器", "重启"),
        "intent": "troubleshooting",
    },
    {
        "match": ("维护",),
        "queries": (
            "维护 保养 滚刷 边刷 滤网 尘盒",
        ),
        "terms": ("维护", "保养", "滚刷", "边刷", "滤网", "尘盒"),
        "intent": "maintenance",
    },
    {
        "match": ("保养",),
        "queries": (
            "保养 维护 滚刷 边刷 滤网 尘盒",
        ),
        "terms": ("维护", "保养", "滚刷", "边刷", "滤网", "尘盒"),
        "intent": "maintenance",
    },
)

DOMAIN_TERMS = (
    "APP",
    "WiFi",
    "2.4G",
    "5G",
    "网络",
    "更新",
    "绑定",
    "重启",
    "大户型",
    "电池",
    "续航",
    "集尘",
    "断点续扫",
    "水箱",
    "出水",
    "出水管",
    "传感器",
    "水位",
    "拖布",
    "电机",
    "维护",
    "保养",
    "滚刷",
    "边刷",
    "滤网",
    "尘盒",
    "吸力",
    "参数",
    "导航",
    "回充",
    "充电座",
    "故障",
    "异常",
)


@dataclass
class SessionState:
    session_id: str
    user_id: str
    agent: ReActAgent
    memory_tool: MemoryTool
    rag_tool: RAGTool
    history: list[Message] = field(default_factory=list)
    lock: RLock = field(default_factory=RLock)


class HelloRagAgentService:
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or load_settings()
        self.memory_store = SQLiteMemoryStore(self.settings.memory.db_path)
        self.memory_manager = MemoryManager(
            store=self.memory_store,
            default_top_k=self.settings.memory.default_top_k,
            working_max_entries=self.settings.memory.working_max_entries,
            working_ttl_minutes=self.settings.memory.working_ttl_minutes,
            profile_enabled=self.settings.memory.profile_enabled,
            profile_max_facts=self.settings.memory.profile_max_facts,
            assistant_memory_min_chars=self.settings.memory.assistant_memory_min_chars,
        )
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

    def create_session(self, user_id: str | None = None) -> str:
        return self._get_or_create_session(user_id=user_id).session_id

    def ask(self, query: str, session_id: str | None = None, user_id: str | None = None) -> tuple[str, str]:
        question = query.strip()
        if not question:
            raise ValueError("Query cannot be empty.")

        session = self._get_or_create_session(session_id, user_id=user_id)
        with session.lock:
            answer = self._answer(question=question, session=session)

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

    def get_user_id(self, session_id: str) -> str | None:
        session = self._sessions.get(session_id)
        return session.user_id if session else None

    def reset_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        with session.lock:
            session.history.clear()
            session.memory_tool.clear_session_context()
            session.agent.clear_history()

    def knowledge_stats(self) -> dict[str, object]:
        return self.knowledge_base.stats()

    def _get_or_create_session(self, session_id: str | None = None, user_id: str | None = None) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]

            new_session_id = session_id or uuid4().hex[:8]
            persisted_user_id = self.memory_store.get_user_id_for_session(new_session_id) if session_id else None
            resolved_user_id = (user_id or persisted_user_id or f"session:{new_session_id}").strip()
            memory_tool = MemoryTool(
                session_id=new_session_id,
                user_id=resolved_user_id,
                default_top_k=max(self.settings.memory.default_top_k, self.settings.agent.history_turns // 2, 2),
                max_entries=self.settings.memory.working_max_entries,
                store=self.memory_store,
                manager=self.memory_manager,
                working_ttl_minutes=self.settings.memory.working_ttl_minutes,
                profile_enabled=self.settings.memory.profile_enabled,
                profile_max_facts=self.settings.memory.profile_max_facts,
                assistant_memory_min_chars=self.settings.memory.assistant_memory_min_chars,
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
                user_id=resolved_user_id,
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

        profile_packet = session.memory_tool.build_profile_packet(
            query=query,
            top_k=min(self.settings.memory.default_top_k, self.settings.memory.profile_max_facts),
        )
        if profile_packet is not None:
            packets.append(profile_packet)

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

    def _answer(self, *, question: str, session: SessionState) -> str:
        memory_first_answer = self._answer_from_memory(question, session)
        if memory_first_answer is not None:
            return memory_first_answer

        if not self._is_greeting(question):
            direct_answer = self._answer_with_retrieval(query=question, session=session)
            if direct_answer.strip() and direct_answer != INSUFFICIENT_ANSWER:
                return direct_answer

        session.agent.clear_history()
        prompt = self._build_context(query=question, session=session)

        with contextlib.redirect_stdout(io.StringIO()):
            answer = session.agent.run(prompt)

        answer = self._normalize_answer(answer)
        if self._needs_fallback(answer):
            answer = self._answer_with_retrieval(query=question, session=session)
        return self._normalize_answer(answer)

    def _answer_with_retrieval(self, query: str, session: SessionState) -> str:
        memory_answer = self._answer_from_memory(query, session)
        if memory_answer is not None:
            return memory_answer

        memory_lines = self._collect_memory_lines(query=query, session=session)
        search_query = self._build_search_query(query=query, memory_lines=memory_lines)
        results = self._search_knowledge(query=query, search_query=search_query)
        if not results:
            if memory_lines:
                return self._build_memory_grounded_answer(query=query, memory_lines=memory_lines)
            return INSUFFICIENT_ANSWER

        evidence_briefs = self._build_structured_evidence_briefs(query=query)
        if not evidence_briefs:
            evidence_briefs = self._build_evidence_briefs(query=search_query, results=results)
        if not evidence_briefs:
            if memory_lines:
                return self._build_memory_grounded_answer(query=query, memory_lines=memory_lines)
            return INSUFFICIENT_ANSWER

        overview_answer = self._build_overview_answer(query=query)
        if overview_answer is not None:
            return overview_answer

        troubleshooting_answer = self._build_troubleshooting_answer(
            query=query,
            memory_lines=memory_lines,
            evidence_briefs=evidence_briefs,
        )
        if troubleshooting_answer is not None:
            return troubleshooting_answer

        extractive_answer = self._build_extractive_answer(
            query=query,
            memory_lines=memory_lines,
            evidence_briefs=evidence_briefs,
        )
        if extractive_answer is not None:
            return extractive_answer

        answer = self._generate_grounded_answer(
            query=query,
            memory_lines=memory_lines,
            evidence_briefs=evidence_briefs,
        )
        normalized = self._normalize_direct_answer(answer)
        if normalized:
            return normalized

        if memory_lines:
            return self._build_memory_grounded_answer(query=query, memory_lines=memory_lines)
        return INSUFFICIENT_ANSWER

    def _answer_from_memory(self, query: str, session: SessionState) -> str | None:
        if not self._is_memory_only_query(query):
            return None

        memory_lines = self._collect_memory_lines(query=query, session=session)
        if not memory_lines:
            return None

        if len(memory_lines) == 1:
            return f"记得。根据当前会话：{memory_lines[0]}"

        formatted = "\n".join(
            f"{index}. {line}"
            for index, line in enumerate(memory_lines, start=1)
        )
        return f"记得。根据当前会话：\n{formatted}"

    def _collect_memory_lines(self, *, query: str, session: SessionState) -> list[str]:
        filtered: list[str] = []
        seen: set[str] = set()

        profile_lines = session.memory_tool.get_profile_lines(
            query=query,
            limit=min(3, self.settings.memory.profile_max_facts),
        )
        if not profile_lines and self._is_mixed_query(query):
            profile_lines = session.memory_tool.get_profile_lines(limit=min(2, self.settings.memory.profile_max_facts))

        for line in profile_lines:
            normalized = self._normalize_memory_line(line)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            filtered.append(normalized)

        entries = session.memory_tool.search(
            query,
            top_k=max(self.settings.agent.history_turns, 4),
        )

        for entry in entries:
            content = entry.content.strip()
            if not content:
                continue
            if entry.role == "assistant" and any(marker in content for marker in LOW_SIGNAL_ASSISTANT_PATTERNS):
                continue

            normalized = self._normalize_memory_line(content)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            filtered.append(normalized)

        return filtered

    @staticmethod
    def _normalize_memory_line(content: str) -> str:
        normalized = re.sub(r"^(请记住|请记一下|帮我记一下|记一下)[，,:：]?\s*", "", content).strip()
        return normalized or content.strip()

    @staticmethod
    def _is_memory_query(query: str) -> bool:
        normalized = query.strip()
        if not normalized:
            return False

        lowered = normalized.lower()
        if "remember" in lowered or "memory" in lowered:
            return True

        return any(pattern in normalized for pattern in MEMORY_QUERY_PATTERNS)

    @classmethod
    def _is_memory_only_query(cls, query: str) -> bool:
        normalized = query.strip()
        if not normalized:
            return False
        if any(pattern in normalized for pattern in EXPLICIT_MEMORY_PATTERNS):
            return True
        return cls._is_memory_query(query) and not cls._is_knowledge_query(query)

    @staticmethod
    def _is_knowledge_query(query: str) -> bool:
        normalized = query.strip()
        if not normalized:
            return False
        return any(pattern in normalized for pattern in KNOWLEDGE_QUERY_PATTERNS)

    @staticmethod
    def _is_greeting(query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return False
        return any(normalized == pattern or normalized.startswith(f"{pattern} ") for pattern in GREETING_PATTERNS)

    @classmethod
    def _is_mixed_query(cls, query: str) -> bool:
        normalized = query.strip()
        if not normalized:
            return False
        return any(pattern in normalized for pattern in FIRST_PERSON_PATTERNS) and cls._is_knowledge_query(query)

    @classmethod
    def _should_use_direct_retrieval(cls, query: str) -> bool:
        if cls._is_greeting(query):
            return False
        return cls._is_mixed_query(query) or cls._is_knowledge_query(query)

    @staticmethod
    def _is_overview_query(query: str) -> bool:
        normalized = query.strip()
        if not normalized:
            return False
        return (
            "知识库" in normalized
            or "主要讲什么" in normalized
            or "主要内容" in normalized
            or "讲什么" in normalized
        )

    @staticmethod
    def _build_search_query(*, query: str, memory_lines: list[str]) -> str:
        if not memory_lines:
            return query
        focus = "；".join(memory_lines[:2])
        return f"{query}\n用户场景：{focus}"

    def _search_knowledge(self, *, query: str, search_query: str) -> list[object]:
        search_queries = self._expand_search_queries(query=query, search_query=search_query)
        strategies = ["keyword"]
        if self.settings.knowledge_base.retrieval_mode != "keyword":
            strategies.append(self.settings.knowledge_base.retrieval_mode)

        candidates: dict[str, dict[str, object]] = {}
        pool_size = max(
            self.settings.knowledge_base.top_k,
            self.settings.knowledge_base.rerank_pool_size,
        )
        for variant_rank, candidate_query in enumerate(search_queries, start=1):
            for strategy in strategies:
                results = self.knowledge_base.search(
                    candidate_query,
                    top_k=pool_size,
                    strategy=strategy,
                )
                for rank, item in enumerate(results, start=1):
                    chunk_id = item.chunk.chunk_id
                    candidate_score = self._score_candidate(
                        query=query,
                        item=item,
                        base_score=float(item.score),
                        rank=rank,
                        variant_rank=variant_rank,
                        candidate_query=str(candidate_query),
                    )
                    payload = candidates.setdefault(
                        chunk_id,
                        {
                            "item": item,
                            "best_rank": rank,
                            "best_score": candidate_score,
                            "query": candidate_query,
                            "strategy": strategy,
                            "variant_rank": variant_rank,
                        },
                    )
                    if rank < int(payload["best_rank"]):
                        payload["best_rank"] = rank
                    if candidate_score > float(payload["best_score"]):
                        payload["best_score"] = candidate_score
                        payload["item"] = item
                        payload["query"] = candidate_query
                        payload["strategy"] = strategy
                        payload["variant_rank"] = variant_rank

        ranked = sorted(
            candidates.values(),
            key=lambda payload: float(payload["best_score"]),
            reverse=True,
        )
        return self._select_diverse_results(query=query, ranked_payloads=ranked)

    def _expand_search_queries(self, *, query: str, search_query: str) -> list[str]:
        queries = [search_query, query.strip()]
        normalized = query.strip()
        lowered = normalized.lower()
        query_terms = self._extract_query_terms(query, search_query)
        for rule in QUERY_EXPANSION_RULES:
            if all(term.lower() in lowered for term in rule["match"]):
                queries.extend(rule["queries"])
                queries.append(self._format_query_variant(normalized, rule["terms"]))

        if self._is_selection_query(normalized):
            queries.append(self._format_query_variant(normalized, ("选购", "参数", *query_terms[:4])))
        if self._is_troubleshooting_query(normalized):
            queries.append(self._format_query_variant(normalized, ("故障", "排查", *query_terms[:4])))
        if self._is_maintenance_query(normalized):
            queries.append(self._format_query_variant(normalized, ("维护", "保养", *query_terms[:4])))
        if query_terms:
            queries.append(self._format_query_variant(normalized, tuple(query_terms[:6])))

        deduped: list[str] = []
        seen: set[str] = set()
        for item in queries:
            cleaned = item.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
            if len(deduped) >= self.settings.knowledge_base.max_query_variants:
                break
        return deduped

    @staticmethod
    def _format_query_variant(base_query: str, terms: tuple[str, ...]) -> str:
        cleaned_terms = [term.strip() for term in terms if isinstance(term, str) and term.strip()]
        if not cleaned_terms:
            return base_query
        return f"{base_query} {' '.join(cleaned_terms)}"

    def _score_candidate(
        self,
        *,
        query: str,
        item: object,
        base_score: float,
        rank: int,
        variant_rank: int,
        candidate_query: str,
    ) -> float:
        chunk = item.chunk
        haystack = f"{chunk.title}\n{chunk.content}"
        heading_text = " ".join(getattr(chunk, "heading_path", ()))
        score = base_score
        score += max(0, 12 - rank) * 2.0
        score += max(0, 8 - variant_rank) * 1.5

        strategy = str(getattr(item, "strategy", ""))
        if strategy == "hybrid":
            score += 10.0
        elif strategy == "vector":
            score += 6.0

        intent = self._detect_query_intent(query)
        for term in self._extract_query_terms(query, candidate_query):
            if term.lower() in chunk.title.lower():
                score += 8.0
            if term.lower() in heading_text.lower():
                score += 6.0
            if term.lower() in haystack.lower():
                score += 3.0

        if intent == "selection":
            if "选购指南" in chunk.title:
                score += 35.0
            if "故障排除" in chunk.title:
                score -= 6.0
        elif intent == "troubleshooting":
            if "故障排除" in chunk.title:
                score += 20.0
            if "选购指南" in chunk.title:
                score -= 6.0
        elif intent == "maintenance":
            if "维护保养" in chunk.title:
                score += 20.0
            if "选购指南" in chunk.title:
                score -= 4.0

        section_type = str(getattr(chunk, "section_type", ""))
        source = str(getattr(chunk, "source", "")).lower()
        if source.endswith(".txt"):
            score += 4.0
        elif source.endswith(".md"):
            score += 2.0
        elif source.endswith(".pdf"):
            score -= 4.0
        if intent == "troubleshooting" and section_type == "qa":
            score += 6.0
        if intent == "selection" and section_type in {"bullet", "paragraph"}:
            score += 4.0
        if query in haystack:
            score += 10.0
        if heading_text and query in heading_text:
            score += 6.0
        score += len(getattr(item, "match_terms", ())) * 2.0
        return score

    def _select_diverse_results(self, *, query: str, ranked_payloads: list[dict[str, object]]) -> list[object]:
        max_per_source = 1 if self._is_overview_query(query) else 2
        source_counts: dict[str, int] = {}
        selected: list[object] = []
        deferred: list[object] = []

        for payload in ranked_payloads:
            item = payload["item"]
            source = str(item.chunk.source)
            current = source_counts.get(source, 0)
            if current >= max_per_source:
                deferred.append(item)
                continue
            source_counts[source] = current + 1
            selected.append(item)
            if len(selected) >= self.settings.knowledge_base.top_k:
                return selected

        for item in deferred:
            selected.append(item)
            if len(selected) >= self.settings.knowledge_base.top_k:
                break
        return selected

    @classmethod
    def _detect_query_intent(cls, query: str) -> str:
        if cls._is_overview_query(query):
            return "overview"
        if cls._is_troubleshooting_query(query):
            return "troubleshooting"
        if cls._is_maintenance_query(query):
            return "maintenance"
        if cls._is_selection_query(query):
            return "selection"
        return "general"

    @staticmethod
    def _extract_query_terms(query: str, candidate_query: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        combined = f"{query} {candidate_query}"
        for term in DOMAIN_TERMS:
            if term.lower() in combined.lower() and term not in seen:
                seen.add(term)
                terms.append(term)
        for token in re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", combined):
            normalized = token.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)
        return terms

    @classmethod
    def _is_selection_query(cls, query: str) -> bool:
        normalized = query.strip()
        if not normalized:
            return False
        markers = ("选购", "推荐", "参数", "适合", "大户型", "怎么选", "哪个好", "值得买")
        if any(marker in normalized for marker in markers):
            return True
        return "是否" in normalized and "吸力" in normalized

    @staticmethod
    def _is_troubleshooting_query(query: str) -> bool:
        normalized = query.strip()
        if not normalized:
            return False
        markers = ("连不上", "故障", "异常", "回充", "出水", "排查", "检查", "重启", "无法", "报警")
        for marker in markers:
            if marker in normalized:
                return True
        return False

    @staticmethod
    def _is_maintenance_query(query: str) -> bool:
        normalized = query.strip()
        if not normalized:
            return False
        markers = ("维护", "保养", "多久", "更换", "清理", "滤网", "滚刷", "边刷", "尘盒")
        for marker in markers:
            if marker in normalized:
                return True
        return False

    def _build_evidence_briefs(self, *, query: str, results: list[object]) -> list[dict[str, object]]:
        query_tokens = set(self._tokenize_query(query))
        briefs: list[dict[str, object]] = []
        for item in results:
            points = self._extract_relevant_points(item.chunk.content, query_tokens)
            if getattr(item, "snippet", ""):
                points = [item.snippet, *points]
            deduped_points: list[str] = []
            seen_points: set[str] = set()
            for point in points:
                normalized = str(point).strip()
                if not normalized or normalized in seen_points:
                    continue
                seen_points.add(normalized)
                deduped_points.append(normalized)
                if len(deduped_points) >= self.settings.knowledge_base.max_evidence_points:
                    break
            points = deduped_points
            if not points:
                continue
            briefs.append(
                {
                    "source": item.chunk.source,
                    "title": item.chunk.title,
                    "citation": getattr(item, "citation", item.chunk.source),
                    "score": round(item.score, 4),
                    "points": points,
                }
            )
        return briefs

    def _build_structured_evidence_briefs(self, *, query: str) -> list[dict[str, object]]:
        ranked_points: list[dict[str, object]] = []
        seen: set[str] = set()
        for chunk in self.knowledge_base._chunks:
            for question, answer in self._extract_chunk_qa_pairs(chunk.content):
                score = self._score_structured_point(
                    query=query,
                    title=chunk.title,
                    question=question,
                    answer=answer,
                )
                if score <= 0:
                    continue
                key = f"{chunk.source}|{question}|{answer}"
                if key in seen:
                    continue
                seen.add(key)
                ranked_points.append(
                    {
                        "source": chunk.source,
                        "title": chunk.title,
                        "citation": self.knowledge_base.format_citation(chunk),
                        "score": round(score, 4),
                        "question": question,
                        "answer": answer,
                        "points": [f"{question}：{answer}"],
                    }
                )

        ranked_points.sort(key=lambda item: float(item["score"]), reverse=True)
        if self._is_troubleshooting_query(query):
            ranked_points = self._filter_troubleshooting_briefs(query=query, briefs=ranked_points)
        return ranked_points[: self.settings.knowledge_base.top_k]

    @staticmethod
    def _tokenize_query(text: str) -> list[str]:
        return [token for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower()) if token.strip()]

    def _extract_relevant_points(self, content: str, query_tokens: set[str]) -> list[str]:
        candidates: list[tuple[float, str]] = []
        heading = ""

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            cleaned = line.strip(" -*\t")
            if not cleaned:
                continue

            if re.match(r"^\d+\.\s*\*\*.*\*\*$", line):
                heading = self._clean_heading(cleaned)
                continue

            text = cleaned
            if heading and (raw_line.lstrip().startswith("-") or raw_line.lstrip().startswith("•")):
                text = f"{heading}：{cleaned}"

            score = self._score_evidence_point(text=text, query_tokens=query_tokens)
            candidates.append((score, text))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        for score, text in candidates:
            normalized = text.strip()
            if normalized in seen:
                continue
            if score <= 0.2 and selected:
                continue
            seen.add(normalized)
            selected.append(normalized)
            if len(selected) >= 5:
                break

        return selected or [text for _, text in candidates[:2]]

    def _extract_chunk_qa_pairs(self, content: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        current_question = ""
        current_answers: list[str] = []

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            heading_match = re.match(r"^\d+\.\s*\*\*(.+?)\*\*$", line)
            if heading_match:
                if current_question and current_answers:
                    pairs.append((current_question, " ".join(current_answers)))
                current_question = heading_match.group(1).strip("？? ").strip()
                current_answers = []
                continue

            inline_match = re.match(r"^\d+\.\s*([^：:]+)[：:]\s*(.+)$", line)
            if inline_match:
                if current_question and current_answers:
                    pairs.append((current_question, " ".join(current_answers)))
                left = inline_match.group(1).strip()
                right = inline_match.group(2).strip()
                current_question = ""
                current_answers = []

                if left == "故障现象" and "；检测：" in right:
                    symptom, rest = right.split("；检测：", 1)
                    pairs.append((symptom.strip(), f"检测：{rest.strip()}"))
                else:
                    pairs.append((left, right))
                continue

            if current_question and raw_line.lstrip().startswith("-"):
                current_answers.append(line.lstrip("- ").strip())
                continue

            if current_question and current_answers and not re.match(r"^\d+\.", line):
                current_answers.append(line)

        if current_question and current_answers:
            pairs.append((current_question, " ".join(current_answers)))

        compact_content = re.sub(r"\r\n?", "\n", content)
        markdown_pairs = re.findall(
            r"\d+\.\s*\*\*(.+?)\*\*\s*[-：:]\s*(.+?)(?=(?:\d+\.\s*\*\*|$))",
            compact_content,
            flags=re.S,
        )
        for question, answer in markdown_pairs:
            normalized_question = re.sub(r"\s+", " ", question).strip("？? ").strip()
            normalized_answer = re.sub(r"\s+", " ", answer).strip(" -\n\t")
            if normalized_question and normalized_answer:
                pairs.append((normalized_question, normalized_answer))

        deduped: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for question, answer in pairs:
            key = (question.strip(), answer.strip())
            if not key[0] or not key[1] or key in seen_pairs:
                continue
            seen_pairs.add(key)
            deduped.append(key)
        return deduped

    def _score_structured_point(
        self,
        *,
        query: str,
        title: str,
        question: str,
        answer: str,
    ) -> float:
        intent = self._detect_query_intent(query)
        terms = self._extract_query_terms(query, query)
        question_text = question.lower()
        answer_text = answer.lower()
        title_text = title.lower()
        score = 0.0

        for term in terms:
            lowered = term.lower()
            if lowered in question_text:
                score += 10.0
            elif lowered in answer_text:
                score += 5.0
            elif lowered in title_text:
                score += 3.0

        if intent == "selection":
            if any(marker in question for marker in ("大户型", "参数", "注意什么", "选购")):
                score += 12.0
            if "选购" in title:
                score += 10.0
        elif intent == "troubleshooting":
            if any(marker in question for marker in ("怎么办", "怎么处理", "怎么解决", "原因", "异常")):
                score += 10.0
            if "故障" in title:
                score += 10.0
            if not self._looks_like_troubleshooting_item(question=question, answer=answer):
                score -= 12.0
        elif intent == "maintenance":
            if any(marker in question for marker in ("多久", "更换", "清理", "保养", "维护")):
                score += 10.0
            if "维护" in title:
                score += 10.0

        normalized_query = query.strip().lower()
        combined = f"{question} {answer}".lower()
        if normalized_query and normalized_query in combined:
            score += 12.0
        return score

    def _filter_troubleshooting_briefs(
        self,
        *,
        query: str,
        briefs: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        filtered: list[dict[str, object]] = []
        for item in briefs:
            question = str(item.get("question", ""))
            answer = str(item.get("answer", ""))
            combined = f"{question} {answer}"
            if "app" in query.lower() and any(marker in combined for marker in ("密码", "登录")):
                continue
            if not self._looks_like_troubleshooting_item(question=question, answer=answer):
                continue
            if self._troubleshooting_relevance(query=query, question=question, answer=answer) <= 0:
                continue
            filtered.append(item)

        if filtered:
            return filtered
        return briefs

    def _troubleshooting_relevance(self, *, query: str, question: str, answer: str) -> float:
        combined = f"{question} {answer}".lower()
        score = 0.0
        for term in self._extract_query_terms(query, query):
            lowered = term.lower()
            if lowered in combined:
                score += 2.0

        topic_terms = self._troubleshooting_topic_terms(query)
        if topic_terms:
            topic_hits = sum(1 for term in topic_terms if term.lower() in combined)
            score += topic_hits * 3.0
            if topic_hits == 0:
                score -= 12.0

        if "app" in query.lower():
            if "app" in combined:
                score += 3.0
            if any(marker in combined for marker in ("wifi", "2.4g", "网络", "绑定", "路由器")):
                score += 4.0
            if any(marker in combined for marker in ("密码", "登录", "高级功能", "语音控制")):
                score -= 10.0
        if "出水" in query:
            if any(marker in combined for marker in ("出水", "水箱", "出水管", "拖布", "水位")):
                score += 5.0
        if "回充" in query:
            if any(marker in combined for marker in ("回充", "充电座", "充电", "障碍物", "传感器")):
                score += 5.0
        return score

    @staticmethod
    def _troubleshooting_topic_terms(query: str) -> tuple[str, ...]:
        if "出水" in query:
            return ("出水", "水箱", "出水管", "拖布", "水位", "拖地")
        if "回充" in query:
            return ("回充", "充电", "充电座", "传感器", "障碍物")
        if "app" in query.lower():
            return ("WiFi", "2.4G", "网络", "绑定", "路由器", "连接")
        return ()

    @staticmethod
    def _looks_like_troubleshooting_item(*, question: str, answer: str) -> bool:
        combined = f"{question} {answer}"
        if any(marker in question for marker in ("怎么办", "怎么处理", "怎么解决", "原因", "异常", "失败", "无法")):
            return True
        return any(
            marker in combined
            for marker in ("检查", "检测", "修复", "重启", "重新", "故障", "异常", "报警", "无法", "失败")
        )

    @staticmethod
    def _score_evidence_point(*, text: str, query_tokens: set[str]) -> float:
        text_tokens = set(HelloRagAgentService._tokenize_query(text))
        overlap = len(text_tokens & query_tokens) if query_tokens else 0
        score = overlap * 2.5
        if "：" in text or ":" in text:
            score += 0.4
        if len(text) <= 90:
            score += 0.2
        return score

    @staticmethod
    def _clean_heading(line: str) -> str:
        cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
        cleaned = cleaned.strip("*")
        cleaned = cleaned.rstrip("？?")
        return cleaned.strip()

    def _build_overview_answer(self, *, query: str) -> str | None:
        if not self._is_overview_query(query):
            return None

        titles = {chunk.title for chunk in self.knowledge_base._chunks}
        topics: list[str] = []
        if any("选购" in title for title in titles):
            topics.append("选购参数和不同家庭场景的机型建议")
        if any("维护" in title for title in titles):
            topics.append("日常维护、耗材清理和保养周期")
        if any("故障" in title for title in titles):
            topics.append("常见故障排查，比如回充、联网和异常报警")
        if any("扫拖" in title for title in titles):
            topics.append("扫拖一体机的使用、出水和拖地相关问题")
        if any("100问" in title for title in titles):
            topics.append("扫地机器人基础原理、功能问答和使用技巧")

        if not topics:
            return None

        bullets = "\n".join(
            f"{index}. {topic}"
            for index, topic in enumerate(topics[:5], start=1)
        )
        return f"这个知识库主要围绕扫地机器人和扫拖一体机的使用知识，重点包括：\n{bullets}"

    def _build_extractive_answer(
        self,
        *,
        query: str,
        memory_lines: list[str],
        evidence_briefs: list[dict[str, object]],
    ) -> str | None:
        points: list[str] = []
        seen: set[str] = set()
        for item in evidence_briefs:
            for point in item["points"]:
                normalized = str(point).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                points.append(normalized)
                if len(points) >= 5:
                    break
            if len(points) >= 5:
                break

        if not points:
            return None

        if memory_lines:
            memory_intro = "结合你当前会话，先注意这些条件：\n" + "\n".join(
                f"{index}. {line}" for index, line in enumerate(memory_lines[:2], start=1)
            )
            evidence_intro = "再结合知识库，可以重点看："
            bullets = "\n".join(f"{index}. {point}" for index, point in enumerate(points, start=1))
            return f"{memory_intro}\n{evidence_intro}\n{bullets}"

        lead = "根据知识库，可以重点看："
        if any(keyword in query for keyword in ("多久", "周期", "多长时间")):
            lead = "根据知识库，相关周期信息主要是："
        elif any(keyword in query for keyword in ("原因", "为什么", "异常")):
            lead = "根据知识库，常见原因主要有："
        elif any(keyword in query for keyword in ("怎么", "如何", "排查")):
            lead = "根据知识库，可以按这些要点排查："
        elif any(keyword in query for keyword in ("哪些", "包括", "关注")):
            lead = "根据知识库，重点包括："

        bullets = "\n".join(f"{index}. {point}" for index, point in enumerate(points, start=1))
        return f"{lead}\n{bullets}"

    def _build_troubleshooting_answer(
        self,
        *,
        query: str,
        memory_lines: list[str],
        evidence_briefs: list[dict[str, object]],
    ) -> str | None:
        if not self._is_troubleshooting_query(query):
            return None

        troubleshooting_briefs = self._filter_troubleshooting_briefs(query=query, briefs=evidence_briefs)
        steps: list[str] = []
        seen: set[str] = set()
        for item in troubleshooting_briefs:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            candidate_steps = self._split_troubleshooting_answer(question=question, answer=answer)
            for step in candidate_steps:
                normalized = step.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                steps.append(normalized)
                if len(steps) >= 4:
                    break
            if len(steps) >= 4:
                break

        if not steps:
            fallback_steps = self._extract_troubleshooting_steps_from_points(query=query, briefs=evidence_briefs)
            for step in fallback_steps:
                normalized = step.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                steps.append(normalized)
                if len(steps) >= 4:
                    break

        if not steps:
            return None

        lead = "根据知识库，可以优先检查："
        if any(marker in query for marker in ("原因", "为什么", "异常")):
            lead = "根据知识库，常见原因和检查点主要是："

        bullets = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
        if memory_lines:
            memory_intro = "\n".join(f"{index}. {line}" for index, line in enumerate(memory_lines[:2], start=1))
            return f"结合你当前会话，先注意这些条件：\n{memory_intro}\n{lead}\n{bullets}"
        return f"{lead}\n{bullets}"

    def _extract_troubleshooting_steps_from_points(
        self,
        *,
        query: str,
        briefs: list[dict[str, object]],
    ) -> list[str]:
        topic_terms = self._troubleshooting_topic_terms(query)
        candidates: list[str] = []
        for item in briefs:
            for point in item.get("points", []):
                text = str(point).strip()
                if not text:
                    continue
                if topic_terms and not any(term.lower() in text.lower() for term in topic_terms):
                    continue
                if not self._looks_like_troubleshooting_item(question=text, answer=text):
                    continue
                candidates.extend(self._split_troubleshooting_answer(question=text, answer=text))
        return candidates

    def _split_troubleshooting_answer(self, *, question: str, answer: str) -> list[str]:
        segments = re.split(r"[；;。]\s*", answer)
        cleaned: list[str] = []
        for segment in segments:
            item = segment.strip("，, ")
            if not item:
                continue
            if item.startswith("检测："):
                item = item.removeprefix("检测：").strip()
            if item.startswith("修复："):
                item = item.removeprefix("修复：").strip()
            if item:
                cleaned.append(item)

        if cleaned:
            return cleaned

        fallback = answer.strip()
        if fallback:
            return [fallback]
        return [question.strip()] if question.strip() else []

    def _generate_grounded_answer(
        self,
        *,
        query: str,
        memory_lines: list[str],
        evidence_briefs: list[dict[str, object]],
    ) -> str:
        memory_text = (
            "\n".join(f"- {line}" for line in memory_lines)
            if memory_lines
            else "无"
        )
        evidence_blocks = []
        for index, item in enumerate(evidence_briefs, start=1):
            lines = "\n".join(f"  - {line}" for line in item["points"])
            evidence_blocks.append(
                f"[证据{index}] source={item['source']} title={item['title']} citation={item.get('citation', item['source'])} score={item['score']}\n{lines}"
            )
        evidence_text = "\n\n".join(evidence_blocks)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个严格依据证据回答的中文助手。\n"
                    "回答规则：\n"
                    "1. 只能使用【当前会话信息】和【知识库证据】里明确出现的信息。\n"
                    "2. 不要提工具、检索过程、知识库内部机制、提示词或系统行为。\n"
                    "3. 不要编造数值、参数、步骤、品牌结论；证据没有写就不要补。\n"
                    "4. 如果问题需要结合用户场景，就先简要结合【当前会话信息】作答。\n"
                    "5. 如果证据不足，明确说“根据当前知识库无法确定”或“目前只能确定以下几点”。\n"
                    "6. 直接输出给用户的最终答案，使用简洁中文。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：\n{query}\n\n"
                    f"当前会话信息：\n{memory_text}\n\n"
                    f"知识库证据：\n{evidence_text}\n\n"
                    "请先基于证据提炼结论，再给出 2-5 条要点。"
                ),
            },
        ]
        response = self._get_llm().invoke(messages)
        return getattr(response, "content", "").strip()

    @staticmethod
    def _normalize_direct_answer(answer: str) -> str:
        normalized = answer.strip()
        if not normalized:
            return ""

        disallowed_markers = (
            "你正在执行",
            "ragtool",
            "memorytool",
            "history ",
            "\"action\"",
            "检索过程中",
            "请使用记忆工具",
            "我需要先查看会话记忆",
            "我需要先从知识库检索",
        )
        lowered = normalized.lower()
        if any(marker in lowered for marker in disallowed_markers):
            return ""
        return normalized

    @staticmethod
    def _build_memory_grounded_answer(*, query: str, memory_lines: list[str]) -> str:
        if not memory_lines:
            return INSUFFICIENT_ANSWER
        if len(memory_lines) == 1:
            return f"结合当前会话，我目前能确定：{memory_lines[0]}"
        bullets = "\n".join(f"{index}. {line}" for index, line in enumerate(memory_lines, start=1))
        return f"结合当前会话，我目前能确定：\n{bullets}"

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
            "无法在限定步数内",
            "限定步数",
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
