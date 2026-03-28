from __future__ import annotations

from collections.abc import Iterator
import json
import re
from typing import Any

from hello_agents.core.message import Message

from hello_rag_agent.agent_service import (
    HelloRagAgentService as BaseHelloRagAgentService,
    INSUFFICIENT_ANSWER,
    SessionState,
)


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

GREETING_PATTERNS = ("你好", "hi", "hello", "在吗", "谢谢", "多谢")

LOW_SIGNAL_ASSISTANT_PATTERNS = (
    "收到，我会记住",
    "我会记住这条信息",
    "好的，我记住了",
    "已记录",
)


class HelloRagAgentService(BaseHelloRagAgentService):
    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 14) -> Iterator[str]:
        normalized = text.strip()
        if not normalized:
            return
        for index in range(0, len(normalized), chunk_size):
            yield normalized[index : index + chunk_size]

    def stream_ask(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> tuple[Iterator[str], str]:
        question = query.strip()
        if not question:
            raise ValueError("Query cannot be empty.")

        session = self._get_or_create_session(session_id, user_id=user_id)

        def _generator() -> Iterator[str]:
            chunks: list[str] = []
            with session.lock:
                for chunk in self._stream_answer(question=question, session=session):
                    chunks.append(chunk)
                    yield chunk

                answer = "".join(chunks).strip()
                user_message = Message(content=question, role="user")
                assistant_message = Message(content=answer, role="assistant")
                session.history.extend([user_message, assistant_message])
                session.memory_tool.remember_message(user_message)
                session.memory_tool.remember_message(assistant_message)

        return _generator(), session.session_id

    def _stream_answer(self, *, question: str, session: SessionState) -> Iterator[str]:
        memory_first_answer = self._answer_from_memory(question, session)
        if memory_first_answer is not None:
            yield from self._chunk_text(memory_first_answer)
            return

        if not self._is_greeting(question):
            stream = self._stream_answer_with_retrieval(query=question, session=session)
            if stream is not None:
                yield from stream
                return

        answer = self._answer(question=question, session=session)
        yield from self._chunk_text(answer)

    def _stream_answer_with_retrieval(self, *, query: str, session: SessionState) -> Iterator[str] | None:
        memory_answer = self._answer_from_memory(query, session)
        if memory_answer is not None:
            return self._chunk_text(memory_answer)

        memory_lines = self._collect_memory_lines(query=query, session=session)
        results = self._search_knowledge_simple(query=query, memory_lines=memory_lines)
        if not results:
            if memory_lines:
                return self._chunk_text(self._build_memory_grounded_answer(query=query, memory_lines=memory_lines))
            return self._chunk_text(INSUFFICIENT_ANSWER)

        evidence_briefs = self._build_evidence_briefs(query=query, results=results)
        if not evidence_briefs:
            if memory_lines:
                return self._chunk_text(self._build_memory_grounded_answer(query=query, memory_lines=memory_lines))
            return self._chunk_text(INSUFFICIENT_ANSWER)

        overview_answer = self._build_overview_answer(query=query)
        if overview_answer is not None:
            return self._chunk_text(overview_answer)

        troubleshooting_answer = self._build_troubleshooting_answer(
            query=query,
            memory_lines=memory_lines,
            evidence_briefs=evidence_briefs,
        )
        if troubleshooting_answer is not None:
            return self._chunk_text(troubleshooting_answer)

        extractive_answer = self._build_extractive_answer(
            query=query,
            memory_lines=memory_lines,
            evidence_briefs=evidence_briefs,
        )
        if extractive_answer is not None:
            return self._chunk_text(extractive_answer)

        return self._stream_grounded_answer(
            query=query,
            memory_lines=memory_lines,
            evidence_briefs=evidence_briefs,
        )

    def _search_knowledge_simple(self, *, query: str, memory_lines: list[str]) -> list[object]:
        variants = [query]
        if memory_lines:
            variants.append(self._build_search_query(query=query, memory_lines=memory_lines))

        strategies = ["keyword"]
        if self.settings.knowledge_base.retrieval_mode != "keyword":
            strategies.append(self.settings.knowledge_base.retrieval_mode)

        best_by_chunk: dict[str, object] = {}
        top_k = self.settings.knowledge_base.top_k
        pool_size = max(top_k, self.settings.knowledge_base.rerank_pool_size)
        for variant in variants:
            for strategy in strategies:
                for item in self.knowledge_base.search(variant, top_k=pool_size, strategy=strategy):
                    chunk_id = item.chunk.chunk_id
                    current = best_by_chunk.get(chunk_id)
                    if current is None or float(item.score) > float(current.score):
                        best_by_chunk[chunk_id] = item

        ranked = sorted(best_by_chunk.values(), key=lambda item: float(item.score), reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _normalize_memory_line(content: str) -> str:
        normalized = re.sub(r"^(请记住|请记一下|帮我记一下|记一下)[：:，,\s]*", "", content).strip()
        return normalized or content.strip()

    @staticmethod
    def _is_memory_query(query: str) -> bool:
        normalized = query.strip()
        if not normalized:
            return False
        lowered = normalized.lower()
        if "remember" in lowered or "memory" in lowered:
            return True
        markers = (
            "你还记得",
            "还记得吗",
            "记不记得",
            "我刚才说",
            "我刚刚说",
            "我之前说",
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
        return any(pattern in normalized for pattern in markers)

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
        markers = (
            "推荐",
            "适合",
            "怎么选",
            "哪些参数",
            "是否",
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
            "连接",
            "WiFi",
            "APP",
            "出水",
            "吸力",
            "滤网",
            "滚刷",
            "边刷",
        )
        return bool(normalized) and any(pattern in normalized for pattern in markers)

    @staticmethod
    def _is_greeting(query: str) -> bool:
        normalized = query.strip().lower()
        return bool(normalized) and any(normalized == pattern or normalized.startswith(f"{pattern} ") for pattern in GREETING_PATTERNS)

    @classmethod
    def _is_mixed_query(cls, query: str) -> bool:
        normalized = query.strip()
        first_person_markers = (
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
        return bool(normalized) and any(pattern in normalized for pattern in first_person_markers) and cls._is_knowledge_query(query)

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

    @classmethod
    def _is_selection_query(cls, query: str) -> bool:
        normalized = query.strip()
        markers = ("选购", "推荐", "参数", "适合", "大户型", "怎么选", "哪个好", "值得买")
        return bool(normalized) and (any(marker in normalized for marker in markers) or ("是否" in normalized and "吸力" in normalized))

    @staticmethod
    def _is_troubleshooting_query(query: str) -> bool:
        normalized = query.strip()
        markers = (
            "连不上",
            "连接不上",
            "连接失败",
            "故障",
            "异常",
            "暂停",
            "停机",
            "停止",
            "中断",
            "突然停止",
            "自动回充",
            "卡住",
            "回充",
            "出水",
            "排查",
            "检查",
            "重启",
            "无法",
            "报警",
            "WiFi",
            "APP",
            "绑定",
            "配网",
        )
        return bool(normalized) and any(marker in normalized for marker in markers)

    @staticmethod
    def _is_maintenance_query(query: str) -> bool:
        normalized = query.strip()
        markers = ("维护", "保养", "多久", "更换", "清理", "滤网", "滚刷", "边刷", "尘盒")
        return bool(normalized) and any(marker in normalized for marker in markers)

    @staticmethod
    def _build_search_query(*, query: str, memory_lines: list[str]) -> str:
        if not memory_lines:
            return query
        focus = "；".join(memory_lines[:3])
        return f"{query}\n用户场景：{focus}"

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
            if normalized and normalized not in seen:
                seen.add(normalized)
                filtered.append(normalized)

        entries = session.memory_tool.search(query, top_k=max(self.settings.agent.history_turns, 4))
        for entry in entries:
            content = entry.content.strip()
            if not content:
                continue
            if entry.role == "assistant" and any(marker in content for marker in LOW_SIGNAL_ASSISTANT_PATTERNS):
                continue
            normalized = self._normalize_memory_line(content)
            if normalized and normalized not in seen:
                seen.add(normalized)
                filtered.append(normalized)
        return filtered

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
        bullets = self._format_answer_bullets(topics[:5])
        return (
            "这个知识库主要就是围绕扫地机器人和扫拖一体机在实际使用里常见的问题来整理的。"
            "如果你想快速了解，可以先看这几块：\n"
            f"{bullets}"
        )

    def _build_evidence_briefs(self, *, query: str, results: list[object]) -> list[dict[str, object]]:
        briefs: list[dict[str, object]] = []
        query_terms = self._tokenize_query(query)
        for item in results:
            points = self._extract_relevant_points(item.chunk.content, set(query_terms))
            if getattr(item, "snippet", ""):
                points = [item.snippet, *points]
            deduped: list[str] = []
            seen: set[str] = set()
            for point in points:
                normalized = str(point).strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    deduped.append(normalized)
                if len(deduped) >= self.settings.knowledge_base.max_evidence_points:
                    break
            if deduped:
                briefs.append(
                    {
                        "source": item.chunk.source,
                        "title": item.chunk.title,
                        "citation": getattr(item, "citation", item.chunk.source),
                        "score": round(item.score, 4),
                        "points": deduped,
                    }
                )
        return briefs

    def _extract_relevant_points(self, content: str, query_tokens: set[str]) -> list[str]:
        candidates: list[tuple[float, str]] = []
        for raw_line in content.splitlines():
            line = raw_line.strip(" -*\t")
            if not line or line.startswith("#"):
                continue
            text_tokens = set(self._tokenize_query(line))
            overlap = len(text_tokens & query_tokens) if query_tokens else 0
            score = overlap * 2.0
            if "：" in line or ":" in line:
                score += 0.4
            if len(line) <= 120:
                score += 0.2
            candidates.append((score, line))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        for score, line in candidates:
            if score <= 0 and selected:
                continue
            if line not in seen:
                seen.add(line)
                selected.append(line)
            if len(selected) >= 5:
                break
        return selected

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
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    humanized = self._humanize_answer_point(normalized)
                    if humanized:
                        points.append(humanized)
                if len(points) >= 5:
                    break
            if len(points) >= 5:
                break
        if not points:
            return None

        if memory_lines:
            memory_intro = self._build_memory_preface(memory_lines, limit=3)
            bullets = self._format_answer_bullets(points)
            return f"{memory_intro}\n再结合知识库，我会更建议你重点看这几项：\n{bullets}"

        lead = "如果是你这个问题，我会先重点看这几项："
        if any(keyword in query for keyword in ("多久", "周期", "多长时间")):
            lead = "如果你主要想确认周期，可以先记这几条："
        elif any(keyword in query for keyword in ("原因", "为什么", "异常")):
            lead = "更常见的原因，基本集中在这几类："
        elif any(keyword in query for keyword in ("怎么", "如何", "排查")):
            lead = "你可以先从这几步看起："
        elif any(keyword in query for keyword in ("哪些", "包括", "关注")):
            lead = "如果要抓重点，我会先看这几项："

        bullets = self._format_answer_bullets(points)
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
        topic_terms = self._troubleshooting_topic_terms(query)
        steps: list[str] = []
        seen: set[str] = set()
        for item in troubleshooting_briefs:
            for point in item["points"]:
                text = str(point).strip()
                if not text:
                    continue
                if topic_terms and not any(term.lower() in text.lower() for term in topic_terms):
                    continue
                if text in seen:
                    continue
                seen.add(text)
                humanized = self._humanize_answer_point(text)
                if humanized:
                    steps.append(humanized)
                if len(steps) >= 4:
                    break
            if len(steps) >= 4:
                break

        if not steps:
            return None

        lead = "你可以先按这个顺序排查："
        if any(marker in query for marker in ("原因", "为什么", "异常")):
            lead = "这类问题更常见的原因和检查点，大多在这几项："
        natural_steps = [self._humanize_troubleshooting_step(index, step) for index, step in enumerate(steps)]
        bullets = self._format_answer_bullets(natural_steps)
        if memory_lines:
            memory_intro = self._build_memory_preface(memory_lines, limit=3)
            return f"{memory_intro}\n{lead}\n{bullets}"
        return f"{lead}\n{bullets}"

    @staticmethod
    def _troubleshooting_topic_terms(query: str) -> tuple[str, ...]:
        if any(marker in query for marker in ("暂停", "停机", "停止", "中断", "突然停止", "自动回充", "卡住")):
            return (
                "暂停",
                "停机",
                "停止",
                "频繁暂停",
                "突然停止清扫",
                "中断",
                "清扫",
                "回充",
                "障碍物",
                "传感器",
                "电量",
                "尘盒",
                "滤网",
                "滚刷",
            )
        if "出水" in query:
            return ("出水", "水箱", "出水管", "拖布", "水位", "拖地")
        if "回充" in query:
            return ("回充", "充电", "充电座", "传感器", "障碍物")
        lowered = query.lower()
        if "app" in lowered or "wifi" in lowered or "连接" in query or "绑定" in query or "配网" in query:
            return ("WiFi", "2.4G", "网络", "绑定", "路由器", "连接")
        return ()

    @staticmethod
    def _troubleshooting_excluded_terms(query: str) -> tuple[str, ...]:
        if any(marker in query for marker in ("暂停", "停机", "停止", "中断", "突然停止", "自动回充", "卡住")):
            return ("拖布", "拖地", "水箱", "出水", "APP", "WiFi", "充电座")
        return ()

    @staticmethod
    def _is_pause_like_query(query: str) -> bool:
        return any(marker in query for marker in ("暂停", "停机", "停止", "中断", "突然停止", "自动回充", "卡住"))

    @staticmethod
    def _split_troubleshooting_answer(*, question: str, answer: str) -> list[str]:
        segments = re.split(r"[；。\n]+", answer)
        cleaned: list[str] = []
        for segment in segments:
            item = segment.strip(" ：:，, ")
            if item:
                cleaned.append(item)
        return cleaned or ([answer.strip()] if answer.strip() else [question.strip()])

    def _grounded_messages(
        self,
        *,
        query: str,
        memory_lines: list[str],
        evidence_briefs: list[dict[str, object]],
    ) -> list[dict[str, str]]:
        memory_text = "\n".join(f"- {line}" for line in memory_lines) if memory_lines else "无"
        evidence_blocks = []
        for index, item in enumerate(evidence_briefs, start=1):
            lines = "\n".join(f"  - {line}" for line in item["points"])
            evidence_blocks.append(
                f"[证据{index}] source={item['source']} title={item['title']} citation={item.get('citation', item['source'])} score={item['score']}\n{lines}"
            )
        evidence_text = "\n\n".join(evidence_blocks)
        return [
            {
                "role": "system",
                "content": (
                    "你是一个严格依据证据、但说话自然的中文助手。\n"
                    "1. 只能使用【当前会话信息】和【知识库证据】里明确出现的信息。\n"
                    "2. 不要提工具、检索过程、系统机制。\n"
                    "3. 没有证据支持的内容不要补充。\n"
                    "4. 先直接回答用户最关心的问题，语气尽量口语化、像日常解释，不要写成报告。\n"
                    "5. 如果问题需要结合用户场景，就先结合当前会话信息再回答。\n"
                    "6. 如果证据不足，要坦诚说明按当前知识库的信息暂时还不能更确定地判断，或者目前只能确定以下几点。\n"
                    "7. 最终答案优先用简短自然的中文；只有在内容本身适合分点时，再列 2-4 条关键点。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：\n{query}\n\n"
                    f"当前会话信息：\n{memory_text}\n\n"
                    f"知识库证据：\n{evidence_text}\n\n"
                    "请先用 1-2 句直接回答用户的问题；如果需要，再补充 2-4 条关键要点。"
                ),
            },
        ]

    def _generate_grounded_answer(
        self,
        *,
        query: str,
        memory_lines: list[str],
        evidence_briefs: list[dict[str, object]],
    ) -> str:
        response = self._get_llm().invoke(self._grounded_messages(query=query, memory_lines=memory_lines, evidence_briefs=evidence_briefs))
        return getattr(response, "content", "").strip()

    def _stream_grounded_answer(
        self,
        *,
        query: str,
        memory_lines: list[str],
        evidence_briefs: list[dict[str, object]],
    ) -> Iterator[str]:
        for chunk in self._get_llm().stream_invoke(
            self._grounded_messages(query=query, memory_lines=memory_lines, evidence_briefs=evidence_briefs)
        ):
            if chunk:
                yield chunk

    @staticmethod
    def _normalize_direct_answer(answer: str) -> str:
        normalized = answer.strip()
        if not normalized:
            return ""
        disallowed_markers = (
            "ragtool",
            "memorytool",
            "history ",
            "\"action\"",
            "检索过程",
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
            return f"我记得你前面提过：{memory_lines[0]}"
        bullets = "\n".join(f"- {line}" for line in memory_lines)
        return f"我记得你前面提过这些：\n{bullets}"

    @staticmethod
    def _select_salient_memory_lines(memory_lines: list[str], limit: int = 3) -> list[str]:
        if not memory_lines:
            return []
        preferred_markers = (
            "偏好",
            "更在意",
            "地面",
            "木地板",
            "宠物",
            "长毛猫",
            "维护状态",
            "过滤网",
            "滚刷",
            "回充",
            "APP",
            "静音",
            "缠毛",
        )
        ranked: list[tuple[int, int, str]] = []
        for index, line in enumerate(memory_lines):
            score = sum(1 for marker in preferred_markers if marker in line)
            score -= min(len(line) // 80, 2)
            ranked.append((score, -index, line))
        ranked.sort(reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        for _, _, line in ranked:
            normalized = line.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                selected.append(normalized)
            if len(selected) >= limit:
                break
        return selected or memory_lines[:limit]

    def _humanize_answer_point(self, point: str) -> str:
        cleaned = self._clean_answer_point(point)
        if not cleaned:
            return ""

        def _soften_clause(text: str, *, drop_prefixes: tuple[str, ...] = ()) -> str:
            softened = text.strip(" ：:；;，,。")
            for prefix in drop_prefixes:
                if softened.startswith(prefix):
                    softened = softened[len(prefix) :].lstrip(" ：:；;，,。")
                    break
            softened = re.sub(r"\s+", " ", softened)
            return softened.strip()

        if "；检测：" in cleaned and "；修复：" in cleaned:
            symptom, rest = cleaned.split("；检测：", 1)
            checks, fixes = rest.split("；修复：", 1)
            symptom = symptom.removeprefix("故障现象：").strip("： ")
            checks = _soften_clause(checks, drop_prefixes=("先确认", "请确认", "确认", "先检查", "检查"))
            fixes = _soften_clause(fixes, drop_prefixes=("可以尝试", "可尝试", "先尝试", "尝试", "建议"))
            if symptom and checks and fixes:
                if any(marker in symptom for marker in ("无法", "失败", "连不上", "异常", "报警", "找不到", "不旋转", "丢失")):
                    return f"如果现在卡在“{symptom}”这一步，先看 {checks}；还是不行的话，再试试 {fixes}"
                return f"先看 {checks}；还是不行的话，再试试 {fixes}"
            if symptom and checks:
                return f"如果现在卡在“{symptom}”这一步，先看 {checks}"

        if cleaned.startswith("检测："):
            step = _soften_clause(cleaned.removeprefix("检测："), drop_prefixes=("先确认", "请确认", "确认", "先检查", "检查"))
            return f"先看 {step}"
        if cleaned.startswith("修复："):
            step = _soften_clause(cleaned.removeprefix("修复："), drop_prefixes=("可以尝试", "可尝试", "先尝试", "尝试", "建议"))
            return f"还是不行的话，再试试 {step}"
        if cleaned.startswith("故障现象："):
            return cleaned.removeprefix("故障现象：").strip()
        if any(marker in cleaned for marker in ("频繁暂停", "突然停止清扫", "暂停清扫")) and "检测：" not in cleaned:
            return f"如果一直扫着扫着就停下来，通常先从这条看：{cleaned}"
        if cleaned.startswith("调小水箱出水量"):
            return "如果你家是木地板，拖地时最好把出水量调小一点，尽量用干拖模式，别让地面积水。"
        if "：" in cleaned:
            label, content = cleaned.split("：", 1)
            label = label.strip()
            content = _soften_clause(content)
            if label == "地面适配" and content:
                if any(marker in content for marker in ("木地板", "地板")):
                    return f"先看地面适配，{content}，因为木地板更怕水痕和返潮，这项会直接影响你家里用起来稳不稳。"
                return f"先看地面适配，{content}，这项最直接决定它在你家好不好用。"
            if label == "清洁覆盖率" and content:
                return f"再看清洁覆盖率，{content}，覆盖稳不稳会直接影响边角和桌椅周围会不会反复漏扫。"
            if label in {"导航类型", "吸力", "电池容量"} and content:
                if label == "导航类型":
                    return f"导航类型这项也别忽略，{content}，导航稳不稳会直接影响它会不会乱跑和重复清扫。"
                if label == "吸力":
                    return f"吸力这项也别忽略，{content}，如果家里有灰尘、毛发或地毯，这项会直接影响清洁效果。"
                if label == "电池容量":
                    return f"电池容量这项也别忽略，{content}，续航够不够会影响它能不能一遍把家里扫完。"
        if "、" in cleaned and not any(marker in cleaned for marker in ("：", "；", "。")) and len(cleaned) <= 40:
            return f"剩下这些参数也别忽略：{cleaned}。"
        return cleaned

    @staticmethod
    def _build_recommendation_reason_points(*, query: str, memory_lines: list[str]) -> list[str]:
        joined = f"{query} {' '.join(memory_lines)}"
        points: list[str] = []
        if "木地板" in joined:
            points.append("你家是木地板，所以更要看出水量调节和拖布抬升，日常用起来更不容易留下水痕。")
        if any(marker in joined for marker in ("猫", "狗", "宠物", "长毛")):
            points.append("家里有宠物的话，防缠毛和边角覆盖会更重要，后面清理毛发会省心很多。")
        if any(marker in joined for marker in ("静音", "噪音", "夜间")):
            points.append("你更在意静音，那就别只看参数表，电机噪音控制和夜间模式也会直接影响实际体验。")
        if any(marker in joined for marker in ("大户型", "120平", "大面积", "大面积户型", "续航")):
            points.append("如果家里面积偏大，续航和断点续扫会更关键，不然一遍扫不完，体验会明显打折。")
        return points[:3]

    @staticmethod
    def _recommendation_scenario_phrases(query: str, memory_lines: list[str]) -> list[str]:
        joined = f"{query} {' '.join(memory_lines)}"
        phrases: list[str] = []
        if "木地板" in joined:
            phrases.append("木地板清洁")
        if any(marker in joined for marker in ("猫", "狗", "宠物", "长毛")):
            phrases.append("宠物毛发处理")
        if any(marker in joined for marker in ("静音", "噪音", "夜间")):
            phrases.append("低噪音体验")
        if any(marker in joined for marker in ("大户型", "120平", "大面积")):
            phrases.append("大面积续航")
        return phrases

    @classmethod
    def _build_recommendation_answer(
        cls,
        *,
        query: str,
        memory_lines: list[str],
        points: list[str],
    ) -> str:
        reason_points = cls._build_recommendation_reason_points(query=query, memory_lines=memory_lines)
        suggestion_points: list[str] = []
        seen: set[str] = set()
        ordered_candidates = (
            [*points[:1], *reason_points, *points[1:]]
            if cls._is_mixed_query(query)
            else [*points, *reason_points]
        )
        extra_bucket_seen = False
        for point in ordered_candidates:
            normalized = str(point).strip()
            if normalized and normalized not in seen:
                if normalized.startswith("剩下这些参数也别忽略："):
                    if extra_bucket_seen:
                        continue
                    extra_bucket_seen = True
                seen.add(normalized)
                suggestion_points.append(normalized)
            if len(suggestion_points) >= 4:
                break

        scenario_phrases = cls._recommendation_scenario_phrases(query, memory_lines)
        if scenario_phrases:
            if len(scenario_phrases) == 1:
                judgement = f"按你这个情况，我会优先选更适合{scenario_phrases[0]}的机型。"
            else:
                judgement = f"按你这个情况，我会优先选同时兼顾{'、'.join(scenario_phrases)}的机型。"
        elif any(marker in query for marker in ("大户型", "120平", "大面积")):
            judgement = "如果你主要是大户型场景，我会优先选续航更稳、覆盖更完整的机型。"
        else:
            judgement = "如果按你的使用场景来选，我会优先挑更省心、也更贴合日常需求的机型。"

        if reason_points:
            explanation = f"原因也很简单，{reason_points[0].rstrip('。')}。"
        elif suggestion_points:
            explanation = f"你可以重点看这几项，因为它们最直接决定实际用起来是不是省心。"
        else:
            explanation = "你可以重点看这几项，因为它们最直接决定实际体验。"

        bullets = cls._format_answer_bullets(suggestion_points)
        if memory_lines:
            return f"{judgement}\n{explanation}\n如果你现在准备选，我会建议优先看这几项：\n{bullets}"
        return f"{judgement}\n{explanation}\n真要开始选的话，我会建议优先看这几项：\n{bullets}"

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
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    humanized = self._humanize_answer_point(normalized)
                    if humanized:
                        points.append(humanized)
                if len(points) >= 5:
                    break
            if len(points) >= 5:
                break
        if not points:
            return None

        if self._is_mixed_query(query) or self._is_selection_query(query):
            return self._build_recommendation_answer(
                query=query,
                memory_lines=memory_lines,
                points=points,
            )

        bullets = self._format_answer_bullets(points)
        if memory_lines:
            memory_intro = self._build_memory_preface(memory_lines, limit=3)
            return f"{memory_intro}\n再结合知识库，我会更建议你重点看这几项：\n{bullets}"

        lead = "如果是你这个问题，我会先重点看这几项："
        if self._is_mixed_query(query):
            lead = "按你家这个情况，我会优先看这几项，因为它们和你平时的使用环境最相关："
        if any(keyword in query for keyword in ("多久", "周期", "多长时间")):
            lead = "如果你主要想确认周期，可以先记这几条："
        elif any(keyword in query for keyword in ("原因", "为什么", "异常")):
            lead = "更常见的原因，基本集中在这几类："
        elif any(keyword in query for keyword in ("怎么", "如何", "排查")):
            lead = "你可以先从这几步看起："
        elif any(keyword in query for keyword in ("哪些", "包括", "关注")):
            lead = "如果只抓重点，我会先看这几项："

        return f"{lead}\n{bullets}"

    def _build_pause_troubleshooting_answer(
        self,
        *,
        memory_lines: list[str],
        evidence_briefs: list[dict[str, object]],
    ) -> str | None:
        joined_points = " ".join(
            str(point).strip()
            for item in evidence_briefs
            for point in item["points"]
            if str(point).strip()
        )
        if not joined_points:
            return None

        steps: list[str] = []
        if any(marker in joined_points for marker in ("尘盒", "滤网", "电量", "传感器")):
            steps.append("先看尘盒和滤网有没有堵，电量够不够，再顺手把机身传感器擦一下。")
        if any(marker in joined_points for marker in ("自动回充", "回充", "故障")):
            steps.append("如果它经常扫着扫着自己停下来回充，再看看是不是开了自动回充，或者机身本身有故障提示。")
        if any(marker in joined_points for marker in ("清理后重试", "重试", "排查故障提示", "充满电")):
            steps.append("前面这些处理完，再让它重新跑一轮；要是还是频繁暂停，就重点看一下机身提示或故障代码。")

        if not steps:
            return None

        bullets = self._format_answer_bullets(steps[:3])
        lead = "别着急，这种“扫着扫着总停下来”的情况，通常先看这几项就够了："
        if memory_lines:
            memory_intro = self._build_memory_preface(memory_lines, limit=2)
            return f"{memory_intro}\n{lead}\n{bullets}"
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
        if self._is_pause_like_query(query):
            pause_answer = self._build_pause_troubleshooting_answer(
                memory_lines=memory_lines,
                evidence_briefs=troubleshooting_briefs,
            )
            if pause_answer:
                return pause_answer
        topic_terms = self._troubleshooting_topic_terms(query)
        excluded_terms = self._troubleshooting_excluded_terms(query)
        steps: list[str] = []
        seen: set[str] = set()
        for item in troubleshooting_briefs:
            for point in item["points"]:
                text = str(point).strip()
                if not text:
                    continue
                if topic_terms and not any(term.lower() in text.lower() for term in topic_terms):
                    continue
                if excluded_terms and any(term.lower() in text.lower() for term in excluded_terms):
                    continue
                if text in seen:
                    continue
                seen.add(text)
                humanized = self._humanize_answer_point(text)
                if humanized:
                    steps.append(humanized)
                if len(steps) >= 4:
                    break
            if len(steps) >= 4:
                break

        if not steps:
            return None

        lead = "别着急，这类问题通常先按这个顺序看就行："
        if any(marker in query for marker in ("原因", "为什么", "异常")):
            lead = "这类情况一般先看这几项，基本就能定位到："
        natural_steps = [self._humanize_troubleshooting_step(index, step) for index, step in enumerate(steps)]
        bullets = self._format_answer_bullets(natural_steps)
        if memory_lines:
            memory_intro = self._build_memory_preface(memory_lines, limit=3)
            return f"{memory_intro}\n{lead}\n{bullets}"
        return f"{lead}\n{bullets}"

    @staticmethod
    def _humanize_troubleshooting_step(index: int, step: str) -> str:
        cleaned = step.strip("：；。 ")
        if not cleaned:
            return ""
        if cleaned.startswith(("先", "再", "如果", "别")):
            return cleaned
        if index == 0:
            return f"先看 {cleaned}"
        if index == 1:
            return f"再看 {cleaned}"
        if index == 2:
            return f"如果前两步都没问题，再看 {cleaned}"
        return f"如果还是不行，再试试 {cleaned}"

    @classmethod
    def _response_style_mode(cls, query: str) -> str:
        if cls._is_overview_query(query):
            return "overview"
        if cls._is_memory_only_query(query):
            return "memory"
        if cls._is_troubleshooting_query(query):
            return "troubleshooting"
        if cls._is_mixed_query(query) or cls._is_selection_query(query):
            return "recommendation"
        return "general"

    @classmethod
    def _style_prompt_hint(cls, query: str) -> str:
        mode = cls._response_style_mode(query)
        if mode == "troubleshooting":
            return "像一个熟悉设备问题的助手在现场指导，先直接判断，再按步骤告诉用户先看什么、再试什么。"
        if mode == "recommendation":
            return "像一个懂产品的顾问在给建议，先结合用户场景说判断，再解释重点，不要只堆参数。"
        if mode == "overview":
            return "像聊天时做概览介绍，先一句话说清主题，再补几个最值得先看的方向。"
        if mode == "memory":
            return "像在自然回忆之前对话，直接告诉用户你记得什么，不要太书面。"
        return "像日常对话一样自然解释，先直接回答，再按需要补几条关键点。"

    @classmethod
    def _apply_answer_style(cls, query: str, answer: str) -> str:
        normalized = str(answer).strip()
        if not normalized:
            return normalized

        mode = cls._response_style_mode(query)
        bullet_lines = [line for line in normalized.splitlines() if line.strip().startswith("- ")]

        if normalized == INSUFFICIENT_ANSWER:
            if mode == "troubleshooting":
                return "按现在这份资料，我这边暂时还不能更确定地判断。你要是愿意，可以把暂停前后的现象再说具体一点，我再帮你一起缩小范围。"
            return "按现在这份资料，我这边暂时还不能更确定地判断。你要是愿意，可以把问题再说具体一点，我再继续帮你一起看。"

        if mode == "troubleshooting":
            if normalized.startswith("你可以先从这几步看起："):
                return normalized.replace("你可以先从这几步看起：", "别急，这种情况一般先从这几步看起：", 1)
            if normalized.startswith("你可以先按这个顺序排查："):
                return normalized.replace("你可以先按这个顺序排查：", "别急，这种情况通常先按这个顺序排查就行：", 1)
            if normalized.startswith("如果是你这个问题，我会先重点看这几项：") and bullet_lines:
                return normalized.replace("如果是你这个问题，我会先重点看这几项：", "别急，这种情况一般先看这几项：", 1)

        if mode == "recommendation":
            if normalized.startswith("如果是你这个问题，我会先重点看这几项："):
                return normalized.replace("如果是你这个问题，我会先重点看这几项：", "如果按你的使用场景来选，我会优先看这几项：", 1)
            if normalized.startswith("如果只抓重点，我会先看这几项："):
                return normalized.replace("如果只抓重点，我会先看这几项：", "如果只抓重点，我会优先看这几项：", 1)
            if normalized.startswith("按你家这个情况，我会优先看这几项："):
                return normalized.replace("按你家这个情况，我会优先看这几项：", "按你家这个情况，我会优先看这几项，因为它们和你平时的使用环境最相关：", 1)

        if mode == "overview" and not normalized.startswith("这个知识库主要"):
            return f"这个知识库主要可以这样理解：\n{normalized}"

        if mode == "memory" and not normalized.startswith(("我记得", "我还记得")):
            return f"我记得你前面提过这些：\n{normalized}"

        if mode == "general" and bullet_lines and not normalized.startswith((
            "如果是你这个问题",
            "你可以先从这几步看起：",
            "别急",
            "按你家这个情况",
            "这个知识库主要",
            "按目前能确定的信息，我会先从这几项跟你说：",
        )):
            return f"按目前能确定的信息，我会先从这几项跟你说：\n{normalized}"

        return normalized

    @staticmethod
    def _extract_profile_values(profile_lines: list[str]) -> list[str]:
        values: list[str] = []
        for line in profile_lines:
            text = str(line).strip()
            if not text:
                continue
            if "：" in text:
                _, value = text.split("：", 1)
                text = value.strip()
            elif ":" in text:
                _, value = text.split(":", 1)
                text = value.strip()
            if text:
                values.append(text)
        return values

    @staticmethod
    def _polish_memory_statement(line: str) -> str:
        text = str(line).strip()
        text = text.replace("请记住，", "").replace("请记住", "")
        text = text.replace("帮我记一下，", "").replace("帮我记一下", "")
        text = text.replace("记一下，", "").replace("记一下", "")
        text = text.replace("太吃", "太吵")
        replacements = {
            "我家里是": "家里是",
            "我家是": "家里是",
            "我更在意": "更在意",
            "我不希望": "不希望",
            "我希望": "希望",
            "我最近": "最近",
            "我上周": "上周",
            "我今晚": "今晚",
            "我已经": "已经",
            "我还没": "还没",
        }
        for source, target in replacements.items():
            if text.startswith(source):
                text = target + text[len(source) :]
                break
        text = re.sub(r"\s+", "", text)
        text = text.rstrip("，,；;")
        if text and text[-1] not in "。！？":
            text += "。"
        return text

    @classmethod
    def _memory_detail_is_redundant(cls, line: str, profile_values: list[str]) -> bool:
        if not line:
            return True
        if re.search(r"\d+\s*平", line):
            return False
        detail_markers = (
            "不希望",
            "希望",
            "晚上",
            "白天",
            "已经",
            "还没",
            "最近",
            "偶尔",
            "经常",
            "上周",
            "今晚",
            "失败",
            "异常",
            "不稳定",
            "问题",
            "想买",
            "适合",
            "先",
            "再",
            "刚",
        )
        if any(marker in line for marker in detail_markers):
            return False
        return any(value and value in line for value in profile_values)

    @staticmethod
    def _trim_profile_prefix_from_memory(line: str, profile_values: list[str]) -> str:
        trimmed = line.strip()
        for value in sorted((value for value in profile_values if value), key=len, reverse=True):
            if not trimmed.startswith(value):
                continue
            remainder = trimmed[len(value) :].lstrip("，,；;、 ")
            if not remainder:
                return ""
            if remainder.startswith("不希望晚上"):
                remainder = "晚上不希望" + remainder[len("不希望晚上") :]
            elif remainder.startswith("和"):
                remainder = "也很在意" + remainder[len("和") :]
            if remainder and remainder[-1] not in "。！？":
                remainder += "。"
            return remainder
        return trimmed

    @staticmethod
    def _parse_profile_lines(profile_lines: list[str]) -> dict[str, list[str]]:
        parsed: dict[str, list[str]] = {}
        for line in profile_lines:
            text = str(line).strip()
            if not text:
                continue
            if ":" in text:
                key, value = text.split(":", 1)
            elif "：" in text:
                key, value = text.split("：", 1)
            else:
                continue
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            bucket = parsed.setdefault(key, [])
            if value not in bucket:
                bucket.append(value)
        return parsed

    @staticmethod
    def _render_memory_theme_lines(profile_lines: list[str], detail_lines: list[str]) -> list[str]:
        parsed = HelloRagAgentService._parse_profile_lines(profile_lines)
        rendered: list[str] = []
        remaining_details = list(detail_lines)

        floor = parsed.get("地面", [])
        pets = parsed.get("宠物", [])
        home_size = parsed.get("户型", []) or parsed.get("home_size", [])
        preferences = parsed.get("偏好", [])
        maintenance = parsed.get("维护状态", [])
        issues = parsed.get("问题", [])

        if floor or pets:
            parts: list[str] = []
            if floor:
                parts.append(f"家里是{floor[0]}")
            if pets:
                parts.append(f"还养了{pets[0]}")
            rendered.append("家庭情况上，我记得你" + "，".join(parts) + "。")

        if home_size:
            size_detail = next((line for line in remaining_details if re.search(r"\d+\s*平", line)), "")
            if size_detail:
                remaining_details.remove(size_detail)
                if size_detail.startswith(("我家", "家里")):
                    rendered.append(f"户型上，我记得{size_detail}")
                else:
                    rendered.append(f"户型上，我记得你{size_detail}")
            else:
                rendered.append(f"户型上，我记得你家{home_size[0]}。")

        if preferences:
            normalized_prefs = [value.replace("更在意", "").strip() for value in preferences]
            normalized_prefs = [value for value in normalized_prefs if value]
            preference_order = {"静音": 0, "APP稳定性": 1}
            normalized_prefs.sort(key=lambda item: preference_order.get(item, 10))
            if normalized_prefs:
                if len(normalized_prefs) == 1:
                    rendered.append(f"偏好上，你现在更在意{normalized_prefs[0]}。")
                else:
                    first, *rest = normalized_prefs
                    tail = "，也很在意".join(rest)
                    rendered.append(f"偏好上，你现在更在意{first}，也很在意{tail}。")

        if maintenance:
            rendered.append(f"维护情况上，你之前提过{maintenance[0]}。")
        if issues:
            rendered.append(f"问题上，你还提过{issues[0]}。")

        rendered.extend(remaining_details)
        return rendered

    @staticmethod
    def _detail_line_is_redundant_with_profile(line: str, profile_lines: list[str]) -> bool:
        compact_line = line.replace(" ", "")
        if compact_line.startswith("也很在意"):
            return True
        for profile in profile_lines:
            compact_profile = str(profile).replace(" ", "")
            if not compact_profile:
                continue
            if ":" in compact_profile:
                _, compact_profile = compact_profile.split(":", 1)
            elif "：" in compact_profile:
                _, compact_profile = compact_profile.split("：", 1)
            if compact_profile and compact_profile in compact_line:
                if any(marker in compact_line for marker in ("晚上", "经常", "120平", "想买", "还没", "已经", "偶尔", "最近")):
                    return False
                return True
        return False

    def _answer_from_memory(self, query: str, session: SessionState) -> str | None:
        if not self._is_memory_only_query(query):
            return None

        profile_lines = session.memory_tool.get_profile_lines(limit=min(4, self.settings.memory.profile_max_facts))
        profile_values = self._extract_profile_values(profile_lines)
        recent_user_lines: list[str] = []
        seen: set[str] = set()

        for message in reversed(session.history):
            if message.role != "user":
                continue
            content = self._normalize_memory_line(message.content)
            if not content or self._is_memory_query(content):
                continue
            polished = self._polish_memory_statement(content)
            polished = self._trim_profile_prefix_from_memory(polished, profile_values)
            if not polished or self._memory_detail_is_redundant(polished, profile_values):
                continue
            if polished in seen:
                continue
            seen.add(polished)
            recent_user_lines.append(polished)
            if len(recent_user_lines) >= 4:
                break

        recent_user_lines.reverse()

        compact_details: list[str] = []
        for line in recent_user_lines:
            normalized = self._normalize_memory_line(line)
            if not normalized:
                continue
            if self._detail_line_is_redundant_with_profile(normalized, profile_lines):
                continue
            compact_details.append(normalized)

        memory_lines = self._render_memory_theme_lines(profile_lines, compact_details)
        memory_lines = [line for line in memory_lines if line]

        if not memory_lines:
            return None

        if len(memory_lines) == 1:
            return f"我还记得一条比较关键的信息：{memory_lines[0]}"

        bullets = "\n".join(f"- {line}" for line in memory_lines[:5])
        return f"我还记得几条关键信息：\n{bullets}"

    def _answer_with_retrieval(self, query: str, session: SessionState) -> str:
        answer = BaseHelloRagAgentService._answer_with_retrieval(self, query=query, session=session)
        return self._apply_answer_style(query, answer)

    def _answer(self, *, question: str, session: SessionState) -> str:
        answer = BaseHelloRagAgentService._answer(self, question=question, session=session)
        return self._apply_answer_style(question, answer)

    def _build_overview_answer(self, *, query: str) -> str | None:
        answer = BaseHelloRagAgentService._build_overview_answer(self, query=query)
        if answer is None:
            return None
        return self._apply_answer_style(query, answer)

    @staticmethod
    def _build_memory_grounded_answer(*, query: str, memory_lines: list[str]) -> str:
        if not memory_lines:
            return INSUFFICIENT_ANSWER
        if len(memory_lines) == 1:
            return f"我还记得你前面提过：{memory_lines[0]}"
        bullets = "\n".join(f"- {line}" for line in memory_lines)
        return f"我还记得你前面提过这些：\n{bullets}"

    def _grounded_messages(
        self,
        *,
        query: str,
        memory_lines: list[str],
        evidence_briefs: list[dict[str, object]],
    ) -> list[dict[str, str]]:
        memory_text = "\n".join(f"- {line}" for line in memory_lines) if memory_lines else "无"
        evidence_blocks = []
        for index, item in enumerate(evidence_briefs, start=1):
            lines = "\n".join(f"  - {line}" for line in item["points"])
            evidence_blocks.append(
                f"[证据{index}] source={item['source']} title={item['title']} citation={item.get('citation', item['source'])} score={item['score']}\n{lines}"
            )
        evidence_text = "\n\n".join(evidence_blocks)
        style_hint = self._style_prompt_hint(query)
        return [
            {
                "role": "system",
                "content": (
                    "你是一个严格依据证据、但说话自然的中文助手。\n"
                    "1. 只能使用【当前会话信息】和【知识库证据】里明确出现的信息。\n"
                    "2. 不要提工具、检索过程、系统机制。\n"
                    "3. 没有证据支持的内容不要补充。\n"
                    f"4. 回答风格要求：{style_hint}\n"
                    "5. 如果问题需要结合用户场景，就先结合当前会话信息再回答。\n"
                    "6. 如果证据不足，要坦诚说明按当前知识库的信息暂时还不能更确定地判断，或者目前只能确定以下几点。\n"
                    "7. 最终答案优先用简短自然的中文；只有在内容本身适合分点时，再列 2-4 条关键点。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：\n{query}\n\n"
                    f"当前会话信息：\n{memory_text}\n\n"
                    f"知识库证据：\n{evidence_text}\n\n"
                    "请先用 1-2 句直接回答用户的问题；如果需要，再补充 2-4 条关键要点。"
                ),
            },
        ]

    def _generate_grounded_answer(
        self,
        *,
        query: str,
        memory_lines: list[str],
        evidence_briefs: list[dict[str, object]],
    ) -> str:
        response = self._get_llm().invoke(
            self._grounded_messages(query=query, memory_lines=memory_lines, evidence_briefs=evidence_briefs)
        )
        return self._apply_answer_style(query, getattr(response, "content", "").strip())

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
            "需要明确问题主题",
        )
        timeout_markers = ("max step", "step limit", "无法在限定步数内", "限定步数")
        return (
            any(marker in lowered for marker in fallback_markers)
            or any(marker in normalized for marker in clarification_markers)
            or any(marker in normalized for marker in timeout_markers)
        )


_SERVICE: HelloRagAgentService | None = None


def get_service() -> HelloRagAgentService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = HelloRagAgentService()
    return _SERVICE
