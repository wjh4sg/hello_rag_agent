from __future__ import annotations

from typing import Any

from hello_agents.context import ContextPacket
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.response import ToolResponse

from hello_rag_agent.knowledge_base import KnowledgeBase, SearchResult
from hello_rag_agent.llm import SafeHelloAgentsLLM


SUPPORTED_SEARCH_STRATEGIES = {"keyword", "vector", "hybrid"}


class RAGTool(Tool):
    """Unified retrieval tool for search, ask, and knowledge-base diagnostics."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm: SafeHelloAgentsLLM,
        system_prompt: str,
        default_top_k: int = 4,
        rag_namespace: str = "default",
        default_search_strategy: str = "hybrid",
    ):
        super().__init__(
            name="rag_tool",
            description=(
                "Access the local knowledge base. Use action=search to inspect evidence, "
                "action=ask to answer using retrieved evidence, and action=stats to inspect the index."
            ),
        )
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.system_prompt = system_prompt
        self.default_top_k = default_top_k
        self.rag_namespace = rag_namespace
        self.default_search_strategy = self._normalize_search_strategy(default_search_strategy)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
        namespace: str | None = None,
    ) -> list[SearchResult]:
        self._normalize_namespace(namespace)
        return self.knowledge_base.search(
            query,
            top_k=top_k or self.default_top_k,
            strategy=self._normalize_search_strategy(strategy),
        )

    def render_context(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
        namespace: str | None = None,
    ) -> str:
        results = self.search(query, top_k=top_k, strategy=strategy, namespace=namespace)
        if not results:
            return "No relevant knowledge was found in the local knowledge base."

        lines = []
        for index, item in enumerate(results, start=1):
            preview = item.snippet or item.chunk.content.strip()
            if len(preview) > 500:
                preview = f"{preview[:500]}..."
            lines.append(f"[Source {index}] file: {item.chunk.source}")
            lines.append(f"title: {item.chunk.title}")
            lines.append(f"citation: {item.citation}")
            lines.append(f"strategy: {item.strategy}")
            lines.append(f"score: {item.score:.4f}")
            if item.match_terms:
                lines.append(f"match_terms: {', '.join(item.match_terms)}")
            lines.append(f"content: {preview}")
            lines.append("")
        return "\n".join(lines).strip()

    def build_context_packet(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
        namespace: str | None = None,
    ) -> ContextPacket | None:
        context = self.render_context(query, top_k=top_k, strategy=strategy, namespace=namespace)
        if context == "No relevant knowledge was found in the local knowledge base.":
            return None
        return ContextPacket(
            content=context,
            metadata={
                "type": "knowledge_base",
                "namespace": self.rag_namespace,
                "strategy": self._normalize_search_strategy(strategy),
            },
        )

    def ask(
        self,
        *,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
        namespace: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ToolResponse:
        resolved_namespace = self._normalize_namespace(namespace)
        resolved_strategy = self._normalize_search_strategy(strategy)
        results = self.search(
            query,
            top_k=top_k,
            strategy=resolved_strategy,
            namespace=resolved_namespace,
        )
        if not results:
            return ToolResponse.success(
                text="根据当前本地知识库，没有找到足够证据来可靠回答这个问题。",
                data={
                    "matches": [],
                    "namespace": resolved_namespace,
                    "strategy": resolved_strategy,
                },
                stats={"match_count": 0},
            )

        evidence_text = self.render_context(
            query,
            top_k=top_k,
            strategy=resolved_strategy,
            namespace=resolved_namespace,
        )
        history_text = self._render_history(history)
        messages = [
            {
                "role": "system",
                "content": (
                    self.system_prompt
                    + "\n\n你正在执行 RAGTool.ask。只能依据检索到的证据回答。"
                    + " 如果证据不足，要明确说明，不要补充知识库里没有的信息。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题:\n{query}\n\n"
                    f"最近对话:\n{history_text}\n\n"
                    f"检索策略: {resolved_strategy}\n"
                    f"命名空间: {resolved_namespace}\n\n"
                    f"证据:\n{evidence_text}\n\n"
                    "请基于这些证据直接回答。"
                ),
            },
        ]
        response = self.llm.invoke(messages)
        answer = getattr(response, "content", "").strip()
        if not answer:
            answer = "根据检索证据，我暂时无法生成可靠答案。"

        return ToolResponse.success(
            text=answer,
            data={
                "matches": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "source": item.chunk.source,
                        "title": item.chunk.title,
                        "citation": item.citation,
                        "content": item.chunk.content,
                        "snippet": item.snippet,
                        "strategy": item.strategy,
                        "match_terms": list(item.match_terms),
                        "score": item.score,
                    }
                    for item in results
                ],
                "namespace": resolved_namespace,
                "strategy": resolved_strategy,
            },
            stats={
                "match_count": len(results),
                "top_score": results[0].score if results else 0.0,
            },
        )

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        action = self._extract_action(parameters)
        namespace = self._extract_namespace(parameters)
        strategy = self._extract_search_strategy(parameters)

        if action == "stats":
            stats = self.knowledge_base.stats()
            stats.update({"namespace": namespace, "strategy": strategy})
            return ToolResponse.success(
                text=(
                    f"Knowledge base stats: {stats['document_count']} documents, "
                    f"{stats['chunk_count']} chunks, retrieval={stats['retrieval_mode']}."
                ),
                data=stats,
            )

        query = self._extract_query(parameters)
        if not query:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"The rag_tool {action} action requires a query.",
            )

        top_k = self._extract_top_k(parameters)
        if action == "search":
            try:
                results = self.search(query, top_k=top_k, strategy=strategy, namespace=namespace)
            except ValueError as exc:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=str(exc),
                )

            if not results:
                return ToolResponse.success(
                    text="No relevant knowledge was found in the local knowledge base.",
                    data={"matches": [], "namespace": namespace, "strategy": strategy},
                )

            return ToolResponse.success(
                text=self.render_context(query, top_k=top_k, strategy=strategy, namespace=namespace),
                data={
                    "matches": [
                        {
                            "chunk_id": item.chunk.chunk_id,
                            "source": item.chunk.source,
                            "title": item.chunk.title,
                            "citation": item.citation,
                            "content": item.chunk.content,
                            "snippet": item.snippet,
                            "strategy": item.strategy,
                            "match_terms": list(item.match_terms),
                            "score": item.score,
                        }
                        for item in results
                    ],
                    "namespace": namespace,
                    "strategy": strategy,
                },
            )

        if action == "ask":
            try:
                return self.ask(
                    query=query,
                    top_k=top_k,
                    strategy=strategy,
                    namespace=namespace,
                    history=self._extract_history(parameters),
                )
            except ValueError as exc:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=str(exc),
                )

        return ToolResponse.error(
            code=ToolErrorCode.INVALID_PARAM,
            message=f"Unsupported rag_tool action: {action}",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="RAG action to execute. Supported values: search, ask, stats.",
                required=True,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="User question or retrieval query, required for search and ask.",
                required=False,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Maximum number of retrieved chunks.",
                required=False,
                default=self.default_top_k,
            ),
            ToolParameter(
                name="namespace",
                type="string",
                description="Knowledge namespace. Only default is currently available.",
                required=False,
                default=self.rag_namespace,
            ),
            ToolParameter(
                name="search_strategy",
                type="string",
                description="Retrieval strategy: keyword, vector, or hybrid.",
                required=False,
                default=self.default_search_strategy,
            ),
            ToolParameter(
                name="history",
                type="array",
                description="Optional recent conversation messages used by action=ask.",
                required=False,
            ),
        ]

    @staticmethod
    def _extract_action(parameters: dict[str, Any]) -> str:
        value = parameters.get("action")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        return "search"

    @staticmethod
    def _extract_query(parameters: dict[str, Any]) -> str:
        for key in ("query", "input", "question"):
            value = parameters.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_top_k(parameters: dict[str, Any]) -> int | None:
        value = parameters.get("top_k")
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _extract_namespace(self, parameters: dict[str, Any]) -> str:
        value = parameters.get("namespace")
        return self._normalize_namespace(value if isinstance(value, str) and value.strip() else None)

    def _extract_search_strategy(self, parameters: dict[str, Any]) -> str:
        value = parameters.get("search_strategy")
        return self._normalize_search_strategy(value if isinstance(value, str) and value.strip() else None)

    @staticmethod
    def _extract_history(parameters: dict[str, Any]) -> list[dict[str, str]] | None:
        value = parameters.get("history")
        if isinstance(value, list):
            normalized = []
            for item in value:
                if isinstance(item, dict):
                    role = item.get("role")
                    content = item.get("content")
                    if isinstance(role, str) and isinstance(content, str) and content.strip():
                        normalized.append({"role": role, "content": content.strip()})
            return normalized or None
        return None

    def _normalize_namespace(self, namespace: str | None) -> str:
        resolved = namespace or self.rag_namespace
        if resolved != self.rag_namespace:
            raise ValueError(
                f"Unsupported namespace '{resolved}'. Only '{self.rag_namespace}' is currently available."
            )
        return resolved

    def _normalize_search_strategy(self, strategy: str | None) -> str:
        resolved = (strategy or self.default_search_strategy).strip().lower()
        if resolved not in SUPPORTED_SEARCH_STRATEGIES:
            return self.default_search_strategy
        return resolved

    @staticmethod
    def _render_history(history: list[dict[str, str]] | None) -> str:
        if not history:
            return "无"

        lines = []
        for index, item in enumerate(history[-6:], start=1):
            lines.append(f"[History {index}] {item['role']}: {item['content']}")
        return "\n".join(lines)
