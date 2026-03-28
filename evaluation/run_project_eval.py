from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
import sys
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EVAL_DIR / "results"
DEFAULT_RETRIEVAL_CASES_PATH = EVAL_DIR / "retrieval_eval_cases.json"
DEFAULT_MEMORY_CASES_PATH = EVAL_DIR / "memory_eval_cases.json"
DEFAULT_SESSION_CASES_PATH = EVAL_DIR / "session_memory_eval_cases.json"
DEFAULT_FUSION_CASES_PATH = EVAL_DIR / "fusion_eval_cases.json"
DEFAULT_ANSWER_CASES_PATH = EVAL_DIR / "rag_eval_cases.json"
DEFAULT_ANSWER_SMOKE_CASES_PATH = EVAL_DIR / "answer_smoke_eval_cases.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from hello_rag_agent import HelloRagAgentService
from hello_rag_agent.config import AppSettings, load_settings
from hello_rag_agent.knowledge_base import KnowledgeBase
from hello_rag_agent.llm import SafeHelloAgentsLLM
from hello_rag_agent.tools.memory_tool import MemoryTool
from hello_agents.core.message import Message
from run_rag_llm_judge import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TIMEOUT,
    average_score,
    build_judge_prompt,
    invoke_judge_with_retries,
    parse_judge_payload,
    render_evidence,
)


TRACE_MARKERS = (
    "thought:",
    "action:",
    "observation:",
    "finish",
    "rag_tool",
    "memory_tool",
    "你正在执行",
    "[history ",
    "请使用记忆工具",
    "先查看会话记忆",
)


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    goal: str
    expected_sources: tuple[str, ...]
    expected_keywords: tuple[str, ...]
    expected_rewrite_substrings: tuple[str, ...]
    forbidden_keywords: tuple[str, ...]
    top_k: int
    strategy: str | None
    min_keyword_hit_rate: float
    max_first_hit_rank: int | None


@dataclass(frozen=True)
class SeedEntry:
    role: str
    content: str
    memory_type: str
    importance: float
    age_seconds: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MemoryCase:
    case_id: str
    query: str
    goal: str
    entries: tuple[SeedEntry, ...]
    top_k: int
    memory_types: tuple[str, ...]
    expected_substrings: tuple[str, ...]
    expected_top1: str


@dataclass(frozen=True)
class SessionCase:
    case_id: str
    goal: str
    user_id: str | None
    turns: tuple[str, ...]
    recall_query: str
    expected_memory_substrings: tuple[str, ...]
    expected_answer_substrings: tuple[str, ...]
    top_k: int
    min_answer_hit_rate: float
    reload_service_before_recall: bool


@dataclass(frozen=True)
class FusionCase:
    case_id: str
    goal: str
    user_id: str | None
    turns: tuple[str, ...]
    query: str
    expected_memory_substrings: tuple[str, ...]
    expected_query_substrings: tuple[str, ...]
    expected_sources: tuple[str, ...]
    expected_keywords: tuple[str, ...]
    expected_answer_substrings: tuple[str, ...]
    top_k: int
    min_answer_hit_rate: float


@dataclass(frozen=True)
class AnswerCase:
    case_id: str
    question: str
    goal: str
    category: str
    expected_substrings: tuple[str, ...]
    min_hit_rate: float
    forbidden_substrings: tuple[str, ...]


@dataclass(frozen=True)
class AnswerSmokeCase:
    case_id: str
    question: str
    goal: str
    expected_substrings: tuple[str, ...]
    min_hit_rate: float
    forbidden_substrings: tuple[str, ...]


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[RetrievalCase] = []
    for item in raw:
        cases.append(
            RetrievalCase(
                case_id=str(item["id"]),
                query=str(item["query"]),
                goal=str(item.get("goal", "")),
                expected_sources=tuple(item.get("expected_sources", [])),
                expected_keywords=tuple(item.get("expected_keywords", [])),
                expected_rewrite_substrings=tuple(item.get("expected_rewrite_substrings", [])),
                forbidden_keywords=tuple(item.get("forbidden_keywords", [])),
                top_k=int(item.get("top_k", 4)),
                strategy=str(item["strategy"]) if item.get("strategy") else None,
                min_keyword_hit_rate=float(item.get("min_keyword_hit_rate", 0.5)),
                max_first_hit_rank=int(item["max_first_hit_rank"]) if item.get("max_first_hit_rank") is not None else None,
            )
        )
    return cases


def load_memory_cases(path: Path) -> list[MemoryCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[MemoryCase] = []
    for item in raw:
        entries = tuple(
            SeedEntry(
                role=str(entry.get("role", "user")),
                content=str(entry["content"]),
                memory_type=str(entry.get("memory_type", "working")),
                importance=float(entry.get("importance", 0.6)),
                age_seconds=int(entry.get("age_seconds", 0)),
                metadata=dict(entry.get("metadata", {})),
            )
            for entry in item.get("entries", [])
        )
        cases.append(
            MemoryCase(
                case_id=str(item["id"]),
                query=str(item["query"]),
                goal=str(item.get("goal", "")),
                entries=entries,
                top_k=int(item.get("top_k", 3)),
                memory_types=tuple(item.get("memory_types", [])),
                expected_substrings=tuple(item.get("expected_substrings", [])),
                expected_top1=str(item.get("expected_top1", "")),
            )
        )
    return cases


def load_session_cases(path: Path) -> list[SessionCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[SessionCase] = []
    for item in raw:
        cases.append(
            SessionCase(
                case_id=str(item["id"]),
                goal=str(item.get("goal", "")),
                user_id=str(item["user_id"]) if item.get("user_id") else None,
                turns=tuple(str(turn) for turn in item.get("turns", [])),
                recall_query=str(item["recall_query"]),
                expected_memory_substrings=tuple(item.get("expected_memory_substrings", [])),
                expected_answer_substrings=tuple(item.get("expected_answer_substrings", [])),
                top_k=int(item.get("top_k", 4)),
                min_answer_hit_rate=float(item.get("min_answer_hit_rate", 1.0)),
                reload_service_before_recall=bool(item.get("reload_service_before_recall", False)),
            )
        )
    return cases


def load_fusion_cases(path: Path) -> list[FusionCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[FusionCase] = []
    for item in raw:
        cases.append(
            FusionCase(
                case_id=str(item["id"]),
                goal=str(item.get("goal", "")),
                user_id=str(item["user_id"]) if item.get("user_id") else None,
                turns=tuple(str(turn) for turn in item.get("turns", [])),
                query=str(item["query"]),
                expected_memory_substrings=tuple(item.get("expected_memory_substrings", [])),
                expected_query_substrings=tuple(item.get("expected_query_substrings", [])),
                expected_sources=tuple(item.get("expected_sources", [])),
                expected_keywords=tuple(item.get("expected_keywords", [])),
                expected_answer_substrings=tuple(item.get("expected_answer_substrings", [])),
                top_k=int(item.get("top_k", 4)),
                min_answer_hit_rate=float(item.get("min_answer_hit_rate", 1.0)),
            )
        )
    return cases


def load_answer_cases(path: Path) -> list[AnswerCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[AnswerCase] = []
    for item in raw:
        cases.append(
            AnswerCase(
                case_id=str(item["id"]),
                question=str(item["question"]),
                goal=str(item.get("goal", "")),
                category=str(item.get("category", "rag_qa")),
                expected_substrings=tuple(item.get("expected_substrings", [])),
                min_hit_rate=float(item.get("min_hit_rate", 0.0)),
                forbidden_substrings=tuple(item.get("forbidden_substrings", [])),
            )
        )
    return cases


def load_answer_smoke_cases(path: Path) -> list[AnswerSmokeCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[AnswerSmokeCase] = []
    for item in raw:
        cases.append(
            AnswerSmokeCase(
                case_id=str(item["id"]),
                question=str(item["question"]),
                goal=str(item.get("goal", "")),
                expected_substrings=tuple(item.get("expected_substrings", [])),
                min_hit_rate=float(item.get("min_hit_rate", 0.67)),
                forbidden_substrings=tuple(item.get("forbidden_substrings", [])),
            )
        )
    return cases


def safe_resolve_api_key(settings: AppSettings) -> str:
    try:
        return settings.resolve_api_key()
    except RuntimeError:
        return ""


def contains_trace_markers(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in TRACE_MARKERS)


def substring_hit_rate(text: str, expected_substrings: tuple[str, ...]) -> float:
    if not expected_substrings:
        return 1.0
    haystack = text.strip()
    if not haystack:
        return 0.0
    hits = sum(1 for item in expected_substrings if item in haystack)
    return hits / len(expected_substrings)


def find_source_ranks(actual_sources: list[str], expected_sources: tuple[str, ...]) -> list[int]:
    if not expected_sources:
        return []
    ranks: list[int] = []
    for index, source in enumerate(actual_sources, start=1):
        if source in expected_sources:
            ranks.append(index)
    return ranks


def invoke_and_parse_judge(
    *,
    judge_llm: SafeHelloAgentsLLM,
    prompt: str,
    retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            response = invoke_judge_with_retries(judge_llm=judge_llm, prompt=prompt)
            return parse_judge_payload(getattr(response, "content", ""))
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def build_retrieval_suite(
    *,
    settings: AppSettings,
    cases_path: Path,
    default_top_k: int,
    api_key: str,
    max_cases: int | None = None,
) -> dict[str, Any]:
    cases = load_retrieval_cases(cases_path)
    if max_cases is not None:
        cases = cases[:max_cases]
    knowledge_base = KnowledgeBase(
        settings.knowledge_base,
        api_key=api_key,
        base_url=settings.llm.base_url,
    )
    results: list[dict[str, Any]] = []

    for case in cases:
        top_k = case.top_k or default_top_k
        evidence = knowledge_base.search(case.query, top_k=top_k, strategy=case.strategy)
        actual_sources = [item.chunk.source for item in evidence]
        combined_text = "\n".join(item.chunk.content for item in evidence)
        snippet_text = "\n".join(item.snippet for item in evidence if item.snippet)
        top1_text = ""
        if evidence:
            top1_text = (evidence[0].snippet or evidence[0].chunk.content).strip()
        matched_ranks = find_source_ranks(actual_sources, case.expected_sources)
        source_hit = bool(matched_ranks) if case.expected_sources else True
        first_hit_rank = min(matched_ranks) if matched_ranks else None
        rank_ok = case.max_first_hit_rank is None or (
            first_hit_rank is not None and first_hit_rank <= case.max_first_hit_rank
        )
        keyword_hit_rate = substring_hit_rate(combined_text, case.expected_keywords)
        snippet_keyword_hit_rate = substring_hit_rate(snippet_text, case.expected_keywords)
        top1_keyword_hit_rate = substring_hit_rate(top1_text, case.expected_keywords)
        trace = knowledge_base.get_last_search_trace()
        rewritten_query = str(trace.get("rewritten_query", "")).strip()
        rewrite_keywords = tuple(str(item) for item in trace.get("rewrite_keywords", []))
        rewrite_mode = str(trace.get("rewrite_mode", "unknown"))
        rewrite_text = "\n".join(filter(None, (rewritten_query, " ".join(rewrite_keywords))))
        rewrite_hit_rate = substring_hit_rate(rewrite_text, case.expected_rewrite_substrings)
        forbidden_hits = [
            item for item in case.forbidden_keywords if item in combined_text or item in snippet_text
        ]
        top1_forbidden_hits = [item for item in case.forbidden_keywords if item in top1_text]
        citation_coverage_rate = (
            sum(1 for item in evidence if getattr(item, "citation", "").strip()) / len(evidence)
            if evidence
            else 0.0
        )
        source_diversity = (len(set(actual_sources)) / len(actual_sources)) if actual_sources else 0.0
        passed = (
            source_hit
            and rank_ok
            and keyword_hit_rate >= case.min_keyword_hit_rate
            and (rewrite_hit_rate >= 1.0 if case.expected_rewrite_substrings else True)
            and not forbidden_hits
        )

        results.append(
            {
                "id": case.case_id,
                "query": case.query,
                "goal": case.goal,
                "top_k": top_k,
                "strategy": case.strategy or settings.knowledge_base.retrieval_mode,
                "expected_sources": list(case.expected_sources),
                "expected_keywords": list(case.expected_keywords),
                "expected_rewrite_substrings": list(case.expected_rewrite_substrings),
                "forbidden_keywords": list(case.forbidden_keywords),
                "actual_sources": actual_sources,
                "source_hit": source_hit,
                "first_hit_rank": first_hit_rank,
                "max_first_hit_rank": case.max_first_hit_rank,
                "rank_ok": rank_ok,
                "keyword_hit_rate": round(keyword_hit_rate, 4),
                "snippet_keyword_hit_rate": round(snippet_keyword_hit_rate, 4),
                "top1_keyword_hit_rate": round(top1_keyword_hit_rate, 4),
                "rewritten_query": rewritten_query,
                "rewrite_keywords": list(rewrite_keywords),
                "rewrite_mode": rewrite_mode,
                "rewrite_hit_rate": round(rewrite_hit_rate, 4),
                "forbidden_hits": forbidden_hits,
                "top1_forbidden_hits": top1_forbidden_hits,
                "citation_coverage_rate": round(citation_coverage_rate, 4),
                "source_diversity": round(source_diversity, 4),
                "passed": passed,
            }
        )

    hit_ranks = [item["first_hit_rank"] for item in results if item["first_hit_rank"] is not None]
    summary = {
        "suite": "retrieval",
        "case_count": len(results),
        "pass_rate": mean(1.0 if item["passed"] else 0.0 for item in results),
        "source_hit_rate": mean(1.0 if item["source_hit"] else 0.0 for item in results),
        "rank_ok_rate": mean(1.0 if item["rank_ok"] else 0.0 for item in results),
        "avg_keyword_hit_rate": mean(item["keyword_hit_rate"] for item in results),
        "avg_snippet_keyword_hit_rate": mean(item["snippet_keyword_hit_rate"] for item in results),
        "avg_top1_keyword_hit_rate": mean(item["top1_keyword_hit_rate"] for item in results),
        "avg_rewrite_hit_rate": mean(item["rewrite_hit_rate"] for item in results),
        "forbidden_hit_rate": mean(1.0 if item["forbidden_hits"] else 0.0 for item in results),
        "top1_forbidden_hit_rate": mean(1.0 if item["top1_forbidden_hits"] else 0.0 for item in results),
        "avg_citation_coverage_rate": mean(item["citation_coverage_rate"] for item in results),
        "avg_source_diversity": mean(item["source_diversity"] for item in results),
        "rewrite_mode_counts": {
            mode: sum(1 for item in results if item["rewrite_mode"] == mode)
            for mode in sorted({item["rewrite_mode"] for item in results})
        },
        "avg_first_hit_rank": round(mean(hit_ranks), 2) if hit_ranks else None,
    }
    return {"summary": summary, "cases": results}


def build_memory_suite(cases_path: Path, max_cases: int | None = None) -> dict[str, Any]:
    cases = load_memory_cases(cases_path)
    if max_cases is not None:
        cases = cases[:max_cases]
    results: list[dict[str, Any]] = []

    for case in cases:
        memory_tool = MemoryTool(session_id=case.case_id, default_top_k=case.top_k)
        memory_tool.clear_all()
        now = datetime.now()
        for entry in case.entries:
            timestamp = now - timedelta(seconds=entry.age_seconds)
            memory_tool.add(
                content=entry.content,
                role=entry.role,
                memory_type=entry.memory_type,
                importance=entry.importance,
                metadata=entry.metadata,
                timestamp=timestamp,
            )

        matches = memory_tool.search(
            case.query,
            top_k=case.top_k,
            memory_types=list(case.memory_types) if case.memory_types else None,
        )
        combined_text = "\n".join(item.content for item in matches)
        hit_rate = substring_hit_rate(combined_text, case.expected_substrings)
        first_hit = matches[0].content if matches else ""
        top1_ok = case.expected_top1 in first_hit if case.expected_top1 else True
        passed = hit_rate >= 1.0 and top1_ok

        results.append(
            {
                "id": case.case_id,
                "query": case.query,
                "goal": case.goal,
                "memory_types": list(case.memory_types),
                "expected_substrings": list(case.expected_substrings),
                "expected_top1": case.expected_top1,
                "match_count": len(matches),
                "matched_contents": [item.content for item in matches],
                "substring_hit_rate": round(hit_rate, 4),
                "top1_ok": top1_ok,
                "passed": passed,
            }
        )

    summary = {
        "suite": "memory",
        "case_count": len(results),
        "pass_rate": mean(1.0 if item["passed"] else 0.0 for item in results),
        "avg_substring_hit_rate": mean(item["substring_hit_rate"] for item in results),
        "top1_ok_rate": mean(1.0 if item["top1_ok"] else 0.0 for item in results),
    }
    return {"summary": summary, "cases": results}


def build_session_suite(
    *,
    settings: AppSettings,
    cases_path: Path,
    api_key: str,
    max_cases: int | None = None,
) -> dict[str, Any]:
    if not api_key:
        return {
            "summary": {
                "suite": "session",
                "skipped": True,
                "reason": "未检测到可用 API Key，无法运行会话级端到端评测。",
            },
            "cases": [],
        }

    cases = load_session_cases(cases_path)
    if max_cases is not None:
        cases = cases[:max_cases]
    service = HelloRagAgentService(settings=settings)
    results: list[dict[str, Any]] = []

    for case in cases:
        session = service._get_or_create_session(user_id=case.user_id)
        session_id = session.session_id
        for turn in case.turns:
            user_message = Message(content=turn, role="user")
            ack_message = Message(content="收到，我会记住这条信息。", role="assistant")
            session.history.extend([user_message, ack_message])
            session.memory_tool.remember_message(user_message)
            session.memory_tool.remember_message(ack_message)

        if case.reload_service_before_recall:
            service = HelloRagAgentService(settings=settings)
            session = service._get_or_create_session(session_id, user_id=case.user_id)

        memory_matches = session.memory_tool.search(case.recall_query, top_k=case.top_k)
        memory_text = "\n".join(item.content for item in memory_matches)
        answer = service._answer_with_retrieval(case.recall_query, session)
        memory_hit_rate = substring_hit_rate(memory_text, case.expected_memory_substrings)
        answer_hit_rate = substring_hit_rate(answer, case.expected_answer_substrings)
        trace_leak = contains_trace_markers(answer)
        passed = (
            memory_hit_rate >= 1.0
            and answer_hit_rate >= case.min_answer_hit_rate
            and not trace_leak
        )

        results.append(
            {
                "id": case.case_id,
                "goal": case.goal,
                "user_id": case.user_id,
                "turns": list(case.turns),
                "recall_query": case.recall_query,
                "expected_memory_substrings": list(case.expected_memory_substrings),
                "expected_answer_substrings": list(case.expected_answer_substrings),
                "reload_service_before_recall": case.reload_service_before_recall,
                "memory_hit_rate": round(memory_hit_rate, 4),
                "answer_hit_rate": round(answer_hit_rate, 4),
                "trace_leak": trace_leak,
                "memory_matches": [item.content for item in memory_matches],
                "answer": answer,
                "passed": passed,
            }
        )
        service.reset_session(session_id)

    summary = {
        "suite": "session",
        "case_count": len(results),
        "pass_rate": mean(1.0 if item["passed"] else 0.0 for item in results),
        "avg_memory_hit_rate": mean(item["memory_hit_rate"] for item in results),
        "avg_answer_hit_rate": mean(item["answer_hit_rate"] for item in results),
        "trace_leak_rate": mean(1.0 if item["trace_leak"] else 0.0 for item in results),
    }
    return {"summary": summary, "cases": results}


def build_fusion_suite(
    *,
    settings: AppSettings,
    cases_path: Path,
    api_key: str,
    max_cases: int | None = None,
) -> dict[str, Any]:
    if not api_key:
        return {
            "summary": {
                "suite": "fusion",
                "skipped": True,
                "reason": "未检测到可用 API Key，无法运行记忆-检索协同评测。",
            },
            "cases": [],
        }

    cases = load_fusion_cases(cases_path)
    if max_cases is not None:
        cases = cases[:max_cases]
    service = HelloRagAgentService(settings=settings)
    results: list[dict[str, Any]] = []

    for case in cases:
        session = service._get_or_create_session(user_id=case.user_id)
        session_id = session.session_id
        for turn in case.turns:
            user_message = Message(content=turn, role="user")
            ack_message = Message(content="收到，我会记住这条信息。", role="assistant")
            session.history.extend([user_message, ack_message])
            session.memory_tool.remember_message(user_message)
            session.memory_tool.remember_message(ack_message)

        memory_lines = service._collect_memory_lines(query=case.query, session=session)
        search_query = service._build_search_query(query=case.query, memory_lines=memory_lines)
        evidence = service._search_knowledge(query=case.query, search_query=search_query)
        answer = service._answer_with_retrieval(case.query, session)

        actual_sources = [item.chunk.source for item in evidence]
        evidence_text = "\n".join(item.chunk.content for item in evidence)
        matched_ranks = find_source_ranks(actual_sources, case.expected_sources)
        memory_hit_rate = substring_hit_rate("\n".join(memory_lines), case.expected_memory_substrings)
        query_hit_rate = substring_hit_rate(search_query, case.expected_query_substrings)
        source_hit = bool(matched_ranks) if case.expected_sources else True
        keyword_hit_rate = substring_hit_rate(evidence_text, case.expected_keywords)
        answer_hit_rate = substring_hit_rate(answer, case.expected_answer_substrings)
        trace_leak = contains_trace_markers(answer)
        passed = (
            memory_hit_rate >= 1.0
            and query_hit_rate >= 1.0
            and source_hit
            and answer_hit_rate >= case.min_answer_hit_rate
            and not trace_leak
        )

        results.append(
            {
                "id": case.case_id,
                "goal": case.goal,
                "user_id": case.user_id,
                "query": case.query,
                "memory_lines": memory_lines,
                "search_query": search_query,
                "actual_sources": actual_sources,
                "expected_sources": list(case.expected_sources),
                "memory_hit_rate": round(memory_hit_rate, 4),
                "query_hit_rate": round(query_hit_rate, 4),
                "source_hit": source_hit,
                "first_hit_rank": min(matched_ranks) if matched_ranks else None,
                "keyword_hit_rate": round(keyword_hit_rate, 4),
                "answer_hit_rate": round(answer_hit_rate, 4),
                "trace_leak": trace_leak,
                "answer": answer,
                "passed": passed,
            }
        )
        service.reset_session(session_id)

    hit_ranks = [item["first_hit_rank"] for item in results if item["first_hit_rank"] is not None]
    summary = {
        "suite": "fusion",
        "case_count": len(results),
        "pass_rate": mean(1.0 if item["passed"] else 0.0 for item in results),
        "avg_memory_hit_rate": mean(item["memory_hit_rate"] for item in results),
        "avg_query_hit_rate": mean(item["query_hit_rate"] for item in results),
        "source_hit_rate": mean(1.0 if item["source_hit"] else 0.0 for item in results),
        "avg_keyword_hit_rate": mean(item["keyword_hit_rate"] for item in results),
        "avg_answer_hit_rate": mean(item["answer_hit_rate"] for item in results),
        "trace_leak_rate": mean(1.0 if item["trace_leak"] else 0.0 for item in results),
        "avg_first_hit_rank": round(mean(hit_ranks), 2) if hit_ranks else None,
    }
    return {"summary": summary, "cases": results}


def build_answer_suite(
    *,
    settings: AppSettings,
    cases_path: Path,
    top_k: int,
    api_key: str,
    judge_model: str,
    max_cases: int | None = None,
) -> dict[str, Any]:
    if not api_key:
        return {
            "summary": {
                "suite": "answer",
                "skipped": True,
                "reason": "未检测到可用 API Key，无法运行回答级 LLM judge 评测。",
            },
            "cases": [],
        }

    cases = load_answer_cases(cases_path)
    if max_cases is not None:
        cases = cases[:max_cases]
    service = HelloRagAgentService(settings=settings)
    knowledge_base = KnowledgeBase(
        settings.knowledge_base,
        api_key=api_key,
        base_url=settings.llm.base_url,
    )
    judge_llm = SafeHelloAgentsLLM(
        model=judge_model,
        api_key=api_key,
        base_url=settings.llm.base_url,
        temperature=0.0,
        max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
        timeout=DEFAULT_JUDGE_TIMEOUT,
    )

    results: list[dict[str, Any]] = []
    for case in cases:
        answer, session_id = service.ask(case.question)
        evidence_results = knowledge_base.search(case.question, top_k=top_k)
        prompt = build_judge_prompt(
            question=case.question,
            goal=case.goal,
            answer=answer,
            evidence=render_evidence(evidence_results),
        )
        payload = invoke_and_parse_judge(judge_llm=judge_llm, prompt=prompt)
        hit_rate = substring_hit_rate(answer, case.expected_substrings)
        trace_leak = contains_trace_markers(answer)
        forbidden_hits = [item for item in case.forbidden_substrings if item in answer]
        rule_pass = hit_rate >= case.min_hit_rate and not trace_leak and not forbidden_hits
        passed = payload.get("verdict") == "pass" and rule_pass
        results.append(
            {
                "id": case.case_id,
                "category": case.category,
                "question": case.question,
                "goal": case.goal,
                "session_id": session_id,
                "answer": answer,
                "expected_substrings": list(case.expected_substrings),
                "substring_hit_rate": round(hit_rate, 4),
                "min_hit_rate": case.min_hit_rate,
                "trace_leak": trace_leak,
                "forbidden_hits": forbidden_hits,
                "rule_pass": rule_pass,
                "evidence": [
                    {
                        "source": item.chunk.source,
                        "title": item.chunk.title,
                        "score": item.score,
                        "citation": item.citation,
                        "snippet": item.snippet,
                        "content": item.chunk.content,
                    }
                    for item in evidence_results
                ],
                "judge": payload,
                "average_score": average_score(payload),
                "passed": passed,
            }
        )

    verdicts = [item["judge"]["verdict"] for item in results]
    summary = {
        "suite": "answer",
        "case_count": len(results),
        "judge_model": judge_model,
        "average_score": mean(item["average_score"] for item in results),
        "pass_rate": sum(verdict == "pass" for verdict in verdicts) / len(verdicts),
        "borderline_rate": sum(verdict == "borderline" for verdict in verdicts) / len(verdicts),
        "fail_rate": sum(verdict == "fail" for verdict in verdicts) / len(verdicts),
        "rule_pass_rate": mean(1.0 if item["rule_pass"] else 0.0 for item in results),
        "avg_substring_hit_rate": mean(item["substring_hit_rate"] for item in results),
        "trace_leak_rate": mean(1.0 if item["trace_leak"] else 0.0 for item in results),
        "forbidden_hit_rate": mean(1.0 if item["forbidden_hits"] else 0.0 for item in results),
        "dimension_averages": {
            "groundedness": mean(item["judge"]["groundedness"] for item in results),
            "relevance": mean(item["judge"]["relevance"] for item in results),
            "completeness": mean(item["judge"]["completeness"] for item in results),
            "clarity": mean(item["judge"]["clarity"] for item in results),
        },
    }
    return {"summary": summary, "cases": results}


def build_answer_smoke_suite(
    *,
    settings: AppSettings,
    cases_path: Path,
    api_key: str,
    max_cases: int | None = None,
) -> dict[str, Any]:
    if not api_key:
        return {
            "summary": {
                "suite": "answer_smoke",
                "skipped": True,
                "reason": "未检测到可用 API Key，无法运行轻量回答冒烟评测。",
            },
            "cases": [],
        }

    cases = load_answer_smoke_cases(cases_path)
    if max_cases is not None:
        cases = cases[:max_cases]
    service = HelloRagAgentService(settings=settings)
    results: list[dict[str, Any]] = []

    for case in cases:
        answer, session_id = service.ask(case.question)
        hit_rate = substring_hit_rate(answer, case.expected_substrings)
        trace_leak = contains_trace_markers(answer)
        forbidden_hits = [item for item in case.forbidden_substrings if item in answer]
        passed = hit_rate >= case.min_hit_rate and not trace_leak and not forbidden_hits
        results.append(
            {
                "id": case.case_id,
                "question": case.question,
                "goal": case.goal,
                "session_id": session_id,
                "answer": answer,
                "expected_substrings": list(case.expected_substrings),
                "substring_hit_rate": round(hit_rate, 4),
                "min_hit_rate": case.min_hit_rate,
                "trace_leak": trace_leak,
                "forbidden_hits": forbidden_hits,
                "passed": passed,
            }
        )
        service.reset_session(session_id)

    summary = {
        "suite": "answer_smoke",
        "case_count": len(results),
        "pass_rate": mean(1.0 if item["passed"] else 0.0 for item in results),
        "avg_substring_hit_rate": mean(item["substring_hit_rate"] for item in results),
        "trace_leak_rate": mean(1.0 if item["trace_leak"] else 0.0 for item in results),
        "forbidden_hit_rate": mean(1.0 if item["forbidden_hits"] else 0.0 for item in results),
    }
    return {"summary": summary, "cases": results}


def build_markdown_report(
    *,
    summary: dict[str, Any],
    suites: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Project Evaluation Report",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Model under test: {summary['model_under_test']}",
        f"- Suites requested: {', '.join(summary['requested_suites'])}",
        f"- Suites completed: {', '.join(summary['completed_suites']) or 'none'}",
        f"- Suites skipped: {', '.join(summary['skipped_suites'].keys()) or 'none'}",
        f"- Overall pass rate: {summary['overall_pass_rate']:.2%}" if summary["overall_pass_rate"] is not None else "- Overall pass rate: n/a",
        "",
        "## Suite Summaries",
        "",
    ]

    for suite_name in summary["requested_suites"]:
        suite_payload = suites.get(suite_name)
        if not suite_payload:
            continue
        suite_summary = suite_payload["summary"]
        lines.append(f"### {suite_name}")
        lines.append("")
        if suite_summary.get("skipped"):
            lines.append("- Status: skipped")
            lines.append(f"- Reason: {suite_summary['reason']}")
            lines.append("")
            continue

        for key, value in suite_summary.items():
            if key == "suite":
                continue
            if isinstance(value, float):
                if key.endswith("rate"):
                    lines.append(f"- {key}: {value:.2%}")
                else:
                    lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")

    lines.extend(["## Failed Cases", ""])
    failed_any = False
    for suite_name, suite_payload in suites.items():
        suite_summary = suite_payload["summary"]
        if suite_summary.get("skipped"):
            continue
        failed_cases = [case for case in suite_payload["cases"] if not case.get("passed", False)]
        if not failed_cases:
            continue
        failed_any = True
        lines.append(f"### {suite_name}")
        lines.append("")
        for case in failed_cases:
            case_title = case.get("query") or case.get("question") or case.get("recall_query") or case.get("id")
            lines.append(f"- {case['id']}: {case_title}")
        lines.append("")

    if not failed_any:
        lines.append("- No failed cases.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_suites(raw: str) -> list[str]:
    allowed = ("retrieval", "memory", "session", "fusion", "answer", "answer_smoke")
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported suites: {', '.join(invalid)}")
    return requested or list(allowed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the project evaluation suites.")
    parser.add_argument(
        "--suites",
        default="retrieval,memory,session,fusion,answer",
        help="Comma-separated suites: retrieval,memory,session,fusion,answer,answer_smoke",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "dev"),
        default="full",
        help="Use 'dev' for a lightweight local feedback loop.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Default evidence size for retrieval/answer suites.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit each suite to the first N cases. Useful for fast local iteration.",
    )
    parser.add_argument("--retrieval-cases", type=Path, default=DEFAULT_RETRIEVAL_CASES_PATH)
    parser.add_argument("--memory-cases", type=Path, default=DEFAULT_MEMORY_CASES_PATH)
    parser.add_argument("--session-cases", type=Path, default=DEFAULT_SESSION_CASES_PATH)
    parser.add_argument("--fusion-cases", type=Path, default=DEFAULT_FUSION_CASES_PATH)
    parser.add_argument("--answer-cases", type=Path, default=DEFAULT_ANSWER_CASES_PATH)
    parser.add_argument("--answer-smoke-cases", type=Path, default=DEFAULT_ANSWER_SMOKE_CASES_PATH)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--output-prefix", default="project_eval")
    args = parser.parse_args()

    requested_suites = parse_suites(args.suites)
    max_cases = args.max_cases
    if args.profile == "dev":
        if args.suites == parser.get_default("suites"):
            requested_suites = ["retrieval", "memory", "session", "fusion", "answer_smoke"]
        if max_cases is None:
            max_cases = 2
    settings = load_settings()
    api_key = safe_resolve_api_key(settings)

    suites: dict[str, dict[str, Any]] = {}
    if "retrieval" in requested_suites:
        suites["retrieval"] = build_retrieval_suite(
            settings=settings,
            cases_path=args.retrieval_cases,
            default_top_k=args.top_k,
            api_key=api_key,
            max_cases=max_cases,
        )
    if "memory" in requested_suites:
        suites["memory"] = build_memory_suite(args.memory_cases, max_cases=max_cases)
    if "session" in requested_suites:
        suites["session"] = build_session_suite(
            settings=settings,
            cases_path=args.session_cases,
            api_key=api_key,
            max_cases=max_cases,
        )
    if "fusion" in requested_suites:
        suites["fusion"] = build_fusion_suite(
            settings=settings,
            cases_path=args.fusion_cases,
            api_key=api_key,
            max_cases=max_cases,
        )
    if "answer" in requested_suites:
        suites["answer"] = build_answer_suite(
            settings=settings,
            cases_path=args.answer_cases,
            top_k=args.top_k,
            api_key=api_key,
            judge_model=args.judge_model,
            max_cases=max_cases,
        )
    if "answer_smoke" in requested_suites:
        suites["answer_smoke"] = build_answer_smoke_suite(
            settings=settings,
            cases_path=args.answer_smoke_cases,
            api_key=api_key,
            max_cases=max_cases,
        )

    completed_suites = [
        name for name, payload in suites.items() if not payload["summary"].get("skipped")
    ]
    skipped_suites = {
        name: payload["summary"]["reason"]
        for name, payload in suites.items()
        if payload["summary"].get("skipped")
    }
    all_case_results = [
        case
        for payload in suites.values()
        if not payload["summary"].get("skipped")
        for case in payload["cases"]
    ]
    overall_pass_rate = (
        mean(1.0 if case.get("passed", False) else 0.0 for case in all_case_results)
        if all_case_results
        else None
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_under_test": settings.llm.model,
        "requested_suites": requested_suites,
        "completed_suites": completed_suites,
        "skipped_suites": skipped_suites,
        "overall_pass_rate": overall_pass_rate,
        "suite_summaries": {name: payload["summary"] for name, payload in suites.items()},
    }

    payload = {"summary": summary, "suites": suites}
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = DEFAULT_OUTPUT_DIR / f"{args.output_prefix}_{timestamp}.json"
    md_path = DEFAULT_OUTPUT_DIR / f"{args.output_prefix}_{timestamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_report(summary=summary, suites=suites), encoding="utf-8")

    print(f"Suites requested: {', '.join(requested_suites)}")
    print(f"Suites completed: {', '.join(completed_suites) or 'none'}")
    if overall_pass_rate is None:
        print("Overall pass rate: n/a")
    else:
        print(f"Overall pass rate: {overall_pass_rate:.2%}")
    if skipped_suites:
        for suite_name, reason in skipped_suites.items():
            print(f"Skipped {suite_name}: {reason}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
