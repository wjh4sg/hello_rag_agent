from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "settings.yml"
DEFAULT_PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"


@dataclass(frozen=True)
class AgentSettings:
    name: str
    max_steps: int
    history_turns: int


@dataclass(frozen=True)
class KnowledgeBaseSettings:
    data_dir: Path
    supported_extensions: tuple[str, ...]
    chunk_size: int
    chunk_overlap: int
    atomic_chunking_enabled: bool
    atomic_chunk_max_chars: int
    top_k: int
    retrieval_mode: str
    rerank_pool_size: int
    reranker_enabled: bool
    reranker_semantic_weight: float
    max_query_variants: int
    query_rewrite_enabled: bool
    query_rewrite_model: str
    query_rewrite_max_keywords: int
    max_evidence_points: int
    snippet_max_chars: int
    vector_store_dir: Path
    collection_name: str
    embedding_model: str
    embedding_dimensions: int | None


@dataclass(frozen=True)
class LLMSettings:
    model: str
    base_url: str
    temperature: float
    max_tokens: int | None


@dataclass(frozen=True)
class MemorySettings:
    db_path: Path
    working_max_entries: int
    working_ttl_minutes: int
    default_top_k: int
    profile_enabled: bool
    profile_max_facts: int
    assistant_memory_min_chars: int


@dataclass(frozen=True)
class AppSettings:
    agent: AgentSettings
    knowledge_base: KnowledgeBaseSettings
    llm: LLMSettings
    memory: MemorySettings
    prompt_path: Path

    def resolve_api_key(self) -> str:
        for env_name in ("LLM_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
            value = os.getenv(env_name)
            if value:
                return value
        raise RuntimeError(
            "未找到可用的 API Key，请在 `.env` 或系统环境变量中配置 "
            "`LLM_API_KEY`、`DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY`。"
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_settings(config_path: Path | None = None) -> AppSettings:
    load_dotenv(BASE_DIR / ".env")

    path = config_path or DEFAULT_CONFIG_PATH
    raw = _read_yaml(path)

    agent_conf = raw.get("agent", {})
    kb_conf = raw.get("knowledge_base", {})
    llm_conf = raw.get("llm", {})
    memory_conf = raw.get("memory", {})

    prompt_path = DEFAULT_PROMPT_PATH

    return AppSettings(
        agent=AgentSettings(
            name=agent_conf.get("name", "hello-rag-agent"),
            max_steps=int(agent_conf.get("max_steps", 4)),
            history_turns=int(agent_conf.get("history_turns", 6)),
        ),
        knowledge_base=KnowledgeBaseSettings(
            data_dir=(BASE_DIR / kb_conf.get("data_dir", "data")).resolve(),
            supported_extensions=tuple(kb_conf.get("supported_extensions", [".txt", ".md", ".pdf"])),
            chunk_size=int(kb_conf.get("chunk_size", 600)),
            chunk_overlap=int(kb_conf.get("chunk_overlap", 120)),
            atomic_chunking_enabled=str(
                os.getenv("KB_ATOMIC_CHUNKING_ENABLED", kb_conf.get("atomic_chunking_enabled", True))
            ).strip().lower()
            not in {"0", "false", "no", "off"},
            atomic_chunk_max_chars=int(
                os.getenv("KB_ATOMIC_CHUNK_MAX_CHARS", kb_conf.get("atomic_chunk_max_chars", 320))
            ),
            top_k=int(kb_conf.get("top_k", 4)),
            retrieval_mode=str(os.getenv("KB_RETRIEVAL_MODE", kb_conf.get("retrieval_mode", "hybrid"))),
            rerank_pool_size=int(os.getenv("KB_RERANK_POOL_SIZE", kb_conf.get("rerank_pool_size", 12))),
            reranker_enabled=str(
                os.getenv("KB_RERANKER_ENABLED", kb_conf.get("reranker_enabled", True))
            ).strip().lower()
            not in {"0", "false", "no", "off"},
            reranker_semantic_weight=float(
                os.getenv("KB_RERANKER_SEMANTIC_WEIGHT", kb_conf.get("reranker_semantic_weight", 0.55))
            ),
            max_query_variants=int(os.getenv("KB_MAX_QUERY_VARIANTS", kb_conf.get("max_query_variants", 6))),
            query_rewrite_enabled=str(
                os.getenv("KB_QUERY_REWRITE_ENABLED", kb_conf.get("query_rewrite_enabled", True))
            ).strip().lower()
            not in {"0", "false", "no", "off"},
            query_rewrite_model=str(
                os.getenv(
                    "KB_QUERY_REWRITE_MODEL",
                    kb_conf.get("query_rewrite_model", os.getenv("LLM_MODEL_ID", "qwen-plus")),
                )
            ),
            query_rewrite_max_keywords=int(
                os.getenv("KB_QUERY_REWRITE_MAX_KEYWORDS", kb_conf.get("query_rewrite_max_keywords", 6))
            ),
            max_evidence_points=int(os.getenv("KB_MAX_EVIDENCE_POINTS", kb_conf.get("max_evidence_points", 4))),
            snippet_max_chars=int(os.getenv("KB_SNIPPET_MAX_CHARS", kb_conf.get("snippet_max_chars", 220))),
            vector_store_dir=(
                BASE_DIR / os.getenv("KB_VECTOR_STORE_DIR", kb_conf.get("vector_store_dir", "data/chroma_db"))
            ).resolve(),
            collection_name=str(
                os.getenv("KB_COLLECTION_NAME", kb_conf.get("collection_name", "hello_rag_agent_kb"))
            ),
            embedding_model=str(
                os.getenv("EMBEDDING_MODEL_ID", kb_conf.get("embedding_model", "text-embedding-v4"))
            ),
            embedding_dimensions=(
                int(os.getenv("EMBEDDING_DIMENSIONS", kb_conf.get("embedding_dimensions")))
                if os.getenv("EMBEDDING_DIMENSIONS") or kb_conf.get("embedding_dimensions") is not None
                else None
            ),
        ),
        llm=LLMSettings(
            model=os.getenv("LLM_MODEL_ID", llm_conf.get("model", "qwen-plus")),
            base_url=os.getenv(
                "LLM_BASE_URL",
                llm_conf.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ),
            temperature=float(os.getenv("LLM_TEMPERATURE", llm_conf.get("temperature", 0.2))),
            max_tokens=(
                int(os.getenv("LLM_MAX_TOKENS", llm_conf.get("max_tokens")))
                if os.getenv("LLM_MAX_TOKENS") or llm_conf.get("max_tokens") is not None
                else None
            ),
        ),
        memory=MemorySettings(
            db_path=(BASE_DIR / os.getenv("MEMORY_DB_PATH", memory_conf.get("db_path", "data/memory.db"))).resolve(),
            working_max_entries=int(
                os.getenv("MEMORY_WORKING_MAX_ENTRIES", memory_conf.get("working_max_entries", 80))
            ),
            working_ttl_minutes=int(
                os.getenv("MEMORY_WORKING_TTL_MINUTES", memory_conf.get("working_ttl_minutes", 240))
            ),
            default_top_k=int(os.getenv("MEMORY_DEFAULT_TOP_K", memory_conf.get("default_top_k", 4))),
            profile_enabled=str(
                os.getenv("MEMORY_PROFILE_ENABLED", memory_conf.get("profile_enabled", True))
            ).strip().lower()
            not in {"0", "false", "no", "off"},
            profile_max_facts=int(os.getenv("MEMORY_PROFILE_MAX_FACTS", memory_conf.get("profile_max_facts", 50))),
            assistant_memory_min_chars=int(
                os.getenv(
                    "MEMORY_ASSISTANT_MIN_CHARS",
                    memory_conf.get("assistant_memory_min_chars", 20),
                )
            ),
        ),
        prompt_path=prompt_path,
    )
