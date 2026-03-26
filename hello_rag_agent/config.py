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
    top_k: int
    retrieval_mode: str
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
class AppSettings:
    agent: AgentSettings
    knowledge_base: KnowledgeBaseSettings
    llm: LLMSettings
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
            top_k=int(kb_conf.get("top_k", 4)),
            retrieval_mode=str(os.getenv("KB_RETRIEVAL_MODE", kb_conf.get("retrieval_mode", "hybrid"))),
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
        prompt_path=prompt_path,
    )
