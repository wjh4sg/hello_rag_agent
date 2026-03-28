from __future__ import annotations

import os

import httpx
from hello_agents import HelloAgentsLLM
from hello_agents.core.llm_adapters import AnthropicAdapter, BaseLLMAdapter, GeminiAdapter, OpenAIAdapter


class DirectOpenAIAdapter(OpenAIAdapter):
    """Use an explicit HTTP client while still honoring local proxy settings by default."""

    @staticmethod
    def _trust_env() -> bool:
        raw = os.getenv("LLM_TRUST_ENV", "true").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def create_client(self):
        from openai import OpenAI

        http_client = httpx.Client(timeout=self.timeout, trust_env=self._trust_env())
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            http_client=http_client,
        )

    def create_async_client(self):
        from openai import AsyncOpenAI

        http_client = httpx.AsyncClient(timeout=self.timeout, trust_env=self._trust_env())
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            http_client=http_client,
        )


def create_direct_adapter(
    api_key: str,
    base_url: str | None,
    timeout: int,
    model: str,
) -> BaseLLMAdapter:
    if base_url:
        base_url_lower = base_url.lower()
        if "anthropic.com" in base_url_lower:
            return AnthropicAdapter(api_key, base_url, timeout, model)
        if "googleapis.com" in base_url_lower or "generativelanguage" in base_url_lower:
            return GeminiAdapter(api_key, base_url, timeout, model)
    return DirectOpenAIAdapter(api_key, base_url, timeout, model)


class SafeHelloAgentsLLM(HelloAgentsLLM):
    """Project-local wrapper that keeps hello_agents while forcing direct model access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapter = create_direct_adapter(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            model=self.model,
        )
