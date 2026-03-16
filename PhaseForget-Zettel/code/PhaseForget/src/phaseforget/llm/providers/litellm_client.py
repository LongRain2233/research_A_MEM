"""
LiteLLM-backed LLM client.

Provides unified access to OpenAI, OpenRouter, local vLLM, and other
backends via the LiteLLM gateway. Aligns with Implementation Plan §5.1.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

import litellm

from phaseforget.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class LiteLLMClient(BaseLLMClient):
    """LLM client using LiteLLM for multi-provider support."""

    DEFAULT_TIMEOUT = 120  # seconds

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self._model = model
        self._timeout = timeout
        if api_key:
            litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url
        litellm.drop_params = True
        logger.info(f"LiteLLMClient initialized: model={model}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text completion via LiteLLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=self._timeout,
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            logger.error(f"LLM generation timed out after {self._timeout}s")
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate a JSON response, parsing the LLM output."""
        raw = await self.generate(prompt, system_prompt, temperature, max_tokens)

        # Strip markdown code fences if present
        text = raw.strip()
        # Handle ```json ... ``` or ``` ... ``` wrapping
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM JSON response: {text[:200]}...")
            return {}
