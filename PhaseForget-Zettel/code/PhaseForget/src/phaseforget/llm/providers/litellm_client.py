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
import time
from typing import Any, Optional

import litellm

from phaseforget.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

# Retryable HTTP status codes
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds, doubles each attempt


def _extract_json_object(text: str) -> str | None:
    """
    Best-effort extraction of a JSON object string from raw LLM output.

    Strategy (mirrors A-mem's analyze_content / _parse_json_response):
      1. Strip leading/trailing whitespace.
      2. Remove markdown code fences (```json ... ``` or ``` ... ```).
      3. Locate the outermost { ... } block to tolerate preamble/postamble text.

    Returns the extracted JSON string, or None if no { ... } block found.
    """
    if not text:
        return None

    # Remove markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()

    # Find outermost { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    return cleaned[start : end + 1]


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

    def _is_retryable(self, exc: Exception) -> bool:
        """Determine whether an exception warrants a retry attempt."""
        status_code = getattr(exc, "status_code", None)
        if status_code in _RETRYABLE_STATUS_CODES:
            return True
        msg = str(exc).lower()
        return any(kw in msg for kw in ("timeout", "timed out", "connection", "rate limit", "rate_limit"))

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Generate text completion via LiteLLM with automatic retry.

        Retries up to _MAX_RETRIES times on transient errors (timeout, 429,
        5xx) with exponential backoff. Non-retryable errors are re-raised
        immediately.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                completion_kwargs: dict[str, Any] = dict(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response_format is not None:
                    completion_kwargs["response_format"] = response_format
                response = await asyncio.wait_for(
                    litellm.acompletion(**completion_kwargs),
                    timeout=self._timeout,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned null content")
                return content
            except asyncio.TimeoutError as e:
                last_exc = e
                wait = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"LLM timeout on attempt {attempt + 1}/{_MAX_RETRIES}, "
                    f"retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)
            except Exception as e:
                if self._is_retryable(e):
                    last_exc = e
                    wait = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"LLM retryable error on attempt {attempt + 1}/{_MAX_RETRIES} "
                        f"({type(e).__name__}: {str(e)[:80]}), retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"LLM generation failed (non-retryable): {e}")
                    raise

        logger.error(f"LLM generation failed after {_MAX_RETRIES} retries: {last_exc}")
        raise last_exc  # type: ignore[misc]

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Generate a JSON response, parsing the LLM output.

        Parsing strategy:
          1. Call generate() with response_format={"type": "json_object"} so that
             ollama/OpenAI-compatible backends enforce JSON-mode output natively,
             eliminating most parse failures before they happen.
          2. Use _extract_json_object() to tolerate any residual preamble/postamble.
          3. json.loads() the extracted string.
          4. Validate root type is dict; treat list/scalar as a parse failure.
          5. On failure return {} and log a warning with a raw preview.
        """
        raw = await self.generate(
            prompt,
            system_prompt,
            temperature,
            max_tokens,
            response_format={"type": "json_object"},
        )

        extracted = _extract_json_object(raw)
        if extracted is None:
            logger.warning(
                f"Failed to locate JSON object in LLM response: {raw[:200]}..."
            )
            return {}

        try:
            result = json.loads(extracted)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse LLM JSON response ({e}): {extracted[:200]}..."
            )
            return {}

        if not isinstance(result, dict):
            logger.warning(
                f"LLM JSON root is {type(result).__name__}, expected dict: {extracted[:200]}..."
            )
            return {}

        return result
