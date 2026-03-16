"""
Abstract base class for LLM clients.

All LLM interactions in PhaseForget are routed through this interface,
enabling backend-agnostic operation (OpenAI, local vLLM, etc.) via LiteLLM.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text completion from the LLM."""
        ...

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate a structured JSON response from the LLM."""
        ...
