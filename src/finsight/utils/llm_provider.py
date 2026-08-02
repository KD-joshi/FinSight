"""Unified LLM provider with automatic fallback.

Strategy
--------
1. **Primary** — Groq Cloud (Fastest inference)
2. **Fallback 1** — Google Gemini Flash (Generous free tier)
3. **Fallback 2** — Cohere Command-R-Plus (Generous trial tier)

All functions return **LangChain-compatible** ``BaseChatModel`` instances.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere

from config.settings import settings

logger = logging.getLogger(__name__)

# Retryable HTTP status codes
_RETRYABLE_STATUS_CODES: set[int] = {429, 500, 502, 503, 504}
_MAX_RETRIES: int = settings.llm_max_retries
_INITIAL_BACKOFF_SECS: float = 1.0
_BACKOFF_MULTIPLIER: float = 2.0


# ======================================================================
# Factory helpers
# ======================================================================

def get_llm(**kwargs) -> BaseChatModel:
    """Return the primary Groq language model."""
    model = settings.groq_model
    temperature = kwargs.pop("temperature", None)
    max_tokens = kwargs.pop("max_tokens", None)
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        max_retries=settings.llm_max_retries,
        **kwargs
    )

def get_fallback_llm(**kwargs) -> BaseChatModel:
    """Return the Gemini fallback model."""
    if not settings.google_api_key:
        raise ValueError("Google API key not configured")
    model = kwargs.pop("model", None)
    temperature = kwargs.pop("temperature", None)
    max_tokens = kwargs.pop("max_tokens", None)
    return ChatGoogleGenerativeAI(
        model=model or settings.gemini_model,
        google_api_key=settings.google_api_key,
        max_output_tokens=max_tokens or settings.llm_max_tokens,
        thinking_level="medium",
        **kwargs
    )

def get_cohere_llm(**kwargs) -> BaseChatModel:
    """Return the Cohere fallback model."""
    if not settings.cohere_api_key:
        raise ValueError("Cohere API key not configured")
    model = kwargs.pop("model", None)
    temperature = kwargs.pop("temperature", None)
    return ChatCohere(
        model=model or "command-r-plus",
        cohere_api_key=settings.cohere_api_key,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        **kwargs
    )


# ======================================================================
# Invocation with automatic waterfall fallback
# ======================================================================

def _is_retryable(exc: Exception) -> bool:
    exc_str = str(exc).lower()
    status: Optional[int] = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status and int(status) in _RETRYABLE_STATUS_CODES:
        return True
    retryable_substrings = ("429", "rate limit", "rate_limit", "resource_exhausted", "too many requests")
    return any(s in exc_str for s in retryable_substrings)

def invoke_with_fallback(
    prompt: Union[str, list[BaseMessage], ChatPromptTemplate],
    *,
    parse_to_str: bool = True,
    **invoke_kwargs: Any,
) -> Union[str, BaseMessage]:
    """Invoke the LLM with a 3-tier waterfall fallback.
    Groq -> Gemini -> Cohere.
    """
    # ── Normalise prompt
    messages: list[BaseMessage]
    if isinstance(prompt, str):
        messages = [HumanMessage(content=prompt)]
    elif isinstance(prompt, ChatPromptTemplate):
        messages = prompt.format_messages(**invoke_kwargs)
        invoke_kwargs = {}
    else:
        messages = list(prompt)

    # ── Collect available providers
    providers = []
    try:
        providers.append(get_llm())
    except ValueError: pass
    try:
        providers.append(get_fallback_llm())
    except ValueError: pass
    try:
        providers.append(get_cohere_llm())
    except ValueError: pass

    if not providers:
        raise RuntimeError("No LLM providers are configured in settings!")

    last_exc: Optional[Exception] = None

    for i, provider in enumerate(providers):
        provider_name = type(provider).__name__
        
        # We only retry the FIRST provider (Groq) to avoid excessive waiting.
        # Fallbacks are attempted exactly once.
        max_attempts = _MAX_RETRIES if i == 0 else 1
        backoff = _INITIAL_BACKOFF_SECS

        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug("Invoking %s (attempt %d/%d)", provider_name, attempt, max_attempts)
                response = provider.invoke(messages, **invoke_kwargs)
                logger.info("Successfully received response from %s", provider_name)
                return response.content if parse_to_str else response

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_retryable(exc) and attempt < max_attempts:
                    logger.warning("Retryable error on %s: %s — backing off %.1fs", provider_name, exc, backoff)
                    time.sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER
                else:
                    logger.warning("%s failed: %s", provider_name, exc)
                    break # Give up on this provider, move to the next in waterfall
    
    # If we get here, all providers failed
    logger.error("All LLM providers in the waterfall failed.")
    raise last_exc # type: ignore


# ======================================================================
# Convenience: chain-compatible wrapper
# ======================================================================

def get_llm_with_fallback(**kwargs) -> BaseChatModel:
    """Return a primary LLM that has a ``.with_fallbacks()`` chain."""
    primary = get_llm(**kwargs)
    fallbacks = []
    try:
        fallbacks.append(get_fallback_llm(**kwargs))
    except ValueError: pass
    try:
        fallbacks.append(get_cohere_llm(**kwargs))
    except ValueError: pass

    if fallbacks:
        return primary.with_fallbacks(fallbacks)
    return primary
