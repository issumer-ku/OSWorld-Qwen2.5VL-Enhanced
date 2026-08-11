import ipaddress
import json
import os
import time
from typing import Dict, Optional
from urllib.parse import urlparse

import openai
from requests.exceptions import SSLError


class RetryableChatCompletionError(RuntimeError):
    """The provider returned a successful HTTP response without a usable answer."""


MAX_RETRY_TIMES = int(os.getenv("OSWORLD_MAX_RETRY_TIMES", "5"))


def is_local_openai_endpoint(base_url: Optional[str]) -> bool:
    """Return whether an OpenAI-compatible endpoint is local to the user."""
    endpoint = (base_url or "").strip()
    if not endpoint:
        return False
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
    host = (parsed.hostname or "").strip().lower()
    if host in {"localhost", "host.docker.internal"} or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(address.is_loopback or address.is_private or address.is_link_local)


def extract_content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(part.get("text", ""))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def extract_message_field(message, field: str):
    value = getattr(message, field, None)
    if value is not None:
        return value

    if hasattr(message, "model_dump"):
        dumped = message.model_dump()
        return dumped.get(field)

    if isinstance(message, dict):
        return message.get(field)

    return None


def merge_reasoning_content(content, reasoning_content) -> str:
    content_text = extract_content_text(content)
    reasoning_text = extract_content_text(reasoning_content).strip()
    if not reasoning_text:
        return content_text
    return f"<think>\n{reasoning_text}\n</think>\n\n{content_text.lstrip()}"


def serialize_native_tool_calls(message) -> str:
    tool_calls = extract_message_field(message, "tool_calls") or []
    serialized = []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        if function is None and isinstance(tool_call, dict):
            function = tool_call.get("function")
        if function is None:
            continue

        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")
        else:
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
        if not name:
            continue

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        serialized.append(
            json.dumps(
                {"name": name, "arguments": arguments},
                ensure_ascii=False,
            )
        )
    return "\n".join(serialized)


def call_openai_compatible(
    payload: Dict,
    model: str,
    *,
    base_url: Optional[str],
    api_key: Optional[str],
    default_max_tokens: int,
    default_temperature: float,
    default_top_p: float,
    logger=None,
) -> str:
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
    provider_default_read_timeout = (
        "600" if "openrouter.ai" in resolved_base_url.lower() else "120"
    )
    default_timeout = str(
        float(os.environ.get("OSWORLD_HTTP_CONNECT_TIMEOUT", "15"))
        + float(
            os.environ.get(
                "OSWORLD_HTTP_READ_TIMEOUT",
                provider_default_read_timeout,
            )
        )
    )
    timeout_s = float(os.environ.get("OSWORLD_OPENAI_TIMEOUT", default_timeout))

    try:
        client = openai.OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            timeout=timeout_s,
            max_retries=0,
        )
    except TypeError:
        client = openai.OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    retryable_types = tuple(
        exc
        for exc in [
            SSLError,
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "BadRequestError", None),
            getattr(openai, "InternalServerError", None),
            RetryableChatCompletionError,
        ]
        if isinstance(exc, type)
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRY_TIMES + 1):
        try:
            create_kwargs = dict(
                model=model,
                messages=payload["messages"],
                max_tokens=payload.get("max_tokens", default_max_tokens),
                temperature=payload.get("temperature", default_temperature),
                top_p=payload.get("top_p", default_top_p),
            )
            extra_body = payload.get("extra_body")
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            if payload.get("tools"):
                create_kwargs["tools"] = payload["tools"]
                create_kwargs["tool_choice"] = payload.get("tool_choice", "auto")
            response = client.chat.completions.create(**create_kwargs)
            choices = getattr(response, "choices", None)
            if not choices:
                raise RetryableChatCompletionError(
                    "OpenAI-compatible endpoint returned no completion choices"
                )
            message = getattr(choices[0], "message", None)
            if message is None:
                raise RetryableChatCompletionError(
                    "OpenAI-compatible endpoint returned a choice without a message"
                )
            content = extract_message_field(message, "content")
            reasoning_content = extract_message_field(message, "reasoning_content")
            merged_content = merge_reasoning_content(content, reasoning_content)
            native_tool_calls = serialize_native_tool_calls(message)
            if native_tool_calls:
                merged_content = "\n\n".join(
                    part for part in (merged_content.strip(), native_tool_calls) if part
                )
            if not merged_content.strip():
                response_id = getattr(response, "id", None)
                finish_reason = getattr(choices[0], "finish_reason", None)
                raise RetryableChatCompletionError(
                    "OpenAI-compatible endpoint returned an empty completion "
                    f"(response_id={response_id}, finish_reason={finish_reason})"
                )
            if logger:
                logger.info(
                    "[QwenAgent] provider response id=%s model=%s finish_reason=%s native_tool_calls=%s",
                    getattr(response, "id", None),
                    getattr(response, "model", None),
                    getattr(choices[0], "finish_reason", None),
                    bool(native_tool_calls),
                )
            return merged_content
        except retryable_types as exc:
            last_err = exc
            error_text = str(exc).lower()
            status_code = getattr(exc, "status_code", None)
            if (
                "maximum context length" in error_text
                or ("input length" in error_text and "exceeds" in error_text)
                or "context_length_exceeded" in error_text
                or (
                    isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code != 429
                )
            ):
                raise
            if logger:
                logger.warning(
                    "[QwenAgent] call_llm failed attempt %d/%d: %s",
                    attempt,
                    MAX_RETRY_TIMES,
                    exc,
                )
            time.sleep(min(5.0 * attempt, 30.0))

    if last_err is not None:
        raise last_err
    return ""
