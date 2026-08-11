"""Shared action protocol helpers for agent-to-runner control signals."""

from typing import Optional


ADAPTER_ERROR_PREFIX = "ADAPTER_ERROR:"
MODEL_ERROR_PREFIX = "MODEL_ERROR:"


def adapter_error_action(reason: str) -> str:
    normalized = "_".join(str(reason or "unknown").strip().lower().split())
    return f"{ADAPTER_ERROR_PREFIX}{normalized or 'unknown'}"


def decode_adapter_error_action(action) -> Optional[str]:
    if not isinstance(action, str) or not action.startswith(ADAPTER_ERROR_PREFIX):
        return None
    reason = action[len(ADAPTER_ERROR_PREFIX):].strip()
    return reason or "unknown"


def model_error_action(reason: str) -> str:
    normalized = "_".join(str(reason or "unknown").strip().lower().split())
    return f"{MODEL_ERROR_PREFIX}{normalized or 'unknown'}"


def decode_model_error_action(action) -> Optional[str]:
    if not isinstance(action, str) or not action.startswith(MODEL_ERROR_PREFIX):
        return None
    reason = action[len(MODEL_ERROR_PREFIX):].strip()
    return reason or "unknown"
