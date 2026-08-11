"""Enhanced Qwen2.5-VL adapter for OSWorld."""

from .agent import QwenAgent
from .images import process_image

__all__ = ["QwenAgent", "process_image"]
__version__ = "0.1.0"
