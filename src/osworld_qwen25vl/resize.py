"""Qwen-compatible image resizing helpers."""

import math


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
    max_long_side: int = 8192,
) -> tuple[int, int]:
    """Resize dimensions while preserving Qwen's patch-grid constraints."""
    if height < 2 or width < 2:
        raise ValueError(f"height={height} and width={width} must both be at least 2")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio is too large: {height}/{width}")
    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)
    resized_height = _round_by_factor(height, factor)
    resized_width = _round_by_factor(width, factor)
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width
