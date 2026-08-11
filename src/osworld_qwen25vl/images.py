import base64
from io import BytesIO
import math
import re
from typing import Tuple

from PIL import Image

from .resize import smart_resize


def process_image(image_bytes: bytes) -> str:
    """Resize + re-encode screenshot and return base64 PNG."""
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800,
    )

    image = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_size_from_bytes(image_bytes: bytes) -> Tuple[int, int]:
    image = Image.open(BytesIO(image_bytes))
    return image.size


def image_size_from_base64(image_b64: str) -> Tuple[int, int]:
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    return image.size


def scale_accessibility_tree_coordinates(
    table: str,
    *,
    coordinate_type: str,
    original_width: int,
    original_height: int,
    processed_width: int,
    processed_height: int,
) -> str:
    """Express accessibility coordinates in the tool's advertised space."""
    pair_pattern = re.compile(
        r"^\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)$"
    )

    if coordinate_type == "relative":
        x_scale = 999.0 / max(1, original_width - 1)
        y_scale = 999.0 / max(1, original_height - 1)
    else:
        x_scale = processed_width / original_width
        y_scale = processed_height / original_height

    def scale_pair(value: str) -> str:
        match = pair_pattern.match(value.strip())
        if not match:
            return value
        x = round(float(match.group(1)) * x_scale)
        y = round(float(match.group(2)) * y_scale)
        return f"({x}, {y})"

    scaled_lines = []
    for line in (table or "").splitlines():
        columns = line.split("\t")
        if len(columns) >= 7 and columns[0].strip().lower() != "tag":
            columns[-2] = scale_pair(columns[-2])
            columns[-1] = scale_pair(columns[-1])
        scaled_lines.append("\t".join(columns))
    return "\n".join(scaled_lines)


def image_perceptual_hash(image_bytes: bytes, hash_size: int = 16) -> int:
    """Return a small average hash suitable for detecting a genuinely static UI."""
    image = Image.open(BytesIO(image_bytes)).convert("L")
    image = image.resize((hash_size, hash_size))
    pixels = list(image.getdata())
    average = sum(pixels) / max(1, len(pixels))
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def perceptual_hash_distance(first: int, second: int) -> int:
    return int(first ^ second).bit_count()


def adjust_coordinates(
    x: float,
    y: float,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
) -> Tuple[int, int]:
    x = float(x)
    y = float(y)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("coordinates must be finite numbers")

    def clamp_to_screen(out_x: float, out_y: float) -> Tuple[int, int]:
        if not (original_width and original_height):
            return int(round(out_x)), int(round(out_y))
        return (
            min(max(int(round(out_x)), 0), int(original_width) - 1),
            min(max(int(round(out_y)), 0), int(original_height) - 1),
        )

    if not (original_width and original_height):
        return clamp_to_screen(x, y)
    if coordinate_type == "absolute":
        if processed_width and processed_height:
            x_scale = original_width / processed_width
            y_scale = original_height / processed_height
            return clamp_to_screen(x * x_scale, y * y_scale)
        return clamp_to_screen(x, y)

    # Qwen2.5-VL providers do not always obey the requested 0..999 relative
    # coordinate space. If either value is outside that space, interpret the
    # pair as coordinates in the processed screenshot instead.
    if (float(x) > 999 or float(y) > 999) and processed_width and processed_height:
        # Providers that ignore the requested relative space generally emit
        # coordinates in the processed image they received.
        x_scale = original_width / processed_width
        y_scale = original_height / processed_height
        return clamp_to_screen(x * x_scale, y * y_scale)

    x_scale = max(0, original_width - 1) / 999
    y_scale = max(0, original_height - 1) / 999
    return clamp_to_screen(x * x_scale, y * y_scale)
