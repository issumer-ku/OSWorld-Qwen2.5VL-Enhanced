import ast
import json
import math
import re
from typing import Dict, List, Optional


def parse_xml_tool_call(xml_content: str) -> Optional[Dict]:
    params: Dict = {}
    func_match = re.search(r"<function=([^>]+)>", xml_content, re.IGNORECASE)
    if not func_match or func_match.group(1).strip().lower() != "computer_use":
        return None

    # Some local Qwen runtimes omit the final </parameter> while still closing
    # the function. Recover only values bounded by the next parameter,
    # </function>, or end-of-input so prose outside the function is not parsed.
    parameter_pattern = (
        r"<parameter(?:=([^>]+)>|>([a-zA-Z_][\w-]*)>)\s*(.*?)"
        r"(?:\s*</parameter>|"
        r"(?=\s*<parameter(?:=|>[a-zA-Z_][\w-]*>)|\s*</function>|\Z))"
    )
    for match in re.finditer(
        parameter_pattern,
        xml_content,
        re.DOTALL | re.IGNORECASE,
    ):
        name = (match.group(1) or match.group(2)).strip().lower()
        value = match.group(3).strip()
        if value.startswith("[") or value.startswith("{"):
            try:
                params[name] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        params[name] = value
    return params


def _iter_json_tool_calls(response: str):
    """Recover OpenAI-style JSON tool calls without interpreting plain prose."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response or ""):
        try:
            value, _ = decoder.raw_decode(response[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if value.get("name") != "computer_use":
            continue
        arguments = value.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(arguments, dict):
            yield arguments


def iter_tool_call_params(response: str):
    # OpenRouter/Qwen sometimes serializes the closing tool-call token as the
    # visible training artefact 📐 or ⚗, or omits only </tool_call> while still
    # closing </function>. Accept those variants without accepting plain prose.
    yielded = set()
    pattern = r"<tool_call>(.*?)(?:</tool_call>|[📐⚗]|(?=\n\s*(?:user|assistant)\s*\n)|$)"
    for tool_call_match in re.finditer(pattern, response, re.DOTALL | re.IGNORECASE):
        params = parse_xml_tool_call(tool_call_match.group(1))
        if params:
            fingerprint = json.dumps(params, sort_keys=True, ensure_ascii=False)
            yielded.add(fingerprint)
            yield params

    # Some Qwen2.5-VL endpoints omit the outer <tool_call> wrapper entirely,
    # while preserving a complete and unambiguous computer_use function block.
    # Parse that block directly, but de-duplicate calls already found above.
    function_pattern = r"(<function=computer_use>.*?</function>)"
    for function_match in re.finditer(function_pattern, response, re.DOTALL | re.IGNORECASE):
        params = parse_xml_tool_call(function_match.group(1))
        if params:
            fingerprint = json.dumps(params, sort_keys=True, ensure_ascii=False)
            if fingerprint not in yielded:
                yielded.add(fingerprint)
                yield params

    for params in _iter_json_tool_calls(response):
        fingerprint = json.dumps(params, sort_keys=True, ensure_ascii=False)
        if fingerprint not in yielded:
            yielded.add(fingerprint)
            yield params


def parse_keys(raw_keys, *, lowercase: bool = False) -> List[str]:
    if isinstance(raw_keys, str):
        try:
            raw_keys = json.loads(raw_keys)
        except Exception:
            try:
                raw_keys = ast.literal_eval(raw_keys)
            except Exception:
                pass

    def clean_key_token(key: object) -> str:
        token = str(key).strip()
        token = token.strip(" \t\r\n[](){}\"'")
        token = token.rstrip(" \t\r\n]")
        token = token.lstrip(" \t\r\n[")
        return token.strip()

    def flatten(keys_obj) -> List[str]:
        if keys_obj is None:
            return []
        if isinstance(keys_obj, list):
            values: List[str] = []
            for item in keys_obj:
                values.extend(flatten(item))
            return values
        values = []
        for part in re.split(r"\s*\+\s*", str(keys_obj).strip()):
            cleaned = clean_key_token(part)
            if cleaned:
                values.append(cleaned)
        return values

    keys = flatten(raw_keys)
    return [key.lower() for key in keys] if lowercase else keys


def parse_coordinate(raw_coord):
    if isinstance(raw_coord, str):
        try:
            raw_coord = json.loads(raw_coord)
        except Exception:
            # Recover common Qwen serialization glitches such as
            # "[742, 1009]]" without accepting non-numeric coordinates.
            numbers = re.findall(r"-?\d+(?:\.\d+)?", raw_coord)
            if len(numbers) >= 2:
                return float(numbers[0]), float(numbers[1])
            return None
    if isinstance(raw_coord, dict):
        if "x" in raw_coord and "y" in raw_coord:
            raw_coord = (raw_coord["x"], raw_coord["y"])
        else:
            return None
    if isinstance(raw_coord, (list, tuple)) and len(raw_coord) >= 2:
        try:
            x = float(raw_coord[0])
            y = float(raw_coord[1])
        except (TypeError, ValueError):
            return None
        if math.isfinite(x) and math.isfinite(y):
            return x, y
        return None
    return None


def parse_number(raw_value, default=0.0) -> float:
    try:
        return float(raw_value)
    except Exception:
        return float(default)


def extract_action_line(response: str, *, preserve_base_split_bug: bool = False) -> str:
    for line in response.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            if preserve_base_split_bug:
                return stripped.split("Action:", 1)[-1].strip()
            return stripped.split(":", 1)[-1].strip()
    return ""


def parse_terminal_status_alias(text: str) -> Optional[str]:
    """Parse only an explicit standalone terminate status shorthand.

    Some OpenAI-compatible local runtimes reproduce the human-readable
    ``terminate=success`` notation from a prompt instead of emitting a tool
    call.  Keep this fallback deliberately narrow: reasoning, evidence prose,
    and action descriptions that merely mention termination must not end a
    task.
    """
    visible = re.sub(
        r"<think>.*?</think>",
        "",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    visible_lines = [line.strip() for line in visible.splitlines() if line.strip()]
    action = extract_action_line(visible).strip()
    if action and len(visible_lines) != 1:
        return None
    candidate = action or visible.strip()
    match = re.fullmatch(
        r"(?:action\s*[:=]\s*)?terminate\s*"
        r"(?:"
        r"=\s*|"
        r"\(\s*(?:status\s*=\s*)?[\"']?|"
        r"\s+"
        r")"
        r"(success|succeeded|failure|fail|failed|error|infeasible)"
        r"[\"']?\s*\)?"
        r"(?:[.!]|\s+(?:the\s+)?task\b.*)?",
        candidate,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def looks_completed_response(
    text: str,
    *,
    allow_local_terminal_alias: bool = False,
) -> bool:
    """Recognize only explicit terminal statements, not incidental uses of 'done'."""
    if allow_local_terminal_alias:
        terminal_status = parse_terminal_status_alias(text)
        if terminal_status in {"success", "succeeded"}:
            return True
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    action = extract_action_line(cleaned).strip().lower()
    if action in {
        "done",
        "task complete",
        "task completed",
        "the task is complete",
        "the task is completed",
    }:
        return True
    return bool(re.fullmatch(r"\s*(?:the\s+)?task\s+(?:is\s+)?(?:complete|completed|done)[.!]?\s*", cleaned, re.IGNORECASE))


def looks_infeasible_response(text: str) -> bool:
    original = text or ""
    lowered = original.lower()
    if "[infeasible]" in lowered:
        return True

    # Reasoning often mentions a temporarily unavailable button or an
    # unsupported intermediate approach before selecting a valid action. Only
    # classify the response as infeasible when its visible terminal statement
    # explicitly says the task itself cannot be completed.
    visible = re.sub(
        r"<think>.*?</think>",
        "",
        original,
        flags=re.DOTALL | re.IGNORECASE,
    )
    visible = re.sub(
        r"<tool_call>.*?(?:</tool_call>|[📐⚗]|$)",
        "",
        visible,
        flags=re.DOTALL | re.IGNORECASE,
    )
    visible = " ".join(visible.lower().split()).strip(" .!,:;-")
    visible = re.sub(r"^action:\s*", "", visible)

    return bool(
        re.fullmatch(
            r"(?:the\s+)?task\s+(?:is\s+)?"
            r"(?:infeasible|impossible|not feasible|not possible|"
            r"cannot be completed|can't be completed|cannot be done)",
            visible,
        )
        or re.fullmatch(
            r"(?:i\s+)?(?:cannot|can't|am unable to)\s+complete"
            r"(?:\s+this|\s+the)?\s+task(?:\s+as\s+(?:requested|specified))?",
            visible,
        )
    )
