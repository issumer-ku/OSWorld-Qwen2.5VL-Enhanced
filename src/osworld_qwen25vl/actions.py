import ast
import json
import re
from typing import Dict, List, Tuple

from .images import adjust_coordinates
from .parser import (
    extract_action_line,
    iter_tool_call_params,
    looks_completed_response,
    looks_infeasible_response,
    parse_coordinate,
    parse_keys,
    parse_number,
    parse_terminal_status_alias,
)


def py_string(text: str) -> str:
    return json.dumps("" if text is None else str(text), ensure_ascii=False)


def _termination_code(status: object) -> str:
    normalized = str(status or "success").strip().lower()
    return "FAIL" if normalized in {"fail", "failed", "failure", "error", "infeasible"} else "DONE"


def _normalize_action_name(action: object) -> str:
    normalized = str(action or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "click": "left_click",
        "move": "mouse_move",
        "move_mouse": "mouse_move",
        "drag": "left_click_drag",
        "hotkey": "key",
        "keypress": "key",
        "type_text": "type",
        "finish": "terminate",
        "done": "terminate",
    }
    return aliases.get(normalized, normalized)


def _normalize_type_text(text: object) -> str:
    """Decode a model-emitted quoted string literal without altering raw text."""
    value = "" if text is None else str(text)
    stripped = value.strip()
    if (
        len(stripped) >= 2
        and stripped[0] == stripped[-1]
        and stripped[0] in {"'", '"'}
    ):
        try:
            decoded = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            decoded = None
        if isinstance(decoded, str):
            return decoded
    return value


def action_description_matches_code(
    description: str,
    pyautogui_code: List[str],
) -> bool:
    """Reject explicit action claims that are absent from the executable call."""
    normalized = " ".join((description or "").lower().split())
    if not normalized or not pyautogui_code:
        return True

    rendered_code = " ".join(pyautogui_code).lower()
    if pyautogui_code in (["WAIT"], ["DONE"], ["FAIL"]):
        return True

    claims_enter = bool(
        re.search(r"\b(?:press|hit|send)\s+(?:the\s+)?(?:enter|return)\b", normalized)
    )
    executes_enter = bool(
        re.search(r"(?:press|keydown|keyup)\([\"'](?:enter|return)[\"']", rendered_code)
        or re.search(r"hotkey\([^)]*[\"'](?:enter|return)[\"']", rendered_code)
    )
    if claims_enter and not executes_enter:
        return False

    claims_click = bool(
        re.search(r"\b(?:click|double-click|right-click|middle-click)\b", normalized)
    )
    executes_click = any(
        marker in rendered_code
        for marker in (
            "pyautogui.click(",
            "pyautogui.rightclick(",
            "pyautogui.middleclick(",
            "pyautogui.doubleclick(",
            "pyautogui.tripleclick(",
        )
    )
    if claims_click and not executes_click:
        return False

    claims_scroll = bool(re.search(r"\bscroll\b", normalized))
    executes_scroll = "pyautogui.scroll(" in rendered_code or "pyautogui.hscroll(" in rendered_code
    if claims_scroll and not executes_scroll:
        return False

    return True


def _recover_unambiguous_natural_action(action_text: str) -> List[str]:
    """Recover only natural-language actions that need no coordinates or text."""
    normalized = re.sub(r"\s+", " ", action_text or "").strip().lower()
    if not normalized:
        return []

    if re.search(r"\bscroll\b.*\b(down|lower|below)\b", normalized):
        return ["pyautogui.scroll(-5)"]
    if re.search(r"\bscroll\b.*\b(up|upper|above)\b", normalized):
        return ["pyautogui.scroll(5)"]
    if re.search(r"\b(wait|pause)\b", normalized):
        return ["WAIT"]

    compact_action = re.sub(r"\s+", " ", action_text or "").strip()
    quoted_type = re.fullmatch(
        r"(?:type|input|write|enter)\s+(['\"])(.+)\1"
        r"(?:\s+(?:into|in)\s+.+)?[.!]?",
        compact_action,
        flags=re.IGNORECASE,
    )
    if quoted_type:
        text = quoted_type.group(2)
        if text:
            return [f"pyautogui.typewrite({py_string(text)})"]

    key_names = {
        "enter": "enter",
        "return": "enter",
        "escape": "esc",
        "esc": "esc",
        "tab": "tab",
        "backspace": "backspace",
        "delete": "delete",
        "space": "space",
        "home": "home",
        "end": "end",
        "page up": "pageup",
        "page down": "pagedown",
        "up arrow": "up",
        "down arrow": "down",
        "left arrow": "left",
        "right arrow": "right",
    }
    key_match = re.fullmatch(
        r"(?:press|hit|send)\s+(?:the\s+)?"
        r"(enter|return|escape|esc|tab|backspace|delete|space|home|end|"
        r"page up|page down|up arrow|down arrow|left arrow|right arrow)"
        r"(?:\s+key)?(?:\s+to\s+.+)?[.!]?",
        normalized,
    )
    if key_match:
        return [f"pyautogui.press({py_string(key_names[key_match.group(1)])})"]

    if re.match(
        r"^(?:close|exit|quit)\s+(?:the\s+)?"
        r"(?:active\s+|current\s+)?(?:application|app|window|program)\b",
        normalized,
    ) or re.match(
        r"^(?:close|exit|quit)\s+(?:the\s+)?[a-z0-9 ._+-]+\s+"
        r"(?:application|app|window|program)\b",
        normalized,
    ):
        return ['pyautogui.hotkey("alt", "f4")']
    return []


def _coord_adjuster(
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
):
    def adjust(x: float, y: float) -> Tuple[int, int]:
        return adjust_coordinates(
            x,
            y,
            coordinate_type=coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

    return adjust


def parse_base_response(
    response: str,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
    allow_local_runtime_compat: bool = False,
) -> Tuple[str, List[str]]:
    low_level_instruction = ""
    pyautogui_code: List[str] = []

    if not response or not response.strip():
        return low_level_instruction, pyautogui_code

    adjust = _coord_adjuster(
        coordinate_type=coordinate_type,
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
    )

    def process_tool_call_params(params: Dict) -> None:
        action = _normalize_action_name(params.get("action"))
        if not action:
            return

        coordinate = parse_coordinate(params.get("coordinate"))
        text = params.get("text")

        def press_modifier_keys() -> None:
            if text:
                for key in str(text).split("+"):
                    key = key.strip().lower()
                    if key:
                        pyautogui_code.append(f"pyautogui.keyDown({py_string(key)})")

        def release_modifier_keys() -> None:
            if text:
                keys = [key.strip().lower() for key in str(text).split("+") if key.strip()]
                for key in reversed(keys):
                    pyautogui_code.append(f"pyautogui.keyUp({py_string(key)})")

        if action == "left_click":
            if not coordinate:
                return
            press_modifier_keys()
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.click({x}, {y})")
            release_modifier_keys()
        elif action == "right_click":
            if not coordinate:
                return
            press_modifier_keys()
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.rightClick({x}, {y})")
            release_modifier_keys()
        elif action == "middle_click":
            if not coordinate:
                return
            press_modifier_keys()
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.middleClick({x}, {y})")
            release_modifier_keys()
        elif action == "double_click":
            if not coordinate:
                return
            press_modifier_keys()
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
            release_modifier_keys()
        elif action == "triple_click":
            if not coordinate:
                return
            press_modifier_keys()
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.tripleClick({x}, {y})")
            release_modifier_keys()
        elif action == "type":
            text = _normalize_type_text(params.get("text", ""))
            if text == "":
                return
            pyautogui_code.append(f"pyautogui.typewrite({py_string(text)})")
        elif action == "key":
            keys = parse_keys(params.get("keys", []))
            if not keys:
                return
            keys_str = ", ".join(py_string(key) for key in keys)
            if len(keys) > 1:
                pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
            else:
                pyautogui_code.append(f"pyautogui.press({keys_str})")
        elif action in {"scroll", "hscroll"}:
            pixels = params.get("pixels", 0)
            try:
                pixels = int(float(pixels))
            except Exception:
                pixels = 0
            if pixels == 0:
                return
            press_modifier_keys()
            pyautogui_code.append(f"pyautogui.scroll({pixels})")
            release_modifier_keys()
        elif action == "wait":
            pyautogui_code.append("WAIT")
        elif action == "terminate":
            pyautogui_code.append(_termination_code(params.get("status", "success")))
        elif action == "answer":
            pyautogui_code.append("DONE")
        elif action == "mouse_move":
            if not coordinate:
                return
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
        elif action == "left_click_drag":
            if not coordinate:
                return
            x, y = adjust(*coordinate)
            duration = 0.5
            if "duration" in params:
                try:
                    duration = float(params["duration"])
                except Exception:
                    duration = 0.5
            pyautogui_code.append(f"pyautogui.dragTo({x}, {y}, duration={duration})")

    low_level_instruction = extract_action_line(response, preserve_base_split_bug=True)

    for params in iter_tool_call_params(response):
        before_count = len(pyautogui_code)
        process_tool_call_params(params)
        if len(pyautogui_code) > before_count:
            break

    if not pyautogui_code and allow_local_runtime_compat:
        terminal_status = parse_terminal_status_alias(response)
        if terminal_status:
            pyautogui_code.append(_termination_code(terminal_status))

    if not low_level_instruction and pyautogui_code:
        low_level_instruction = _instruction_from_first_code(pyautogui_code[0])

    return low_level_instruction, pyautogui_code


def parse_internal_response(
    response: str,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
    allow_local_runtime_compat: bool = False,
) -> Tuple[str, List[str]]:
    low_level_instruction = ""
    pyautogui_code: List[str] = []

    if not response or not response.strip():
        return low_level_instruction, pyautogui_code

    adjust = _coord_adjuster(
        coordinate_type=coordinate_type,
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
    )
    infeasible_response = looks_infeasible_response(response)

    def process_tool_call_params(params: Dict) -> None:
        action = _normalize_action_name(params.get("action"))
        if not action:
            return

        coordinate = parse_coordinate(params.get("coordinate"))

        if action == "left_click":
            if not coordinate:
                return
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.click({x}, {y})")
        elif action == "right_click":
            if not coordinate:
                return
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.rightClick({x}, {y})")
        elif action == "middle_click":
            if not coordinate:
                return
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.middleClick({x}, {y})")
        elif action == "double_click":
            if not coordinate:
                return
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
        elif action == "triple_click":
            if not coordinate:
                return
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.tripleClick({x}, {y})")
        elif action == "type":
            text = _normalize_type_text(params.get("text", ""))
            if text == "":
                return
            normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
            if "\n" not in normalized_text:
                pyautogui_code.append(f"pyautogui.typewrite({py_string(normalized_text)})")
            else:
                chunks = normalized_text.split("\n")
                for idx, chunk in enumerate(chunks):
                    if chunk:
                        pyautogui_code.append(f"pyautogui.typewrite({py_string(chunk)})")
                    if idx < len(chunks) - 1:
                        pyautogui_code.append(f"pyautogui.press({py_string('enter')})")
        elif action == "key":
            keys = parse_keys(params.get("keys", []), lowercase=True)
            if not keys:
                return
            keys_str = ", ".join(py_string(key) for key in keys)
            if len(keys) > 1:
                pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
            else:
                pyautogui_code.append(f"pyautogui.press({keys_str})")
        elif action == "key_down":
            for key in parse_keys(params.get("keys", []), lowercase=True):
                pyautogui_code.append(f"pyautogui.keyDown({py_string(key)})")
        elif action == "key_up":
            for key in parse_keys(params.get("keys", []), lowercase=True):
                pyautogui_code.append(f"pyautogui.keyUp({py_string(key)})")
        elif action == "scroll":
            pixels = int(parse_number(params.get("pixels", 0), default=0))
            if pixels == 0:
                return
            pyautogui_code.append(f"pyautogui.scroll({pixels})")
        elif action == "hscroll":
            pixels = int(parse_number(params.get("pixels", 0), default=0))
            if pixels == 0:
                return
            pyautogui_code.append(f"pyautogui.hscroll({pixels})")
        elif action == "wait":
            pyautogui_code.append("WAIT")
        elif action == "terminate":
            pyautogui_code.append(_termination_code(params.get("status", "success")))
        elif action == "call_user":
            # OSWorld is autonomous. Give the format-recovery loop a chance to
            # choose an executable action instead of falsely reporting success.
            return
        elif action == "screenshot":
            pyautogui_code.append("WAIT")
        elif action == "mouse_move":
            if not coordinate:
                return
            x, y = adjust(*coordinate)
            pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
        elif action == "left_click_drag":
            if not coordinate:
                return
            x, y = adjust(*coordinate)
            duration = parse_number(params.get("duration", 0.5), default=0.5)
            pyautogui_code.append(f"pyautogui.dragTo({x}, {y}, duration={duration})")
        elif action == "left_mouse_down":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            pyautogui_code.append("pyautogui.mouseDown(button='left')")
        elif action == "left_mouse_up":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            pyautogui_code.append("pyautogui.mouseUp(button='left')")

    low_level_instruction = extract_action_line(response)

    for params in iter_tool_call_params(response):
        before_count = len(pyautogui_code)
        process_tool_call_params(params)
        if len(pyautogui_code) > before_count:
            break

    if not pyautogui_code and allow_local_runtime_compat:
        terminal_status = parse_terminal_status_alias(response)
        if terminal_status:
            pyautogui_code.append(_termination_code(terminal_status))

    if not pyautogui_code and not infeasible_response:
        pyautogui_code.extend(_recover_unambiguous_natural_action(low_level_instruction))

    if not pyautogui_code and looks_completed_response(
        response,
        allow_local_terminal_alias=allow_local_runtime_compat,
    ):
        pyautogui_code.append("DONE")

    # A response without a parseable tool call is not task completion.  Returning
    # an empty list lets the agent request a corrected response instead of
    # silently converting natural-language advice into DONE.
    if not pyautogui_code and infeasible_response:
        pyautogui_code.append("FAIL")

    if not low_level_instruction and pyautogui_code:
        first_code = pyautogui_code[0]
        if first_code == "FAIL":
            low_level_instruction = "Need user input"
        else:
            low_level_instruction = _instruction_from_first_code(first_code)

    return low_level_instruction, pyautogui_code


def _instruction_from_first_code(first_code: str) -> str:
    if first_code == "DONE":
        return "Task completed"
    if first_code == "WAIT":
        return "Waiting"
    if "." in first_code:
        return f"Performing {first_code.split('.', 1)[1].split('(', 1)[0]} action"
    return "Performing action"
