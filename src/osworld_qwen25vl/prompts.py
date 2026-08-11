import json
from datetime import datetime
from typing import Dict


BASE_ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type the exact raw string in `text`. Do not add surrounding quotation marks unless they must appear in the UI.
* `mouse_move`: Move the cursor to a specified (x, y) coordinate in the tool's advertised coordinate space.
* `left_click`: Click the left mouse button at a specified (x, y) coordinate in the tool's advertised coordinate space. Optional `text` parameter can specify modifier keys (e.g., "ctrl", "shift", "ctrl+shift") that will be held during the click.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) coordinate in the tool's advertised coordinate space. Optional `text` parameter can specify modifier keys that will be held during the click.
* `middle_click`: Click the middle mouse button at a specified (x, y) coordinate in the tool's advertised coordinate space. Optional `text` parameter can specify modifier keys that will be held during the click.
* `double_click`: Double-click the left mouse button at a specified (x, y) coordinate in the tool's advertised coordinate space. Optional `text` parameter can specify modifier keys that will be held during the click.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) coordinate in the tool's advertised coordinate space. Optional `text` parameter can specify modifier keys that will be held during the click.
* `scroll`: Scroll vertically. Use a small positive value (usually 3 to 7) to scroll up and a small negative value to scroll down.
* `hscroll`: Scroll horizontally. Use a small positive value to scroll right and a small negative value to scroll left.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""


INTERNAL_ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and hold a single key without releasing it.
* `key_up`: Release a previously held single key.
* `left_mouse_down`: Press and hold the left mouse button.
* `left_mouse_up`: Release the left mouse button.
* `type`: Type the exact raw string in `text`. Do not add surrounding quotation marks unless they must appear in the UI.
* `mouse_move`: Move the cursor to a specified (x, y) coordinate in the tool's advertised coordinate space.
* `left_click`: Click the left mouse button at the required coordinate.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate in the tool's advertised coordinate space.
* `right_click`: Click the right mouse button at the required coordinate.
* `middle_click`: Click the middle mouse button at the required coordinate.
* `double_click`: Double-click the left mouse button at the required coordinate.
* `triple_click`: Triple-click the left mouse button at the required coordinate.
* `scroll`: Scroll vertically. Use a small positive value (usually 3 to 7) to scroll up and a small negative value to scroll down.
* `hscroll`: Scroll horizontally. Use a small positive value to scroll right and a small negative value to scroll left.
* `screenshot`: Capture a new screenshot of the current screen.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `call_user`: Ask user for information or confirmation.
""".strip()


STATE_GROUNDING_PROMPT = """
# State grounding and completion
- Before every action, identify the active application from the visible title bar,
  window contents, and active dock/taskbar indicator. Do not infer the active
  application from the task instruction or from an earlier screenshot.
- Being in the wrong application is a navigation problem, not evidence that the
  task is infeasible. Switch to or launch the required application and continue.
- Treat every previous action as an attempt, not proof of success. Verify its
  visible result in the latest observation before planning the next action.
- Use a computer_use call with action=terminate and status=success only when
  every requested state or artifact is visibly present in the latest
  observation. If completion is not visually verifiable, inspect the relevant
  application or file before terminating.
- Do not repeatedly click the same location on an unchanged screen. Change the
  approach, use keyboard navigation, switch applications, or reopen the relevant
  UI surface.
""".strip()


def build_description_prompt(processed_width: int, processed_height: int, coordinate_type: str) -> str:
    resolution = (
        (
            f"* Use screenshot pixel coordinates: x ranges from 0 through "
            f"{max(0, processed_width - 1)} and y ranges from 0 through "
            f"{max(0, processed_height - 1)}. Do not normalize coordinates."
        )
        if coordinate_type == "absolute"
        else (
            "* Use normalized integer coordinates from 0 through 999 on both "
            "axes: (0, 0) is the top-left and (999, 999) is the bottom-right. "
            "Do not use screenshot pixel coordinates."
        )
    )
    return "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. Interact only through the visible UI using mouse and keyboard actions. You may open the application menu, use dock/taskbar icons, or launch a visible Terminal when the task permits; do not assume hidden host, filesystem, process, or shell access.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            resolution,
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
            "* Emit exactly one logical computer action per response. Observe the resulting screen before choosing the next action.",
            "* This benchmark is autonomous. Do not ask the user for help; use terminate=failure only when the task itself is explicitly impossible.",
        ]
    )


def build_coordinate_schema(
    processed_width: int,
    processed_height: int,
    coordinate_type: str,
) -> Dict:
    if coordinate_type == "absolute":
        description = (
            "(x, y) screenshot pixel coordinates. "
            f"x: 0..{max(0, processed_width - 1)}; "
            f"y: 0..{max(0, processed_height - 1)}."
        )
        maximum = max(0, processed_width - 1, processed_height - 1)
    else:
        description = (
            "(x, y) normalized coordinates. Both values must be between "
            "0 and 999; do not provide screenshot pixel coordinates."
        )
        maximum = 999
    return {
        "type": "array",
        "description": description,
        "minItems": 2,
        "maxItems": 2,
        "items": {
            "type": "number",
            "minimum": 0,
            "maximum": maximum,
        },
    }


def build_base_tools_def(processed_width: int, processed_height: int, coordinate_type: str) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": build_description_prompt(processed_width, processed_height, coordinate_type),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": BASE_ACTION_DESCRIPTION_PROMPT,
                        "enum": [
                            "key",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "triple_click",
                            "scroll",
                            "hscroll",
                            "wait",
                            "terminate",
                            "answer",
                        ],
                    },
                    "keys": {"type": "array", "description": "Required only by `action=key`."},
                    "text": {
                        "type": "string",
                        "description": "Required by `action=type` and `action=answer`. Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) to specify modifier keys (e.g., 'ctrl', 'shift', 'ctrl+shift'). Optional for scroll actions (scroll, hscroll) to specify a modifier key (e.g., 'shift', 'ctrl') to hold during scrolling.",
                    },
                    "coordinate": build_coordinate_schema(
                        processed_width,
                        processed_height,
                        coordinate_type,
                    ),
                    "pixels": {"type": "number", "description": "Scroll amount."},
                    "time": {"type": "number", "description": "Seconds to wait."},
                    "status": {
                        "type": "string",
                        "description": "Task status for terminate.",
                        "enum": ["success", "failure"],
                    },
                    "evidence": {
                        "type": "string",
                        "description": "For action=terminate with status=success, describe the concrete result visibly present in the latest observation.",
                    },
                },
            },
        },
    }


def build_internal_tools_def(processed_width: int, processed_height: int, coordinate_type: str) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": build_description_prompt(processed_width, processed_height, coordinate_type),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": INTERNAL_ACTION_DESCRIPTION_PROMPT,
                        "enum": [
                            "key",
                            "key_down",
                            "key_up",
                            "left_mouse_down",
                            "left_mouse_up",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "triple_click",
                            "scroll",
                            "hscroll",
                            "screenshot",
                            "wait",
                            "terminate",
                            "call_user",
                        ],
                    },
                    "keys": {
                        "type": "array",
                        "description": "Required only by `action=key`, `action=key_down`, or `action=key_up`.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Required only by `action=type` and `action=call_user`.",
                    },
                    "coordinate": build_coordinate_schema(
                        processed_width,
                        processed_height,
                        coordinate_type,
                    ),
                    "pixels": {
                        "type": "number",
                        "description": "Scroll amount. Required only by `action=scroll` or `action=hscroll`.",
                    },
                    "time": {
                        "type": "number",
                        "description": "Seconds to wait. Required only by `action=wait`.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Task status for terminate.",
                        "enum": ["success", "failure"],
                    },
                    "evidence": {
                        "type": "string",
                        "description": "For action=terminate with status=success, describe the concrete result visibly present in the latest observation.",
                    },
                },
            },
        },
    }


def build_base_system_prompt(tools_def: Dict, collapse_text: str) -> str:
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(tools_def)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "Response format for every step:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "Rules:\n"
        "- Output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything else outside those parts.\n"
        "- Emit exactly one tool call containing exactly one logical action per response.\n"
        "- If finishing, use action=terminate in the tool call."
    )


def build_internal_system_prompt(tools_def: Dict, collapse_text: str) -> str:
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(tools_def, ensure_ascii=False)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "For normal UI interaction steps:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "For terminal steps, you may either:\n"
        "- output a final natural-language response with no tool call, or\n"
        "- use a terminal tool call such as call_user or terminate.\n\n"
        "Rules:\n"
        "- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything after a tool call.\n"
        "- Emit exactly one tool call containing exactly one logical action per response.\n"
        "- A natural-language action without a <tool_call> is INVALID and will not be executed.\n"
        "- For every click, drag, or mouse move, inspect the screenshot and include a coordinate parameter.\n"
        "- Do not merely describe where to click. Emit the computer_use call that performs the click.\n"
        "- Do not use terminate until the requested result is visibly complete in the latest screenshot.\n"
        "- The benchmark is autonomous: do not use call_user. Reassess the UI and continue independently.\n"
        "- Use terminate when you want to explicitly end the task with a success or failure status.\n"
        "- If the task is infeasible, say so explicitly in the response.\n\n"
        "Valid click example:\n"
        "Action: Click the menu button.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\n"
        "left_click\n"
        "</parameter>\n"
        "<parameter=coordinate>\n"
        "[950, 120]\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "Invalid example: Action: Click the menu button. (No tool call, so nothing would happen.)"
    )


def build_instruction_prompt(instruction: str, previous_actions_str: str) -> str:
    return (
        "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n"
        f"{previous_actions_str}"
    )


def build_native_system_prompt(collapse_text: str) -> str:
    return (
        "You are an autonomous desktop agent. Use the computer_use function to "
        "complete the user's task through the visible GUI.\n\n"
        "Rules:\n"
        "- Call exactly one computer_use function with one logical action per response.\n"
        "- Observe the resulting screen before selecting the next action.\n"
        "- For clicks, drags, and mouse moves, include numeric coordinates.\n"
        "- Do not ask the user for help.\n"
        "- Use terminate=failure only when the task itself is intrinsically impossible.\n"
        "- Use action=terminate with status=success only when every requested result is visibly present "
        "in the latest observation, and include concrete visible evidence in the "
        "evidence argument.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}."
    )
