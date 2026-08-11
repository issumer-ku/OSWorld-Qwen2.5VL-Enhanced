import json
import logging
import os
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from .accessibility import (
    linearize_accessibility_tree,
    tag_screenshot,
    trim_accessibility_tree,
)

from .protocol import (
    adapter_error_action,
    decode_adapter_error_action,
    decode_model_error_action,
    model_error_action,
)

from .actions import (
    action_description_matches_code,
    parse_base_response,
    parse_internal_response,
    py_string,
)
from .client import call_openai_compatible, is_local_openai_endpoint
from .history import (
    build_messages,
    dump_debug_messages,
    ensure_empty_think_prefix,
    previous_actions_text,
    update_folding_state,
)
from .images import (
    image_perceptual_hash,
    image_size_from_base64,
    image_size_from_bytes,
    perceptual_hash_distance,
    process_image,
    scale_accessibility_tree_coordinates,
)
from .prompts import (
    STATE_GROUNDING_PROMPT,
    build_base_system_prompt,
    build_base_tools_def,
    build_instruction_prompt,
    build_internal_system_prompt,
    build_internal_tools_def,
    build_native_system_prompt,
)
from .parser import iter_tool_call_params, parse_coordinate


logger = None


class _QwenBaseAgent:
    """
    Shared implementation for Qwen computer-use agents.

    Characteristics:
    - OpenAI-compatible API only.
    - XML tool-call output format.
    - History truncation by `history_n`.
    - Old screenshot folding by `image_max` / `fold_size`.
    """

    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "qwen-vl",
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        history_n: int = 100,
        add_thought_prefix: bool = False,
        coordinate_type: str = "relative",
        api_backend: str = "openai",
        image_max: int = 20,
        fold_size: int = 10,
        collapse_text: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        a11y_tree_max_tokens: int = 3500,
        multimodal_history_n: int = 3,
        a11y_history_n: int = 1,
        loop_action_limit: int = 3,
        format_failure_limit: int = 3,
        completion_verification: bool = True,
        native_tools: bool = False,
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.history_n = history_n
        self.add_thought_prefix = add_thought_prefix
        self.coordinate_type = coordinate_type
        self.effective_coordinate_type = coordinate_type
        self.api_backend = api_backend
        self.image_max = int(image_max)
        self.fold_size = int(fold_size)
        self.collapse_text = collapse_text or self.COLLAPSED_SCREENSHOT_TEXT
        self.base_url = base_url
        self.api_key = api_key
        resolved_base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        )
        self.local_runtime_compat = is_local_openai_endpoint(resolved_base_url)
        self.a11y_tree_max_tokens = int(a11y_tree_max_tokens)
        self.multimodal_history_n = int(multimodal_history_n)
        self.a11y_history_n = int(a11y_history_n)
        self.loop_action_limit = int(loop_action_limit)
        self.format_failure_limit = int(format_failure_limit)
        self.completion_verification = bool(completion_verification)
        self.native_tools = bool(native_tools)

        if action_space != "pyautogui":
            raise ValueError("QwenAgent supports only pyautogui action space")
        if coordinate_type not in {"absolute", "relative"}:
            raise ValueError("coordinate_type must be 'absolute' or 'relative'")
        supported_observations = {
            "screenshot",
            "a11y_tree",
            "screenshot_a11y_tree",
            "som",
        }
        if observation_type not in supported_observations:
            raise ValueError(
                f"QwenAgent observation_type must be one of {sorted(supported_observations)}"
            )
        if api_backend != "openai":
            raise ValueError("QwenAgent supports only OpenAI-compatible APIs")
        if self.image_max < 1:
            raise ValueError("image_max must be >= 1")
        if self.fold_size < 1:
            raise ValueError("fold_size must be >= 1")
        if self.multimodal_history_n < 1:
            raise ValueError("multimodal_history_n must be >= 1")
        if self.a11y_history_n < 1:
            raise ValueError("a11y_history_n must be >= 1")
        if self.loop_action_limit < 2:
            raise ValueError("loop_action_limit must be >= 2")
        if self.format_failure_limit < 1:
            raise ValueError("format_failure_limit must be >= 1")

        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []
        self.responses: List[str] = []
        self.screenshots: List[Optional[str]] = []
        self.observation_texts: List[str] = []
        self.action_records: List[Tuple[Tuple[str, ...], str, int]] = []
        self.consecutive_format_failures = 0
        self.consecutive_stagnation_failures = 0
        self.consecutive_completion_failures = 0
        self.consecutive_completion_audit_failures = 0
        self.folded_prefix_k = 0
        self.pending_recovery_feedback = ""

    @staticmethod
    def _py_string(text: str) -> str:
        return py_string(text)

    def _build_tools_def(self, processed_width: int, processed_height: int) -> Dict:
        return build_base_tools_def(
            processed_width,
            processed_height,
            self.effective_coordinate_type,
        )

    def _build_system_prompt(self, tools_def: Dict) -> str:
        prompt = (
            build_native_system_prompt(self.collapse_text)
            if self.native_tools
            else build_base_system_prompt(tools_def, self.collapse_text)
        )
        prompt += "\n\n" + STATE_GROUNDING_PROMPT
        prompt += self._local_runtime_prompt_suffix()
        if self.add_thought_prefix:
            prompt += (
                "\n\n# Deliberation\n"
                "Before the Action line, briefly analyze the latest UI state in "
                "a <think>...</think> block. Verify what changed after the prior "
                "action and identify the single safest next action."
            )
        if self.observation_type in {"a11y_tree", "screenshot_a11y_tree", "som"}:
            prompt += (
                "\n\n# Accessibility observations\n"
                "The accessibility tree is a tab-separated table. Position is the "
                "element's top-left (x, y), and size is (width, height). Click the "
                "center: (x + width/2, y + height/2). Prefer named actionable "
                "elements from the tree over guessing coordinates."
            )
        if self.observation_type == "som":
            prompt += (
                "\nThe screenshot is Set-of-Marks tagged. Use the tag labels and "
                "the accompanying accessibility table to identify the target."
            )
        return prompt

    def _local_runtime_prompt_suffix(self) -> str:
        if not self.local_runtime_compat:
            return ""
        return (
            "\n\n# Local-runtime output contract\n"
            "- Past action records are read-only context. Never copy their "
            "format, commands, or an `| executed:` suffix into your response.\n"
            "- Describe and emit the single next action, not an action that has "
            "already been attempted.\n"
            "- Every non-terminal action requires a computer_use tool call. Do "
            "not output PyAutoGUI code or prose in place of that call.\n"
            "- Match the action to the operation: use `type` after focusing a "
            "text field, `key` for Enter or shortcuts, and mouse actions only "
            "for visible pointer targets."
        )

    def _response_transform(self, response: str) -> str:
        return response

    def _coordinate_type_for_response(
        self,
        response: str,
        *,
        processed_width: int,
        processed_height: int,
    ) -> str:
        if (
            self.coordinate_type != "relative"
            or self.effective_coordinate_type == "absolute"
            or not processed_width
            or not processed_height
        ):
            return self.effective_coordinate_type

        for params in iter_tool_call_params(response):
            coordinate = parse_coordinate(params.get("coordinate"))
            if not coordinate:
                continue
            x, y = coordinate
            is_inside_processed_image = (
                0 <= x < processed_width
                and 0 <= y < processed_height
            )
            if is_inside_processed_image and (x > 999 or y > 999):
                self.effective_coordinate_type = "absolute"
                active_logger = logger or logging.getLogger(
                    "desktopenv.qwen_agent"
                )
                active_logger.warning(
                    "%s emitted screenshot-pixel coordinates while relative "
                    "coordinates were requested; locking this task to absolute "
                    "coordinates after observing (%s, %s) in a %sx%s image.",
                    self._log_prefix(),
                    int(round(x)),
                    int(round(y)),
                    processed_width,
                    processed_height,
                )
                break
        return self.effective_coordinate_type

    def _debug_message_filename(self, step_idx: int) -> str:
        return f"qwen_messages_{os.getpid()}_step_{step_idx}.json"

    def _build_payload(
        self,
        messages: List[Dict],
        tools_def: Optional[Dict] = None,
    ) -> Dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }
        if self.native_tools and tools_def:
            payload["tools"] = [tools_def]
            payload["tool_choice"] = "auto"
        return payload

    def _log_prefix(self) -> str:
        return "Qwen"

    @staticmethod
    def _normalize_action_description(description: str) -> str:
        return " ".join((description or "").lower().split())

    def _static_action_window(
        self,
        observation_signature: int,
    ) -> List[Tuple[str, ...]]:
        max_hash_distance = int(
            os.environ.get("OSWORLD_QWEN_STATIC_HASH_DISTANCE", "8")
        )
        max_records = max(self.loop_action_limit * 2, self.loop_action_limit + 1)
        matching_suffix = []
        for code, _, signature in reversed(self.action_records[-max_records:]):
            if (
                perceptual_hash_distance(observation_signature, signature)
                > max_hash_distance
            ):
                break
            matching_suffix.append(code)
        matching_suffix.reverse()
        if len(matching_suffix) < self.loop_action_limit:
            return []
        return matching_suffix

    @staticmethod
    def _mouse_action_parts(code: str) -> Optional[Tuple[str, int, int, str]]:
        match = re.fullmatch(
            r"(pyautogui\.(?:click|rightClick|middleClick|doubleClick|"
            r"tripleClick|moveTo|dragTo))\(\s*(-?\d+)\s*,\s*(-?\d+)\s*"
            r"(,.*)?\)",
            code.strip(),
        )
        if not match:
            return None
        return (
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
            match.group(4) or "",
        )

    def _actions_equivalent(
        self,
        left: Tuple[str, ...],
        right: Tuple[str, ...],
    ) -> bool:
        """Compare actions while tolerating harmless local-model cursor jitter."""
        if left == right:
            return True
        if len(left) != len(right):
            return False
        tolerance = max(
            0,
            int(os.environ.get("OSWORLD_QWEN_LOOP_COORD_TOLERANCE", "12")),
        )
        if tolerance == 0:
            return False
        for left_code, right_code in zip(left, right):
            if left_code == right_code:
                continue
            left_mouse = self._mouse_action_parts(left_code)
            right_mouse = self._mouse_action_parts(right_code)
            if left_mouse is None or right_mouse is None:
                return False
            left_name, left_x, left_y, left_suffix = left_mouse
            right_name, right_x, right_y, right_suffix = right_mouse
            if left_name != right_name or left_suffix != right_suffix:
                return False
            if abs(left_x - right_x) > tolerance or abs(left_y - right_y) > tolerance:
                return False
        return True

    def _tail_cycle(self, actions: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
        """Return a repeated multi-action tail cycle, if one is complete."""
        for period in range(2, len(actions) // 2 + 1):
            previous = actions[-2 * period : -period]
            current = actions[-period:]
            if all(
                self._actions_equivalent(left, right)
                for left, right in zip(previous, current)
            ):
                return actions[-period:]
        return []

    def _tail_identical_count(self, actions: List[Tuple[str, ...]]) -> int:
        if not actions:
            return 0
        candidate = actions[-1]
        count = 0
        for action in reversed(actions):
            if not self._actions_equivalent(action, candidate):
                break
            count += 1
        return count

    def _is_repeated_action(
        self,
        pyautogui_code: List[str],
        low_level_instruction: str,
        observation_signature: int,
    ) -> bool:
        """Reject an action that cycles on an unchanged visible UI."""
        candidate = tuple(code.strip() for code in pyautogui_code if code.strip())
        if not candidate or candidate in {("WAIT",), ("DONE",), ("FAIL",)}:
            return False
        static_actions = self._static_action_window(observation_signature)
        if not static_actions:
            return False
        proposed_sequence = static_actions + [candidate]
        if (
            self._tail_identical_count(proposed_sequence)
            > self.loop_action_limit
        ):
            return True
        return bool(self._tail_cycle(proposed_sequence))

    def _stagnation_context(self, observation_signature: int) -> str:
        """Describe an ineffective action before the model repeats it again."""
        static_actions = self._static_action_window(observation_signature)
        if not static_actions:
            return ""
        identical_count = self._tail_identical_count(static_actions)
        cycle = self._tail_cycle(static_actions)
        if identical_count >= self.loop_action_limit:
            rendered_actions = "; ".join(static_actions[-1])
            action_summary = (
                f"repeatedly executing `{rendered_actions}`"
            )
        elif cycle:
            rendered_actions = " | ".join(
                "; ".join(action) for action in cycle
            )
            action_summary = (
                "executing this action cycle: "
                f"`{rendered_actions}`"
            )
        else:
            return ""
        return (
            "STAGNATION WARNING: The visible screen has remained unchanged after "
            f"{action_summary}. Do not repeat any action from that cycle on this "
            "screen. Re-check the active application and use a "
            "materially different route, such as keyboard navigation, switching "
            "applications, reopening the relevant UI, or targeting a different "
            "control."
        )

    def _last_action_left_screen_unchanged(
        self,
        observation_signature: int,
    ) -> bool:
        if not self.action_records:
            return False
        _, _, before_action_signature = self.action_records[-1]
        max_hash_distance = int(
            os.environ.get("OSWORLD_QWEN_STATIC_HASH_DISTANCE", "8")
        )
        return (
            perceptual_hash_distance(
                observation_signature,
                before_action_signature,
            )
            <= max_hash_distance
        )

    @staticmethod
    def _completion_audit_prompt(
        instruction: str,
        *,
        last_action_left_screen_unchanged: bool = False,
    ) -> str:
        prompt = (
            "COMPLETION AUDIT: The preceding action=terminate, status=success request is "
            "provisional and must be checked against the latest observation.\n\n"
            f"Exact task: {instruction}\n\n"
            "Re-read every requested condition and inspect only what is visibly "
            "present now. Attempts, intentions, or an earlier statement of success "
            "are not evidence. Being in the wrong application is not infeasibility; "
            "navigate to the required application and continue.\n\n"
            "Evidence must identify the visible UI surface and concrete value or "
            "artifact that proves the result, for example a named settings field, "
            "dialog, terminal output, selected file, title bar, or status message. "
            "Merely restating the requested goal as if it were complete is not "
            "evidence. For multi-item tasks, verify every requested item.\n\n"
            "For outcomes that are not directly visible after execution, such as "
            "clipboard copies, closing a process, restoring or moving a file, use "
            "the matching executed action together with the latest UI state as "
            "evidence. Do not request an already executed operation again merely "
            "because its hidden state cannot be read from the screenshot.\n\n"
            "If every requested result is visibly confirmed, emit a new "
            "computer_use tool call with action=terminate and status=success, "
            "name the concrete visible evidence "
            "in the Action line, and include the same evidence in the tool call's "
            "`evidence` parameter. Otherwise emit the single next executable "
            "computer_use action that advances or verifies the task. If and only "
            "if the task itself is intrinsically impossible, use terminate(failure)."
        )
        if last_action_left_screen_unchanged:
            prompt += (
                "\n\nThe latest executable action produced no detectable visible "
                "screen change. For an action whose effect should be visible, treat "
                "that as evidence that it did not take effect and verify it through "
                "another UI surface. Clipboard copies and command-line process "
                "changes may legitimately have no visual delta, but they require "
                "matching executed actions and concrete evidence rather than a "
                "generic success claim."
            )
        return prompt

    def _build_completion_audit_messages(
        self,
        *,
        system_prompt: str,
        audit_prompt: str,
        processed_b64: Optional[str],
        observation_text: str,
    ) -> List[Dict]:
        """Build a fresh verifier context without the agent's completion claim."""
        verifier_system = (
            system_prompt
            + "\n\n# Independent completion verification\n"
            "Act as a skeptical verifier. Judge the task from the latest UI "
            "observation and the record of actions that actually executed. Do not "
            "trust or repeat an earlier success claim. A completed verdict requires "
            "specific UI evidence; otherwise return one executable action that "
            "checks or advances the task."
        )
        history_label, action_history = self._render_executed_action_history()
        verifier_text = audit_prompt + f"\n\n{history_label}:\n" + action_history
        user_content: List[Dict] = []
        if observation_text:
            user_content.append({"type": "text", "text": observation_text})
        if processed_b64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{processed_b64}"
                    },
                }
            )
        user_content.append({"type": "text", "text": verifier_text})
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": verifier_system}],
            },
            {"role": "user", "content": user_content},
        ]

    def _build_recovery_messages(
        self,
        *,
        system_prompt: str,
        instruction: str,
        processed_b64: Optional[str],
        observation_text: str,
        reason: str,
        rejected_action: Tuple[str, ...] = (),
        rejected_response: str = "",
    ) -> List[Dict]:
        """Build a fresh context that is not anchored to a failed plan."""
        recovery_system = (
            system_prompt
            + "\n\n# Independent recovery\n"
            "Discard the failed plan and independently re-ground on the latest "
            "observation. Do not imitate a rejected response. First decide whether "
            "the exact task is already complete. If it is complete, use "
            "action=terminate with status=success and concrete evidence. Otherwise "
            "return exactly "
            "one executable computer_use action that makes progress."
        )
        history_label, action_history = self._render_executed_action_history()

        if reason == "stagnation":
            rejected = "; ".join(rejected_action) or "unknown"
            recovery_instruction = (
                "STAGNATION RECOVERY: The following action was rejected because it "
                "had already been executed while the visible UI remained unchanged: "
                f"`{rejected}`. Do not repeat that action, its coordinates, or an "
                "equivalent click on the same control. Reassess the current state. "
                "If the task is not already complete, choose a different control, "
                "keyboard route, verification surface, or application-navigation path."
            )
        else:
            rejected_detail = rejected_response
            if self.local_runtime_compat and "| executed:" in rejected_detail:
                rejected_detail = (
                    "The rejected response copied a past-action record instead of "
                    "emitting a computer_use tool call; its text is omitted to "
                    "avoid reinforcing that invalid format."
                )
            recovery_instruction = (
                "FORMAT RECOVERY: The preceding response contained no executable "
                "computer_use call. Convert the intended operation into exactly one "
                "valid tool call. Mouse actions require numeric coordinates obtained "
                "from the latest observation. Do not respond with prose alone.\n\n"
                f"Rejected response:\n{rejected_detail}"
            )
            if self.local_runtime_compat:
                recovery_instruction += (
                    "\n\nDo not output `| executed:`, PyAutoGUI history notation, or "
                    "a description of a past action. Return the requested XML "
                    "computer_use tool call."
                )

        recovery_text = (
            f"Exact task: {instruction}\n\n"
            f"{history_label}:\n{action_history}\n\n"
            f"{recovery_instruction}"
        )
        user_content: List[Dict] = []
        if observation_text:
            user_content.append({"type": "text", "text": observation_text})
        if processed_b64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{processed_b64}"
                    },
                }
            )
        user_content.append({"type": "text", "text": recovery_text})
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": recovery_system}],
            },
            {"role": "user", "content": user_content},
        ]

    def _render_executed_action_history(self) -> Tuple[str, str]:
        records = self.action_records[-8:]
        if not self.local_runtime_compat:
            rendered = [
                f"- {description or 'action'} | executed: {'; '.join(code)}"
                for code, description, _ in records
            ]
            return (
                "Recent actions confirmed as executed",
                "\n".join(rendered) if rendered else "- None",
            )

        rendered = [
            json.dumps(
                {
                    "description": description or "action",
                    "commands": list(code),
                },
                ensure_ascii=False,
            )
            for code, description, _ in records
        ]
        body = "\n".join(rendered) if rendered else "[]"
        return (
            "Past action records (read-only; do not copy into the response)",
            f"<executed_action_history>\n{body}\n</executed_action_history>",
        )

    @staticmethod
    def _completion_tool_evidence(response: str) -> str:
        for params in iter_tool_call_params(response or ""):
            action = str(params.get("action", "")).strip().lower()
            status = str(params.get("status", "success")).strip().lower()
            if action == "terminate" and status == "success":
                return " ".join(str(params.get("evidence", "")).split()).strip(" .!")
        return ""

    @staticmethod
    def _completion_response_is_unverified(response: str) -> bool:
        normalized = " ".join((response or "").lower().split())
        contradictory_patterns = (
            "task is impossible",
            "task appears to be impossible",
            "task is infeasible",
            "task cannot be completed",
            "cannot complete the task",
            "wrong application",
            "not suitable for this task",
        )
        if any(pattern in normalized for pattern in contradictory_patterns):
            return True

        tool_evidence = _QwenBaseAgent._completion_tool_evidence(response)

        action_line = ""
        for line in (response or "").splitlines():
            if line.strip().lower().startswith("action:"):
                action_line = " ".join(
                    line.split(":", 1)[-1].strip().lower().split()
                ).strip(" .!")
                break
        generic_completion_actions = {
            "done",
            "task complete",
            "task completed",
            "the task is complete",
            "the task is completed",
            "the task is completed successfully",
            "the task has been completed successfully",
        }
        tool_has_evidence = len(tool_evidence) >= 12
        pending_control_pattern = re.compile(
            r"\b(?:done|save|submit|apply|confirm|install|create|delete|add|ok)"
            r"\s+button\b"
        )
        if pending_control_pattern.search(
            f"{action_line} {tool_evidence}".lower()
        ):
            return True
        if not action_line:
            return not tool_has_evidence
        if action_line in generic_completion_actions and not tool_has_evidence:
            return True

        grounded_markers = (
            "latest screen",
            "screen shows",
            "window shows",
            "dialog shows",
            "panel shows",
            "field shows",
            "terminal shows",
            "document shows",
            "editor shows",
            "desktop shows",
            "canvas shows",
            "file list shows",
            "status bar",
            "title bar",
            "visible",
            "visibly",
            "displayed",
            "listed",
            "selected",
        )
        combined_evidence = f"{action_line} {tool_evidence}".lower()
        grounded = any(
            marker in combined_evidence for marker in grounded_markers
        )
        if not grounded:
            grounded = bool(
                re.search(
                    r"\b(?:screen|window|dialog|panel|field|terminal|document|"
                    r"editor|desktop|canvas|file list|status bar|title bar)\b"
                    r".{0,100}\b(?:show|shows|showing|display|displays|displayed|"
                    r"list|lists|listed|contain|contains|visible)\b",
                    combined_evidence,
                )
            )
        return not (tool_has_evidence and grounded)

    def _completion_has_trajectory_support(
        self,
        instruction: str,
        response: str,
    ) -> bool:
        """Recognize hidden outcomes only when the executed trajectory supports them."""
        normalized_instruction = " ".join((instruction or "").lower().split())
        evidence = self._completion_tool_evidence(response).lower()
        recent_code = "\n".join(
            code
            for action, _, _ in self.action_records[-8:]
            for code in action
        ).lower()

        clipboard_task = "clipboard" in normalized_instruction
        copied_to_clipboard = (
            "clipboard" in evidence or "copied" in evidence
        )
        copy_action = bool(
            re.search(
                r"pyautogui\.hotkey\([^)]*[\"']ctrl[\"'][^)]*[\"']c[\"']",
                recent_code,
            )
        )
        recent_descriptions = "\n".join(
            description for _, description, _ in self.action_records[-8:]
        ).lower()
        copy_action = copy_action or bool(
            re.search(
                r"\b(?:copy location|copy file location|copy path|copy file path)\b",
                recent_descriptions,
            )
            and "pyautogui.click" in recent_code
        )
        if clipboard_task and copied_to_clipboard and copy_action:
            return True

        process_task = (
            "force quit" in normalized_instruction
            or bool(
                re.search(
                    r"\b(?:kill|terminate)\b.*"
                    r"\b(?:process|application|app)\b",
                    normalized_instruction,
                )
            )
        )
        kill_action = (
            'typewrite("kill' in recent_code
            or "typewrite('kill" in recent_code
        )
        submitted = bool(
            re.search(r"pyautogui\.press\([\"']enter[\"']\)", recent_code)
        )
        terminal_evidence = "terminal" in evidence and (
            "not running" in evidence
            or "no longer" in evidence
            or "terminated" in evidence
            or "executed" in evidence
        )
        return process_task and kill_action and submitted and terminal_evidence

    def _parse_response(
        self,
        response: str,
        *,
        original_width: int,
        original_height: int,
        processed_width: int,
        processed_height: int,
    ) -> Tuple[str, List[str]]:
        coordinate_type = self._coordinate_type_for_response(
            response,
            processed_width=processed_width,
            processed_height=processed_height,
        )
        return parse_base_response(
            response,
            coordinate_type=coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            allow_local_runtime_compat=self.local_runtime_compat,
        )

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        screenshot_bytes = obs["screenshot"]
        observation_signature = image_perceptual_hash(screenshot_bytes)

        original_width, original_height = image_size_from_bytes(screenshot_bytes)
        image_bytes = screenshot_bytes
        observation_text = ""
        observation_label = ""

        if self.observation_type in {"a11y_tree", "screenshot_a11y_tree", "som"}:
            raw_tree = obs.get("accessibility_tree")
            if not raw_tree:
                raise ValueError(
                    f"{self.observation_type} requires obs['accessibility_tree']"
                )
            linearized_tree = linearize_accessibility_tree(
                raw_tree, platform=self.platform
            )
            linearized_tree = trim_accessibility_tree(
                linearized_tree, self.a11y_tree_max_tokens
            )
            observation_text = linearized_tree
            observation_label = "Accessibility tree for the current screen:\n"

            if self.observation_type == "som":
                _, _, tagged_screenshot, linearized_som_tree = tag_screenshot(
                    screenshot_bytes, raw_tree, self.platform
                )
                if isinstance(tagged_screenshot, (bytes, bytearray)):
                    image_bytes = bytes(tagged_screenshot)
                else:
                    buffer = BytesIO()
                    tagged_screenshot.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                if linearized_som_tree:
                    linearized_som_tree = trim_accessibility_tree(
                        linearized_som_tree, self.a11y_tree_max_tokens
                    )
                    observation_text = linearized_som_tree
                    observation_label = (
                        "Accessibility tree matching the tagged screenshot:\n"
                    )

        if self.observation_type == "a11y_tree":
            processed_b64 = None
            processed_width, processed_height = original_width, original_height
        else:
            processed_b64 = process_image(image_bytes)
            processed_width, processed_height = image_size_from_base64(processed_b64)

        if observation_text:
            observation_text = scale_accessibility_tree_coordinates(
                observation_text,
                coordinate_type=self.effective_coordinate_type,
                original_width=original_width,
                original_height=original_height,
                processed_width=processed_width,
                processed_height=processed_height,
            )
            observation_text = observation_label + observation_text

        recovery_feedback = self.pending_recovery_feedback
        self.pending_recovery_feedback = ""
        if recovery_feedback:
            feedback_text = (
                "Execution feedback from the preceding step:\n"
                f"{recovery_feedback}"
            )
            observation_text = (
                f"{feedback_text}\n\n{observation_text}"
                if observation_text
                else feedback_text
            )

        self.screenshots.append(processed_b64)
        self.observation_texts.append(observation_text)
        total_steps = len(self.screenshots)
        self.folded_prefix_k = update_folding_state(
            total_steps,
            self.folded_prefix_k,
            self.image_max,
            self.fold_size,
        )

        effective_history_n = self.history_n
        if self.observation_type != "screenshot":
            effective_history_n = min(
                effective_history_n,
                self.multimodal_history_n,
            )
        start_step = max(1, total_steps - effective_history_n + 1)
        previous_actions_str = previous_actions_text(self.actions, start_step)

        tools_def = self._build_tools_def(processed_width, processed_height)
        system_prompt = self._build_system_prompt(tools_def)
        instruction_prompt = build_instruction_prompt(instruction, previous_actions_str)
        stagnation_context = self._stagnation_context(observation_signature)
        if stagnation_context:
            instruction_prompt += "\n\n" + stagnation_context

        self.observations.append({"screenshot": processed_b64})
        messages = build_messages(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            screenshots=self.screenshots,
            observation_texts=self.observation_texts,
            observation_text_history_n=(
                self.a11y_history_n
                if self.observation_type != "screenshot"
                else None
            ),
            responses=self.responses,
            start_step=start_step,
            total_steps=total_steps,
            folded_prefix_k=self.folded_prefix_k,
            collapse_text=self.collapse_text,
            response_transform=self._response_transform,
        )

        step_idx = total_steps - 1
        dump_debug_messages(messages, self._debug_message_filename(step_idx), logger)

        response = ""
        low_level_instruction = ""
        pyautogui_code: List[str] = []
        format_retries = max(0, int(os.environ.get("OSWORLD_QWEN_FORMAT_RETRIES", "2")))
        saw_stagnation_rejection = False
        saw_completion_rejection = False
        saw_action_contract_rejection = False
        saw_completion_audit_failure = False
        completion_audit_performed = False
        last_rejected_action: Tuple[str, ...] = ()
        completion_blocked_by_static_action = (
            self._last_action_left_screen_unchanged(observation_signature)
        )

        for format_attempt in range(format_retries + 1):
            repetition_rejected = False
            completion_rejected = False
            action_contract_rejected = False
            rejected_action: Tuple[str, ...] = ()
            payload = self._build_payload(messages, tools_def)
            if format_attempt > 0 and float(payload.get("temperature", 0.0)) <= 0:
                payload["temperature"] = float(
                    os.environ.get("OSWORLD_QWEN_RETRY_TEMPERATURE", "0.2")
                )
            response = self.call_llm(payload, self.model)

            if logger:
                logger.info(
                    "%s Output (format attempt %d/%d): %s",
                    self._log_prefix(),
                    format_attempt + 1,
                    format_retries + 1,
                    response,
                )

            low_level_instruction, pyautogui_code = self._parse_response(
                response or "",
                original_width=original_width,
                original_height=original_height,
                processed_width=processed_width,
                processed_height=processed_height,
            )

            if (
                pyautogui_code == ["DONE"]
                and self.completion_verification
                and not completion_audit_performed
            ):
                completion_audit_performed = True
                audit_messages = self._build_completion_audit_messages(
                    system_prompt=system_prompt,
                    audit_prompt=self._completion_audit_prompt(
                        instruction,
                        last_action_left_screen_unchanged=(
                            completion_blocked_by_static_action
                        ),
                    ),
                    processed_b64=processed_b64,
                    observation_text=observation_text,
                )
                audit_payload = self._build_payload(audit_messages, tools_def)
                audit_payload["temperature"] = 0.0
                audit_payload["max_tokens"] = min(
                    self.max_tokens,
                    max(
                        256,
                        int(
                            os.environ.get(
                                "OSWORLD_QWEN_COMPLETION_AUDIT_MAX_TOKENS",
                                "1200",
                            )
                        ),
                    ),
                )
                try:
                    audit_response = self.call_llm(audit_payload, self.model)
                except Exception as exc:
                    completion_rejected = True
                    saw_completion_rejection = True
                    saw_completion_audit_failure = True
                    low_level_instruction = (
                        "Completion audit unavailable after provider retries"
                    )
                    pyautogui_code = []
                    if logger:
                        logger.warning(
                            "Completion audit provider call failed after retries; "
                            "rejecting provisional completion and continuing bounded "
                            "recovery: %s",
                            exc,
                        )
                    break
                else:
                    audit_instruction, audit_code = self._parse_response(
                        audit_response or "",
                        original_width=original_width,
                        original_height=original_height,
                        processed_width=processed_width,
                        processed_height=processed_height,
                    )
                    trajectory_support = self._completion_has_trajectory_support(
                        instruction,
                        audit_response or "",
                    )
                    audit_unverified = self._completion_response_is_unverified(
                        audit_response
                    )
                    if audit_code == ["DONE"] and (
                        audit_unverified
                        and not trajectory_support
                    ):
                        completion_rejected = True
                        saw_completion_rejection = True
                        if logger:
                            logger.warning(
                                "Rejected unverified or self-contradictory completion audit."
                            )
                        audit_code = []
                    response = audit_response or ""
                    low_level_instruction = audit_instruction
                    pyautogui_code = audit_code
                    messages = audit_messages
                    if logger:
                        logger.info(
                            "%s completion audit output: %s",
                            self._log_prefix(),
                            response,
                        )
            elif (
                pyautogui_code == ["DONE"]
                and self.completion_verification
                and (
                    self._completion_response_is_unverified(response)
                    and not self._completion_has_trajectory_support(
                        instruction,
                        response,
                    )
                )
            ):
                completion_rejected = True
                saw_completion_rejection = True
                pyautogui_code = []

            action_contract_rejected = bool(
                pyautogui_code
                and not action_description_matches_code(
                    low_level_instruction,
                    pyautogui_code,
                )
            )
            if action_contract_rejected:
                saw_action_contract_rejection = True
                rejected_action = tuple(
                    code.strip() for code in pyautogui_code if code.strip()
                )
                last_rejected_action = rejected_action
                if logger:
                    logger.warning(
                        "Rejected action because the Action description claimed "
                        "an operation absent from the executable tool call."
                    )
                pyautogui_code = []

            repetition_rejected = self._is_repeated_action(
                pyautogui_code,
                low_level_instruction,
                observation_signature,
            )
            if repetition_rejected:
                saw_stagnation_rejection = True
                rejected_action = tuple(
                    code.strip() for code in pyautogui_code if code.strip()
                )
                last_rejected_action = rejected_action
                if logger:
                    logger.warning(
                        "Rejected repeated low-level action because the visible "
                        "screen remained unchanged."
                    )
                pyautogui_code = []

            if pyautogui_code:
                break

            if format_attempt < format_retries:
                if logger:
                    if repetition_rejected:
                        logger.warning(
                            "Qwen repeated an action on an unchanged screen; "
                            "requesting a materially different action."
                        )
                    elif completion_rejected:
                        logger.warning(
                            "Qwen did not provide concrete visible completion "
                            "evidence; requesting a next action."
                        )
                    elif action_contract_rejected:
                        logger.warning(
                            "Qwen action description did not match the executable "
                            "tool call; requesting one atomic action."
                        )
                    else:
                        logger.warning(
                            "Qwen returned no executable computer_use tool call; "
                            "requesting a corrected response."
                        )
                if repetition_rejected:
                    messages = self._build_recovery_messages(
                        system_prompt=system_prompt,
                        instruction=instruction,
                        processed_b64=processed_b64,
                        observation_text=observation_text,
                        reason="stagnation",
                        rejected_action=rejected_action,
                    )
                elif (
                    not completion_rejected
                    and not action_contract_rejected
                ):
                    messages = self._build_recovery_messages(
                        system_prompt=system_prompt,
                        instruction=instruction,
                        processed_b64=processed_b64,
                        observation_text=observation_text,
                        reason="format",
                        rejected_response=response or "",
                    )
                else:
                    retry_text = (
                        "COMPLETION NOT VERIFIED: The latest observation does not "
                        "support the proposed action=terminate with status=success, "
                        "or the Action line "
                        "did not name concrete visible evidence. Do not repeat a "
                        "generic completion claim. Inspect the current application and "
                        "emit one executable action that advances or verifies every "
                        "remaining requirement."
                        if completion_rejected
                        else (
                            "ACTION CONTRACT ERROR: The Action line claimed an "
                            "operation that the tool call would not execute. "
                            f"The rejected executable action was: {rejected_action}. "
                            "Reply with one atomic action whose description exactly "
                            "matches the single computer_use call. Do not describe "
                            "pressing Enter, clicking, or scrolling unless that same "
                            "operation is present in the tool call."
                        )
                    )
                    messages.extend(
                        [
                            {"role": "assistant", "content": response or ""},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": retry_text,
                                    }
                                ],
                            },
                        ]
                    )

        self.responses.append(response or "")

        if not pyautogui_code:
            if not saw_completion_audit_failure:
                self.consecutive_completion_audit_failures = 0
            if saw_completion_audit_failure:
                self.consecutive_format_failures = 0
                self.consecutive_stagnation_failures = 0
                self.consecutive_completion_failures = 0
                self.consecutive_completion_audit_failures += 1
                audit_failure_limit = max(
                    1,
                    int(
                        os.environ.get(
                            "OSWORLD_QWEN_COMPLETION_AUDIT_FAILURE_LIMIT",
                            str(self.format_failure_limit),
                        )
                    ),
                )
                if (
                    self.consecutive_completion_audit_failures
                    >= audit_failure_limit
                ):
                    low_level_instruction = (
                        "Completion audit provider failed after bounded recovery"
                    )
                    pyautogui_code = [
                        adapter_error_action("provider_completion_audit")
                    ]
                else:
                    low_level_instruction = (
                        "Completion audit unavailable; provisional success rejected"
                    )
                    pyautogui_code = ["WAIT"]
                    self.pending_recovery_feedback = (
                        "The preceding action=terminate with status=success was NOT "
                        "executed because "
                        "the independent completion audit was unavailable. Continue "
                        "the task or inspect the visible result before proposing "
                        "completion again."
                    )
            elif saw_stagnation_rejection:
                self.consecutive_format_failures = 0
                self.consecutive_completion_failures = 0
                self.consecutive_stagnation_failures += 1
                stagnation_failure_limit = max(
                    1,
                    int(
                        os.environ.get(
                            "OSWORLD_QWEN_STAGNATION_FAILURE_LIMIT",
                            str(self.format_failure_limit),
                        )
                    ),
                )
                if self.consecutive_stagnation_failures >= stagnation_failure_limit:
                    low_level_instruction = (
                        "Model repeated ineffective actions after bounded recovery"
                    )
                    pyautogui_code = [model_error_action("stagnation")]
                else:
                    low_level_instruction = (
                        "Repeated action rejected; requesting a fresh observation"
                    )
                    pyautogui_code = ["WAIT"]
                    rendered_action = (
                        "; ".join(last_rejected_action)
                        if last_rejected_action
                        else "the repeated action"
                    )
                    self.pending_recovery_feedback = (
                        "The preceding proposed action was NOT executed because "
                        "it had already failed repeatedly while the screen stayed "
                        f"unchanged: `{rendered_action}`. Do not emit it again. "
                        "Re-ground on the current screen and use a different "
                        "control, keyboard route, or application-navigation path."
                    )
            elif saw_completion_rejection:
                self.consecutive_format_failures = 0
                self.consecutive_stagnation_failures = 0
                self.consecutive_completion_failures += 1
                completion_failure_limit = max(
                    1,
                    int(
                        os.environ.get(
                            "OSWORLD_QWEN_COMPLETION_FAILURE_LIMIT",
                            str(self.format_failure_limit),
                        )
                    ),
                )
                if self.consecutive_completion_failures >= completion_failure_limit:
                    low_level_instruction = (
                        "Repeated unverified completion claims; preserving the "
                        "current task state for another observation"
                    )
                    pyautogui_code = ["WAIT"]
                    self.consecutive_completion_failures = 0
                    self.pending_recovery_feedback = (
                        "Repeated completion claims were not accepted because they "
                        "did not cite concrete UI evidence. Preserve the completed "
                        "state, inspect a visible verification surface, and then "
                        "provide an action=terminate, status=success call with "
                        "specific evidence."
                    )
                else:
                    low_level_instruction = (
                        "Completion was not verified; requesting a fresh observation"
                    )
                    pyautogui_code = ["WAIT"]
                    self.pending_recovery_feedback = (
                        "The preceding action=terminate with status=success was NOT "
                        "executed because "
                        "the latest screen did not provide concrete completion "
                        "evidence. Inspect or finish the requested result and emit "
                        "an executable action instead of another generic success claim."
                    )
            elif saw_action_contract_rejection:
                self.consecutive_stagnation_failures = 0
                self.consecutive_completion_failures = 0
                self.consecutive_format_failures += 1
                if self.consecutive_format_failures >= self.format_failure_limit:
                    low_level_instruction = (
                        "Model repeated action-contract mismatches after bounded "
                        "recovery"
                    )
                    pyautogui_code = [model_error_action("action_contract")]
                else:
                    low_level_instruction = (
                        "Action contract mismatch; requesting a fresh observation"
                    )
                    pyautogui_code = ["WAIT"]
                    self.pending_recovery_feedback = (
                        "The preceding response was NOT executed because its Action "
                        "description did not match its tool call. Emit one atomic "
                        "computer_use call that performs exactly the described action."
                    )
            else:
                self.consecutive_stagnation_failures = 0
                self.consecutive_completion_failures = 0
                self.consecutive_format_failures += 1
                if self.consecutive_format_failures >= self.format_failure_limit:
                    low_level_instruction = (
                        "Model returned malformed responses after bounded recovery"
                    )
                    pyautogui_code = [model_error_action("format")]
                else:
                    low_level_instruction = (
                        "No executable action parsed; requesting a fresh observation"
                    )
                    pyautogui_code = ["WAIT"]
                    self.pending_recovery_feedback = (
                        "The preceding response was NOT executed because it contained "
                        "no valid computer_use tool call. Repeat the intended operation "
                        "as exactly one executable tool call with all required parameters."
                    )
        else:
            self.consecutive_format_failures = 0
            self.consecutive_stagnation_failures = 0
            self.consecutive_completion_failures = 0
            self.consecutive_completion_audit_failures = 0

        if logger:
            logger.info("Low level instruction: %s", low_level_instruction)
            logger.info("Pyautogui code: %s", pyautogui_code)

        self.actions.append(low_level_instruction)
        if pyautogui_code not in (["FAIL"], ["DONE"], ["WAIT"]) and not (
            len(pyautogui_code) == 1
            and (
                decode_adapter_error_action(pyautogui_code[0]) is not None
                or decode_model_error_action(pyautogui_code[0]) is not None
            )
        ):
            self.action_records.append(
                (
                    tuple(code.strip() for code in pyautogui_code if code.strip()),
                    self._normalize_action_description(low_level_instruction),
                    observation_signature,
                )
            )
        return response or "", pyautogui_code

    def parse_response(
        self,
        response: str,
        original_width: int = None,
        original_height: int = None,
        processed_width: int = None,
        processed_height: int = None,
    ) -> Tuple[str, List[str]]:
        coordinate_type = self._coordinate_type_for_response(
            response,
            processed_width=processed_width,
            processed_height=processed_height,
        )
        return parse_base_response(
            response,
            coordinate_type=coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            allow_local_runtime_compat=self.local_runtime_compat,
        )

    def call_llm(self, payload: Dict, model: str) -> str:
        return call_openai_compatible(
            payload,
            model,
            base_url=self.base_url,
            api_key=self.api_key,
            default_max_tokens=self.max_tokens,
            default_temperature=self.temperature,
            default_top_p=self.top_p,
            logger=logger,
        )

    def reset(self, _logger=None, *args, **kwargs):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.qwen_agent")
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []
        self.observation_texts = []
        self.action_records = []
        self.consecutive_format_failures = 0
        self.consecutive_stagnation_failures = 0
        self.consecutive_completion_failures = 0
        self.consecutive_completion_audit_failures = 0
        self.folded_prefix_k = 0
        self.pending_recovery_feedback = ""
        # The worker reuses this agent with the same model and API provider, so
        # keep an inferred provider coordinate convention across task resets.


class QwenAgent(_QwenBaseAgent):
    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(
        self,
        *args,
        enable_thinking: bool = False,
        observation_type: str = "screenshot",
        **kwargs,
    ):
        super().__init__(*args, observation_type=observation_type, **kwargs)
        self.enable_thinking = enable_thinking
        self.observation_type = observation_type

    def _build_tools_def(self, processed_width: int, processed_height: int) -> Dict:
        return build_internal_tools_def(
            processed_width,
            processed_height,
            self.effective_coordinate_type,
        )

    def _build_system_prompt(self, tools_def: Dict) -> str:
        prompt = (
            build_native_system_prompt(self.collapse_text)
            if self.native_tools
            else build_internal_system_prompt(tools_def, self.collapse_text)
        )
        prompt += "\n\n" + STATE_GROUNDING_PROMPT
        prompt += self._local_runtime_prompt_suffix()
        if self.add_thought_prefix:
            prompt += (
                "\n\n# Deliberation\n"
                "Before the Action line, briefly analyze the latest UI state in "
                "a <think>...</think> block. Verify what changed after the prior "
                "action and identify the single safest next action."
            )
        if self.observation_type in {"a11y_tree", "screenshot_a11y_tree", "som"}:
            prompt += (
                "\n\n# Accessibility observations\n"
                "The accessibility tree is a tab-separated table. Position is the "
                "element's top-left (x, y), and size is (width, height). Click the "
                "center: (x + width/2, y + height/2). Prefer named actionable "
                "elements from the tree over guessing coordinates."
            )
        if self.observation_type == "som":
            prompt += (
                "\nThe screenshot is Set-of-Marks tagged. Use tag labels and the "
                "accompanying accessibility table to identify the target."
            )
        return prompt

    def _response_transform(self, response: str) -> str:
        return ensure_empty_think_prefix(response)

    def _debug_message_filename(self, step_idx: int) -> str:
        return f"qwen_messages_{os.getpid()}_step_{step_idx}.json"

    def _build_payload(
        self,
        messages: List[Dict],
        tools_def: Optional[Dict] = None,
    ) -> Dict:
        payload = super()._build_payload(messages, tools_def)
        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL", "")
        if "dashscope" in base_url.lower():
            extra_body = dict(payload.get("extra_body") or {})
            extra_body["enable_thinking"] = bool(self.enable_thinking)
            payload["extra_body"] = extra_body
        return payload

    def _log_prefix(self) -> str:
        return "Qwen"

    def _parse_response(
        self,
        response: str,
        *,
        original_width: int,
        original_height: int,
        processed_width: int,
        processed_height: int,
    ) -> Tuple[str, List[str]]:
        coordinate_type = self._coordinate_type_for_response(
            response,
            processed_width=processed_width,
            processed_height=processed_height,
        )
        return parse_internal_response(
            response,
            coordinate_type=coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            allow_local_runtime_compat=self.local_runtime_compat,
        )

    def parse_response(
        self,
        response: str,
        original_width: int = None,
        original_height: int = None,
        processed_width: int = None,
        processed_height: int = None,
    ) -> Tuple[str, List[str]]:
        coordinate_type = self._coordinate_type_for_response(
            response,
            processed_width=processed_width,
            processed_height=processed_height,
        )
        return parse_internal_response(
            response,
            coordinate_type=coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            allow_local_runtime_compat=self.local_runtime_compat,
        )
