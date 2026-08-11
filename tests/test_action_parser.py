import unittest

from osworld_qwen25vl.actions import (
    action_description_matches_code,
    parse_base_response,
    parse_internal_response,
)
from osworld_qwen25vl.parser import looks_infeasible_response, parse_keys


def tool_call(*params):
    body = "\n".join(
        f"<parameter={name}>{value}</parameter>"
        for name, value in params
    )
    return (
        "Action: test\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        f"{body}\n"
        "</function>\n"
        "</tool_call>"
    )


class QwenActionParserTest(unittest.TestCase):
    def test_parse_keys_accepts_python_literal_lists(self):
        self.assertEqual(parse_keys("['ctrl', 's']", lowercase=True), ["ctrl", "s"])
        self.assertEqual(parse_keys("[\"ctrl\", \"shift\", \"t\"]", lowercase=True), ["ctrl", "shift", "t"])
        self.assertEqual(parse_keys([["ctrl"], "s"], lowercase=True), ["ctrl", "s"])

    def test_parse_keys_splits_plus_separated_shortcuts(self):
        self.assertEqual(parse_keys("ctrl+shift+t", lowercase=True), ["ctrl", "shift", "t"])

    def test_internal_parser_handles_non_success_terminate_aliases(self):
        for status in ["fail", "failed", "failure", "error", "infeasible"]:
            _, code = parse_internal_response(
                tool_call(("action", "terminate"), ("status", status)),
                coordinate_type="relative",
            )
            self.assertEqual(code, ["FAIL"])

    def test_existing_structured_terminate_success_remains_supported(self):
        _, code = parse_internal_response(
            tool_call(("action", "terminate"), ("status", "success")),
            coordinate_type="relative",
        )
        self.assertEqual(code, ["DONE"])

    def test_local_terminal_shorthand_is_a_bounded_compatibility_alias(self):
        for response in [
            "Action: terminate=success",
            "<think>I checked the visible result.</think>\nAction: terminate(success).",
            'terminate(status="success")',
            "terminate success The task has completed successfully.",
        ]:
            _, code = parse_internal_response(
                response,
                coordinate_type="relative",
                allow_local_runtime_compat=True,
            )
            self.assertEqual(code, ["DONE"])

        _, code = parse_internal_response(
            "Action: terminate=failure",
            coordinate_type="relative",
            allow_local_runtime_compat=True,
        )
        self.assertEqual(code, ["FAIL"])

    def test_local_terminal_shorthand_does_not_change_remote_parser(self):
        for response in [
            "Action: terminate=success",
            'terminate(status="success")',
            "terminate success The task has completed successfully.",
        ]:
            _, code = parse_internal_response(
                response,
                coordinate_type="relative",
                allow_local_runtime_compat=False,
            )
            self.assertEqual(code, [])

    def test_terminal_shorthand_is_not_inferred_from_reasoning_or_prose(self):
        for response in [
            "<think>I may use terminate=success later.</think>\nAction: Click Save.",
            "Action: Click Save, then use terminate=success.",
            "Action: terminate=success\nHowever, the result is not verified.",
            "The next step should be terminate(success) after verification.",
        ]:
            _, code = parse_internal_response(
                response,
                coordinate_type="relative",
                allow_local_runtime_compat=True,
            )
            self.assertEqual(code, [])

    def test_base_parser_handles_non_success_terminate_aliases(self):
        _, code = parse_base_response(
            tool_call(("action", "terminate"), ("status", "fail")),
            coordinate_type="relative",
        )
        self.assertEqual(code, ["FAIL"])

    def test_internal_parser_generates_hotkeys_from_python_literal_lists(self):
        _, code = parse_internal_response(
            tool_call(("action", "key"), ("keys", "['ctrl', 's']")),
            coordinate_type="relative",
        )
        self.assertEqual(code, ['pyautogui.hotkey("ctrl", "s")'])

    def test_internal_parser_accepts_json_tool_call(self):
        response = (
            'Action: Click Save\n'
            '<tool_call>{"name":"computer_use","arguments":'
            '{"action":"left_click","coordinate":[500,500]}}</tool_call>'
        )
        _, code = parse_internal_response(
            response,
            coordinate_type="relative",
            original_width=1920,
            original_height=1080,
        )
        self.assertEqual(code, ["pyautogui.click(960, 540)"])

    def test_internal_parser_recovers_unclosed_final_parameter(self):
        response = (
            "Action: Click the browser menu\n"
            "<function=computer_use>\n"
            "<parameter=action>\n"
            "left_click\n"
            "</parameter>\n"
            "<parameter=coordinate>\n"
            "[1062, 38]\n"
            "</function>"
        )
        _, code = parse_internal_response(
            response,
            coordinate_type="absolute",
            original_width=1920,
            original_height=1080,
            processed_width=1920,
            processed_height=1080,
        )
        self.assertEqual(code, ["pyautogui.click(1062, 38)"])

    def test_internal_parser_accepts_explicit_plain_completion(self):
        for response in [
            "Action: Task completed",
            "The task is complete.",
            "<think>I verified the result.</think>\nTask completed.",
        ]:
            _, code = parse_internal_response(
                response,
                coordinate_type="relative",
            )
            self.assertEqual(code, ["DONE"])

    def test_internal_parser_does_not_treat_incidental_done_as_completion(self):
        _, code = parse_internal_response(
            "Action: Click Done after saving the file.",
            coordinate_type="relative",
        )
        self.assertEqual(code, [])

    def test_internal_parser_executes_only_first_valid_tool_call(self):
        response = "\n".join(
            [
                tool_call(("action", "mouse_move"), ("coordinate", "[100, 200]")),
                tool_call(("action", "scroll"), ("pixels", "-5")),
            ]
        )
        _, code = parse_internal_response(
            response,
            coordinate_type="absolute",
            original_width=1920,
            original_height=1080,
            processed_width=1920,
            processed_height=1080,
        )
        self.assertEqual(code, ["pyautogui.moveTo(100, 200)"])

    def test_internal_parser_rejects_click_without_coordinate(self):
        _, code = parse_internal_response(
            tool_call(("action", "left_click")),
            coordinate_type="relative",
        )
        self.assertEqual(code, [])

    def test_internal_parser_accepts_common_click_alias(self):
        _, code = parse_internal_response(
            tool_call(("action", "click"), ("coordinate", "[500, 500]")),
            coordinate_type="relative",
            original_width=1920,
            original_height=1080,
        )
        self.assertEqual(code, ["pyautogui.click(960, 540)"])

    def test_internal_parser_decodes_wrapping_quotes_from_type_text(self):
        _, code = parse_internal_response(
            tool_call(("action", "type"), ("text", "'Favorites'")),
            coordinate_type="relative",
        )
        self.assertEqual(code, ['pyautogui.typewrite("Favorites")'])

    def test_internal_parser_preserves_unwrapped_text(self):
        _, code = parse_internal_response(
            tool_call(("action", "type"), ("text", "Favorites")),
            coordinate_type="relative",
        )
        self.assertEqual(code, ['pyautogui.typewrite("Favorites")'])

    def test_internal_parser_recovers_unambiguous_enter_without_tool_call(self):
        _, code = parse_internal_response(
            "Action: Press the Enter key.",
            coordinate_type="relative",
        )
        self.assertEqual(code, ['pyautogui.press("enter")'])

    def test_internal_parser_recovers_enter_with_a_purpose_clause(self):
        _, code = parse_internal_response(
            "Action: Press Enter to execute the command in the terminal.",
            coordinate_type="relative",
        )
        self.assertEqual(code, ['pyautogui.press("enter")'])

    def test_internal_parser_recovers_unambiguous_close_without_tool_call(self):
        _, code = parse_internal_response(
            "Action: Close the LibreOffice Calc application.",
            coordinate_type="relative",
        )
        self.assertEqual(code, ['pyautogui.hotkey("alt", "f4")'])

    def test_internal_parser_rejects_call_user_in_autonomous_run(self):
        _, code = parse_internal_response(
            tool_call(("action", "call_user"), ("text", "Please sign in")),
            coordinate_type="relative",
        )
        self.assertEqual(code, [])

    def test_infeasible_detection_ignores_incidental_reasoning(self):
        self.assertFalse(
            looks_infeasible_response(
                "<think>The old option is not available, so I will use Settings.</think>\n"
                + tool_call(
                    ("action", "left_click"),
                    ("coordinate", "[500, 500]"),
                )
            )
        )
        self.assertTrue(looks_infeasible_response("Action: The task is infeasible."))

    def test_action_contract_detects_missing_enter(self):
        self.assertFalse(
            action_description_matches_code(
                "Type the URL and press Enter.",
                ['pyautogui.typewrite("https://example.com")'],
            )
        )
        self.assertTrue(
            action_description_matches_code(
                "Press Enter.",
                ['pyautogui.press("enter")'],
            )
        )

    def test_action_contract_detects_non_executed_click(self):
        self.assertFalse(
            action_description_matches_code(
                "Click the Save button.",
                ['pyautogui.typewrite("Save")'],
            )
        )
        self.assertTrue(
            action_description_matches_code(
                "Click the Save button.",
                ["pyautogui.click(500, 500)"],
            )
        )


if __name__ == "__main__":
    unittest.main()
