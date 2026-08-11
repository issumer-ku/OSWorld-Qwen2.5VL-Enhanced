from io import BytesIO
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from osworld_qwen25vl.client import call_openai_compatible, is_local_openai_endpoint
from osworld_qwen25vl.history import build_messages
from osworld_qwen25vl.agent import QwenAgent
from osworld_qwen25vl.images import (
    adjust_coordinates,
    image_perceptual_hash,
    perceptual_hash_distance,
    scale_accessibility_tree_coordinates,
)


class QwenClientReliabilityTest(unittest.TestCase):
    def test_local_runtime_detection_excludes_openrouter(self):
        self.assertTrue(is_local_openai_endpoint("http://127.0.0.1:1234/v1"))
        self.assertTrue(is_local_openai_endpoint("http://192.168.50.159:1234/v1"))
        self.assertTrue(is_local_openai_endpoint("http://host.docker.internal:1234/v1"))
        self.assertFalse(is_local_openai_endpoint("https://openrouter.ai/api/v1"))
        self.assertFalse(
            is_local_openai_endpoint(
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
        )
    def test_empty_environment_api_key_uses_dummy_for_unauthenticated_local_server(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok",
                        reasoning_content=None,
                    )
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: response,
                )
            )
        )
        payload = {
            "messages": [{"role": "user", "content": "ping"}],
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with patch(
                "osworld_qwen25vl.client.openai.OpenAI",
                return_value=client,
            ) as openai_client:
                result = call_openai_compatible(
                    payload,
                    "local-model",
                    base_url="http://127.0.0.1:1234/v1",
                    api_key=None,
                    default_max_tokens=16,
                    default_temperature=0.0,
                    default_top_p=0.9,
                )

        self.assertEqual(result, "ok")
        self.assertEqual(openai_client.call_args.kwargs["api_key"], "dummy")

    def test_empty_choices_are_retried(self):
        empty_response = SimpleNamespace(choices=None)
        valid_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="recovered",
                        reasoning_content=None,
                    )
                )
            ]
        )
        create = Mock(side_effect=[empty_response, valid_response])
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )
        payload = {
            "messages": [{"role": "user", "content": "ping"}],
        }

        with patch(
            "osworld_qwen25vl.client.openai.OpenAI",
            return_value=client,
        ), patch("osworld_qwen25vl.client.time.sleep"):
            result = call_openai_compatible(
                payload,
                "remote-model",
                base_url="https://openrouter.ai/api/v1",
                api_key="test-key",
                default_max_tokens=16,
                default_temperature=0.0,
                default_top_p=0.9,
            )

        self.assertEqual(result, "recovered")
        self.assertEqual(create.call_count, 2)

    def test_native_tool_call_is_returned_as_parser_compatible_json(self):
        response = SimpleNamespace(
            id="response-1",
            model="qwen-test",
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="computer_use",
                                    arguments='{"action":"key","keys":["enter"]}',
                                )
                            )
                        ],
                    ),
                )
            ],
        )
        create = Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )
        tool = {
            "type": "function",
            "function": {
                "name": "computer_use",
                "parameters": {"type": "object"},
            },
        }

        with patch(
            "osworld_qwen25vl.client.openai.OpenAI",
            return_value=client,
        ):
            result = call_openai_compatible(
                {
                    "messages": [{"role": "user", "content": "press enter"}],
                    "tools": [tool],
                    "tool_choice": "auto",
                },
                "remote-model",
                base_url="https://openrouter.ai/api/v1",
                api_key="test-key",
                default_max_tokens=16,
                default_temperature=0.0,
                default_top_p=0.9,
            )

        self.assertIn('"name": "computer_use"', result)
        self.assertIn('"action": "key"', result)
        self.assertEqual(create.call_args.kwargs["tools"], [tool])
        self.assertEqual(create.call_args.kwargs["tool_choice"], "auto")


class QwenAdapterReliabilityTest(unittest.TestCase):
    @staticmethod
    def _multimodal_observation():
        image = Image.new("RGB", (320, 240), "white")
        ImageDraw.Draw(image).rectangle((80, 70, 220, 130), fill="blue")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        tree = (
            '<root xmlns:state="https://accessibility.ubuntu.example.org/ns/state" '
            'xmlns:component="https://accessibility.ubuntu.example.org/ns/component">'
            '<button name="Save" state:showing="true" state:visible="true" '
            'state:enabled="true" component:screencoord="(80, 70)" '
            'component:size="(140, 60)">Save</button></root>'
        )
        return {
            "screenshot": buffer.getvalue(),
            "accessibility_tree": tree,
        }

    def test_relative_a11y_coordinates_match_tool_space(self):
        table = (
            "tag\tname\ttext\tclass\tdescription\tposition\tsize\n"
            "button\tSave\t\t\t\t(960, 540)\t(192, 108)"
        )
        scaled = scale_accessibility_tree_coordinates(
            table,
            coordinate_type="relative",
            original_width=1920,
            original_height=1080,
            processed_width=1920,
            processed_height=1080,
        )
        self.assertIn("(500, 500)", scaled)
        self.assertIn("(100, 100)", scaled)

    def test_absolute_a11y_coordinates_match_processed_image(self):
        table = (
            "tag\tname\ttext\tclass\tdescription\tposition\tsize\n"
            "button\tSave\t\t\t\t(960, 540)\t(192, 108)"
        )
        scaled = scale_accessibility_tree_coordinates(
            table,
            coordinate_type="absolute",
            original_width=1920,
            original_height=1080,
            processed_width=1280,
            processed_height=720,
        )
        self.assertIn("(640, 360)", scaled)
        self.assertIn("(128, 72)", scaled)

    def test_old_a11y_trees_are_omitted_but_latest_is_retained(self):
        messages = build_messages(
            system_prompt="system",
            instruction_prompt="instruction",
            screenshots=[None, None, None],
            observation_texts=["tree one", "tree two", "tree three"],
            observation_text_history_n=1,
            responses=["action one", "action two"],
            start_step=1,
            total_steps=3,
            folded_prefix_k=0,
            collapse_text="collapsed",
        )
        rendered = str(messages)
        self.assertNotIn("tree one", rendered)
        self.assertNotIn("tree two", rendered)
        self.assertIn("tree three", rendered)
        self.assertIn("Earlier accessibility tree omitted", rendered)

    def test_perceptual_hash_distance_is_zero_for_identical_values(self):
        self.assertEqual(perceptual_hash_distance(12345, 12345), 0)

    def test_relative_coordinate_endpoints_stay_inside_screen(self):
        self.assertEqual(
            adjust_coordinates(
                999,
                999,
                coordinate_type="relative",
                original_width=1920,
                original_height=1080,
                processed_width=1920,
                processed_height=1088,
            ),
            (1919, 1079),
        )

    def test_coordinates_are_clamped_after_provider_overshoot(self):
        self.assertEqual(
            adjust_coordinates(
                2500,
                1400,
                coordinate_type="relative",
                original_width=1920,
                original_height=1080,
                processed_width=1920,
                processed_height=1088,
            ),
            (1919, 1079),
        )

    def test_relative_mode_locks_to_absolute_after_pixel_coordinate_evidence(self):
        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            coordinate_type="relative",
        )
        first_response = (
            "Action: Open the browser menu.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>left_click</parameter>"
            "<parameter=coordinate>[1882, 124]</parameter>"
            "</function></tool_call>"
        )
        second_response = (
            "Action: Open Search engine settings.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>left_click</parameter>"
            "<parameter=coordinate>[177, 427]</parameter>"
            "</function></tool_call>"
        )

        _, first = agent.parse_response(
            first_response,
            original_width=1920,
            original_height=1080,
            processed_width=1920,
            processed_height=1088,
        )
        _, second = agent.parse_response(
            second_response,
            original_width=1920,
            original_height=1080,
            processed_width=1920,
            processed_height=1088,
        )

        self.assertEqual(first, ["pyautogui.click(1882, 123)"])
        self.assertEqual(second, ["pyautogui.click(177, 424)"])
        self.assertEqual(agent.effective_coordinate_type, "absolute")
        absolute_tool = agent._build_tools_def(1920, 1088)
        self.assertIn(
            "Use screenshot pixel coordinates",
            absolute_tool["function"]["description"],
        )

    def test_coordinate_mode_lock_survives_reset_for_same_provider(self):
        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            coordinate_type="relative",
        )
        response = (
            "Action: Open the browser menu.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>left_click</parameter>"
            "<parameter=coordinate>[1882, 124]</parameter>"
            "</function></tool_call>"
        )
        agent.parse_response(
            response,
            original_width=1920,
            original_height=1080,
            processed_width=1920,
            processed_height=1088,
        )

        agent.reset()

        self.assertEqual(agent.effective_coordinate_type, "absolute")

    def test_relative_coordinate_prompt_has_one_unambiguous_contract(self):
        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            coordinate_type="relative",
        )
        tool = agent._build_tools_def(1920, 1088)
        function = tool["function"]
        coordinate = function["parameters"]["properties"]["coordinate"]

        self.assertIn("normalized integer coordinates", function["description"])
        self.assertIn("Do not use screenshot pixel coordinates", function["description"])
        self.assertNotIn(
            "pixel coordinate",
            function["parameters"]["properties"]["action"]["description"],
        )
        self.assertEqual(coordinate["items"]["maximum"], 999)
        self.assertIn("normalized coordinates", coordinate["description"])

    def test_loop_guard_requires_same_low_level_action_and_static_screen(self):
        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            loop_action_limit=3,
        )
        code = ("pyautogui.click(500, 500)",)
        agent.action_records = [
            (code, "click the save button", 12345),
            (code, "click the save button", 12345),
            (code, "click the save button", 12345),
        ]

        self.assertTrue(
            agent._is_repeated_action(
                list(code),
                "Click the Save button",
                12345,
            )
        )
        self.assertTrue(
            agent._is_repeated_action(
                list(code),
                "Adjust the brightness slider",
                12345,
            )
        )
        self.assertFalse(
            agent._is_repeated_action(
                list(code),
                "Click the Save button",
                (1 << 256) - 1,
            )
        )

    def test_loop_guard_treats_small_mouse_coordinate_jitter_as_same_action(self):
        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            loop_action_limit=3,
        )
        agent.action_records = [
            (("pyautogui.click(540, 72)",), "click the same button", 12345),
            (("pyautogui.click(540, 71)",), "click the same button", 12345),
            (("pyautogui.click(540, 70)",), "click the same button", 12345),
        ]

        self.assertTrue(
            agent._is_repeated_action(
                ["pyautogui.click(540, 69)"],
                "Click the same button",
                12345,
            )
        )
        self.assertFalse(
            agent._is_repeated_action(
                ["pyautogui.click(580, 69)"],
                "Click a different control",
                12345,
            )
        )

    def test_loop_guard_detects_alternating_cycle_on_static_screen(self):
        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            loop_action_limit=3,
        )
        click_delete = ("pyautogui.click(1176, 799)",)
        click_cancel = ("pyautogui.click(1085, 799)",)
        agent.action_records = [
            (click_delete, "click delete data", 12345),
            (click_cancel, "click cancel", 12345),
            (click_delete, "click delete data", 12345),
        ]

        self.assertTrue(
            agent._is_repeated_action(
                list(click_cancel),
                "Click Cancel",
                12345,
            )
        )
        self.assertFalse(
            agent._is_repeated_action(
                ["pyautogui.hotkey(\"alt\", \"left\")"],
                "Navigate back",
                12345,
            )
        )
        self.assertFalse(
            agent._is_repeated_action(
                list(click_cancel),
                "Click Cancel",
                (1 << 256) - 1,
            )
        )

        self.assertEqual(agent._stagnation_context(12345), "")
        agent.action_records.append(
            (click_cancel, "click cancel", 12345)
        )
        context = agent._stagnation_context(12345)
        self.assertIn("action cycle", context)
        self.assertIn("pyautogui.click(1176, 799)", context)
        self.assertIn("pyautogui.click(1085, 799)", context)

    def test_loop_guard_allows_enter_after_text_entry_on_static_screen(self):
        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            loop_action_limit=3,
        )
        press_enter = ('pyautogui.press("enter")',)
        agent.action_records = [
            (press_enter, "execute the first command", 12345),
            (('pyautogui.typewrite("kill -9 3541")',), "type kill", 12345),
            (
                ('pyautogui.typewrite("ps aux | grep libreoffice")',),
                "type process check",
                12345,
            ),
        ]

        self.assertFalse(
            agent._is_repeated_action(
                list(press_enter),
                "Press Enter to execute the command",
                12345,
            )
        )

    def test_malformed_output_yields_before_bounded_failure(self):
        image = Image.new("RGB", (320, 240), "white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        observation = {"screenshot": buffer.getvalue()}

        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            format_failure_limit=3,
        )
        agent.call_llm = lambda payload, model: "Action: Click the Save button."

        _, first = agent.predict("Save the document", observation)
        _, second = agent.predict("Save the document", observation)
        _, third = agent.predict("Save the document", observation)

        self.assertEqual(first, ["WAIT"])
        self.assertEqual(second, ["WAIT"])
        self.assertEqual(third, ["MODEL_ERROR:format"])

    def test_unambiguous_quoted_type_action_is_recovered(self):
        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
        )
        instruction, actions = agent.parse_response(
            "Action: Type 'cd ~/Desktop/project' into the terminal.",
            original_width=1920,
            original_height=1080,
            processed_width=1920,
            processed_height=1088,
        )

        self.assertIn("Type", instruction)
        self.assertEqual(actions, ['pyautogui.typewrite("cd ~/Desktop/project")'])

    def test_completion_audit_replaces_premature_done_with_next_action(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        responses = iter(
            [
                (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: Open the required application from the dock.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>left_click</parameter>"
                    "<parameter=coordinate>[100, 500]</parameter>"
                    "</function></tool_call>"
                ),
            ]
        )
        agent.call_llm = lambda payload, model: next(responses)

        response, actions = agent.predict("Create and save the requested PDF", observation)

        self.assertIn("Open the required application", response)
        self.assertEqual(actions, ["pyautogui.click(32, 120)"])

    def test_completion_audit_accepts_visibly_confirmed_done(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        responses = iter(
            [
                (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: The latest screen visibly shows the saved PDF on the desktop.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "<parameter=evidence>"
                    "The latest screen shows report.pdf listed on the desktop."
                    "</parameter>"
                    "</function></tool_call>"
                ),
            ]
        )
        captured_payloads = []

        def fake_call(payload, model):
            captured_payloads.append(payload)
            return next(responses)

        agent.call_llm = fake_call

        response, actions = agent.predict("Create and save the requested PDF", observation)

        self.assertIn("visibly shows", response)
        self.assertEqual(actions, ["DONE"])
        audit_messages = captured_payloads[1]["messages"]
        self.assertEqual([message["role"] for message in audit_messages], ["system", "user"])
        self.assertNotIn(
            "Action: The task is completed successfully",
            str(audit_messages),
        )
        self.assertIn("Past action records (read-only", str(audit_messages))
        self.assertNotIn(" | executed: pyautogui", str(audit_messages))

    def test_openrouter_history_and_parser_behavior_remain_unchanged(self):
        remote_agent = QwenAgent(
            model="test",
            base_url="https://openrouter.ai/api/v1",
            observation_type="screenshot",
        )
        remote_agent.action_records = [
            (("pyautogui.click(100, 200)",), "Click Save", 123)
        ]

        label, history = remote_agent._render_executed_action_history()

        self.assertFalse(remote_agent.local_runtime_compat)
        self.assertEqual(label, "Recent actions confirmed as executed")
        self.assertEqual(
            history,
            "- Click Save | executed: pyautogui.click(100, 200)",
        )
        _, actions = remote_agent.parse_response(
            "terminate success The task has completed successfully.",
            original_width=1920,
            original_height=1080,
            processed_width=1920,
            processed_height=1080,
        )
        self.assertEqual(actions, [])
        remote_prompt = remote_agent._build_system_prompt(
            remote_agent._build_tools_def(1920, 1088)
        )
        self.assertNotIn("Local-runtime output contract", remote_prompt)
        self.assertNotIn("| executed: suffix", remote_prompt)

    def test_local_recovery_omits_copied_execution_record(self):
        local_agent = QwenAgent(
            model="test",
            base_url="http://192.168.50.159:1234/v1",
            observation_type="screenshot",
        )
        local_agent.action_records = [
            (("pyautogui.click(100, 200)",), "Click Save", 123)
        ]

        messages = local_agent._build_recovery_messages(
            system_prompt="system",
            instruction="Save the settings",
            processed_b64=None,
            observation_text="latest observation",
            reason="format",
            rejected_response=(
                "Action: Click Save | executed: pyautogui.click(100, 200)"
            ),
        )
        rendered = str(messages)

        self.assertTrue(local_agent.local_runtime_compat)
        local_prompt = local_agent._build_system_prompt(
            local_agent._build_tools_def(1920, 1088)
        )
        self.assertIn("Local-runtime output contract", local_prompt)
        self.assertIn("use `type` after focusing a text field", local_prompt)
        self.assertIn("<executed_action_history>", rendered)
        self.assertIn("copied a past-action record", rendered)
        self.assertNotIn(
            "Action: Click Save | executed: pyautogui.click(100, 200)",
            rendered,
        )

    def test_completion_audit_accepts_concrete_document_evidence(self):
        response = (
            "Action: The document visibly shows the completed replacements.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>terminate</parameter>"
            "<parameter=status>success</parameter>"
            "<parameter=evidence>"
            "The document shows every instance of text replaced with test."
            "</parameter></function></tool_call>"
        )

        self.assertFalse(QwenAgent._completion_response_is_unverified(response))

    def test_completion_audit_accepts_panel_showing_specific_artifact(self):
        response = (
            "Action: The requested folder is open in VS Code.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>terminate</parameter>"
            "<parameter=status>success</parameter>"
            "<parameter=evidence>"
            "The Explorer panel is showing the files from ~/Desktop/project."
            "</parameter></function></tool_call>"
        )

        self.assertFalse(QwenAgent._completion_response_is_unverified(response))

    def test_completion_evidence_recovers_misplaced_parameter_separator(self):
        response = (
            "Action: The requested folder is open in VS Code.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>terminate</parameter>"
            "<parameter=status>success</parameter>"
            "<parameter>evidence>"
            "The Explorer panel is showing the files from ~/Desktop/project."
            "</parameter></function></tool_call>"
        )

        self.assertEqual(
            QwenAgent._completion_tool_evidence(response),
            "The Explorer panel is showing the files from ~/Desktop/project",
        )
        self.assertFalse(QwenAgent._completion_response_is_unverified(response))

    def test_completion_audit_rejects_goal_restatement_as_evidence(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        responses = iter(
            [
                (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "<parameter=evidence>"
                    "The line length for code wrapping is set to 50 characters."
                    "</parameter></function></tool_call>"
                ),
                (
                    "Action: Open the VS Code settings editor.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>key</parameter>"
                    "<parameter=keys>[\"ctrl\", \",\"]</parameter>"
                    "</function></tool_call>"
                ),
            ]
        )
        agent.call_llm = lambda payload, model: next(responses)

        _, actions = agent.predict(
            "Set the line length for code wrapping to 50 characters",
            observation,
        )

        self.assertEqual(actions, ['pyautogui.hotkey("ctrl", ",")'])

    def test_clipboard_completion_can_use_matching_trajectory_support(self):
        observation = self._multimodal_observation()
        signature = image_perceptual_hash(observation["screenshot"])
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        agent.action_records = [
            (
                ('pyautogui.hotkey("ctrl", "c")',),
                "copy the selected file path",
                signature,
            )
        ]
        responses = iter(
            [
                (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: The selected path was copied to the clipboard.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "<parameter=evidence>"
                    "The selected file path was copied to the clipboard."
                    "</parameter></function></tool_call>"
                ),
            ]
        )
        agent.call_llm = lambda payload, model: next(responses)

        _, actions = agent.predict(
            "Find secret.docx and copy its path to the clipboard",
            observation,
        )

        self.assertEqual(actions, ["DONE"])

    def test_clipboard_completion_accepts_copy_location_menu_action(self):
        observation = self._multimodal_observation()
        signature = image_perceptual_hash(observation["screenshot"])
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        agent.action_records = [
            (
                ("pyautogui.click(420, 387)",),
                "click copy location to copy the file path",
                signature,
            )
        ]
        response = (
            "Action: The path was copied to the clipboard.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>terminate</parameter>"
            "<parameter=status>success</parameter>"
            "<parameter=evidence>"
            "The selected file path was copied to the clipboard."
            "</parameter></function></tool_call>"
        )

        self.assertTrue(
            agent._completion_has_trajectory_support(
                "Find secret.docx and copy its path to the clipboard",
                response,
            )
        )

    def test_process_completion_can_use_matching_terminal_trajectory(self):
        observation = self._multimodal_observation()
        signature = image_perceptual_hash(observation["screenshot"])
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        agent.action_records = [
            (
                ('pyautogui.typewrite("kill -9 3541")',),
                "type the kill command",
                signature,
            ),
            (
                ('pyautogui.press("enter")',),
                "execute the kill command",
                signature,
            ),
        ]
        responses = iter(
            [
                (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: The terminal shows that the process is no longer running.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "<parameter=evidence>"
                    "The terminal shows the Writer process is no longer running."
                    "</parameter></function></tool_call>"
                ),
            ]
        )
        agent.call_llm = lambda payload, model: next(responses)

        _, actions = agent.predict(
            "Force quit the frozen LibreOffice Writer from the command line",
            observation,
        )

        self.assertEqual(actions, ["DONE"])

    def test_completion_audit_rejects_bare_success_tool_call(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        responses = iter(
            [
                (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: Inspect the output file list.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>key</parameter>"
                    "<parameter=keys>[\"ctrl\", \"o\"]</parameter>"
                    "</function></tool_call>"
                ),
            ]
        )
        agent.call_llm = lambda payload, model: next(responses)

        _, actions = agent.predict("Create and save the requested PDF", observation)

        self.assertEqual(actions, ['pyautogui.hotkey("ctrl", "o")'])

    def test_native_completion_requires_evidence_argument(self):
        bare = (
            '{"name":"computer_use","arguments":'
            '{"action":"terminate","status":"success"}}'
        )
        evidenced = (
            '{"name":"computer_use","arguments":'
            '{"action":"terminate","status":"success",'
            '"evidence":"The latest screen shows report.pdf in the output folder."}}'
        )

        self.assertTrue(QwenAgent._completion_response_is_unverified(bare))
        self.assertFalse(
            QwenAgent._completion_response_is_unverified(evidenced)
        )

    def test_completion_evidence_cannot_rely_on_pending_action_button(self):
        response = (
            "Action: The task is completed successfully.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>terminate</parameter>"
            "<parameter=status>success</parameter>"
            "<parameter=evidence>"
            "The fields are filled and the Done button is present."
            "</parameter></function></tool_call>"
        )
        self.assertTrue(QwenAgent._completion_response_is_unverified(response))

    def test_static_last_action_does_not_override_concrete_visible_evidence(self):
        observation = self._multimodal_observation()
        signature = image_perceptual_hash(observation["screenshot"])
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        agent.action_records = [
            (
                ("pyautogui.click(200, 120)",),
                "click the done button",
                signature,
            )
        ]
        responses = iter(
            [
                (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: The latest screen visibly shows the completed result.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "<parameter=evidence>"
                    "The latest screen visibly shows the completed result."
                    "</parameter></function></tool_call>"
                ),
            ]
        )
        captured = {}

        def fake_call(payload, model):
            captured["last_payload"] = payload
            return next(responses)

        agent.call_llm = fake_call
        with patch.dict(os.environ, {"OSWORLD_QWEN_FORMAT_RETRIES": "0"}):
            _, actions = agent.predict("Complete the requested task", observation)

        self.assertEqual(actions, ["DONE"])
        self.assertIn(
            "produced no detectable visible screen change",
            str(captured["last_payload"]["messages"]),
        )

    def test_repeated_unverified_completion_preserves_state_without_model_error(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
            format_failure_limit=2,
        )
        generic_done = (
            "Action: The task is completed successfully.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>terminate</parameter>"
            "<parameter=status>success</parameter>"
            "</function></tool_call>"
        )
        agent.call_llm = lambda payload, model: generic_done

        with patch.dict(os.environ, {"OSWORLD_QWEN_FORMAT_RETRIES": "0"}):
            _, first = agent.predict("Create and save the PDF", observation)
            _, second = agent.predict("Create and save the PDF", observation)

        self.assertEqual(first, ["WAIT"])
        self.assertEqual(second, ["WAIT"])
        self.assertEqual(agent.consecutive_completion_failures, 0)
        self.assertIn("Preserve the completed state", agent.pending_recovery_feedback)

    def test_completion_audit_provider_failure_uses_bounded_recovery(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        calls = 0

        def fake_call(payload, model):
            nonlocal calls
            calls += 1
            if calls == 1:
                return (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                )
            raise RuntimeError("provider unavailable")

        agent.call_llm = fake_call
        _, actions = agent.predict("Create and save the PDF", observation)

        self.assertEqual(actions, ["WAIT"])
        self.assertEqual(agent.consecutive_completion_audit_failures, 1)
        self.assertEqual(agent.consecutive_completion_failures, 0)
        self.assertEqual(agent.consecutive_format_failures, 0)
        self.assertIn(
            "independent completion audit was unavailable",
            agent.pending_recovery_feedback,
        )

    def test_repeated_completion_audit_failure_is_bounded_adapter_error(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
            format_failure_limit=2,
        )
        generic_done = (
            "Action: The task is completed successfully.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>terminate</parameter>"
            "<parameter=status>success</parameter>"
            "</function></tool_call>"
        )
        calls = 0

        def fake_call(payload, model):
            nonlocal calls
            calls += 1
            if calls % 2:
                return generic_done
            raise RuntimeError("provider unavailable")

        agent.call_llm = fake_call
        _, first = agent.predict("Create and save the PDF", observation)
        _, second = agent.predict("Create and save the PDF", observation)

        self.assertEqual(first, ["WAIT"])
        self.assertEqual(
            second,
            ["ADAPTER_ERROR:provider_completion_audit"],
        )
        self.assertEqual(agent.consecutive_completion_audit_failures, 2)

    def test_completion_audit_can_be_disabled_for_ablation(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=False,
        )
        call_count = 0

        def fake_call(payload, model):
            nonlocal call_count
            call_count += 1
            return (
                "Action: The task is completed successfully.\n"
                "<tool_call><function=computer_use>"
                "<parameter=action>terminate</parameter>"
                "<parameter=status>success</parameter>"
                "</function></tool_call>"
            )

        agent.call_llm = fake_call
        _, actions = agent.predict("Create and save the requested PDF", observation)

        self.assertEqual(actions, ["DONE"])
        self.assertEqual(call_count, 1)

    def test_local_completion_alias_still_requires_completion_audit(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        responses = iter(
            [
                "Action: terminate=success",
                (
                    "Action: The latest screen visibly shows the completed result.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "<parameter=evidence>"
                    "The latest screen visibly shows the completed result."
                    "</parameter></function></tool_call>"
                ),
            ]
        )
        call_count = 0

        def fake_call(payload, model):
            nonlocal call_count
            call_count += 1
            return next(responses)

        agent.call_llm = fake_call
        _, actions = agent.predict("Complete the requested task", observation)

        self.assertEqual(actions, ["DONE"])
        self.assertEqual(call_count, 2)

    def test_self_contradictory_completion_audit_is_rejected(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        responses = iter(
            [
                (
                    "Action: The task is completed successfully.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: The task is impossible because this is the wrong application.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>terminate</parameter>"
                    "<parameter=status>success</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: Switch to the required application.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>key</parameter>"
                    "<parameter=keys>[\"alt\", \"tab\"]</parameter>"
                    "</function></tool_call>"
                ),
            ]
        )
        agent.call_llm = lambda payload, model: next(responses)

        _, actions = agent.predict("Create and save the requested PDF", observation)

        self.assertEqual(actions, ['pyautogui.hotkey("alt", "tab")'])

    def test_unverified_completion_uses_its_own_failure_counter(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            completion_verification=True,
        )
        generic_done = (
            "Action: The task is completed successfully.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>terminate</parameter>"
            "<parameter=status>success</parameter>"
            "</function></tool_call>"
        )
        agent.call_llm = lambda payload, model: generic_done

        with patch.dict(os.environ, {"OSWORLD_QWEN_FORMAT_RETRIES": "0"}):
            _, actions = agent.predict("Create and save the requested PDF", observation)

        self.assertEqual(actions, ["WAIT"])
        self.assertEqual(agent.consecutive_completion_failures, 1)
        self.assertEqual(agent.consecutive_format_failures, 0)
        self.assertEqual(agent.consecutive_stagnation_failures, 0)

    def test_static_repetition_is_exposed_in_next_prompt(self):
        observation = self._multimodal_observation()
        signature = image_perceptual_hash(observation["screenshot"])
        repeated_code = ("pyautogui.click(32, 120)",)
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            loop_action_limit=3,
        )
        agent.action_records = [
            (repeated_code, "click the dock icon", signature),
            (repeated_code, "click the dock icon", signature),
            (repeated_code, "click the dock icon", signature),
        ]
        captured = {}

        def fake_call(payload, model):
            captured["payload"] = payload
            return (
                "Action: Switch applications with the keyboard.\n"
                "<tool_call><function=computer_use>"
                "<parameter=action>key</parameter>"
                "<parameter=keys>[\"alt\", \"tab\"]</parameter>"
                "</function></tool_call>"
            )

        agent.call_llm = fake_call
        _, actions = agent.predict("Open the presentation application", observation)

        rendered = str(captured["payload"]["messages"])
        self.assertIn("STAGNATION WARNING", rendered)
        self.assertIn("pyautogui.click(32, 120)", rendered)
        self.assertEqual(actions, ['pyautogui.hotkey("alt", "tab")'])

    def test_rejected_action_feedback_survives_into_next_observation(self):
        observation = self._multimodal_observation()
        signature = image_perceptual_hash(observation["screenshot"])
        repeated_code = ("pyautogui.click(32, 120)",)
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            loop_action_limit=2,
        )
        agent.action_records = [
            (repeated_code, "click the dock icon", signature),
            (repeated_code, "click the dock icon", signature),
        ]
        repeated_response = (
            "Action: Click the dock icon.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>left_click</parameter>"
            "<parameter=coordinate>[100, 500]</parameter>"
            "</function></tool_call>"
        )
        captured = {}

        with patch.dict(os.environ, {"OSWORLD_QWEN_FORMAT_RETRIES": "0"}):
            agent.call_llm = lambda payload, model: repeated_response
            _, first = agent.predict("Open the application", observation)

            def capture_second(payload, model):
                captured["payload"] = payload
                return (
                    "Action: Switch applications with the keyboard.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>key</parameter>"
                    "<parameter=keys>[\"alt\", \"tab\"]</parameter>"
                    "</function></tool_call>"
                )

            agent.call_llm = capture_second
            _, second = agent.predict("Open the application", observation)

        self.assertEqual(first, ["WAIT"])
        self.assertEqual(second, ['pyautogui.hotkey("alt", "tab")'])
        rendered = str(captured["payload"]["messages"])
        self.assertIn("preceding proposed action was NOT executed", rendered)
        self.assertIn("pyautogui.click(32, 120)", rendered)

    def test_stagnation_retry_uses_fresh_recovery_context(self):
        observation = self._multimodal_observation()
        signature = image_perceptual_hash(observation["screenshot"])
        repeated_code = ("pyautogui.click(32, 120)",)
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
            loop_action_limit=2,
        )
        agent.action_records = [
            (repeated_code, "click the dock icon", signature),
            (repeated_code, "click the dock icon", signature),
        ]
        responses = iter(
            [
                (
                    "Action: Click the dock icon.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>left_click</parameter>"
                    "<parameter=coordinate>[100, 500]</parameter>"
                    "</function></tool_call>"
                ),
                (
                    "Action: Switch applications with the keyboard.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>key</parameter>"
                    "<parameter=keys>[\"alt\", \"tab\"]</parameter>"
                    "</function></tool_call>"
                ),
            ]
        )
        payloads = []

        def fake_call(payload, model):
            payloads.append(payload)
            return next(responses)

        agent.call_llm = fake_call
        with patch.dict(os.environ, {"OSWORLD_QWEN_FORMAT_RETRIES": "1"}):
            _, actions = agent.predict("Open the application", observation)

        self.assertEqual(actions, ['pyautogui.hotkey("alt", "tab")'])
        recovery_messages = payloads[1]["messages"]
        self.assertEqual(
            [message["role"] for message in recovery_messages],
            ["system", "user"],
        )
        rendered = str(recovery_messages)
        self.assertIn("Independent recovery", rendered)
        self.assertIn("STAGNATION RECOVERY", rendered)
        self.assertIn("Exact task: Open the application", rendered)
        self.assertIn("pyautogui.click(32, 120)", rendered)

    def test_format_retry_uses_fresh_recovery_context(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=512,
            observation_type="screenshot",
        )
        responses = iter(
            [
                "Action: Click the Utilities folder.",
                (
                    "Action: Open the Utilities folder.\n"
                    "<tool_call><function=computer_use>"
                    "<parameter=action>left_click</parameter>"
                    "<parameter=coordinate>[500, 500]</parameter>"
                    "</function></tool_call>"
                ),
            ]
        )
        payloads = []

        def fake_call(payload, model):
            payloads.append(payload)
            return next(responses)

        agent.call_llm = fake_call
        with patch.dict(os.environ, {"OSWORLD_QWEN_FORMAT_RETRIES": "1"}):
            _, actions = agent.predict("Open Utilities", observation)

        self.assertEqual(actions, ["pyautogui.click(160, 120)"])
        recovery_messages = payloads[1]["messages"]
        self.assertEqual(
            [message["role"] for message in recovery_messages],
            ["system", "user"],
        )
        rendered = str(recovery_messages)
        self.assertIn("Independent recovery", rendered)
        self.assertIn("FORMAT RECOVERY", rendered)
        self.assertIn("Action: Click the Utilities folder.", rendered)

    def test_stagnation_and_format_failures_use_separate_counters(self):
        image = Image.new("RGB", (320, 240), "white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        screenshot = buffer.getvalue()
        observation = {"screenshot": screenshot}

        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            loop_action_limit=2,
            format_failure_limit=2,
        )
        repeated_response = (
            "Action: Click the Save button.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>left_click</parameter>"
            "<parameter=coordinate>[500, 500]</parameter>"
            "</function></tool_call>"
        )
        signature = image_perceptual_hash(screenshot)
        code = ("pyautogui.click(160, 120)",)
        agent.action_records = [
            (code, "click the save button", signature),
            (code, "click the save button", signature),
        ]
        agent.call_llm = lambda payload, model: repeated_response

        with patch.dict(os.environ, {"OSWORLD_QWEN_FORMAT_RETRIES": "0"}):
            _, stagnant = agent.predict("Save the document", observation)

        self.assertEqual(stagnant, ["WAIT"])
        self.assertEqual(agent.consecutive_stagnation_failures, 1)
        self.assertEqual(agent.consecutive_format_failures, 0)

        agent.call_llm = lambda payload, model: "Action: Click the Save button."
        with patch.dict(os.environ, {"OSWORLD_QWEN_FORMAT_RETRIES": "0"}):
            _, malformed = agent.predict("Save the document", observation)

        self.assertEqual(malformed, ["WAIT"])
        self.assertEqual(agent.consecutive_stagnation_failures, 0)
        self.assertEqual(agent.consecutive_format_failures, 1)

    def test_action_contract_mismatch_is_not_silently_executed(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=256,
            observation_type="screenshot",
            format_failure_limit=2,
        )
        response = (
            "Action: Type the URL and press Enter.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>type</parameter>"
            "<parameter=text>https://example.com</parameter>"
            "</function></tool_call>"
        )
        agent.call_llm = lambda payload, model: response

        with patch.dict(os.environ, {"OSWORLD_QWEN_FORMAT_RETRIES": "0"}):
            _, first = agent.predict("Open the website", observation)
            _, second = agent.predict("Open the website", observation)

        self.assertEqual(first, ["WAIT"])
        self.assertEqual(second, ["MODEL_ERROR:action_contract"])

    def test_native_tools_are_sent_as_structured_api_schema(self):
        observation = self._multimodal_observation()
        agent = QwenAgent(
            model="test",
            max_tokens=256,
            observation_type="screenshot",
            native_tools=True,
            completion_verification=False,
        )
        captured = {}

        def fake_call(payload, model):
            captured["payload"] = payload
            return (
                '{"name":"computer_use","arguments":'
                '{"action":"left_click","coordinate":[500,500]}}'
            )

        agent.call_llm = fake_call
        _, actions = agent.predict("Click Save", observation)

        self.assertEqual(actions, ["pyautogui.click(160, 120)"])
        self.assertEqual(
            captured["payload"]["tools"][0]["function"]["name"],
            "computer_use",
        )
        self.assertEqual(captured["payload"]["tool_choice"], "auto")
        self.assertNotIn("<tool_call>", str(captured["payload"]["messages"][0]))

    def test_thought_prefix_adds_deliberation_instruction(self):
        agent = QwenAgent(
            model="test",
            max_tokens=64,
            observation_type="screenshot",
            add_thought_prefix=True,
        )
        prompt = agent._build_system_prompt(agent._build_tools_def(1920, 1088))
        self.assertIn("<think>...</think>", prompt)
        self.assertIn("launch a visible Terminal", prompt)
        self.assertIn("Being in the wrong application is a navigation problem", prompt)

    def test_all_four_observation_modes_build_runnable_payloads(self):
        response = (
            "Action: Wait for the UI.\n"
            "<tool_call><function=computer_use>"
            "<parameter=action>wait</parameter>"
            "<parameter=time>1</parameter>"
            "</function></tool_call>"
        )
        for observation_type in [
            "screenshot",
            "a11y_tree",
            "screenshot_a11y_tree",
            "som",
        ]:
            with self.subTest(observation_type=observation_type):
                agent = QwenAgent(
                    model="test",
                    max_tokens=64,
                    observation_type=observation_type,
                )
                captured = {}

                def fake_call(payload, model):
                    captured["payload"] = payload
                    return response

                agent.call_llm = fake_call
                _, actions = agent.predict(
                    "Click Save",
                    self._multimodal_observation(),
                )
                self.assertEqual(actions, ["WAIT"])
                rendered = str(captured["payload"]["messages"])
                has_image = "data:image/png;base64," in rendered
                has_tree = (
                    "Accessibility tree" in rendered
                    or "index\\ttag\\tname\\ttext" in rendered
                )
                self.assertEqual(
                    has_image,
                    observation_type != "a11y_tree",
                )
                self.assertEqual(
                    has_tree,
                    observation_type != "screenshot",
                )


if __name__ == "__main__":
    unittest.main()
