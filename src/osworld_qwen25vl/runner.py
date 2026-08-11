"""Standalone multi-environment OSWorld runner for enhanced Qwen2.5-VL."""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import logging
import os
import queue
import shutil
import signal
import sys
import time
import traceback
from multiprocessing import Manager, Process, current_process
from pathlib import Path
from typing import Any

from .agent import QwenAgent
from .protocol import decode_adapter_error_action, decode_model_error_action


logger = logging.getLogger("osworld_qwen25vl.runner")
processes: list[Process] = []
is_terminating = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate enhanced Qwen2.5-VL against an external OSWorld checkout."
    )
    parser.add_argument("--osworld-root", required=True, help="Path to an OSWorld checkout")
    parser.add_argument("--path_to_vm", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--action_space", default="pyautogui", choices=["pyautogui"])
    parser.add_argument(
        "--observation_type", default="screenshot",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
    )
    parser.add_argument("--sleep_after_execution", type=float, default=3.0)
    parser.add_argument("--reset_wait", type=float, default=60.0)
    parser.add_argument("--evaluation_wait", type=float, default=20.0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--test_config_base_dir", default=None)
    parser.add_argument("--examples_subdir", default="examples")
    parser.add_argument("--test_all_meta_path", default=None)
    parser.add_argument("--domain", default="all")

    parser.add_argument("--model", default="qwen2.5-vl-7b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--history_n", type=int, default=6)
    parser.add_argument("--multimodal_history_n", type=int, default=3)
    parser.add_argument("--a11y_history_n", type=int, default=1)
    parser.add_argument("--a11y_tree_max_tokens", type=int, default=3500)
    parser.add_argument("--coord", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--image_max", type=int, default=6)
    parser.add_argument("--fold_size", type=int, default=3)
    parser.add_argument("--add_thought_prefix", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--native_tools", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loop_action_limit", type=int, default=3)
    parser.add_argument("--format_failure_limit", type=int, default=3)
    parser.add_argument(
        "--completion_verification",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)

    parser.add_argument("--result_dir", default="./results")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--simple_path", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--task_retries", type=int, default=1)
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--provider_name",
        default="docker",
        choices=["aws", "virtualbox", "vmware", "docker", "azure"],
    )
    parser.add_argument("--client_password", default="")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--use_public_ip", action="store_true")
    parser.add_argument(
        "--enable_proxy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    invocation_dir = Path.cwd()
    root = Path(args.osworld_root).expanduser().resolve()
    if not (root / "desktop_env").is_dir() or not (root / "evaluation_examples").is_dir():
        raise ValueError(f"Not an OSWorld checkout: {root}")
    args.osworld_root = str(root)
    args.result_dir = str((invocation_dir / args.result_dir).resolve())
    args.log_dir = str((invocation_dir / args.log_dir).resolve())
    args.test_config_base_dir = str(
        Path(args.test_config_base_dir).expanduser().resolve()
        if args.test_config_base_dir
        else root / "evaluation_examples"
    )
    args.test_all_meta_path = str(
        Path(args.test_all_meta_path).expanduser().resolve()
        if args.test_all_meta_path
        else root / "evaluation_examples" / "test_nogdrive.json"
    )
    if args.num_envs < 1:
        raise ValueError("--num_envs must be >= 1")
    return args


def _configure_logging(args: argparse.Namespace) -> None:
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.log_level))
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(name)s/%(lineno)d-%(processName)s] %(message)s"
    )
    file_handler = logging.FileHandler(
        Path(args.log_dir) / f"qwen25vl-{datetime.datetime.now():%Y%m%d@%H%M%S}.log",
        encoding="utf-8",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(getattr(logging, args.log_level))
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def _load_desktop_env(osworld_root: str):
    if osworld_root not in sys.path:
        sys.path.insert(0, osworld_root)
    os.chdir(osworld_root)
    from desktop_env.desktop_env import DesktopEnv

    return DesktopEnv


def _result_dir(args: argparse.Namespace, domain: str, example_id: str) -> str:
    if args.simple_path:
        return os.path.join(args.result_dir, domain, example_id)
    return os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        domain,
        example_id,
    )


def _config_path(args: argparse.Namespace, domain: str, example_id: str) -> str:
    return os.path.join(
        args.test_config_base_dir,
        args.examples_subdir,
        domain,
        f"{example_id}.json",
    )


def _runtime_logger(example: dict[str, Any], result_dir: str) -> logging.Logger:
    task_logger = logging.getLogger(f"osworld_qwen25vl.task.{example['id']}")
    task_logger.setLevel(logging.DEBUG)
    task_logger.handlers.clear()
    task_logger.addHandler(logging.FileHandler(os.path.join(result_dir, "runtime.log")))
    return task_logger


def _archive_failed_attempt(result_dir: str, attempt: int) -> str:
    attempts_dir = os.path.join(result_dir, "attempts")
    os.makedirs(attempts_dir, exist_ok=True)
    archive = os.path.join(attempts_dir, f"attempt_{attempt}")
    suffix = 2
    while os.path.exists(archive):
        archive = os.path.join(attempts_dir, f"attempt_{attempt}_{suffix}")
        suffix += 1
    os.makedirs(archive)
    for entry in os.scandir(result_dir):
        if entry.name != "attempts":
            shutil.move(entry.path, os.path.join(archive, entry.name))
    return archive


def _run_single(
    agent: QwenAgent,
    env,
    example: dict[str, Any],
    args: argparse.Namespace,
    result_dir: str,
    scores,
) -> None:
    task_logger = _runtime_logger(example, result_dir)
    env.reset(task_config=example)
    agent.reset(task_logger, vm_ip=getattr(env, "vm_ip", None))
    time.sleep(max(0, args.reset_wait))
    obs = env._get_obs()
    done = False
    step_index = 0
    termination = "max_steps"
    termination_reason = None
    env.controller.start_recording()

    while not done and step_index < args.max_steps:
        response, actions = agent.predict(example["instruction"], obs)
        for action in actions:
            timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            model_error = decode_model_error_action(action)
            adapter_error = decode_adapter_error_action(action)
            if model_error is not None or adapter_error is not None:
                reason = model_error or adapter_error
                reward, done = 0.0, True
                kind = "model_error" if model_error is not None else "adapter_error"
                info = {kind: True, "reason": reason}
                termination, termination_reason = kind, reason
            else:
                obs, reward, done, info = env.step(action, args.sleep_after_execution)
                if action == "DONE":
                    termination = "model_success"
                elif action == "FAIL":
                    termination = "model_infeasible"
            screenshot_name = f"step_{step_index + 1}_{timestamp}.png"
            with open(os.path.join(result_dir, screenshot_name), "wb") as file_obj:
                file_obj.write(obs["screenshot"])
            with open(os.path.join(result_dir, "traj.jsonl"), "a", encoding="utf-8") as file_obj:
                file_obj.write(
                    json.dumps(
                        {
                            "step_num": step_index + 1,
                            "action_timestamp": timestamp,
                            "action": action,
                            "response": response,
                            "reward": reward,
                            "done": done,
                            "info": info,
                            "termination_provenance": termination if done else None,
                            "termination_reason": termination_reason if done else None,
                            "screenshot_file": screenshot_name,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
            if done:
                break
        step_index += 1

    if not done:
        termination = "max_steps"
    time.sleep(max(0, args.evaluation_wait))
    result = float(env.evaluate())
    scores.append(result)
    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as file_obj:
        file_obj.write(f"{result}\n")
    with open(os.path.join(result_dir, "termination.json"), "w", encoding="utf-8") as file_obj:
        json.dump(
            {"provenance": termination, "reason": termination_reason, "step_count": step_index},
            file_obj,
            ensure_ascii=False,
            indent=2,
        )
        file_obj.write("\n")
    env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))


def _worker(task_queue, args: argparse.Namespace, scores) -> None:
    env = None
    try:
        DesktopEnv = _load_desktop_env(args.osworld_root)
        snapshot_name = "init_state"
        screen_size = (args.screen_width, args.screen_height)
        if args.provider_name == "aws":
            from desktop_env.providers.aws.manager import IMAGE_ID_MAP

            snapshot_name = IMAGE_ID_MAP[args.region].get(
                screen_size,
                IMAGE_ID_MAP[args.region][(1920, 1080)],
            )
        env_kwargs = {
            "path_to_vm": args.path_to_vm,
            "action_space": args.action_space,
            "provider_name": args.provider_name,
            "region": args.region,
            "snapshot_name": snapshot_name,
            "screen_size": screen_size,
            "headless": args.headless,
            "os_type": "Ubuntu",
            "require_a11y_tree": args.observation_type
            in {"a11y_tree", "screenshot_a11y_tree", "som"},
            "enable_proxy": args.enable_proxy,
            "client_password": args.client_password,
        }
        if "use_public_ip" in inspect.signature(DesktopEnv).parameters:
            env_kwargs["use_public_ip"] = args.use_public_ip
        env = DesktopEnv(**env_kwargs)
        agent = QwenAgent(
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            action_space=args.action_space,
            observation_type=args.observation_type,
            history_n=args.history_n,
            multimodal_history_n=args.multimodal_history_n,
            a11y_history_n=args.a11y_history_n,
            a11y_tree_max_tokens=args.a11y_tree_max_tokens,
            coordinate_type=args.coord,
            image_max=args.image_max,
            fold_size=args.fold_size,
            add_thought_prefix=args.add_thought_prefix,
            enable_thinking=args.enable_thinking,
            native_tools=args.native_tools,
            base_url=args.base_url,
            api_key=args.api_key,
            loop_action_limit=args.loop_action_limit,
            format_failure_limit=args.format_failure_limit,
            completion_verification=args.completion_verification,
        )
        while True:
            try:
                domain, example_id, attempt = task_queue.get(timeout=5)
            except queue.Empty:
                break
            result_dir = _result_dir(args, domain, example_id)
            os.makedirs(result_dir, exist_ok=True)
            try:
                with open(_config_path(args, domain, example_id), encoding="utf-8") as file_obj:
                    example = json.load(file_obj)
                logger.info(
                    "[%s] %s/%s: %s",
                    current_process().name,
                    domain,
                    example_id,
                    example["instruction"],
                )
                _run_single(agent, env, example, args, result_dir, scores)
            except Exception as exc:
                logger.error("Task %s/%s failed: %s\n%s", domain, example_id, exc, traceback.format_exc())
                try:
                    env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
                except Exception:
                    pass
                with open(os.path.join(result_dir, "traj.jsonl"), "a", encoding="utf-8") as file_obj:
                    file_obj.write(json.dumps({"error": str(exc), "attempt": attempt + 1}) + "\n")
                if attempt < max(0, args.task_retries):
                    _archive_failed_attempt(result_dir, attempt + 1)
                    task_queue.put((domain, example_id, attempt + 1))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _unfinished(args: argparse.Namespace, manifest: dict[str, list[str]]) -> dict[str, list[str]]:
    target = (
        args.result_dir
        if args.simple_path
        else os.path.join(args.result_dir, args.action_space, args.observation_type, args.model)
    )
    if not os.path.isdir(target):
        return manifest
    remaining: dict[str, list[str]] = {}
    for domain, ids in manifest.items():
        left = [
            example_id
            for example_id in ids
            if not os.path.isfile(os.path.join(target, domain, example_id, "result.txt"))
        ]
        if left:
            remaining[domain] = left
    return remaining


def _run_workers(args: argparse.Namespace, manifest: dict[str, list[str]]) -> None:
    global processes
    tasks = [(domain, example_id, 0) for domain, ids in manifest.items() for example_id in ids]
    logger.info("Starting %d tasks with %d environment(s)", len(tasks), args.num_envs)
    if not tasks:
        return
    with Manager() as manager:
        scores = manager.list()
        task_queue = manager.Queue()
        for task in tasks:
            task_queue.put(task)
        processes = []
        for index in range(args.num_envs):
            process = Process(
                target=_worker,
                args=(task_queue, args, scores),
                name=f"EnvProcess-{index + 1}",
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        values = list(scores)
        logger.info("Completed %d tasks; mean score %.6f", len(values), sum(values) / len(values) if values else 0)


def _signal_handler(signum, _frame) -> None:
    global is_terminating
    if is_terminating:
        return
    is_terminating = True
    logger.warning("Received signal %s; terminating workers", signum)
    for process in processes:
        if process.is_alive():
            process.terminate()
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> None:
    args = _normalize_args(_parser().parse_args(argv))
    _configure_logging(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    with open(args.test_all_meta_path, encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    if args.domain != "all":
        if args.domain not in manifest:
            raise KeyError(f"Domain not present in manifest: {args.domain}")
        manifest = {args.domain: manifest[args.domain]}
    manifest = _unfinished(args, manifest)

    args_dir = (
        args.result_dir
        if args.simple_path
        else os.path.join(args.result_dir, args.action_space, args.observation_type, args.model)
    )
    os.makedirs(args_dir, exist_ok=True)
    serialized = dict(vars(args))
    if serialized.get("api_key"):
        serialized["api_key"] = "<redacted>"
    with open(os.path.join(args_dir, "args.json"), "w", encoding="utf-8") as file_obj:
        json.dump(serialized, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    logger.info("Remaining tasks: %s", {key: len(value) for key, value in manifest.items()})
    _run_workers(args, manifest)


if __name__ == "__main__":
    main()
