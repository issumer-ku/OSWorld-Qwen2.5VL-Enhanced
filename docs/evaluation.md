# OSWorld evaluation

## Smoke test first

Create a small OSWorld manifest with one to three representative tasks and pass it
through `--test_all_meta_path`. Confirm that every task produces `traj.jsonl`,
`termination.json`, `recording.mp4`, and `result.txt` before starting all 369 tasks.

## Full run

The command resumes automatically: a task with `result.txt` is considered complete.
Caught setup, controller, and API exceptions are archived and retried according to
`--task_retries`.

Important defaults are `--max_steps 15`, `--history_n 6`,
`--multimodal_history_n 3`, `--coord relative`, deterministic temperature, and
completion verification enabled. Explicitly pass any values that must match an
older experiment.

## Comparison checklist

Keep these fixed between Qwen variants:

- OSWorld commit, manifest, examples, Docker image, and evaluator
- screen resolution, observation type, maximum steps, and reset/evaluation waits
- temperature, top-p, maximum output tokens, history limits, and tool mode
- number of environments and model-server concurrency

Report separately:

- OSWorld score and successful task count
- model errors, adapter errors, infeasible terminations, and max-step terminations
- median task duration and model-call latency
- quantization, runtime, accelerator, and peak memory

Do not compare a two-environment quantized run directly with a one-environment
safetensors run without labeling that systems difference.
