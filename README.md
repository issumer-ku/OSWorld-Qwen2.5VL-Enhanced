# OSWorld Qwen2.5-VL Enhanced

An independently packaged, OpenAI-compatible-runtime-flexible Qwen2.5-VL
computer-use agent for [OSWorld](https://github.com/xlang-ai/OSWorld). It
preserves the enhanced adapter used in our experiments while allowing the
OSWorld checkout and model runtime to be managed separately.

This repository contains no model weights and does not vendor a full OSWorld
checkout. The runner loads an external OSWorld installation through
`--osworld-root`, while the agent connects to LM Studio, vLLM, or another
OpenAI-compatible multimodal chat-completions endpoint.

This is an independent research project, not an official OSWorld component.
Parts of the adapter and evaluation helpers are derived from and modified from
Apache-2.0-licensed OSWorld code; see [NOTICE](NOTICE).

## What is enhanced

- grounded XML and optional native structured tool calls with validated parsing
  and bounded format recovery
- relative/absolute coordinate normalization and provider convention detection
- screenshot folding plus bounded multimodal and accessibility history
- semantic loop detection and completion verification
- explicit model/adapter failure provenance instead of silently scoring failures
- resumable multi-environment OSWorld evaluation with per-task result artifacts
- four observation modes: `screenshot`, `a11y_tree`,
  `screenshot_a11y_tree`, and `som`

## Requirements

- Python 3.10 or later
- a separately installed OSWorld checkout and the prerequisites for the selected
  OSWorld provider, such as Docker, VMware, AWS, Azure, or VirtualBox
- an OpenAI-compatible endpoint that accepts multimodal chat-completions requests
  with image input
- a Qwen2.5-VL model whose served model identifier is accepted by that endpoint

Accessibility-tree and SOM observations also require a compatible OSWorld
environment that returns `accessibility_tree` observations.

## Install

```bash
git clone https://github.com/issumer-ku/OSWorld-Qwen2.5VL-Enhanced.git
cd OSWorld-Qwen2.5VL-Enhanced
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

Install OSWorld separately by following its setup guide and the instructions for
the provider you intend to use. The model server and OSWorld environments may run
on different machines as long as the evaluation host can reach the model endpoint.

## LM Studio / MLX

Load a Qwen2.5-VL model in LM Studio, enable its OpenAI-compatible server, and
allow enough concurrent requests for the selected `--num_envs`. Use the exact
model identifier returned by the server's `/v1/models` endpoint.

```bash
osworld-qwen25vl \
  --osworld-root /home/user/projects/OSWorld \
  --provider_name docker \
  --base_url http://MODEL_HOST:1234/v1 \
  --api_key lm-studio \
  --model qwen2.5-vl-7b \
  --observation_type screenshot \
  --coord relative \
  --num_envs 2 \
  --test_all_meta_path /home/user/projects/OSWorld/evaluation_examples/test_nogdrive.json \
  --result_dir ./results/qwen25vl-enhanced
```

`test_nogdrive.json` contains 361 tasks and excludes eight Google Drive tasks.
Use `test_all.json` with the required Google credentials and setup for the full
369-task evaluation.

Start with a one-to-three-task smoke manifest. Increase `--num_envs` only after
confirming that the endpoint sustains the same number of simultaneous multimodal
generations without malformed responses or severe latency growth.

## Research-lab server / vLLM

On the GPU server, install a recent vLLM release that supports Qwen2.5-VL and run:

```bash
MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
SERVED_MODEL_NAME=qwen2.5-vl-7b-instruct \
TENSOR_PARALLEL_SIZE=1 \
./scripts/serve_vllm.sh
```

Only the endpoint arguments change on the evaluation host:

```bash
osworld-qwen25vl \
  --osworld-root /home/user/projects/OSWorld \
  --provider_name docker \
  --base_url http://LAB_SERVER_IP:8000/v1 \
  --api_key EMPTY \
  --model qwen2.5-vl-7b-instruct \
  --num_envs 2
```

For authenticated remote endpoints, prefer setting `OPENAI_API_KEY` in the
environment instead of placing a real key in a command-line argument.

See [docs/evaluation.md](docs/evaluation.md) for the comparison protocol and
[docs/runtime-switching.md](docs/runtime-switching.md) for endpoint guidance.

## Python API

```python
from osworld_qwen25vl import QwenAgent

agent = QwenAgent(
    model="qwen2.5-vl-7b-instruct",
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    observation_type="screenshot",
    coordinate_type="relative",
)
```

## OSWorld compatibility

The runner uses OSWorld's `DesktopEnv` and provider interfaces from the checkout
given by `--osworld-root`. Packaging and CLI integration were verified against an
enhanced OSWorld checkout based on commit
[`b7db4d8`](https://github.com/xlang-ai/OSWorld/commit/b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf).
Other revisions may work, but should be validated with a smoke test before a full
evaluation.

## Reproducibility

For a fair comparison, keep the following fixed and report them with the result:

- OSWorld commit, task manifest, examples, evaluator, and environment image
- model checkpoint, precision or quantization, and serving runtime version
- screen size, maximum steps, reset/evaluation waits, and observation type
- temperature, top-p, maximum output tokens, history limits, and tool mode
- number of OSWorld environments and model-server concurrency

The runner writes redacted arguments, trajectories, termination provenance,
evaluator results, and recordings when the selected OSWorld environment supports
recording. A task with `result.txt` is treated as complete when a run is resumed.

## Tests and validation scope

```bash
pip install -e '.[dev]'
pytest -q
python -m build
```

The automated tests validate parsing, coordinate handling, history management,
all four observation payloads, and failure handling. CI also builds the package
on Python 3.10 and 3.11. It does not launch a real OSWorld VM/container or model
server. Validate each OSWorld/provider/model-server combination with a small
end-to-end smoke run before starting a benchmark-scale evaluation.

## License and upstream projects

Apache-2.0. This project does not redistribute Qwen model weights. Review the
licenses and citation requirements of [OSWorld](https://github.com/xlang-ai/OSWorld)
and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) before redistribution or
publication.
