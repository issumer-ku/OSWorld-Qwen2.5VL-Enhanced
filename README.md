# OSWorld Qwen2.5-VL Enhanced

An independently packaged, provider-neutral Qwen2.5-VL computer-use agent for
[OSWorld](https://github.com/xlang-ai/OSWorld). It preserves the enhanced adapter
used in our experiments while keeping the OSWorld checkout and the model runtime
replaceable.

The repository contains no model weights and does not fork OSWorld. Point the
runner at any compatible OSWorld checkout with `--osworld-root`, and point the
agent at LM Studio, vLLM, or another OpenAI-compatible multimodal endpoint.

## What is enhanced

- grounded XML or native structured tool calls with strict action parsing
- relative/absolute coordinate normalization and provider convention detection
- screenshot folding plus bounded multimodal and accessibility history
- semantic loop detection, format recovery, and completion verification
- explicit model/adapter failure provenance instead of silently scoring failures
- resumable multi-environment OSWorld evaluation with isolated result artifacts

## Install

```bash
git clone https://github.com/issumer-ku/OSWorld-Qwen2.5VL-Enhanced.git
cd OSWorld-Qwen2.5VL-Enhanced
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install OSWorld separately following its Docker provider instructions. The model
server and OSWorld environments can be on different machines.

## LM Studio / MLX

Load the Qwen2.5-VL model in LM Studio, enable its OpenAI-compatible server, and
allow enough parallel requests for the selected `--num_envs`. Then run:

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

Start with a 1–3 task smoke manifest before the 369-task run. Increase
`--num_envs` only after confirming the endpoint sustains that many simultaneous
multimodal generations without malformed responses or severe latency growth.

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

See [docs/evaluation.md](docs/evaluation.md) for the full comparison protocol and
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

## Reproducibility

For a fair baseline comparison, keep the OSWorld commit, 369-task manifest,
Docker image, screen size, maximum steps, decoding parameters, observation type,
and environment concurrency fixed. Record model quantization and server version
alongside each run. The runner writes redacted arguments, trajectories, recording,
termination provenance, and evaluator results under the selected result directory.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
python -m build
```

## License and upstream projects

Apache-2.0. This project does not redistribute Qwen model weights. Also review the
licenses and citation requirements of [OSWorld](https://github.com/xlang-ai/OSWorld)
and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).
