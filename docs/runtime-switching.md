# Switching model runtimes

The agent speaks the OpenAI chat-completions multimodal protocol. Runtime-specific
configuration is limited to the endpoint, API key, served model name, and server
capacity.

## LM Studio

Use the exact identifier exposed by `GET /v1/models`. If OSWorld runs inside a
container, `127.0.0.1` refers to that container; use the reachable host address or
`host.docker.internal` where supported.

For MLX quantized models, record the quantization level and LM Studio build. Test
one and two concurrent requests against identical prompts before selecting
`--num_envs`.

## Lab GPU server

Serve the official safetensors with an OpenAI-compatible runtime such as vLLM.
Expose the port only to the trusted lab network, and use authentication or a
reverse proxy when the endpoint is reachable outside it.

Set `--model` to the server's served model name, not necessarily the Hugging Face
repository path. Probe `/v1/models` and one image request before launching OSWorld.

## Capacity rule

`--num_envs` controls concurrent OSWorld workers and therefore roughly bounds
simultaneous model requests. Choose it from measured server throughput and memory,
not GPU utilization alone. For a 7B VLM, begin with two workers, examine malformed
output and latency, then increase only if quality and stability remain unchanged.
