# Jev Ultrafast

A browser agent that chooses instead of generating. Each step atomically observes the visible page,
the decision model picks an operation and an observed element in one request, and code executes it.
A small LLM runs only when the chosen operation is `TYPE_TEXT`. Model output never becomes
selectors, coordinates, or executable code.

## Installation

```sh
uv sync
cp .env.example .env
```

Chrome connects through [browser-harness](https://github.com/browser-use/browser-harness); run
`uv run browser-harness --doctor` if it needs connecting. `.env` selects the providers (see below).

## Quick Start

```python
from source import Agent

with Agent("https://en.wikipedia.org/wiki/Main_Page",
           "Find and open the article about Gödel's incompleteness theorems.") as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
```

`uv run pytest -m integration` drives realistic fixture tasks end to end and verifies each
actual outcome — a `DONE` choice is not proof.

## Providers

Decisions and text are configured independently. `DECISION_MODEL` and `TEXT_MODEL` are required;
`.env.example` carries starting values.

| `DECISION_PROVIDER` | Backend |
| --- | --- |
| `typesafe` | TypeSafe System One |
| `zen` | OpenCode Zen System One (`/v1/systemone`) |
| `openrouter` | OpenRouter decisions (`/api/alpha/decisions`) |
| `cygnet` | local llama.cpp letter choice (completions style) |
| `decider` | local llama.cpp trained typed options (systemone style) |

| `TEXT_PROVIDER` | Backend |
| --- | --- |
| `zen` | OpenCode Zen `/v1/chat/completions` |
| `openrouter` | OpenRouter `/api/v1` |
| `local` | OpenAI-compatible endpoint via `TEXT_BASE_URL` |

Free Zen models and loopback endpoints need no key.

## Local models

`uv run python scripts/serve.py <stack.yaml>` starts one llama-server per declared model (plus an
optional debug Chrome), waits for health, and prints the matching `DECISION_*`/`TEXT_*` env lines.
Stack entries take `path`, `name`, `role` (`decide`/`chat`), `port`, `gpu_layers`,
`context_tokens`, `kv_quant`, `swa_full`.

## Development

```sh
uv run ruff check .
uv run pytest
node --check source/snapshot.js
uv build
```

Unit tests are offline. `uv run pytest` also runs the live integration tasks when `.env` and
Chrome are present: four increasingly complex fixture goals through the configured providers,
each verified against the final page rather than the model's `DONE`. `uv run pytest -m "not
integration"` stays offline. `uv run python scripts/check_guards.py` exercises real controls in
a local browser without model calls.

## License

MIT
