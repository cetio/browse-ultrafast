"""Remote backends: System One decision endpoints and OpenAI-compatible text helpers."""

import time
from urllib.parse import urlparse

import httpx

CLIENT = httpx.Client(http2=True, timeout=25)


def post_json(url, key, body):
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers={"Authorization": f"Bearer {key}"} if key else {})
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


class SystemOne:
    """A TypeSafe wire-format endpoint: state plus typed questions in, probability answers out.

    Covers api.typesafe.ai, OpenCode Zen's /v1/systemone, and OpenRouter's alpha decisions
    endpoint; they differ only in base URL, key, and model name.
    """

    def __init__(self, url, key, model):
        self.url, self.key, self.model = url, key, model

    def decide(self, state, questions):
        return post_json(self.url, self.key, {"model": self.model, "state": state, "questions": questions})


def loopback(url):
    return urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"}


class ChatText:
    """An OpenAI-compatible /chat/completions helper that returns one message's content."""

    def __init__(self, base, key, model, *, reasoning=None, json_mode=True):
        self.base, self.key, self.model = base.rstrip("/"), key, model
        self.reasoning = {} if reasoning is None else {"reasoning": reasoning}
        self.json_mode = {"response_format": {"type": "json_object"}} if json_mode else {}
        # A local helper extracts a field value; greedy decoding keeps it deterministic.
        self.sampling = {"temperature": 0} if loopback(self.base) else {}

    def complete(self, system, user):
        started = time.perf_counter()
        result = post_json(
            self.base + "/chat/completions",
            self.key,
            {
                "model": self.model,
                "max_tokens": 1024,
                **self.json_mode,
                **self.reasoning,
                **self.sampling,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError("Text helper returned no message content; nothing typed.") from None
        return content, {
            "model": self.model,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "usage": result.get("usage", {}),
        }
