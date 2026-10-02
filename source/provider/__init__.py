"""Model sources: a System One API or a local readout for decisions; a chat API for text.

DECISION_PROVIDER picks the decision backend: the remote providers (typesafe, zen, openrouter)
speak the TypeSafe wire format and differ only in base URL and key; cygnet and decider are the
local llama.cpp readouts, one per wire format. TEXT_PROVIDER picks the TYPE_TEXT helper (zen,
openrouter, local), an OpenAI-compatible endpoint. DECISION_/TEXT_ MODEL are required — .env.example
carries the concrete starting values. BASE_URL and API_KEY apply uniformly; free Zen models and
loopback endpoints need no key.
"""

import os

from . import cygnet, decider, remote
from .remote import loopback

# provider -> (default base, endpoint path). No path means a llama.cpp readout.
DECISION_PROVIDERS = {
    "typesafe": ("https://api.typesafe.ai", "/v1/systemone"),
    "zen": ("https://opencode.ai/zen", "/v1/systemone"),
    "openrouter": ("https://openrouter.ai", "/api/alpha/decisions"),
    "cygnet": ("http://127.0.0.1:8110", None),
    "decider": ("http://127.0.0.1:8110", None),
}
TEXT_PROVIDERS = {
    "zen": "https://opencode.ai/zen/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "local": "",
}


def _named(env, choices, default):
    name = os.environ.get(env, "").strip().lower() or default
    if name not in choices:
        raise ValueError(f"{env} must be one of {', '.join(choices)}")
    return name


def _required(env, purpose):
    value = os.environ.get(env, "").strip()
    if not value:
        raise ValueError(f"{env} must name the {purpose} model.")
    return value


def decision():
    """The backend answering state + typed questions: a System One endpoint or a local readout."""
    name = _named("DECISION_PROVIDER", DECISION_PROVIDERS, "typesafe")
    default_base, path = DECISION_PROVIDERS[name]
    base = (os.environ.get("DECISION_BASE_URL") or default_base).rstrip("/")
    key = os.environ.get("DECISION_API_KEY", "")
    if not key and not loopback(base) and name != "zen":
        raise ValueError(f"DECISION_PROVIDER={name} needs DECISION_API_KEY.")
    model = _required("DECISION_MODEL", "decision")
    if name == "cygnet":
        return cygnet.Cygnet(base, model)
    if name == "decider":
        return decider.Decider(base, model)
    return remote.SystemOne(base + path, key, model)


def text():
    """The OpenAI-compatible helper writing TYPE_TEXT values."""
    override = os.environ.get("TEXT_BASE_URL", "").rstrip("/")
    name = _named("TEXT_PROVIDER", TEXT_PROVIDERS, "local" if override and loopback(override) else "openrouter")
    base = override or TEXT_PROVIDERS[name]
    if not base:
        raise ValueError("TEXT_PROVIDER=local needs TEXT_BASE_URL, an OpenAI-compatible base.")
    key = os.environ.get("TEXT_API_KEY", "")
    if not key and not loopback(base) and name != "zen":
        raise ValueError(f"TYPE_TEXT needs TEXT_API_KEY for {name}; no text is hardcoded or guessed.")
    model = _required("TEXT_MODEL", "text")
    reasoning = {"enabled": False} if name == "zen" else {"effort": "low"}
    if os.environ.get("TEXT_REASONING") == "none":
        reasoning = {"enabled": False}
    # Some helpers reject response_format; the JSON object is still validated after it returns.
    json_mode = os.environ.get("TEXT_JSON_MODE") != "off"
    return remote.ChatText(base, key, model, reasoning=reasoning, json_mode=json_mode)
