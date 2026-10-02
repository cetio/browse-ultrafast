"""Typed decisions from a decider-family model on llama.cpp's /completion.

The prompt is the plain layout the model was trained on (Context / Question / Options / Answer
slot). llama.cpp reports the top logprobs at that slot, so the readout is the option-label
columns of those logits, restricted to the offered options. Nothing is generated or parsed as
text.
"""

import json
import math
import string

import httpx

CLIENT = httpx.Client(timeout=120)
INDEX_MIN = 8
MAX_OPTIONS = 255
LABELS = {}


def post(url, body):
    try:
        response = CLIENT.post(url, json=body)
    except httpx.HTTPError:
        raise RuntimeError("Local decider connection failed; no action executed.") from None
    if response.is_error:
        raise RuntimeError(f"Local decider returned HTTP {response.status_code}; no action executed.")
    return response.json()


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def annotate(value):
    """Write each element's position into long arrays, as the trained state renderer does."""
    if isinstance(value, dict):
        return {key: annotate(item) for key, item in value.items()}
    if not isinstance(value, list):
        return value
    if len(value) < INDEX_MIN:
        return [annotate(item) for item in value]
    return [_index(index, annotate(item)) for index, item in enumerate(value)]


def _index(index, item):
    """One annotated array element: its position merged into a mapping, or wrapped otherwise."""
    return {"_index": index, **item} if isinstance(item, dict) else {"_index": index, "value": item}


def render_state(state):
    return json.dumps(annotate(state), ensure_ascii=False)


def label_list(base):
    """The single-token labels the model was trained on: A..Z, then AA, AB, ... skipping multi-token names."""
    if base not in LABELS:
        names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        tokens = post(base + "/tokenize", {"content": "\n".join(names), "add_special": False})["tokens"]
        newline = post(base + "/tokenize", {"content": "\n", "add_special": False})["tokens"][0]
        found, index, size = [], 0, 0
        for token in [*tokens, newline]:
            if token != newline:
                size += 1
                continue
            if size == 1:
                found.append(names[index])
            index, size = index + 1, 0
        LABELS[base] = found[:MAX_OPTIONS]
    return LABELS[base]


def labels(base, count):
    return label_list(base)[:count]


def criteria_options(criteria):
    """Option lines as the trained renderer writes them: the name alone, or name and its description."""
    return [name if criteria[name] in (None, "") else f"{name}: {text(criteria[name])}" for name in criteria]


def render(base, state, instructions, options):
    lines = ["Context:", state, "", f"Question: {text(instructions)}", "Options:"]
    lines.extend(f"({label}) {option}" for label, option in zip(labels(base, len(options)), options))
    lines.append("Answer: (")
    return "\n".join(lines)


def read(base, prompt, count):
    """Probabilities over the offered options, from the label columns of the answer slot."""
    result = post(
        base + "/completion",
        {"prompt": prompt, "n_predict": 1, "n_probs": MAX_OPTIONS, "temperature": 0},
    )
    positions = result.get("completion_probabilities") or []
    top = positions[0].get("top_logprobs") if positions else None
    if not top:
        raise RuntimeError("Local decider returned no answer slot; no action executed.")
    offered = labels(base, count)
    logprobs = {}
    for candidate in top:
        token = str(candidate["token"]).strip()
        if token in offered and token not in logprobs:
            logprobs[token] = float(candidate["logprob"])
    if not logprobs:
        raise RuntimeError("Local decider returned no offered label; no action executed.")
    total = sum(math.exp(value) for value in logprobs.values())
    return [math.exp(logprobs[label]) / total if label in logprobs else 0.0 for label in offered]


def confidence(probabilities):
    """TypeSafe's Choice confidence: 0 for a flat distribution, 1 when all mass is on one option."""
    count = len(probabilities)
    if count <= 1:
        return 1.0
    return min(1.0, max(0.0, (count * max(probabilities) - 1) / (count - 1)))


class Decider:
    """The decider wire format behind llama-server: typed answers in the remote response shape."""

    def __init__(self, base, model):
        self.base, self.model = base.rstrip("/"), model

    def decide(self, state, questions):
        rendered, result = render_state(state), {}
        for name, question in questions.items():
            criteria = question.get("criteria", {})
            options = criteria_options(criteria)
            probs = read(self.base, render(self.base, rendered, question["instructions"], options), len(options))
            index = max(range(len(probs)), key=probs.__getitem__)
            names = list(criteria)
            result[name] = {
                "choice": names[index],
                "confidence": round(confidence(probs), 4),
                "probabilities": {label: round(value, 4) for label, value in zip(names, probs)},
            }
        return {"answers": result, "model": self.model}
