"""Local typed decisions from a Cygnet-layout model on llama.cpp's /completion.

Cygnet reads a stock instruct model's option-letter logits at the answer slot — one forward pass,
no generated text. The prompt is the recipe's own scaffold (vendored under cygnet/: the chat
template markers, option labels, and the measured calibration temperature), the state is rendered
as indented JSON, and each letter's probability mass is summed over its surface token forms.
Questions with more options than one pass can carry are read in groups, then a final pass over the
group winners, matching the recipe's decision server.
"""

import json
import math
import re
from pathlib import Path

import httpx

CLIENT = httpx.Client(timeout=120)
DATA = Path(__file__).with_name("data")
DECISION = json.loads((DATA / "cygnet-decision.json").read_text())
TEMPERATURE = json.loads((DATA / "cygnet-calibration.json").read_text())["temperature"][0]
LETTERS = DECISION["labels"]["strings"]
PASS_OPTIONS = DECISION["max_options"]
LETTER_TOKEN = re.compile(r"^[\s(\[{'\"]*([A-Za-z])[\s.,:)\]}'\"]*$")

# The measured scaffold from the cygnet recipe's shim (cygnet_shim.py): without it the model
# continues in prose instead of answering with an option letter.
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)
ANSWER_LINE = "Answer with the letter of exactly one option, and nothing else:"


def post(url, body):
    try:
        response = CLIENT.post(url, json=body)
    except httpx.HTTPError:
        raise RuntimeError("Local decider connection failed; no action executed.") from None
    if response.is_error:
        raise RuntimeError(f"Local decider returned HTTP {response.status_code}; no action executed.")
    return response.json()


def shown(label, description):
    """An option as the model sees it: bare name when empty, text verbatim, JSON compacted."""
    if description is None or (isinstance(description, str) and not description.strip()):
        return label
    if isinstance(description, str):
        return description
    return f"{label}: {json.dumps(description, ensure_ascii=False)}"


def render(state, instructions, options):
    """The recipe's user prompt: state, instructions, lettered options, answer slot."""
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, indent=1)
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    lines = [state.rstrip(), "", instructions.rstrip(), "", "Options:"]
    lines.extend(f"{letter}. {text}" for letter, text in zip(LETTERS, options))
    lines.extend(["", ANSWER_LINE])
    return "\n".join(lines)


def read_pass(base, state, instructions, options):
    """Letter probabilities for one question pass over at most len(LETTERS) options."""
    marks = DECISION["template"]
    prompt = marks["pre"] + SYSTEM + marks["mid"] + render(state, instructions, options) + marks["post"]
    if DECISION.get("add_bos"):
        tokens = post(base + "/tokenize", {"content": prompt, "add_special": True})["tokens"]
        prompt = tokens
    result = post(
        base + "/completion",
        {"prompt": prompt, "n_predict": 1, "n_probs": 255, "temperature": 0},
    )
    positions = result.get("completion_probabilities") or []
    top = positions[0].get("top_logprobs") if positions else None
    if not top:
        raise RuntimeError("Local decider returned no answer slot; no action executed.")
    mass = {}
    for candidate in top:
        match = LETTER_TOKEN.match(str(candidate["token"]))
        if not match:
            continue
        index = LETTERS.index(match.group(1).upper()) if match.group(1).upper() in LETTERS else None
        if index is not None and index < len(options):
            mass[index] = mass.get(index, 0.0) + math.exp(candidate["logprob"])
    if not mass:
        raise RuntimeError("Local decider returned no offered letter; no action executed.")
    total = sum(mass.values())
    return [mass.get(index, 0.0) / total for index in range(len(options))]


def probabilities(base, state, instructions, labels_options):
    """P(option) for one question, grouped past one pass's option budget.

    Groups of PASS_OPTIONS are read one pass each; a final pass over each group's winner
    resolves P(winner), and P(option) = P(winner) x P(option | group). The calibration
    temperature is applied once to the combined distribution.
    """
    options = [shown(label, desc) for label, desc in labels_options]
    if len(options) == 1:
        return [1.0]
    if len(options) <= PASS_OPTIONS:
        dist = read_pass(base, state, instructions, options)
    else:
        groups = [options[index : index + PASS_OPTIONS] for index in range(0, len(options), PASS_OPTIONS)]
        if len(groups) > len(LETTERS):
            raise RuntimeError(f"Local decider cannot offer {len(options)} options; no action executed.")
        slices = [[1.0] if len(group) == 1 else read_pass(base, state, instructions, group) for group in groups]
        winners = [max(range(len(probs)), key=probs.__getitem__) for probs in slices]
        final = read_pass(
            base, state, instructions, [groups[index][winner] for index, winner in enumerate(winners)]
        )
        dist = [0.0] * len(options)
        for index, (group, probs, winner) in enumerate(zip(groups, slices, winners)):
            for offset, value in enumerate(probs):
                dist[index * PASS_OPTIONS + offset] = final[index] * value
    calibrated = [max(value, 1e-12) ** (1.0 / TEMPERATURE) for value in dist]
    total = sum(calibrated)
    return [value / total for value in calibrated]


def confidence(probabilities):
    """TypeSafe's Choice confidence: 0 for a flat distribution, 1 when all mass is on one option."""
    count = len(probabilities)
    if count <= 1:
        return 1.0
    return min(1.0, max(0.0, (count * max(probabilities) - 1) / (count - 1)))


class Cygnet:
    """The Cygnet-layout readout behind llama-server: typed answers in the remote response shape."""

    def __init__(self, base, model):
        self.base, self.model = base.rstrip("/"), model

    def decide(self, state, questions):
        result = {}
        for name, question in questions.items():
            criteria = question.get("criteria", {})
            probs = probabilities(self.base, state, question["instructions"], list(criteria.items()))
            names = list(criteria)
            index = max(range(len(probs)), key=probs.__getitem__)
            result[name] = {
                "choice": names[index],
                "confidence": round(confidence(probs), 4),
                "probabilities": {label: round(value, 4) for label, value in zip(names, probs)},
            }
        return {"answers": result, "model": self.model}
