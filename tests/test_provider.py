"""Offline contracts for provider backends. No paid APIs."""

import json
from unittest.mock import Mock

import pytest

from source import model, provider
from source.provider import cygnet, decider, remote


def page():
    return {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "fingerprint": "x",
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("DECISION_PROVIDER", "DECISION_BASE_URL", "DECISION_MODEL", "DECISION_API_KEY",
                "TEXT_PROVIDER", "TEXT_BASE_URL", "TEXT_MODEL", "TEXT_API_KEY",
                "TEXT_JSON_MODE", "TEXT_REASONING"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("DECISION_API_KEY", "test")
    monkeypatch.setenv("DECISION_MODEL", "test")
    monkeypatch.setattr(remote, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("DECISION_API_KEY", "test")
    monkeypatch.setenv("DECISION_MODEL", "test")
    monkeypatch.setattr(remote, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("DECISION_API_KEY", "test")
    monkeypatch.setenv("DECISION_MODEL", "test")
    monkeypatch.setattr(remote, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_choose_posts_the_fanout_to_openrouter_decisions(monkeypatch):
    monkeypatch.setenv("DECISION_PROVIDER", "openrouter")
    monkeypatch.setenv("DECISION_API_KEY", "test")
    monkeypatch.setenv("DECISION_MODEL", "inception/mercury-decide:free")
    captured = []

    def post(url, key, body):
        captured.append((url, key, body))
        return {"model": "inception/mercury-decide:free", "answers": {
            "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
            "click_target": choice(body["questions"]["click_target"]["criteria"], "2"),
        }}

    monkeypatch.setattr(remote, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert d["operation"] == "CLICK" and d["target"] == "2" and d["choice"] == "e3"
    url, key, body = captured[0]
    assert url == "https://openrouter.ai/api/alpha/decisions" and key == "test"
    assert body["model"] == "inception/mercury-decide:free"
    assert set(body["questions"]) == {"operation", "click_target", "type_text_target"}


def test_a_decisions_answer_without_normalized_probabilities_is_rejected(monkeypatch):
    # A chat model replying with a bare letter cannot stand in for a probability vector.
    monkeypatch.setenv("DECISION_PROVIDER", "zen")
    monkeypatch.setenv("DECISION_MODEL", "jev-1.13-free")
    monkeypatch.setattr(
        remote, "post_json",
        lambda *_a: {"model": "m", "answers": {"operation": {"choice": "CLICK"}}},
    )
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_zen_decision_endpoint_needs_no_key(monkeypatch):
    monkeypatch.setenv("DECISION_PROVIDER", "zen")
    monkeypatch.setenv("DECISION_MODEL", "jev-1.13-free")
    backend = provider.decision()
    assert backend.url == "https://opencode.ai/zen/v1/systemone"
    assert backend.model == "jev-1.13-free" and backend.key == ""


def test_choose_reads_the_local_cygnet_when_configured(monkeypatch):
    monkeypatch.setenv("DECISION_PROVIDER", "cygnet")
    monkeypatch.setenv("DECISION_BASE_URL", "http://127.0.0.1:8110")
    monkeypatch.setenv("DECISION_MODEL", "cygnet-12b")
    captured = []

    def decide(self, state, questions):
        captured.append((self.base, self.model))
        return {"answers": {
            "operation": choice(questions["operation"]["criteria"], "TYPE_TEXT"),
            "type_text_target": choice(questions["type_text_target"]["criteria"], "1"),
        }, "model": self.model}

    monkeypatch.setattr(cygnet.Cygnet, "decide", decide)
    monkeypatch.setattr(remote, "post_json", Mock(side_effect=AssertionError("the cloud must not be called")))
    d = model.choose(page(), "Find a book", [])
    assert d["operation"] == "TYPE_TEXT" and d["choice"] == "e1"
    assert captured == [("http://127.0.0.1:8110", "cygnet-12b")]


def test_decider_prompt_is_the_trained_layout(monkeypatch):
    monkeypatch.setattr(decider, "label_list", lambda base: ["A", "B"])
    rendered = decider.render("http://x", '{"a": 1}', {"goal": "g"}, ["CLICK: click", "DONE: done"])
    assert rendered == (
        'Context:\n{"a": 1}\n\nQuestion: {"goal": "g"}\nOptions:\n'
        "(A) CLICK: click\n(B) DONE: done\nAnswer: ("
    )


def test_decider_read_restricts_to_offered_labels(monkeypatch):
    monkeypatch.setattr(decider, "label_list", lambda base: ["A", "B", "C"])
    payload = {"completion_probabilities": [{"top_logprobs": [
        {"token": "B", "logprob": 0.0}, {"token": "A", "logprob": -2.0}, {"token": "junk", "logprob": -9.0},
    ]}]}
    seen = []
    monkeypatch.setattr(decider, "post", lambda url, body: seen.append(body) or payload)
    probabilities = decider.read("http://x", "prompt", 3)
    assert seen[0]["n_predict"] == 1 and seen[0]["n_probs"] == 255
    assert probabilities[1] > probabilities[0] and probabilities[2] == 0.0
    assert abs(sum(probabilities) - 1) < 1e-6


def test_decider_answers_are_valid_choices(monkeypatch):
    monkeypatch.setattr(decider, "label_list", lambda base: ["A", "B"])
    payload = {"completion_probabilities": [{"top_logprobs": [
        {"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -1.0}]}]}
    monkeypatch.setattr(decider, "post", lambda url, body: payload)
    result = decider.Decider("http://x", "m").decide(
        {"a": 1},
        {"operation": {"criteria": {"CLICK": "click", "DONE": "done"}, "instructions": {"goal": "g"}}},
    )
    model.validate_choice(result["answers"]["operation"], {"CLICK", "DONE"})
    assert result["answers"]["operation"]["choice"] == "CLICK"


def test_cygnet_prompt_is_the_recipe_layout():
    rendered = cygnet.render('{"a": 1}', {"goal": "g"}, ["click", "done"])
    assert rendered == (
        '{"a": 1}\n\n{"goal": "g"}\n\nOptions:\nA. click\nB. done\n\n'
        "Answer with the letter of exactly one option, and nothing else:"
    )


def test_cygnet_read_restricts_to_offered_letters(monkeypatch):
    payload = {"completion_probabilities": [{"top_logprobs": [
        {"token": "B", "logprob": 0.0}, {"token": " B", "logprob": -1.0}, {"token": "junk", "logprob": -9.0},
    ]}]}
    seen = []

    def post(url, body):
        seen.append(url)
        return {"tokens": [1, 2]} if url.endswith("/tokenize") else payload

    monkeypatch.setattr(cygnet, "post", post)
    probs = cygnet.probabilities("http://x", "state", "instructions", [("a", "click"), ("b", "done")])
    assert seen[0].endswith("/tokenize") and seen[1].endswith("/completion")
    assert probs[1] > probs[0] and abs(sum(probs) - 1) < 1e-6


def test_cygnet_answers_are_valid_choices(monkeypatch):
    payload = {"completion_probabilities": [{"top_logprobs": [
        {"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -1.0}]}]}

    def post(url, body):
        return {"tokens": [1, 2]} if url.endswith("/tokenize") else payload

    monkeypatch.setattr(cygnet, "post", post)
    result = cygnet.Cygnet("http://x", "cygnet-12b").decide(
        {"a": 1},
        {"operation": {"criteria": {"CLICK": "click", "DONE": "done"}, "instructions": {"goal": "g"}}},
    )
    model.validate_choice(result["answers"]["operation"], {"CLICK", "DONE"})
    assert result["answers"]["operation"]["choice"] == "CLICK"


def test_cygnet_groups_options_beyond_one_pass(monkeypatch):
    prompts = []

    def post(url, body):
        if url.endswith("/tokenize"):
            return {"tokens": [1]}
        prompts.append(body["prompt"])
        # Every pass puts its mass on letter A: the first group's first option and the
        # first group winner carry the final distribution.
        return {"completion_probabilities": [{"top_logprobs": [{"token": "A", "logprob": 0.0}]}]}

    monkeypatch.setattr(cygnet, "post", post)
    options = [(f"o{i}", f"opt {i}") for i in range(cygnet.PASS_OPTIONS + 1)]
    probs = cygnet.probabilities("http://x", "state", "instructions", options)
    assert len(probs) == len(options)
    assert probs[0] == max(probs) and abs(sum(probs) - 1) < 1e-6
    assert len(prompts) == 2  # one option-group pass + one winners pass; a single option needs no pass


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_API_KEY", "test")
    monkeypatch.setenv("TEXT_MODEL", "test-model")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(remote, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_decision_model_is_required(monkeypatch):
    monkeypatch.setenv("DECISION_PROVIDER", "zen")
    with pytest.raises(ValueError, match="DECISION_MODEL"):
        provider.decision()


def test_text_model_is_required(monkeypatch):
    monkeypatch.setenv("TEXT_PROVIDER", "zen")
    with pytest.raises(ValueError, match="TEXT_MODEL"):
        provider.text()


def test_missing_text_credential_stops_before_guessing():
    with pytest.raises(ValueError, match="TEXT_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_API_KEY", "test")
    monkeypatch.setenv("TEXT_MODEL", "test-model")
    monkeypatch.setattr(remote, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_text_helper_can_drop_json_mode_for_models_that_reject_it(monkeypatch):
    monkeypatch.setenv("TEXT_API_KEY", "test")
    monkeypatch.setenv("TEXT_MODEL", "test-model")
    monkeypatch.setenv("TEXT_JSON_MODE", "off")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(remote, "post_json", post)
    assert model.field_text({"goal": "Fly from Zurich"})[0] == "Zurich"
    assert "response_format" not in post.call_args.args[2]


def test_text_helper_sends_json_mode_by_default(monkeypatch):
    monkeypatch.setenv("TEXT_API_KEY", "test")
    monkeypatch.setenv("TEXT_MODEL", "test-model")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(remote, "post_json", post)
    model.field_text({"goal": "Fly from Zurich"})
    assert post.call_args.args[2]["response_format"] == {"type": "json_object"}


def test_loopback_text_helper_needs_no_key(monkeypatch):
    monkeypatch.setenv("TEXT_PROVIDER", "local")
    monkeypatch.setenv("TEXT_BASE_URL", "http://127.0.0.1:8111/v1")
    monkeypatch.setenv("TEXT_MODEL", "test-model")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"x"}'}}]})
    monkeypatch.setattr(remote, "post_json", post)
    assert model.field_text({"goal": "g"})[0] == "x"
    assert post.call_args.args[2]["temperature"] == 0


def test_zen_text_helper_defaults(monkeypatch):
    monkeypatch.setenv("TEXT_PROVIDER", "zen")
    monkeypatch.setenv("TEXT_MODEL", "space-bunny-free")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"x"}'}}]})
    monkeypatch.setattr(remote, "post_json", post)
    assert model.field_text({"goal": "g"})[0] == "x"
    url, key, body = post.call_args.args
    assert url == "https://opencode.ai/zen/v1/chat/completions" and key == ""
    assert body["model"] == "space-bunny-free"
    assert body["reasoning"] == {"enabled": False}
