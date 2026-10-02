"""Live end-to-end tasks on real sites: a real Chrome and the .env decision/text providers.

Four realistic goals of increasing difficulty — a click-through, an autocomplete search, a
filtered search, and a highly dynamic flight search — exercise the whole
observe -> decide -> execute loop against production pages. A DONE choice is never proof:
each test verifies the final page state independently. `uv run pytest` runs these when .env,
Chrome, and the configured endpoints are reachable; `-m "not integration"` stays offline.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from source import Agent, provider
from source.browser import Browser
from source.provider.remote import loopback

pytestmark = pytest.mark.integration


def load_env():
    path = Path(__file__).parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


@pytest.fixture(scope="module")
def live():
    load_env()
    try:
        backends = [provider.decision(), provider.text()]
    except ValueError as exc:
        pytest.skip(f"integration tests need the .env provider configuration: {exc}")
    for backend in backends:
        endpoint = getattr(backend, "url", getattr(backend, "base", ""))
        if not endpoint:
            continue
        probe = endpoint.rstrip("/") + "/health" if loopback(endpoint) else endpoint
        try:
            httpx.get(probe, timeout=5, follow_redirects=True)
        except httpx.HTTPError as exc:
            pytest.skip(f"configured endpoint is unreachable: {exc}")
    try:
        Browser("about:blank").close()
    except Exception as exc:
        pytest.skip(f"Chrome via browser-harness is unavailable: {exc}")


def run_task(url, goal, max_actions, deadline_seconds=240):
    """Drive one realistic goal; return the final snapshot and live page facts to verify."""
    with Agent(url, goal) as agent:
        deadline = time.monotonic() + deadline_seconds
        state = agent.snapshot()
        failure = None
        try:
            for state in agent.run():
                if len(state["history"]) >= max_actions or time.monotonic() > deadline:
                    break
        except Exception as exc:
            failure = exc
        try:
            facts = {
                "url": agent.state["browser"].evaluate("location.href"),
                "text": agent.state["browser"].evaluate("document.body.innerText"),
            }
        except Exception:
            facts = {"url": "", "text": ""}
        trace = Path("artifacts/tests") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        trace.mkdir(parents=True, exist_ok=True)
        (trace / "state.json").write_text(json.dumps(agent.snapshot(), indent=2))
        if failure is None and state["status"] == "done":
            return state, facts
        detail = failure or f"status {state['status']} after {len(state['history'])} actions"
        pytest.fail(f"{detail}; trace in {trace}")


def test_opens_the_top_story_comments(live):
    state, facts = run_task(
        "https://news.ycombinator.com/",
        "Open the comments thread of the top-ranked story using its comments link "
        "in the first story row.",
        6,
        90,
    )
    assert re.search(r"item\?id=\d+", facts["url"])
    assert "Hacker News" in facts["text"]


def test_wikipedia_search_opens_an_article(live):
    state, facts = run_task(
        "https://en.wikipedia.org/wiki/Main_Page",
        'Search Wikipedia for the "Smalltalk" programming language and open its article.',
        10,
        150,
    )
    assert "/wiki/Smalltalk" in facts["url"]
    assert "Smalltalk" in facts["text"]
    assert state["text_calls"]


def test_wikipedia_portal_switches_language_and_searches(live):
    state, facts = run_task(
        "https://www.wikipedia.org/",
        'Set the Wikipedia language to Deutsch, then search for "Zürich".',
        10,
        150,
    )
    assert "de.wikipedia.org" in facts["url"]
    assert "Zürich" in facts["text"]
    assert any("zürich" in call["value"].lower() for call in state["text_calls"])


def test_google_flights_round_trip_with_dates_and_filter(live):
    state, facts = run_task(
        "https://www.google.com/travel/flights?hl=en",
        "Search Google Flights for round-trip flights from Zurich to Lisbon, departing "
        "November 12 and returning November 15. Type both dates into the date fields. "
        "Show nonstop flights only, then open the first flight option to see booking "
        "choices. If a consent dialog appears, accept it first.",
        30,
        300,
    )
    assert "flights" in facts["url"]
    assert "Lisbon" in facts["text"]
    assert re.search(r"(CHF|EUR|USD|[$€£])\s?\d", facts["text"])
    assert "Nonstop" in facts["text"]
    assert "book" in facts["text"].lower()
    assert len(state["history"]) >= 12
