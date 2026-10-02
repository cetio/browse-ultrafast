"""The complete agent loop. Typed choices, observable state, bounded execution."""

import time

from .browser import Browser, StalePage
from .model import choose, field_context, field_text
from .questions import MAX_STEPS


class Agent:
    """Observe -> decide -> execute, until the decision model says DONE or BLOCKED."""

    def __init__(self, url, goals):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        self.pending_text = None
        browser = Browser(url)
        try:
            page = browser.observe()
        except Exception:
            browser.close()
            raise
        self.state = dict(
            browser=browser,
            goal=task,
            page=page,
            decision=None,
            history=[],
            decisions=[],
            text_calls=[],
            status="ready",
            started_at=None,
            elapsed_ms=0,
        )

    def elapsed(self):
        if self.state["started_at"] is None:
            return 0
        return round((time.perf_counter() - self.state["started_at"]) * 1000)

    def observe(self):
        self.state["page"] = self.state["browser"].observe()
        return self.state["page"]

    def snapshot(self):
        return {k: v for k, v in self.state.items() if k != "browser"}

    def execute(self, decision):
        """Run the chosen action against the page it was decided on. Raises StalePage if it moved."""
        state, page = self.state, self.state["page"]
        selected = decision["choice"]
        if selected in {"DONE", "BLOCKED"}:
            if not state["browser"].fresh(page):
                raise StalePage("Page changed since the decision. Choose again.")
            state["status"] = "done" if selected == "DONE" else "blocked"
            return
        action = next(a for a in page["actions"] if a["id"] == selected)
        text, helper = None, None
        if action["kind"] == "fill":
            if not state["browser"].fresh(page):
                raise StalePage("Page changed before text generation. Choose again.")
            context = field_context(state["goal"], action, page, state["history"])
            if self.pending_text and self.pending_text[0] == context:
                _, text, helper = self.pending_text
            else:
                text, helper = field_text(context)
                self.pending_text = (context, text, helper)
                state["text_calls"].append({**helper, "field": action["label"], "value": text})
        # Browser.act checks freshness immediately before input, including after text generation.
        state["browser"].act(action, page, text=text)
        self.pending_text = None
        # Record execution before observing. A stale post-action observation must not erase the action.
        state["history"].append(
            {
                "step": len(state["history"]) + 1,
                "action": action["label"],
                "kind": action["kind"],
                "choice": selected,
                "probability": decision["probabilities"][selected],
                "confidence": decision["confidence"],
                "latency_ms": decision["latency_ms"],
                "text": text,
                "text_helper": helper["model"] if helper else None,
                "text_latency_ms": helper["latency_ms"] if helper else 0,
                "operation": decision["operation"],
                "target": decision["target"],
                "page_changed": None,
                "url": page["url"],
                "usage": decision["usage"],
                "executed_ms": self.elapsed(),
            }
        )
        state["page"] = state["browser"].observe()
        state["history"][-1].update(
            page_changed=state["page"]["fingerprint"] != page["fingerprint"],
            url=state["page"]["url"],
            elapsed_ms=self.elapsed(),
        )
        repeated = state["history"][-3:]
        if len(repeated) == 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated):
            state["status"] = "blocked"

    def step(self):
        """One decide -> execute cycle. Returns False once the run is done or blocked."""
        state = self.state
        if state["status"] in {"done", "blocked"}:
            return False
        if state["started_at"] is None:
            state["started_at"] = time.perf_counter()
        if len(state["history"]) >= MAX_STEPS:
            state["status"] = "blocked"
            raise ValueError(f"Stopped at the {MAX_STEPS}-action budget")
        try:
            stale = not state["browser"].fresh(state["page"])
        except StalePage:
            stale = True
        if stale:
            self.observe()
        while True:
            page = state["page"]
            if len(state["decisions"]) >= MAX_STEPS * 2:
                state["status"] = "blocked"
                raise ValueError("Reached the model-call budget")
            decision = choose(page, state["goal"], state["history"])
            state["decisions"].append(
                {**decision, "fingerprint": page["fingerprint"], "elapsed_ms": self.elapsed()}
            )
            try:
                self.execute(decision)
            except StalePage:
                # The page moved before input; reobserve and decide again. An executed action is
                # never retried: it is already in history and the new decision sees its result.
                self.observe()
                continue
            state["elapsed_ms"] = self.elapsed()
            return state["status"] not in {"done", "blocked"}

    def run(self):
        while True:
            if not self.step():
                break
            yield self.snapshot()
        yield self.snapshot()

    def close(self):
        self.state["browser"].close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
