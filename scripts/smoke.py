"""Explicit live smoke: an inline fixture page + real model calls. Not run by pytest.

Run with: uv run --env-file .env python scripts/smoke.py
"""

import argparse
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from source import Agent

GOAL = (
    "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, "
    "then open Casa Flora."
)

HTML = """<!doctype html><title>Stays</title>
<style>body{margin:24px;font-family:sans-serif}label{display:block;margin:8px 0}</style>
<h1>Find a stay</h1>
<label>Destination <input id="dest"></label>
<label><input type="checkbox" id="design">Design</label>
<label><input type="checkbox" id="free">Free cancellation</label>
<button id="search" onclick="go()">Search</button>
<p id="filters"></p>
<div id="results"></div>
<script>
function go(){
  const d=document.getElementById('design').checked, f=document.getElementById('free').checked;
  const dest=document.getElementById('dest').value||'anywhere';
  document.getElementById('filters').textContent='Your filters: '+(d?'Design · ':'')+
    'Free cancellation '+(f?'enabled':'off')+' · Destination '+dest;
  document.getElementById('results').innerHTML=
    '<a href="#casa-flora">Casa Flora</a><br><a href="#hotel-rio">Hotel Rio</a>';
}
</script>"""


class Fixture(BaseHTTPRequestHandler):
    def do_GET(self):
        body = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--goal", default=GOAL)
    args = parser.parse_args()
    output = Path("artifacts/dynamic/fixture") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Trace: {output}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with Agent(f"http://127.0.0.1:{server.server_port}/", args.goal) as agent:
        try:
            for state in agent.run():
                history = state["history"]
                print(
                    state["elapsed_ms"], "ms", len(history), "actions",
                    history[-1]["action"] if history else "", flush=True,
                )
                (output / "state.json").write_text(json.dumps(state, indent=2))
                if len(history) >= args.max_actions:
                    raise RuntimeError(f"Diagnostic stopped at {args.max_actions} actions")
        finally:
            state = agent.snapshot()
            try:
                state["verification_text"] = agent.state["browser"].evaluate("document.body.innerText")
            finally:
                (output / "state.json").write_text(json.dumps(state, indent=2))
        assert state["status"] == "done"
        assert state["page"]["url"].endswith("#casa-flora")
        assert "Your filters: Design · Free cancellation enabled · Destination Lisbon" in state["verification_text"]
        result = {
            "ms": state["elapsed_ms"],
            "verified": True,
            "decisions": len(state["decisions"]),
            "actions": len(state["history"]),
        }
        print(json.dumps(result, indent=2))
        (output / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
