"""Serve a stack: spawn one llama-server per declared model, an optional debug
Chrome, wait for health, and keep them up until interrupted.

Endpoints are written to runs/endpoints.json (ephemeral, gitignored) and
printed as provider env lines for shell use.

Do NOT run this under a virtual-memory cap (ulimit -v): the limit is inherited
by the llama-server children and breaks model loading with std::bad_alloc.

Model entries may set: name, role ("decide" | "chat"), port, gpu_layers,
context_tokens, kv_quant, swa_full. Defaults: port 8100+index, all layers
offloaded, f16 KV.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

LLAMA_SERVER = Path.home() / "llama.cpp/build-cuda/bin/llama-server"
CUDA_LIBS = Path.home() / ".local/cuda-rpm-extract/usr/local/cuda-13.4/lib64"
CHROME = "/opt/google/chrome/chrome"
ENDPOINT_FILE = Path("runs/endpoints.json")
HEALTH_TIMEOUT_S = 300


def free_vram():
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(output.stdout.strip().splitlines()[0]) * 2**20
    except (OSError, ValueError, IndexError):
        return None


def spawn_model(entry, index):
    """Start one llama-server for a stack entry. Returns (proc, url)."""
    port = entry.get("port", 8100 + index)
    path = str(Path(entry["path"]).expanduser())
    size = Path(path).stat().st_size
    free = free_vram()
    if free and entry.get("gpu_layers", 99) and size > free:
        print(f"warning: {Path(path).name} is {size / 2**30:.1f} GiB with {free / 2**30:.1f} GiB VRAM free", flush=True)
    cmd = [
        str(LLAMA_SERVER), "--model", path, "--host", "127.0.0.1", "--port", str(port),
        "-c", str(entry.get("context_tokens", 8192)), "-ngl", str(entry.get("gpu_layers", 99)),
    ]
    if entry.get("swa_full"):
        cmd.append("--swa-full")
    kv_quant = entry.get("kv_quant")
    if kv_quant:
        cmd += ["-ctk", kv_quant, "-ctv", kv_quant]
    env = dict(os.environ, LD_LIBRARY_PATH=str(CUDA_LIBS))
    log = open(f"/tmp/llama-{Path(path).stem}-{port}.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), f"http://127.0.0.1:{port}"


def spawn_chrome(config):
    port = config["port"]
    cmd = [
        CHROME, f"--remote-debugging-port={port}", f"--user-data-dir={config['user_data_dir']}",
        "--no-first-run", "--no-default-browser-check", "--disable-session-crashed-bubble",
    ]
    log = open("/tmp/llama-chrome.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), f"http://127.0.0.1:{port}"


def wait_health(url, proc, timeout=HEALTH_TIMEOUT_S):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{url} exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError(f"{url} not healthy within {timeout}s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stack", help="stack YAML")
    parser.add_argument("--once", action="store_true",
                        help="verify health of every model, print endpoints, exit")
    args = parser.parse_args(argv)

    with open(Path(args.stack).expanduser()) as file:
        config = yaml.safe_load(file)

    procs, endpoints = [], {}
    try:
        for index, entry in enumerate(config.get("models", [])):
            proc, url = spawn_model(entry, index)
            procs.append(proc)
            wait_health(url, proc)
            name = entry.get("name") or Path(entry["path"]).stem
            endpoints[name] = {
                "url": url,
                "role": entry.get("role", "chat"),
                "port": entry.get("port"),
                "provider": entry.get("provider", "decider"),
            }
            print(f"up: {name} [{endpoints[name]['role']}] at {url}", flush=True)
        if config.get("chrome") and shutil.which(CHROME):
            proc, url = spawn_chrome(config["chrome"])
            procs.append(proc)
            print(f"up: chrome at {url}", flush=True)

        ENDPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENDPOINT_FILE.write_text(json.dumps(endpoints, indent=2))
        for name, ep in endpoints.items():
            if ep["role"] == "decide":
                print(f"DECISION_PROVIDER={ep['provider']} DECISION_BASE_URL={ep['url']}  # {name}", flush=True)
            else:
                print(f"TEXT_PROVIDER=local TEXT_BASE_URL={ep['url']}/v1  # {name}", flush=True)

        if args.once:
            return 0
        print("serving; Ctrl-C stops everything", flush=True)
        signal.pause()
        return 0
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
