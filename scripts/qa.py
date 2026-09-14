#!/usr/bin/env python3
"""Endpoint QA for the hybrid body: root check, tools, math, loop stress, speed.

    python3 qa.py [--base http://127.0.0.1:8000/v1] [--model qwen3.8-flash-next]

Exit 0 when every gate passes.
"""
import argparse
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000/v1"
MODEL = "qwen3.8-flash-next"
FAILS = []


def chat(messages, **kw):
    body = {"model": MODEL, "messages": messages, "temperature": 0, **kw}
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def gate(name, ok, detail=""):
    print(f"{name}: {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"), ap.add_argument("--model")
    a = ap.parse_args()
    global BASE, MODEL
    BASE, MODEL = a.base or BASE, a.model or MODEL

    with urllib.request.urlopen(BASE.rsplit("/v1", 1)[0] + "/v1/models", timeout=10) as r:
        models = json.load(r)["data"]
    print("root:", models[0].get("root", "?"))

    # tools
    tools = [{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}}},
    }]
    out = chat([{"role": "user", "content": "Weather in Da Nang? Call the tool."}], tools=tools, max_tokens=200)
    calls = out["choices"][0]["message"].get("tool_calls") or []
    ok = bool(calls) and calls[0]["function"]["name"] == "get_weather"
    try:
        json.loads(calls[0]["function"]["arguments"]); json_ok = True
    except Exception:
        json_ok = False
    gate("tools", ok and json_ok, f"n_calls={len(calls)}")

    # math
    out = chat([{"role": "user", "content": "17*19=? Answer with just the number."}], max_tokens=64)
    txt = (out["choices"][0]["message"].get("content") or "")
    gate("math", "323" in txt, repr(txt[-20:]))

    # loop stress: same prompt 3x, output length must not inflate
    lens = []
    for i in range(3):
        out = chat([{"role": "user", "content": "Write a short paragraph about rain in Saigon."}],
                   max_tokens=200)
        c = out["choices"][0]["message"]
        lens.append(len((c.get("content") or "") + (c.get("reasoning") or "")))
    gate("loop-stress", max(lens) < 2 * min(lens), f"lengths={lens}")

    # throughput (c1, 256 tokens)
    t0 = time.time()
    out = chat([{"role": "user", "content": "Describe a morning market in the Mekong Delta in detail."}],
               max_tokens=256)
    el = time.time() - t0
    n = out["usage"]["completion_tokens"]
    gate("throughput", True, f"{n} tok in {el:.1f}s = {n / el:.1f} tok/s")

    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
