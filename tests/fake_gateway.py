"""A local stand-in for an OpenAI-compatible gateway.

Not a mock of the *model* -- that is `providers/mock.py`. This is a mock of the
*wire*: a real HTTP server that speaks the chat-completions API, so the real
`OpenAICompatClient` can be driven end to end without a network or a key.

It exists because the offline client bypasses every part of the stack that the
first paid run depends on: the HTTP layer, native `tools` / `tool_calls`
round-tripping, the assistant/tool message pairing, usage accounting, retries
and the error mapping. Those paths only ever get exercised against a server, and
"it worked against the relay" is an expensive way to find out they don't.

    python -m werewolf_harness.tests.fake_gateway          # serve on :8900
    python -m werewolf_harness.tests.fake_gateway --check   # run a game through it

The server plays by delegating to the offline client, so a game against it is a
real game over real HTTP -- and just as scripted, and just as excluded from any
reported result.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..harness.providers.mock import MockClient

MODELS = ["fake-native-4o", "fake-json-only", "fake-flaky"]


class Handler(BaseHTTPRequestHandler):
    client = MockClient(seed=0)
    calls: list[dict] = []
    fail_next = 0  # set by the flaky model to exercise the retry path

    def log_message(self, *args):  # keep the test output readable
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list",
                        "data": [{"id": m, "object": "model"} for m in MODELS]})
        else:
            self._json({"error": {"message": "not found"}}, status=404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.calls.append(body)

        if not (self.headers.get("Authorization") or "").startswith("Bearer sk-"):
            return self._json({"error": {"message": "Invalid token"}}, status=401)

        model = body.get("model", "")
        if model not in MODELS:
            return self._json(
                {"error": {"message": f"No available channel for model {model}"}},
                status=400,
            )
        if model == "fake-flaky" and Handler.fail_next > 0:
            Handler.fail_next -= 1
            return self._json({"error": {"message": "upstream busy"}}, status=503)

        response = Handler.client.chat(body["messages"], tools=body.get("tools"))
        call = response.tool_calls[0] if response.tool_calls else None

        if model == "fake-json-only":
            # A model with no native tool calling: answer in prose, as one does.
            message = {"role": "assistant",
                       "content": json.dumps({"thought": "…",
                                              "action": call.name if call else "speak",
                                              "args": call.arguments if call else {}})}
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_%d" % len(Handler.calls),
                    "type": "function",
                    "function": {"name": call.name,
                                 "arguments": json.dumps(call.arguments)},
                }] if call else [],
            }

        self._json({
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if call else "stop"}],
            "usage": {"prompt_tokens": response.prompt_tokens,
                      "completion_tokens": response.completion_tokens,
                      "total_tokens": response.total_tokens},
        })

    def _json(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(port: int = 8900) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _check(port: int = 8900) -> int:
    """Run a full game through the real HTTP client and report what it proved."""
    from ..evalkit.runner import RunConfig, run_game

    server = serve(port)
    base = f"http://127.0.0.1:{port}/v1"
    failures = []

    for model, mode in (("fake-native-4o", "native"), ("fake-json-only", "json_prompt")):
        Handler.calls.clear()
        log = run_game(RunConfig(
            seed=5,
            model={"model_name": model, "display_name": model, "tool_mode": mode,
                   "api_key": "sk-local-fake", "base_url": base},
            guard_layers=("L1", "L2", "L3"),
            attack_enabled=True,
        ))
        turns = sum(len(r["agents"]) + len(r.get("night_turns", []))
                    for r in log["rounds"])
        print(f"{model} ({mode}): crashed={log['outcome']['crashed']} "
              f"winner={log['outcome']['winner']} turns={turns} "
              f"http_calls={len(Handler.calls)} "
              f"tokens={log['outcome']['total_prompt_tokens']}")
        if log["outcome"]["crashed"]:
            failures.append(f"{model}: {log['outcome']['crash_reason']}")

        # The pairing rule real gateways enforce.
        for body in Handler.calls:
            for i, m in enumerate(body["messages"]):
                if m.get("role") == "tool":
                    prev = body["messages"][i - 1]
                    if prev.get("role") != "assistant" or not prev.get("tool_calls"):
                        failures.append(f"{model}: orphaned tool message at {i}")
                    elif m["tool_call_id"] not in {c["id"] for c in prev["tool_calls"]}:
                        failures.append(f"{model}: tool_call_id does not match")
        sent_tools = any("tools" in b for b in Handler.calls)
        if mode == "native" and not sent_tools:
            failures.append("native mode sent no tools field")
        if mode == "json_prompt" and sent_tools:
            failures.append("json mode sent a tools field")

    # Retry path, and the two error mappings that matter most.
    from ..harness.providers import OpenAICompatClient, ProviderError, probe_model

    Handler.fail_next = 2
    client = OpenAICompatClient(model="fake-flaky", api_key="sk-local-fake", base_url=base)
    result = probe_model(client, check_temperature=False)
    print(f"probe(fake-flaky after 2x503): reachable={result.reachable} "
          f"tool_mode={result.tool_mode}")

    for model, key, expect in (("no-such-model", "sk-x", "channel group"),
                               ("fake-native-4o", "bad", "token")):
        try:
            OpenAICompatClient(model=model, api_key=key, base_url=base).chat(
                [{"role": "user", "content": "hi"}])
            failures.append(f"{model}/{key}: no error raised")
        except ProviderError as exc:
            print(f"error mapping [{model} + {key}]: {exc.hint}")
            if expect not in (exc.hint or ""):
                failures.append(f"{model}: hint did not mention {expect!r}")

    server.shutdown()
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nthe real HTTP path is sound: both tool modes, message pairing, "
          "usage accounting, retries and error mapping")
    return 0


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        raise SystemExit(_check())
    serve()
    print("fake gateway on http://127.0.0.1:8900/v1  (ctrl-c to stop)")
    threading.Event().wait()
