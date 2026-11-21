"""A fake OpenAI-compatible gateway for offline tests: answers /chat/completions from rules, records every request.

    with FakeGateway(reply) as gw:      # reply(messages, body) -> str; "TOOL <name> <json args>" = a tool call
        os.environ["LITELLM_BASE_URL"] = gw.url
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self


class FakeGateway:
    def __init__(self, reply: Callable[[list[dict[str, Any]], dict[str, Any]], str]) -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []
        gw = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                gw.requests.append({"path": self.path, "body": body})
                if self.path.endswith("/embeddings"):
                    inputs = body.get("input") or []
                    data = [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]}
                            for i, _ in enumerate(inputs if isinstance(inputs, list) else [inputs])]
                    out = {"object": "list", "data": data, "model": body.get("model"),
                           "usage": {"prompt_tokens": 1, "total_tokens": 1}}
                else:
                    text = gw.reply(body.get("messages") or [], body)
                    message: dict[str, Any] = {"role": "assistant", "content": text}
                    finish = "stop"
                    if text.startswith("TOOL "):           # "TOOL <name> <json args>": answer with one tool call
                        _, name, args = text.split(" ", 2)
                        message = {"role": "assistant", "content": None, "tool_calls": [{
                            "id": f"call_{len(gw.requests)}", "type": "function",
                            "function": {"name": name, "arguments": args}}]}
                        finish = "tool_calls"
                    out = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
                           "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
                raw = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> Self:
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a: object) -> None:
        self.server.shutdown()
