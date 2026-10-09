"""Minimal HTTP server exposing reset/step/state (OpenEnv-style), stdlib only.

  python -m k2_tasks_env.server --port 8000
  POST /reset {"task_id": "slug-bug"} -> observation
  POST /step  {"tool": "shell", "args": {"cmd": "ls"}} | {"final": "..."} -> {observation, reward, done, info}
  GET  /state
  GET  /tasks
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .env import CodingTaskEnv
from .tasks import list_tasks


def make_handler(env: CodingTaskEnv):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body):
            data = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/state":
                return self._send(200, env.state())
            if self.path == "/tasks":
                return self._send(200, [{"id": t.id, "prompt": t.prompt, "variant": t.variant} for t in list_tasks()])
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._body()
                if self.path == "/reset":
                    return self._send(200, asdict(env.reset(body["task_id"])))
                if self.path == "/step":
                    r = env.step(body)
                    return self._send(200, {"observation": asdict(r.observation), "reward": r.reward, "done": r.done, "info": r.info})
                self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):  # quiet
            pass
    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--trace", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=30)
    a = ap.parse_args(argv)
    env = CodingTaskEnv(max_steps=a.max_steps, trace_path=a.trace)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(env))
    print(f"k2-tasks-env serving on :{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
