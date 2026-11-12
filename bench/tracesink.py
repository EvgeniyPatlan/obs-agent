"""Trace sink: receives span batches and counts spans and bytes per service.
GET /stats returns running totals (the lab reads deltas)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

lock = threading.Lock()
totals = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            batch = json.loads(raw)
        except ValueError:
            self._send(400, {"error": "bad json"})
            return
        per_span = len(raw) / max(len(batch), 1)
        with lock:
            for span in batch:
                t = totals.setdefault(span.get("service", "?"), {"spans": 0, "bytes": 0.0})
                t["spans"] += 1
                t["bytes"] += per_span
        self._send(200, {"accepted": len(batch)})

    def do_GET(self):
        with lock:
            self._send(200, totals)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    srv.daemon_threads = True
    srv.serve_forever()
