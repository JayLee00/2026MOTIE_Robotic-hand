"""Tiny npz-over-HTTP RPC (stdlib only) shared by model services + orchestrator.

Payloads are numpy .npz blobs so we can ship point clouds / transforms with no
JSON/base64 overhead and no extra deps. Each model microservice runs in its own
conda env; the orchestrator calls `post_npz(url, **arrays)`.
"""
import io
import urllib.error
import urllib.request

import numpy as np


def pack(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def unpack(blob):
    return dict(np.load(io.BytesIO(blob), allow_pickle=False))


def post_npz(url, timeout=600, **arrays):
    """POST arrays as npz, return the response npz as a dict of arrays. On an HTTP error the
    server put its Python traceback in the response BODY (see make_handler) — surface it
    instead of the bare 'HTTP Error 500' so the real cause reaches the caller/timeline."""
    req = urllib.request.Request(url, data=pack(**arrays),
                                 headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return unpack(r.read())
    except urllib.error.HTTPError as e:                      # 500 body carries the server traceback
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"{url} -> HTTP {e.code}\n{body}") from None


def make_handler(predict_fn):
    """Build a BaseHTTPRequestHandler class that pipes /predict npz -> predict_fn."""
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b"ok")
            else:
                self._send(404, b"not found")

        def do_POST(self):
            if self.path != "/predict":
                self._send(404, b"not found")
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                inp = unpack(self.rfile.read(n))
                out = predict_fn(inp)
                self._send(200, pack(**out))
            except Exception as e:  # surface errors to the client
                import traceback
                self._send(500, ("ERR: %s\n%s" % (e, traceback.format_exc())).encode())

        def _send(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(predict_fn, port, name="service"):
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(predict_fn))
    print(f"[{name}] ready on http://127.0.0.1:{port}  (POST /predict npz)", flush=True)
    httpd.serve_forever()
