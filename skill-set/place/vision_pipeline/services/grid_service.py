"""Live image-grid web service (host, env `anyplace_cu128`). The pipeline POSTs stage
images (npz {name, image}) to /put; this composes them into the A|B grid and serves an
auto-refreshing page a browser shows. Opens the browser automatically on startup.

    python vision_pipeline/services/grid_service.py --port 8815

Layout (small margins) — row1 = 5 cells, row2 = 4 cells:
  a11 rgb_parent   a12 depth_parent   a14 mask_parent   b11 local_place (all)   b21 local_place (selected)
  a21 rgb_child    a22 depth_child    a23 rgb+point_child   a24 mask_child
(the old parent Molmo place-point cell 'a13' is dropped — unused in the fruit demo.)
"""
import argparse
import io
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw

CW, CH, MARGIN, GAP = 320, 240, 10, 6
# One uniform grid. Row 1 (5): parent rgb/depth/mask, molmo local_place all + selected.
# Row 2 (4): child rgb/depth/point/mask. a13 (parent place-point) removed.
ROWS = [
    ["a11", "a12", "a14", "b11", "b21"],
    ["a21", "a22", "a23", "a24"],
]
TITLES = {
    "a11": "rgb_parent", "a12": "depth_parent", "a14": "mask_parent",
    "b11": "molmo local_place (all)", "b21": "local_place (selected)",
    "a21": "rgb_child", "a22": "depth_child", "a23": "rgb + point_child", "a24": "mask_child",
}

_state = {}                                   # cell -> HxWx3 uint8
_lock = threading.Lock()


def _fit(img, w, h):
    im = Image.fromarray(np.asarray(img, np.uint8)).convert("RGB")
    im.thumbnail((w, h), Image.BILINEAR)
    bg = Image.new("RGB", (w, h), (20, 20, 20))
    bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bg


def compose():
    ncols = max(len(row) for row in ROWS)
    nrows = len(ROWS)
    grid_w = ncols * CW + (ncols - 1) * GAP
    grid_h = nrows * CH + (nrows - 1) * GAP
    total_w = MARGIN + grid_w + MARGIN
    total_h = MARGIN + grid_h + MARGIN
    canvas = Image.new("RGB", (total_w, total_h), (35, 35, 38))
    draw = ImageDraw.Draw(canvas)

    def put(name, x, y):
        with _lock:
            img = _state.get(name)
        if img is None:
            cell = Image.new("RGB", (CW, CH), (55, 55, 58))
            ImageDraw.Draw(cell).text((8, CH // 2 - 4), f"(waiting) {name}", fill=(150, 150, 150))
        else:
            cell = _fit(img, CW, CH)
        canvas.paste(cell, (x, y))
        draw.rectangle([x, y, x + CW - 1, y + CH - 1], outline=(90, 90, 95))
        draw.rectangle([x, y, x + CW - 1, y + 15], fill=(0, 0, 0))
        draw.text((x + 4, y + 3), TITLES.get(name, name), fill=(255, 235, 60))

    y0 = MARGIN
    for r, row in enumerate(ROWS):                     # left-aligned; row 2 has one fewer cell
        for c, name in enumerate(row):
            put(name, MARGIN + c * (CW + GAP), y0 + r * (CH + GAP))
    return canvas


_PAGE = b"""<!doctype html><html><head><title>place pipeline grid</title>
<style>body{background:#222;margin:0;text-align:center}img{max-width:100%}</style></head>
<body><img id=g src="/grid.png"><script>
setInterval(()=>{document.getElementById('g').src='/grid.png?t='+Date.now()},500)
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, b"ok", "text/plain")
        if self.path.startswith("/grid.png"):
            buf = io.BytesIO()
            compose().save(buf, format="PNG")
            return self._send(200, buf.getvalue(), "image/png")
        return self._send(200, _PAGE, "text/html")

    def do_POST(self):
        if self.path != "/put":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            d = dict(np.load(io.BytesIO(self.rfile.read(n)), allow_pickle=False))
            name = str(d["name"])
            with _lock:
                _state[name] = np.asarray(d["image"], np.uint8)
            self._send(200, b"ok", "text/plain")
        except Exception as e:                                 # noqa: BLE001
            self._send(500, str(e).encode(), "text/plain")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8815)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[grid] ready on {url}  (POST /put npz{{name,image}}; browser auto-refresh)", flush=True)
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:                                      # noqa: BLE001
            print(f"[grid] open {url} in a browser", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
