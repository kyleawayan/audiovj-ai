"""Serve a session folder with HTTP Range support, then open the review page.

``python -m http.server`` ignores the Range header and always returns the whole
file with a 200. Browsers need 206 partial responses to seek inside a media
file, so with the stock server a 116 MB audio.wav can only ever play from the
beginning -- clicking anywhere restarts it.

Usage: uv run python experiments/_serve.py <session-dir> [port]
"""
import functools, http.server, os, re, socketserver, sys, webbrowser
from pathlib import Path

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        m = RANGE_RE.match(rng.strip())
        if not m:
            return super().send_head()
        size = os.path.getsize(path)
        first, last = m.group(1), m.group(2)
        if first == "":                      # suffix form: bytes=-N
            length = int(last or 0)
            start, end = max(0, size - length), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _Limited(f, end - start + 1)


class _Limited:
    """File wrapper that stops after n bytes, so copyfile sends only the range."""

    def __init__(self, f, n):
        self._f, self._left = f, n

    def read(self, size=-1):
        if self._left <= 0:
            return b""
        if size < 0 or size > self._left:
            size = self._left
        b = self._f.read(size)
        self._left -= len(b)
        return b

    def close(self):
        self._f.close()


class Threaded(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    sess = Path(sys.argv[1]).expanduser().resolve()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8412
    handler = functools.partial(RangeHandler, directory=str(sess))
    with Threaded(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/review.html"
        print(f"serving {sess}\n  {url}\n  Ctrl+C to stop")
        webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
