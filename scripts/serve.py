"""Local static server that declares UTF-8, like GitHub Pages does.

`python -m http.server` sends `text/markdown` without a charset, so browsers
render the .md twins as windows-1252 mojibake. Pages sends `charset=utf-8`.
"""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # no-cache = always revalidate (304 when unchanged), so a rebuild never
        # needs a hard reload; production busts caches via ?v= instead.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        ctype = super().guess_type(path)
        if ctype.startswith("text/") or ctype in ("application/xml", "application/json"):
            return f"{ctype}; charset=utf-8"
        return ctype


if __name__ == "__main__":
    port = int(sys.argv[1])
    ThreadingHTTPServer(("", port), partial(Handler, directory="website")).serve_forever()
