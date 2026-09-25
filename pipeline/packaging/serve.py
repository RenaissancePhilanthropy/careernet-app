"""Serve the CareerNet app to this machine only, and stop when the window closes.

Started by start.bat / start.command. Two things it does that `python -m http.server`
does not:

  * binds 127.0.0.1 rather than 0.0.0.0, so the corpus is not offered to everyone else
    on the network you happen to be on
  * exits when the window that launched it goes away, instead of being left listening
"""
import ctypes
import functools
import http.server
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser

HOST = "127.0.0.1"
FIRST_PORT = 8123
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def port_in_use(port):
    """Ask whether anything answers, rather than whether we can bind.

    On Windows a successful bind() does not prove the port is free: http.server sets
    SO_REUSEADDR, so two servers can end up sharing 8123 and the second one quietly
    receives nothing. Connecting is the honest test.
    """
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex((HOST, port)) == 0


def already_ours(port):
    """Is CareerNet already being served on this port by an earlier run?"""
    try:
        with urllib.request.urlopen(
                f"http://{HOST}:{port}/search/data/manifest.json", timeout=2) as r:
            return "soc_major" in json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return False


def next_free(start, tries=25):
    for port in range(start + 1, start + tries):
        if not port_in_use(port):
            return port
    raise SystemExit(f"No free port between {start} and {start + tries - 1}.")


if os.name == "nt":
    _k32 = ctypes.windll.kernel32

    def parent_alive(pid):
        h = _k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = _k32.GetExitCodeProcess(h, ctypes.byref(code))
        _k32.CloseHandle(h)
        return bool(ok) and code.value == 259      # STILL_ACTIVE
else:
    def parent_alive(pid):
        return os.getppid() == pid


def watchdog(pid):
    while True:
        time.sleep(2)
        if not parent_alive(pid):
            os._exit(0)


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass  # the console is for instructions, not a request log


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"Cannot find the app files. Expected a 'static' folder next to "
                         f"this script, at:\n  {ROOT}\n"
                         f"Unzip the whole download and keep the files together.")
    port = FIRST_PORT
    if port_in_use(port):
        if already_ours(port):
            # Reuse the run that is already going. Starting a second server on another port
            # would make a new browser origin, and the browser stores the 300MB model per
            # origin - so a second copy would be downloaded and kept.
            url = f"http://{HOST}:{port}/search/"
            print()
            print("  CareerNet is already running at " + url)
            print("  Opening it. Close the other window when you are finished.")
            print()
            webbrowser.open(url)
            return
        port = next_free(FIRST_PORT)
        print()
        print(f"  Port {FIRST_PORT} is taken by something else, so this is on {port} instead.")
        print("  Your browser treats that as a different site, so it will download the")
        print("  300MB model again for this address. To avoid that, close whatever is using")
        print(f"  port {FIRST_PORT} and run this again.")
    url = f"http://{HOST}:{port}/search/"

    threading.Thread(target=watchdog, args=(os.getppid(),), daemon=True).start()

    handler = functools.partial(Handler, directory=ROOT)
    httpd = http.server.ThreadingHTTPServer((HOST, port), handler)

    print()
    print("  CareerNet is running at " + url)
    print()
    print("  Leave this window open while you use it.")
    print("  Close the window, or press Ctrl+C, to stop.")
    print("  Reachable only from this computer.")
    print()
    threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
