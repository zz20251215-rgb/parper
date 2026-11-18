import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.monitoring import TrainingMonitor


def make_handler(monitor):
    assets = {'/': ('index.html', 'text/html; charset=utf-8'),
              '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
              '/styles.css': ('styles.css', 'text/css; charset=utf-8')}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/state':
                body = json.dumps(monitor.state(), allow_nan=False).encode()
                mime = 'application/json; charset=utf-8'
            elif path in assets:
                filename, mime = assets[path]
                try:
                    body = (PROJECT / 'src/dashboard' / filename).read_bytes()
                except OSError:
                    self.send_error(404, 'Dashboard asset unavailable')
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            # Polling is frequent; retain only unexpected responses in server logs.
            if len(args) > 1 and str(args[1]) != '200':
                super().log_message(format, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('outputs/runs/tack_baseline'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    run = (PROJECT / args.run).resolve()
    allowed = (PROJECT / 'outputs/runs').resolve()
    if not run.is_relative_to(allowed) or not run.is_dir():
        parser.error('--run must be an existing directory inside outputs/runs')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    monitor = TrainingMonitor(run)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(monitor))
    monitor.start()
    print(f'Dashboard: http://127.0.0.1:{args.port} | run={run.name}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()
        server.server_close()


if __name__ == '__main__':
    main()
