"""Mock Lambda Runtime API: spawns a bootstrap, feeds it Function-URL events, times it.

usage: mock.py BOOTSTRAP CPU|- WARM_N [ENV=VALUE ...]
Prints init time (process start to the first /next), cold time (process start to the first move
answered) and warm per-move times of the four example modes (event handed out -> response posted).
"""
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

events = queue.Queue()
results = queue.Queue()
stamps = {}


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def setup(self):
        super().setup()
        import socket
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_GET(self):
        if self.path.endswith('/invocation/next'):
            stamps.setdefault('first_next', time.perf_counter())
            ev = events.get()
            body = json.dumps(ev['payload']).encode()
            self.send_response(200)
            hdr = {
                'Lambda-Runtime-Aws-Request-Id': ev['id'],
                'Lambda-Runtime-Deadline-Ms': str(int(time.time() * 1000) + 60000),
                'Lambda-Runtime-Invoked-Function-Arn': 'arn:aws:lambda:us-east-1:123456789012:function:f',
                'Content-Type': 'application/json',
                'Content-Length': str(len(body)),
            }
            for k, v in hdr.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            stamps['sent_' + ev['id']] = time.perf_counter()
        else:
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(n)
        t = time.perf_counter()
        self.send_response(202)
        self.send_header('Content-Length', '0')
        self.end_headers()
        if '/response' in self.path:
            rid = self.path.split('/')[-2]
            results.put((t, rid, data))


def event(path, method, body, i):
    return {'id': f'req-{i}', 'payload': {
        'version': '2.0', 'routeKey': '$default', 'rawPath': path, 'rawQueryString': '',
        'headers': {'content-type': 'application/json', 'host': 'x.lambda-url.us-east-1.on.aws'},
        'requestContext': {
            'accountId': 'anonymous', 'apiId': 'x', 'domainName': 'x.lambda-url.us-east-1.on.aws',
            'domainPrefix': 'x',
            'http': {'method': method, 'path': path, 'protocol': 'HTTP/1.1', 'sourceIp': '1.2.3.4',
                     'userAgent': 'mock'},
            'requestId': f'req-{i}', 'routeKey': '$default', 'stage': '$default',
            'time': '15/Oct/2026:19:00:00 +0000', 'timeEpoch': 1},
        'body': body, 'isBase64Encoded': False}}


def run(bootstrap, env_extra, reqs, warm_n, cpu=None):
    srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = dict(os.environ, AWS_LAMBDA_RUNTIME_API=f'127.0.0.1:{port}', AWS_LAMBDA_FUNCTION_NAME='f',
               AWS_LAMBDA_FUNCTION_VERSION='1', AWS_LAMBDA_FUNCTION_MEMORY_SIZE='1769',
               AWS_LAMBDA_LOG_GROUP_NAME='g', AWS_LAMBDA_LOG_STREAM_NAME='s', **env_extra)
    cmd = (['taskset', '-c', cpu] if cpu else []) + [bootstrap]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, stdout=open(os.environ.get('MOCK_STDOUT', os.devnull), 'w'), stderr=subprocess.DEVNULL)
    out = {}
    try:
        i = 0
        events.put(event(reqs[0][1], 'POST', reqs[0][2], i))
        i += 1
        t, rid, data = results.get(timeout=30)
        out['cold_first_answer_ms'] = (t - t0) * 1e3
        out['init_ms'] = (stamps['first_next'] - t0) * 1e3
        out['first_answer'] = json.loads(json.loads(data)['body'])
        warm = {}
        for name, path, body in reqs:
            ts = []
            for _ in range(warm_n):
                rid_want = f'req-{i}'
                events.put(event(path, 'POST', body, i))
                i += 1
                t, rid, data = results.get(timeout=30)
                assert rid == rid_want, (rid, rid_want)
                ts.append((t - stamps['sent_' + rid]) * 1e3)
            warm[name] = ts
        out['warm'] = warm
    finally:
        p.terminate()
        p.wait()
        srv.shutdown()
    return out


if __name__ == '__main__':
    bootstrap, cpu, warm_n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    envx = dict(a.split('=', 1) for a in sys.argv[4:])
    ex = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server/data/examples.json')))
    reqs = [(m, '/slinky/move', json.dumps(ex['requests'][m]))
            for m in ['standard', 'duel', 'royale', 'constrictor']]
    res = run(bootstrap, envx, reqs, warm_n, None if cpu == '-' else cpu)
    print('first answer', res['first_answer'])
    print('init (process start to first /next): %.1f ms; cold (process start to first move answered): %.1f ms'
          % (res['init_ms'], res['cold_first_answer_ms']))
    for m, ts in res['warm'].items():
        ts = sorted(ts)
        print('%-11s warm n=%d mean %.2f median %.2f p90 %.2f max %.2f ms'
              % (m, len(ts), statistics.mean(ts), statistics.median(ts), ts[int(0.9 * len(ts)) - 1], ts[-1]))
