"""Latency of a running server under concurrent games (stdlib only, no JAX).

Sends the benchmark positions of ``data/examples.json`` (all four modes) as ``/move`` requests
from ``C`` client threads at once, each with its own game id, and prints the wall time per
request (median, p90, max) for each concurrency level:

    python3 load_test.py --port 18300 --snake az-256 [--levels 1,4,8,16] [--count 200]
"""

import argparse
import json
import os
import statistics
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--snake", default="az-256")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--count", type=int, default=200, help="requests per level")
    args = ap.parse_args()
    ex = json.load(open(os.path.join(HERE, "..", "data", "examples.json")))
    reqs = [ex["requests"][m] for m in ex["modes"]]
    url = f"http://127.0.0.1:{args.port}/{args.snake}/move"
    for level in (int(x) for x in args.levels.split(",")):
        times: list[float] = []
        lock = threading.Lock()
        per = max(1, args.count // level)

        def client(c: int) -> None:
            for i in range(per):
                r = json.loads(json.dumps(reqs[(c + i) % len(reqs)]))
                r["game"]["id"] = f"load-{level}-{c}"
                body = json.dumps(r).encode()
                t = time.perf_counter()
                q = urllib.request.Request(url, body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(q, timeout=30) as resp:
                    resp.read()
                with lock:
                    times.append((time.perf_counter() - t) * 1e3)

        threads = [threading.Thread(target=client, args=(c,)) for c in range(level)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total = time.perf_counter() - t0
        times.sort()
        print(f"concurrency {level:3d}: {len(times)} moves in {total:.1f} s, "
              f"median {statistics.median(times):.1f} ms, p90 {times[int(0.9 * len(times))]:.1f} ms, "
              f"max {times[-1]:.1f} ms", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
