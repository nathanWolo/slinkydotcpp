"""Write the startup self-benchmark's requests (server crate, data/examples.json).

The Python server benchmarks each snake at startup on ``server.example_request(MODE_CONFIGS[m])``
for the four modes (20 duel moves, 5 of each other mode), and those moves advance its
tie-noise move counter. The native server plays the same requests the same number of times,
so its counter (and so its search keys) line up with a Python server started with the same
snakes. Run (CPU, memory capped):

    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/crates/server/py/make_examples.py [OUT]
"""

import json
import os
import sys

from slinky import modes, server

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "examples.json"
)
examples = {m: server.example_request(server.MODE_CONFIGS[m]) for m in modes.MODE_NAMES}
with open(out, "w") as f:
    json.dump({"modes": list(modes.MODE_NAMES), "requests": examples}, f, indent=1)
    f.write("\n")
print("wrote", out, list(examples))
