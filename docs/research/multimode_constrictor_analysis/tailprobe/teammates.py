"""Teammate head-to-heads: games where both of a policy's snakes die on the same turn with
their heads on the same cell (they collided with each other). Usage: python teammates.py games.npz..."""
import sys
import numpy as np
for f in sys.argv[1:]:
    d = np.load(f)
    fa, et, sg, h, body = d["final_alive"], d["final_elim_turn"], d["seat_g"], d["head"], d["body"]
    G, T = h.shape[:2]
    sig = np.array([hash(body[g].tobytes() + sg[g].tobytes()) for g in range(G)])
    _, first = np.unique(sig, return_index=True)
    out = {}
    for p in ("G", "S"):
        n = 0
        for g in first:
            i, j = np.nonzero(sg[g] if p == "G" else ~sg[g])[0]
            if not fa[g, i] and not fa[g, j] and et[g, i] == et[g, j] and et[g, i] < T:
                n += int((h[g, et[g, i], i] == h[g, et[g, i], j]).all())
        out[p] = n
    print(f"{f}: distinct games {len(first)}; games where the two teammates killed each other head-on: {out}")
