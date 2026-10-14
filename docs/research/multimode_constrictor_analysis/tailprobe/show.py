"""ASCII view of a decision: python show.py games.npz decisions.npz idx..."""
import sys
import numpy as np
g = np.load(sys.argv[1]); d = np.load(sys.argv[2])
for i in map(int, sys.argv[3:]):
    gi, ti, si = d['game'][i], d['turn'][i], d['snake'][i]
    body = g['body'][gi, ti]; alive = g['alive'][gi, ti]; head = g['head'][gi, ti]
    H, W = body.shape[1:]
    print(f"dec {i} game {gi} turn {ti} mover {si} (G seat: {d['mover_is_g'][i]}) lengths {g['length'][gi,ti]} alive {alive}")
    print(" legal", d['legal'][i].astype(int), "a_con", d['a_con'][i], "a_std", d['a_std'][i], "G", d['c_lg'][i], "S", d['c_ls'][i], "G_far", d['c_lg_far'][i])
    for y in range(H - 1, -1, -1):
        row = ""
        for x in range(W):
            c = "."
            for s in range(body.shape[0]):
                if alive[s] and body[s, y, x] > 0:
                    v = body[s, y, x]
                    if (head[s] == [x, y]).all():
                        c = "ABCD"[s] if s == si else "abcd"[s]
                    else:
                        c = str(min(v, 9)) if s == si else "#"
            row += c + " "
        print("   ", row)
