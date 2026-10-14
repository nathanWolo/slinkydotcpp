"""Where is the deficit created? Hybrid teams in constrictor, played against pure S.

Team 0 (seated as G in the G,S,G,S line-up, rotated) plays one policy's greedy move for the first K
moves and the other policy's afterwards; team 1 is pure S. Variants:
  S->G(K): S's moves for moves < K, then G's   ('G with an S opening')
  G->S(K): G's moves for moves < K, then S's   ('S with a G opening')
K=0 gives pure G for S->G and pure S for G->S; K large gives pure S for S->G and pure G for G->S.
Same 512 games as before (all 96 distinct (placement, seat-parity) configurations), deduplicated; the
reported share is the unweighted mean over those 96 games (the exact population value).
"""
import json, sys, time
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
sys.path.insert(0, "/home/user/slinkydotcpp/baselines")
import jax, jax.numpy as jnp, numpy as np
import simulate as sim
from common import *

mode = "constrictor"
games = 512
ids = jnp.arange(games, dtype=jnp.int32)
key = sim.cell_key(0, mode)
out = {}
Ks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 100]
for first in ("S", "G"):
    t0 = time.time()
    game, run = sim.build(mode, 80, False, "runs/ppo-modes-seed0", f"runs/ppo-modes-{mode}-seed0", hyb_first=first)
    name = "S->G" if first == "S" else "G->S"
    for K in Ks:
        t, states, bufs = run(key, ids, jnp.int32(K))
        jax.block_until_ready(bufs)
        t = int(t)
        raw = {k: np.asarray(v)[:, :t] for k, v in bufs.items()}
        raw.update(final_elim_turn=np.asarray(states.elim_turn), final_elim_cause=np.asarray(states.elim_cause), final_length=np.asarray(states.length), final_alive=np.asarray(states.alive), final_turn=np.asarray(states.turn), final_done=np.asarray(states.done))
        d = contest_outcome(load(raw, dedupe=True))
        boot = Boot(d["G"], cluster=d["cluster"])
        sh = d["share"][:, 0]
        rec = dict(K=K, share=boot.ratio(sh), win=boot.ratio((sh == 1).astype(float)), loss=boot.ratio((sh == 0).astype(float)), draw=boot.ratio(d["draw"].astype(float)),
                   n_win=int((sh == 1).sum()), n_loss=int((sh == 0).sum()), n_draw=int(d["draw"].sum()), n_games=int(d["G"]))
        # sample (multiplicity-weighted, the 512 games) share for reference
        w = d["mult"]; rec["share_512"] = float((sh * w).sum() / w.sum())
        # team-kill: same-team head-on in contested play (hybrid team vs S): count games where team0 snakes die at the same move by head-on with each other
        out.setdefault(name, []).append(rec)
        print(f"{name} K={K:3d}: G-team share {fmt(rec['share'])} (exact mean over 96 starts {rec['share'][0]:.4f}; 512-sample {rec['share_512']:.4f})  wins/draws/losses {rec['n_win']}/{rec['n_draw']}/{rec['n_loss']}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open("%s/hybrid_constrictor.json" % "docs/research/multimode_constrictor_analysis/behaviour", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
