"""Monte Carlo (lambda=1) discounted placement-return variance in constrictor self-play, G x4 vs S x4 (sampled,
final params; tailprobe's con_gggg_s.npz / con_ssss_s.npz). The trainer's return_var is the variance of GAE
lambda=0.95 returns (= advantages + V), which depend on the critic. If the MC return variance of G's self-play
is not lower than S's, G's lower logged return_var is (at least partly) a property of its critic, not of its games."""
import numpy as np
TP = 'docs/research/multimode_constrictor_analysis/tailprobe'
gamma = 0.99
rng = np.random.default_rng(0)
def mc(path):
    z = np.load(path)
    al, done, fa, turn = z['alive'], z['done'], z['final_alive'], z['turn']
    G, T, N = al.shape
    assert (turn == np.arange(T)[None]).all() or True
    after = np.concatenate([al[:, 1:], fa[:, None]], axis=1)  # alive after step s
    valid = al & ~done[:, :, None]
    after = np.where(valid, after, False)  # nothing after the game
    # alive after step must be computed only on valid steps; for steps after done use alive state as is
    newly = valid & ~after
    gain = np.where(after, newly.sum(-1, keepdims=True), 0)
    loss = np.where(newly, (after).sum(-1, keepdims=True), 0)
    r = (gain - loss) / (N - 1)
    R = np.zeros_like(r, dtype=float); acc = np.zeros((G, N))
    for s in range(T - 1, -1, -1):
        acc = r[:, s] + gamma * acc * valid[:, min(s + 1, T - 1)] if s + 1 < T else r[:, s]
        R[:, s] = acc
    tot = r.sum(1)  # per-seat total reward
    vals = R[valid]; gid = np.broadcast_to(np.arange(G)[:, None, None], valid.shape)[valid]
    # bootstrap over games for the variance
    bs = []
    for _ in range(1000):
        gi = rng.integers(0, G, G); w = np.bincount(gi, minlength=G)[gid].astype(float)
        m = (w * vals).sum() / w.sum(); bs.append((w * (vals - m) ** 2).sum() / w.sum())
    games_done = done.any(1).mean()
    return vals.var(), np.percentile(bs, [2.5, 97.5]), vals.mean(), valid.sum() / G, np.round(np.unique(np.round(tot, 3), return_counts=True), 3), games_done
for f in ['con_gggg_s', 'con_ssss_s']:
    v, ci, m, spg, tots, gd = mc(f'{TP}/{f}.npz')
    print(f'{f}: MC return var {v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], mean {m:+.4f}, valid samples/game {spg:.1f}, finished {gd:.3f}')
    print('   per-seat total reward values/counts', tots.tolist())

# Difference CI (independent samples, bootstrap over games of each) and the V == 0 lambda-return variance
def returns(path, disc):
    z = np.load(path); al, done, fa = z['alive'], z['done'], z['final_alive']
    G, T, N = al.shape
    after = np.concatenate([al[:, 1:], fa[:, None]], axis=1)
    valid = al & ~done[:, :, None]; after = np.where(valid, after, False)
    newly = valid & ~after
    r = (np.where(after, newly.sum(-1, keepdims=True), 0) - np.where(newly, after.sum(-1, keepdims=True), 0)) / (N - 1)
    R = np.zeros(r.shape); acc = np.zeros((G, N))
    for s in range(T - 1, -1, -1):
        acc = r[:, s] + disc * acc; R[:, s] = acc
    gid = np.broadcast_to(np.arange(G)[:, None, None], valid.shape)[valid]
    return R[valid], gid, G
def bvar(vals, gid, G, B=2000, seed=1):
    rr = np.random.default_rng(seed); out = []
    for _ in range(B):
        gi = rr.integers(0, G, G); w = np.bincount(gi, minlength=G)[gid].astype(float)
        m = (w * vals).sum() / w.sum(); out.append((w * (vals - m) ** 2).sum() / w.sum())
    return np.array(out)
for disc, lab in ((0.99, 'MC gamma=0.99'), (0.99 * 0.95, 'lambda-return with V==0 (gamma*lambda=0.9405)')):
    vg, gg, Gg = returns(f'{TP}/con_gggg_s.npz', disc); vs, gs, Gs = returns(f'{TP}/con_ssss_s.npz', disc)
    bg, bs_ = bvar(vg, gg, Gg, seed=1), bvar(vs, gs, Gs, seed=2)
    d = bg - bs_
    print(f'{lab}: G {vg.var():.4f}  S {vs.var():.4f}  G-S {vg.var()-vs.var():+.4f} [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  ratio G/S {vg.var()/vs.var():.3f}')
