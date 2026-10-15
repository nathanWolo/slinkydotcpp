//! The network forward pass, config-driven and in plain `f32`.
//!
//! Ported from `az_modes.network` (`baselines/az_modes.py:198-206`), `unified.conv_features`
//! (`src/slinky/unified.py:431-445`), `unified.torso` (`unified.py:461-470`) and the SAME-padding
//! convolution `unified.conv3x3` (`unified.py:338-348`, mirroring XLA). See `SPEC.md` section 2.
//!
//! The convolutions are direct (no im2col copy): for every output pixel and every in-bounds
//! kernel tap, each non-zero input activation is scattered into the `cout` outputs as one
//! contiguous multiply-add over a weight row, which the compiler vectorises. Skipping zero
//! activations makes the sparse first layer and the post-ReLU layers cheaper without changing
//! any result. Summation order differs from XLA, so results agree to about 1e-6, not bitwise.

use crate::config::{
    FeatureMode, NUM_ACTIONS, NUM_FEATURES, NetConfig, OBS_CHANNELS, OBS_COLS, OBS_LEN, OBS_ROWS,
};
use crate::error::NetError;
use crate::npy::{Array, parse_npz};
use std::collections::HashMap;
use std::path::Path;

/// `unified._NO_MODE_KEEP`: features kept for `features="no_mode"` (`turn`, `opponents_alive`).
const NO_MODE_KEEP: [f32; NUM_FEATURES] = [0., 0., 0., 0., 1., 0., 1.];

struct Conv {
    cin: usize,
    cout: usize,
    stride: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    /// Zeros added before the first row / column (`pad_total / 2`).
    pad_lo: usize,
    /// HWIO, flattened `[dy][dx][cin][cout]`.
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Conv {
    fn out_len(&self) -> usize {
        self.out_h * self.out_w * self.cout
    }

    /// `relu(conv(x) + b)` for one board; `x` is `[in_h][in_w][cin]`, `out` is `[out_h][out_w][cout]`.
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        let (cin, cout) = (self.cin, self.cout);
        out.fill(0.0);
        for oy in 0..self.out_h {
            for ox in 0..self.out_w {
                let acc = &mut out[(oy * self.out_w + ox) * cout..][..cout];
                for dy in 0..3 {
                    let Some(iy) = (oy * self.stride + dy).checked_sub(self.pad_lo) else {
                        continue;
                    };
                    if iy >= self.in_h {
                        continue;
                    }
                    for dx in 0..3 {
                        let Some(ix) = (ox * self.stride + dx).checked_sub(self.pad_lo) else {
                            continue;
                        };
                        if ix >= self.in_w {
                            continue;
                        }
                        let px = &x[(iy * self.in_w + ix) * cin..][..cin];
                        let wt = &self.w[(dy * 3 + dx) * cin * cout..][..cin * cout];
                        for (&a, wrow) in px.iter().zip(wt.chunks_exact(cout)) {
                            if a != 0.0 {
                                for (o, &wv) in acc.iter_mut().zip(wrow) {
                                    *o += a * wv;
                                }
                            }
                        }
                    }
                }
                for (o, &bv) in acc.iter_mut().zip(&self.b) {
                    *o = (*o + bv).max(0.0);
                }
            }
        }
    }
}

/// A loaded network. Immutable, so one value can serve many threads; use one [`Scratch`] each.
pub struct Network {
    cfg: NetConfig,
    convs: Vec<Conv>,
    flat: usize,
    hidden_w: Vec<f32>,
    hidden_b: Vec<f32>,
    feat_w: Option<Vec<f32>>,
    policy_w: Vec<f32>,
    policy_b: [f32; NUM_ACTIONS],
    value_w: Vec<f32>,
    value_b: f32,
}

/// Reusable work buffers for [`Network::forward_into`], so an evaluation allocates nothing.
pub struct Scratch {
    ping: Vec<f32>,
    pong: Vec<f32>,
    hid: Vec<f32>,
    term: Vec<f32>,
}

/// Policy logits and value of a batch, as returned by [`Network::forward`].
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// Raw policy logits per row, in the order UP, DOWN, LEFT, RIGHT (no masking, no softmax).
    pub logits: Vec<[f32; NUM_ACTIONS]>,
    /// `tanh` value per row, in (-1, 1).
    pub values: Vec<f32>,
}

fn take(m: &mut HashMap<String, Array>, key: &str, shape: &[usize]) -> Result<Vec<f32>, NetError> {
    let a = m
        .remove(key)
        .ok_or_else(|| NetError::Shape(format!("missing array {key}")))?;
    if a.shape != shape {
        return Err(NetError::Shape(format!(
            "{key}: file has shape {:?}, config needs {shape:?}",
            a.shape
        )));
    }
    Ok(a.data)
}

impl Network {
    /// Load `config.json` and `params.npz` from a run dir.
    pub fn load(run_dir: impl AsRef<Path>) -> Result<Network, NetError> {
        let dir = run_dir.as_ref();
        let read = |name: &str| {
            let p = dir.join(name);
            std::fs::read(&p).map_err(|e| NetError::Io(p.display().to_string(), e))
        };
        let cfg = NetConfig::from_json(&String::from_utf8_lossy(&read("config.json")?))?;
        Network::from_npz_bytes(cfg, &read("params.npz")?)
    }

    /// Build from a parsed config and the bytes of a `params.npz`.
    pub fn from_npz_bytes(cfg: NetConfig, npz: &[u8]) -> Result<Network, NetError> {
        Network::from_arrays(cfg, parse_npz(npz)?)
    }

    /// Build from a config and named arrays (`"conv0/w"`, ...), validating every shape.
    pub fn from_arrays(cfg: NetConfig, mut m: HashMap<String, Array>) -> Result<Network, NetError> {
        let (mut h, mut w, mut cin) = (OBS_ROWS, OBS_COLS, OBS_CHANNELS);
        let mut convs = Vec::new();
        for (i, (&cout, &stride)) in cfg.conv_channels.iter().zip(&cfg.conv_strides).enumerate() {
            let (oh, ow) = (h.div_ceil(stride), w.div_ceil(stride));
            let pad_total = ((oh - 1) * stride + 3).saturating_sub(h);
            convs.push(Conv {
                cin,
                cout,
                stride,
                in_h: h,
                in_w: w,
                out_h: oh,
                out_w: ow,
                pad_lo: pad_total / 2,
                w: take(&mut m, &format!("conv{i}/w"), &[3, 3, cin, cout])?,
                b: take(&mut m, &format!("conv{i}/b"), &[cout])?,
            });
            (h, w, cin) = (oh, ow, cout);
        }
        let flat = h * w * cin;
        let hid = cfg.hidden;
        let hidden_w = take(&mut m, "hidden/w", &[flat, hid])?;
        let hidden_b = take(&mut m, "hidden/b", &[hid])?;
        let feat_w = match cfg.features {
            FeatureMode::None => None,
            _ => Some(take(&mut m, "feat/w", &[NUM_FEATURES, hid])?),
        };
        let policy_w = take(&mut m, "policy/w", &[hid, NUM_ACTIONS])?;
        let mut policy_b = [0.0; NUM_ACTIONS];
        policy_b.copy_from_slice(&take(&mut m, "policy/b", &[NUM_ACTIONS])?);
        let value_w = take(&mut m, "value/w", &[hid, 1])?;
        let value_b = take(&mut m, "value/b", &[1])?[0];
        Ok(Network {
            cfg,
            convs,
            flat,
            hidden_w,
            hidden_b,
            feat_w,
            policy_w,
            policy_b,
            value_w,
            value_b,
        })
    }

    /// The config this network was built from.
    pub fn config(&self) -> &NetConfig {
        &self.cfg
    }

    /// Number of parameters.
    pub fn num_params(&self) -> usize {
        self.convs
            .iter()
            .map(|c| c.w.len() + c.b.len())
            .sum::<usize>()
            + self.hidden_w.len()
            + self.hidden_b.len()
            + self.feat_w.as_ref().map_or(0, Vec::len)
            + self.policy_w.len()
            + NUM_ACTIONS
            + self.value_w.len()
            + 1
    }

    /// Fresh work buffers sized for this network.
    pub fn scratch(&self) -> Scratch {
        let widest = self.convs.iter().map(Conv::out_len).max().unwrap_or(0);
        Scratch {
            ping: vec![0.0; widest],
            pong: vec![0.0; widest],
            hid: vec![0.0; self.cfg.hidden],
            term: vec![0.0; self.cfg.hidden],
        }
    }

    /// Evaluate a batch. `obs` is `[B][21][21][13]` row-major (`OBS_LEN` floats per row) and
    /// `feats` is `[B][7]`. Allocates; hot loops should use [`Network::forward_into`].
    pub fn forward(&self, obs: &[f32], feats: &[f32]) -> Result<Output, NetError> {
        let b = feats.len() / NUM_FEATURES;
        let mut out = Output {
            logits: vec![[0.0; NUM_ACTIONS]; b],
            values: vec![0.0; b],
        };
        self.forward_into(
            &mut self.scratch(),
            obs,
            feats,
            &mut out.logits,
            &mut out.values,
        )?;
        Ok(out)
    }

    /// Evaluate a batch into caller-provided buffers without allocating. The batch size is
    /// `values.len()`; `logits` must have the same length. Rows are independent.
    pub fn forward_into(
        &self,
        scratch: &mut Scratch,
        obs: &[f32],
        feats: &[f32],
        logits: &mut [[f32; NUM_ACTIONS]],
        values: &mut [f32],
    ) -> Result<(), NetError> {
        let b = values.len();
        if logits.len() != b || obs.len() != b * OBS_LEN || feats.len() != b * NUM_FEATURES {
            return Err(NetError::Input(format!(
                "batch {b}: got {} obs floats (need {}), {} feature floats (need {}), {} logit rows",
                obs.len(),
                b * OBS_LEN,
                feats.len(),
                b * NUM_FEATURES,
                logits.len()
            )));
        }
        let hid = self.cfg.hidden;
        if scratch.hid.len() != hid
            || scratch.term.len() != hid
            || scratch.ping.len() < self.convs.iter().map(Conv::out_len).max().unwrap_or(0)
        {
            *scratch = self.scratch();
        }
        for r in 0..b {
            let board = &obs[r * OBS_LEN..][..OBS_LEN];
            let (mut cur, mut next) = (&mut scratch.ping, &mut scratch.pong);
            for (i, conv) in self.convs.iter().enumerate() {
                let src: &[f32] = if i == 0 {
                    board
                } else {
                    &cur[..self.convs[i - 1].out_len()]
                };
                conv.apply(src, &mut next[..conv.out_len()]);
                std::mem::swap(&mut cur, &mut next);
            }
            let flat = &cur[..self.flat];

            // h = relu((flat @ W + b) + feats @ F), in that order of float additions.
            let acc = &mut scratch.hid;
            acc.fill(0.0);
            for (&a, wrow) in flat.iter().zip(self.hidden_w.chunks_exact(hid)) {
                if a != 0.0 {
                    for (o, &wv) in acc.iter_mut().zip(wrow) {
                        *o += a * wv;
                    }
                }
            }
            for (o, &bv) in acc.iter_mut().zip(&self.hidden_b) {
                *o += bv;
            }
            if let Some(fw) = &self.feat_w {
                let f = &feats[r * NUM_FEATURES..][..NUM_FEATURES];
                let term = &mut scratch.term;
                term.fill(0.0);
                for (k, (&fv, frow)) in f.iter().zip(fw.chunks_exact(hid)).enumerate() {
                    let fv = if self.cfg.features == FeatureMode::NoMode {
                        fv * NO_MODE_KEEP[k]
                    } else {
                        fv
                    };
                    for (o, &wv) in term.iter_mut().zip(frow) {
                        *o += fv * wv;
                    }
                }
                for (o, &t) in acc.iter_mut().zip(term.iter()) {
                    *o += t;
                }
            }
            for o in acc.iter_mut() {
                *o = o.max(0.0);
            }

            // Heads.
            let mut lg = self.policy_b;
            for (&hv, wrow) in acc.iter().zip(self.policy_w.as_chunks::<NUM_ACTIONS>().0) {
                for (l, &wv) in lg.iter_mut().zip(wrow) {
                    *l += hv * wv;
                }
            }
            logits[r] = lg;
            let v: f32 = acc
                .iter()
                .zip(&self.value_w)
                .map(|(&a, &w)| a * w)
                .sum::<f32>()
                + self.value_b;
            values[r] = v.tanh();
        }
        Ok(())
    }
}
