//! `cargo run --release -p slinky-net --example bench -- <run dir> [golden cases.npz]`:
//! microseconds per network evaluation (one board), single core.
use slinky_net::{NUM_FEATURES, Network, OBS_LEN, npy};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = Network::load(&args[1]).expect("load");
    let (obs, feats) = match args.get(2) {
        Some(p) => {
            let c = npy::parse_npz(&std::fs::read(p).unwrap()).unwrap();
            (c["obs"].data.clone(), c["feats"].data.clone())
        }
        None => (vec![0.3; 64 * OBS_LEN], vec![0.5; 64 * NUM_FEATURES]),
    };
    let n = feats.len() / NUM_FEATURES;
    let mut s = net.scratch();
    let (mut lg, mut v) = (vec![[0f32; 4]; n], vec![0f32; n]);
    for batch in [1usize, 4] {
        let reps = 20_000 / batch.max(1);
        let t = Instant::now();
        let mut sink = 0f32;
        for r in 0..reps {
            let i = (r * batch) % (n - batch + 1);
            net.forward_into(
                &mut s,
                &obs[i * OBS_LEN..(i + batch) * OBS_LEN],
                &feats[i * NUM_FEATURES..(i + batch) * NUM_FEATURES],
                &mut lg[..batch],
                &mut v[..batch],
            )
            .unwrap();
            sink += v[0];
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / (reps * batch) as f64;
        println!("batch {batch}: {us:.1} us per evaluation (sink {sink})");
    }
}
