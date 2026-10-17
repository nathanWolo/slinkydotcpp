//! `cargo run --release -p slinky-net --example bench -- <run dir> [golden cases.npz] [reps]`:
//! microseconds per `forward_into` call at batch sizes 1, 2, 4 and 8, on whatever core the
//! process runs (pin it: `taskset -c 2 ...`). With a golden `cases.npz` the boards are its real
//! game states (`kind < 4`); otherwise a dense synthetic board. Prints the best and median of
//! `reps` timed rounds of about 0.2 s each. `INT8=1` benchmarks the experimental int8 hidden
//! layer ([`Network::quantize_hidden_int8`]) and prints its error against the `f32` network.
use slinky_net::{NUM_FEATURES, Network, OBS_LEN, npy};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut net = Network::load(&args[1]).expect("load");
    let (obs, feats) = match args.get(2) {
        Some(p) => {
            let c = npy::parse_npz(&std::fs::read(p).unwrap()).unwrap();
            let kind = &c["kind"].data;
            let (mut o, mut f) = (Vec::new(), Vec::new());
            for (i, &k) in kind.iter().enumerate() {
                if k < 3.5 {
                    o.extend_from_slice(&c["obs"].data[i * OBS_LEN..(i + 1) * OBS_LEN]);
                    f.extend_from_slice(&c["feats"].data[i * NUM_FEATURES..(i + 1) * NUM_FEATURES]);
                }
            }
            (o, f)
        }
        None => (vec![0.3; 64 * OBS_LEN], vec![0.5; 64 * NUM_FEATURES]),
    };
    if std::env::var("INT8").is_ok_and(|v| v == "1") {
        let n = feats.len() / NUM_FEATURES;
        let want = net.forward(&obs, &feats).unwrap();
        net.quantize_hidden_int8(true);
        let got = net.forward(&obs, &feats).unwrap();
        let (mut dl, mut dv, mut ml, mut flips) = (0f32, 0f32, 0f32, 0);
        let argmax = |l: &[f32; 4]| (0..4).fold(0, |b, a| if l[a] > l[b] { a } else { b });
        for r in 0..n {
            flips += (argmax(&got.logits[r]) != argmax(&want.logits[r])) as usize;
            for a in 0..4 {
                dl = dl.max((got.logits[r][a] - want.logits[r][a]).abs());
                ml = ml.max(want.logits[r][a].abs());
            }
            dv = dv.max((got.values[r] - want.values[r]).abs());
        }
        println!(
            "int8 hidden layer: max |dlogit| {dl:.3e} (max |logit| {ml:.1}), max |dvalue| {dv:.3e}, policy argmax changed on {flips} of {n} boards"
        );
    }
    let reps: usize = args.get(3).map_or(7, |s| s.parse().unwrap());
    let n = feats.len() / NUM_FEATURES;
    let mut s = net.scratch();
    let (mut lg, mut v) = (vec![[0f32; 4]; 8], vec![0f32; 8]);
    let mut sink = 0f32;
    let mut call = |i: usize, batch: usize, sink: &mut f32| {
        net.forward_into(
            &mut s,
            &obs[i * OBS_LEN..(i + batch) * OBS_LEN],
            &feats[i * NUM_FEATURES..(i + batch) * NUM_FEATURES],
            &mut lg[..batch],
            &mut v[..batch],
        )
        .unwrap();
        *sink += v[0];
    };
    println!("{n} boards");
    for batch in [1usize, 2, 4, 8] {
        // Calibrate the call count to ~0.2 s per round.
        let t = Instant::now();
        let mut calls = 0;
        while t.elapsed().as_secs_f64() < 0.05 {
            call((calls * batch) % (n - batch + 1), batch, &mut sink);
            calls += 1;
        }
        let calls = calls * 4;
        let mut times: Vec<f64> = (0..reps)
            .map(|_| {
                let t = Instant::now();
                for c in 0..calls {
                    call((c * batch) % (n - batch + 1), batch, &mut sink);
                }
                t.elapsed().as_secs_f64() * 1e6 / calls as f64
            })
            .collect();
        times.sort_by(f64::total_cmp);
        println!(
            "B={batch}: best {:.2} us/call, median {:.2} us/call ({:.2} us/row)",
            times[0],
            times[reps / 2],
            times[reps / 2] / batch as f64
        );
    }
    println!("(sink {sink})");
}
