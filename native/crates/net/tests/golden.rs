//! Parity of the Rust forward pass with the jitted JAX network on golden vectors written by
//! `native/golden/net_golden.py` (see its docstring). Skips when the files are missing.

use slinky_net::{NUM_FEATURES, Network, OBS_LEN, npy};
use std::path::PathBuf;

fn golden_root() -> PathBuf {
    let base = std::env::var("NAZ_GOLDEN").unwrap_or_else(|_| "/dev/shm/naz-golden".into());
    PathBuf::from(base).join("net")
}

#[test]
fn matches_jax() {
    let root = golden_root();
    let Ok(rd) = std::fs::read_dir(&root) else {
        eprintln!(
            "SKIP golden parity: {} missing (run native/golden/net_golden.py)",
            root.display()
        );
        return;
    };
    let mut names: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    names.sort();
    assert!(!names.is_empty());
    for dir in names {
        let net = Network::load(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let cases = npy::parse_npz(&std::fs::read(dir.join("cases.npz")).unwrap()).unwrap();
        let (obs, feats, logits, value) = (
            &cases["obs"],
            &cases["feats"],
            &cases["logits"],
            &cases["value"],
        );
        let n = value.data.len();
        assert_eq!(obs.data.len(), n * OBS_LEN);
        // Run in uneven batches to exercise batching.
        let (mut max_l, mut max_v, mut max_rel, mut max_real, mut max_mag) =
            (0f32, 0f32, 0f32, 0f32, 0f32);
        let kind = &cases["kind"];
        let mut scratch = net.scratch();
        let mut i = 0;
        let mut step = 1;
        while i < n {
            let b = step.min(n - i);
            let mut lg = vec![[0f32; 4]; b];
            let mut v = vec![0f32; b];
            net.forward_into(
                &mut scratch,
                &obs.data[i * OBS_LEN..(i + b) * OBS_LEN],
                &feats.data[i * NUM_FEATURES..(i + b) * NUM_FEATURES],
                &mut lg,
                &mut v,
            )
            .unwrap();
            for r in 0..b {
                for (a, l) in lg[r].iter().enumerate() {
                    let want = logits.data[(i + r) * 4 + a];
                    let e = (l - want).abs();
                    max_l = max_l.max(e);
                    max_mag = max_mag.max(want.abs());
                    max_rel = max_rel.max(e / want.abs().max(1.0));
                    if kind.data[i + r] < 3.5 {
                        max_real = max_real.max(e);
                    }
                }
                max_v = max_v.max((v[r] - value.data[i + r]).abs());
            }
            i += b;
            step = step % 7 + 1;
        }
        println!(
            "{}: {n} cases, {} params, max |dlogit| {max_l:.3e} (real states {max_real:.3e}, relative {max_rel:.3e}, max |logit| {max_mag:.1}), max |dvalue| {max_v:.3e}",
            dir.display(),
            net.num_params()
        );
        assert!(
            max_rel < 2e-4 && max_real < 1e-4 && max_v < 1e-4,
            "{}: parity {max_l} {max_v}",
            dir.display()
        );
    }
}

#[test]
fn bad_run_dirs_are_errors() {
    assert!(Network::load("/nonexistent/run").is_err());
    let root = golden_root().join("M");
    if !root.exists() {
        return;
    }
    // A config whose shapes disagree with the weights must fail cleanly.
    let tmp = std::env::temp_dir().join(format!("naz-net-bad-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::copy(root.join("params.npz"), tmp.join("params.npz")).unwrap();
    let cfg = std::fs::read_to_string(root.join("config.json"))
        .unwrap()
        .replace("\"hidden\": 256", "\"hidden\": 128");
    std::fs::write(tmp.join("config.json"), cfg).unwrap();
    assert!(matches!(
        Network::load(&tmp),
        Err(slinky_net::NetError::Shape(_))
    ));
    let cfg = std::fs::read_to_string(root.join("config.json"))
        .unwrap()
        .replace("float32", "bfloat16");
    std::fs::write(tmp.join("config.json"), cfg).unwrap();
    assert!(matches!(
        Network::load(&tmp),
        Err(slinky_net::NetError::Unsupported(_))
    ));
    let _ = std::fs::remove_dir_all(&tmp);
}
