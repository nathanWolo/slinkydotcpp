//! The step allocates nothing on the heap: a counting global allocator sees no allocation over
//! thousands of reset/step/policy calls in every ladder mode (and slinky's standard-map
//! constrictor).

use slinky_env::{EnvConfig, Rng, random_legal};
use slinky_rules::Mode;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwards the caller's layout to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System.alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn steps_do_not_allocate() {
    let mut envs: Vec<EnvConfig> = Mode::ALL
        .iter()
        .map(|&m| EnvConfig::ladder(m).unwrap())
        .collect();
    envs.push(EnvConfig::for_mode(Mode::Constrictor, None).unwrap());
    let mut rng = Rng::new(3);
    let before = ALLOCS.load(Ordering::Relaxed);
    let mut steps = 0;
    for env in &envs {
        for _ in 0..50 {
            let mut s = env.reset(&mut rng);
            while !s.done {
                let a = random_legal(&mut rng, env.rules(), &s);
                env.step(&mut rng, &mut s, &a);
                let _ = env.step_autoreset(&mut rng, &mut s.clone(), &a);
                steps += 1;
            }
        }
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    assert!(steps > 1000);
    assert_eq!(
        after - before,
        0,
        "{} allocations in {steps} steps",
        after - before
    );
}
