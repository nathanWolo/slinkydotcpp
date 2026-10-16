//! Root-parallel search: several independent trees of the same position, one per thread, each
//! with its own noise stream; their root statistics (visits and value sums per seat and move)
//! are summed and the final moves are chosen from the sum by the usual rule.
//!
//! Not part of the Python agent and off by default. One thread is exactly [`Search::run`] with
//! the noise of [`thread_noise`]`(seed, 0)` (= [`JaxNoise::from_seed`]), so a one-thread call is
//! the faithful search. With more threads the result is deterministic without a deadline (each
//! thread runs a fixed share of the simulations with a fixed noise stream, and the sums are
//! taken in thread order), but it is a different search from one tree with as many simulations.
//!
//! Measured (`examples/quality.rs`, see the crate docs): `k` trees of `n` simulations choose
//! moves about as well as one tree of `n`, so the extra cores buy almost nothing. Not
//! recommended for the ladder; kept for experiments.
//!
//! The simulation count of the [`Budget`] is the total over all threads: thread `t` runs
//! `simulations / k` (the first `simulations % k` threads one more). A deadline applies to every
//! thread. Each workspace keeps its own memory limit.

use crate::SearchError;
use crate::eval::Evaluator;
use crate::noise::{JaxKey, JaxNoise, NoiseSource};
use crate::search::{Budget, RootStats, Search, SearchConfig, SearchOutput, summarize};
use slinky_rules::{GameConfig, GameState, MAX_SNAKES};

/// The noise stream of thread `thread` of a root-parallel search seeded with `seed`: thread 0
/// gets `mcts.search(jax.random.key(seed))`'s stream ([`JaxNoise::from_seed`]), thread `t > 0`
/// that of `fold_in(key(seed), t)`.
pub fn thread_noise(seed: u32, thread: usize) -> JaxNoise {
    if thread == 0 {
        JaxNoise::from_seed(seed)
    } else {
        JaxKey::new(seed).fold_in(thread as u32).search_noise()
    }
}

/// Search `state` with `k = min(workspaces, evaluators, noises)` independent trees in parallel
/// (thread 0 on the calling thread) and combine their roots. With `k == 1` this is
/// [`Search::run`]. Threads whose share of the simulations would be 0 are not started. The
/// returned `nodes_used` and `simulations` are totals over the trees, `depth` the deepest.
pub fn run_root_parallel<E, N>(
    workspaces: &mut [Search],
    config: &GameConfig,
    state: &GameState,
    settings: &SearchConfig,
    budget: Budget,
    evaluators: &mut [E],
    noises: &mut [N],
) -> Result<SearchOutput, SearchError>
where
    E: Evaluator + Send,
    N: NoiseSource + Send,
{
    let k = workspaces
        .len()
        .min(evaluators.len())
        .min(noises.len())
        .min(budget.simulations.max(1));
    if k == 0 {
        return Err(SearchError::Unsupported(
            "a root-parallel search needs at least one workspace, evaluator and noise source"
                .into(),
        ));
    }
    if k == 1 {
        return workspaces[0].run(
            config,
            state,
            settings,
            budget,
            &mut evaluators[0],
            &mut noises[0],
        );
    }
    let share = |t: usize| budget.simulations / k + usize::from(t < budget.simulations % k);
    let mut results: Vec<Result<SearchOutput, SearchError>> = Vec::with_capacity(k);
    std::thread::scope(|scope| {
        let mut jobs = workspaces
            .iter_mut()
            .zip(evaluators.iter_mut())
            .zip(noises.iter_mut())
            .take(k)
            .enumerate();
        let first = jobs.next();
        let handles: Vec<_> = jobs
            .map(|(t, ((ws, ev), no))| {
                let b = Budget {
                    simulations: share(t),
                    deadline: budget.deadline,
                };
                scope.spawn(move || ws.run(config, state, settings, b, ev, no))
            })
            .collect();
        if let Some((t, ((ws, ev), no))) = first {
            let b = Budget {
                simulations: share(t),
                deadline: budget.deadline,
            };
            results.push(ws.run(config, state, settings, b, ev, no));
        }
        for h in handles {
            results.push(h.join().unwrap_or_else(|_| {
                Err(SearchError::Evaluator("a search thread panicked".into()))
            }));
        }
    });
    let mut outs = Vec::with_capacity(k);
    for r in results {
        outs.push(r?);
    }
    let mut root: Option<RootStats> = None;
    for ws in workspaces.iter().take(k) {
        let Some(r) = ws.root_stats() else { continue };
        match &mut root {
            None => root = Some(r),
            Some(acc) => {
                for p in 0..MAX_SNAKES {
                    for a in 0..slinky_rules::NUM_ACTIONS {
                        acc.visits[p][a] = acc.visits[p][a].saturating_add(r.visits[p][a]);
                        acc.value_sum[p][a] += r.value_sum[p][a];
                        acc.value_sum_wide[p][a] += r.value_sum_wide[p][a];
                    }
                }
            }
        }
    }
    let Some(root) = root else {
        return Err(SearchError::Unsupported("no tree was searched".into()));
    };
    let nodes = outs.iter().map(|o| o.nodes_used).sum();
    let depth = outs.iter().map(|o| o.depth).max().unwrap_or(0);
    let sims = outs.iter().map(|o| o.simulations).sum();
    Ok(summarize(&root, config.num_snakes(), nodes, depth, sims))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::UniformEvaluator;
    use slinky_rules::Mode;

    fn duel() -> (GameConfig, GameState) {
        let cfg = GameConfig::for_mode(Mode::Duel, None).unwrap();
        let mut s = GameState::new(&cfg);
        s.set_snake(&cfg, 0, &[(1, 1), (1, 1), (1, 1)], 100, None)
            .unwrap();
        s.set_snake(&cfg, 1, &[(9, 9), (9, 9), (9, 9)], 100, None)
            .unwrap();
        s.refresh_done(&cfg);
        (cfg, s)
    }

    #[test]
    fn one_thread_is_the_plain_search() {
        let (cfg, s) = duel();
        let set = SearchConfig::default();
        let plain = Search::new()
            .run(
                &cfg,
                &s,
                &set,
                Budget::simulations(100),
                &mut UniformEvaluator,
                &mut JaxNoise::from_seed(3),
            )
            .unwrap();
        let par = run_root_parallel(
            &mut [Search::new()],
            &cfg,
            &s,
            &set,
            Budget::simulations(100),
            &mut [UniformEvaluator],
            &mut [thread_noise(3, 0)],
        )
        .unwrap();
        assert_eq!(plain, par);
    }

    #[test]
    fn threads_split_and_sum() {
        let (cfg, s) = duel();
        let set = SearchConfig::default();
        let mut ws: Vec<Search> = (0..3).map(|_| Search::new()).collect();
        let mut evs = [UniformEvaluator; 3];
        let mut noises: Vec<JaxNoise> = (0..3).map(|t| thread_noise(9, t)).collect();
        let a = run_root_parallel(
            &mut ws,
            &cfg,
            &s,
            &set,
            Budget::simulations(100),
            &mut evs,
            &mut noises,
        )
        .unwrap();
        assert_eq!(a.simulations, 100);
        for p in 0..2 {
            assert_eq!(a.visits[p].iter().sum::<i32>(), 100);
            assert!(a.legal[p][a.action[p] as usize]);
        }
        let shares: Vec<usize> = ws.iter().map(|w| w.stats().simulations).collect();
        assert_eq!(shares, vec![34, 33, 33]);
        // Deterministic: the same call again gives the same result.
        let mut noises: Vec<JaxNoise> = (0..3).map(|t| thread_noise(9, t)).collect();
        let b = run_root_parallel(
            &mut ws,
            &cfg,
            &s,
            &set,
            Budget::simulations(100),
            &mut evs,
            &mut noises,
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_is_an_error() {
        let (cfg, s) = duel();
        let mut noises: [JaxNoise; 0] = [];
        let r = run_root_parallel(
            &mut [Search::new()],
            &cfg,
            &s,
            &SearchConfig::default(),
            Budget::simulations(10),
            &mut [UniformEvaluator],
            &mut noises,
        );
        assert!(r.is_err());
    }
}
