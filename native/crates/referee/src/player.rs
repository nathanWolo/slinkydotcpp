//! Agents that play in a match: a native az_modes agent (a run dir with a simulation or time
//! budget) or the `random_legal` baseline.
//!
//! Spec grammar (the canonical form is printed and stored):
//!
//! * `random_legal`: uniformly random over each seat's legal moves (`random_legal_policy`).
//! * `az_modes:<run dir>[:simulations=N][:time_ms=T][:max_simulations=N][:threads=K][:memory_mb=M]`,
//!   or a bare `<run dir>`: the az_modes agent of that run. The default is 256 simulations
//!   (the ladder agent). `time_ms=T` searches each move for `T` ms (not deterministic), at most
//!   `max_simulations`. `threads=K` uses K root-parallel trees (not recommended).

use std::hash::Hasher;
use std::path::{Path, PathBuf};

use slinky_search::{Agent, DEFAULT_MEMORY_LIMIT, MAX_SIMULATIONS, MoveBudget};

use crate::util::Fnv;

/// A parsed agent spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlayerSpec {
    /// Uniformly random legal moves.
    RandomLegal,
    /// A native az_modes agent.
    AzModes {
        /// The run dir (`config.json` + `params.npz`).
        dir: PathBuf,
        /// The per-move budget.
        budget: MoveBudget,
        /// Root-parallel trees per search.
        threads: usize,
        /// Tree memory cap per tree, in bytes.
        memory_limit: usize,
    },
}

impl PlayerSpec {
    /// Parse a spec (module docs).
    pub fn parse(text: &str) -> Result<PlayerSpec, String> {
        let text = text.trim();
        if text == "random_legal" {
            return Ok(PlayerSpec::RandomLegal);
        }
        let rest = text.strip_prefix("az_modes:").unwrap_or(text);
        let mut parts = rest.split(':');
        let dir = parts.next().unwrap_or_default();
        if dir.is_empty() {
            return Err(format!("{text:?}: no run dir"));
        }
        let mut sims: Option<usize> = None;
        let mut time_ms: Option<u64> = None;
        let mut cap: Option<usize> = None;
        let mut threads = 1usize;
        let mut memory_limit = DEFAULT_MEMORY_LIMIT;
        for p in parts {
            let (key, value) = p
                .split_once('=')
                .ok_or_else(|| format!("{text:?}: expected field=value, got {p:?}"))?;
            let num = |v: &str| -> Result<u64, String> {
                v.parse::<u64>()
                    .map_err(|_| format!("{text:?}: {key} must be a non-negative integer"))
            };
            match key {
                "simulations" | "sims" => sims = Some(num(value)? as usize),
                "time_ms" | "ms" => time_ms = Some(num(value)?),
                "max_simulations" => cap = Some(num(value)? as usize),
                "threads" => threads = (num(value)? as usize).max(1),
                "memory_mb" => memory_limit = (num(value)? as usize).saturating_mul(1 << 20),
                _ => return Err(format!("{text:?}: unknown field {key:?}")),
            }
        }
        let budget = match (sims, time_ms) {
            (Some(_), Some(_)) => {
                return Err(format!("{text:?}: give simulations or time_ms, not both"));
            }
            (Some(n), None) => {
                if cap.is_some() {
                    return Err(format!("{text:?}: max_simulations goes with time_ms"));
                }
                if n == 0 || n > MAX_SIMULATIONS {
                    return Err(format!(
                        "{text:?}: simulations must be 1 to {MAX_SIMULATIONS}"
                    ));
                }
                MoveBudget::Simulations(n)
            }
            (None, Some(ms)) => MoveBudget::Time {
                ms,
                max_simulations: cap.unwrap_or(MAX_SIMULATIONS).clamp(1, MAX_SIMULATIONS),
            },
            (None, None) => {
                if cap.is_some() {
                    return Err(format!("{text:?}: max_simulations goes with time_ms"));
                }
                MoveBudget::LADDER
            }
        };
        Ok(PlayerSpec::AzModes {
            dir: PathBuf::from(dir),
            budget,
            threads,
            memory_limit,
        })
    }

    /// The canonical name (only non-default fields beyond the budget).
    pub fn name(&self) -> String {
        match self {
            PlayerSpec::RandomLegal => "random_legal".into(),
            PlayerSpec::AzModes {
                dir,
                budget,
                threads,
                memory_limit,
            } => {
                let mut s = format!("az_modes:{}", dir.display());
                match *budget {
                    MoveBudget::Simulations(n) => s += &format!(":simulations={n}"),
                    MoveBudget::Time {
                        ms,
                        max_simulations,
                    } => {
                        s += &format!(":time_ms={ms}");
                        if max_simulations != MAX_SIMULATIONS {
                            s += &format!(":max_simulations={max_simulations}");
                        }
                    }
                }
                if *threads > 1 {
                    s += &format!(":threads={threads}");
                }
                if *memory_limit != DEFAULT_MEMORY_LIMIT {
                    s += &format!(":memory_mb={}", memory_limit >> 20);
                }
                s
            }
        }
    }

    /// A short label for logs and tables: the run dir's last component and the budget
    /// (`az-modes-M-s0@256`, `az-modes-M-s0@350ms`).
    pub fn short_name(&self) -> String {
        match self {
            PlayerSpec::RandomLegal => "random_legal".into(),
            PlayerSpec::AzModes {
                dir,
                budget,
                threads,
                ..
            } => {
                let base = dir
                    .file_name()
                    .map_or_else(|| dir.display().to_string(), |f| f.to_string_lossy().into());
                let mut s = match *budget {
                    MoveBudget::Simulations(n) => format!("{base}@{n}"),
                    MoveBudget::Time { ms, .. } => format!("{base}@{ms}ms"),
                };
                if *threads > 1 {
                    s += &format!("x{threads}");
                }
                s
            }
        }
    }

    /// Whether the agent's moves depend only on the game (not on timing).
    pub fn is_deterministic(&self) -> bool {
        match self {
            PlayerSpec::RandomLegal => true,
            PlayerSpec::AzModes { budget, .. } => matches!(budget, MoveBudget::Simulations(_)),
        }
    }
}

/// A loaded agent.
pub struct Player {
    /// What it was built from.
    pub spec: PlayerSpec,
    /// The native agent (`None` for `random_legal`).
    pub agent: Option<Agent>,
    /// A hash of the weights (`config.json` and `params.npz` bytes), for the matchup identity.
    pub weights_hash: Option<String>,
}

impl Player {
    /// Load the agent a spec names.
    pub fn load(spec: PlayerSpec) -> Result<Player, String> {
        match &spec {
            PlayerSpec::RandomLegal => Ok(Player {
                spec,
                agent: None,
                weights_hash: None,
            }),
            PlayerSpec::AzModes {
                dir,
                budget,
                threads,
                memory_limit,
            } => {
                let agent = Agent::load(dir)
                    .map_err(|e| format!("{}: {e}", dir.display()))?
                    .with_budget(*budget)
                    .with_threads(*threads)
                    .with_memory_limit(*memory_limit);
                let weights_hash = Some(weights_hash(dir)?);
                Ok(Player {
                    spec,
                    agent: Some(agent),
                    weights_hash,
                })
            }
        }
    }

    /// The canonical name.
    pub fn name(&self) -> String {
        self.spec.name()
    }
}

/// FNV-1a of a run dir's `config.json` and `params.npz`.
fn weights_hash(dir: &Path) -> Result<String, String> {
    let mut h = Fnv::new();
    for f in ["config.json", "params.npz"] {
        let p = dir.join(f);
        let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        h.write(&bytes);
        h.write_u8(0);
    }
    Ok(format!("{:016x}", h.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_parse_and_print_canonically() {
        assert_eq!(
            PlayerSpec::parse("random_legal").unwrap(),
            PlayerSpec::RandomLegal
        );
        let s = PlayerSpec::parse("/r/m").unwrap();
        assert_eq!(s.name(), "az_modes:/r/m:simulations=256");
        assert_eq!(s.short_name(), "m@256");
        assert!(s.is_deterministic());
        let s = PlayerSpec::parse("az_modes:/r/m:sims=1024").unwrap();
        assert_eq!(s.name(), "az_modes:/r/m:simulations=1024");
        let s =
            PlayerSpec::parse("az_modes:/r/m:time_ms=350:max_simulations=8192:threads=2").unwrap();
        assert_eq!(
            s.name(),
            "az_modes:/r/m:time_ms=350:max_simulations=8192:threads=2"
        );
        assert!(!s.is_deterministic());
        assert_eq!(s.short_name(), "m@350msx2");
        assert_eq!(PlayerSpec::parse(&s.name()).unwrap(), s);
        let s = PlayerSpec::parse("az_modes:/r/m:time_ms=100:memory_mb=64").unwrap();
        assert_eq!(s.name(), "az_modes:/r/m:time_ms=100:memory_mb=64");
        for bad in [
            "az_modes:",
            "az_modes:/r:simulations=0",
            "az_modes:/r:simulations=5:time_ms=3",
            "az_modes:/r:max_simulations=5",
            "az_modes:/r:bogus=1",
            "az_modes:/r:simulations",
            "az_modes:/r:simulations=-3",
        ] {
            assert!(PlayerSpec::parse(bad).is_err(), "{bad}");
        }
    }
}
