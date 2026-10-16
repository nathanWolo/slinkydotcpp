//! One game between two agents, A in one seat and B in all others, fixed by `(seed, game id)`.
//!
//! As `evaluate.run_match` does, game `g` puts A in seat `g % N` and B in every other seat,
//! and each agent is asked once per turn for all its seats (B's three seats in a 4-snake game
//! come from one search of the joint position, as B's policy in slinky does). Everything
//! random derives from the seed and the game id: the environment's generator (start position,
//! food, royale sides), each agent's search noise per turn, and `random_legal`'s choices. So a
//! game is the same whatever the thread count, the order games run in, or which other games
//! are played, as long as both agents have deterministic budgets.
//!
//! Searches per turn: with `per_seat` (the CLI default) each seat is searched separately with
//! its own noise, as independent ladder snakes are. Without it an agent searches once per turn
//! and plays all its seats from that one tree, as slinky's `run_match` does. That matters with
//! 4 snakes: B's three seats taken from one tree move consistently with each other's predicted
//! moves, which A's separate tree does not share, and it biases a self-match against A
//! (measured, M at 256 simulations, constrictor, 512 games: share 0.187 +- 0.028 per agent,
//! 0.255 +- 0.034 per seat; parity 0.25). An agent with one seat searches the same way either
//! way, so duel games do not depend on it. [`View::Request`] is always per seat.
//!
//! Views: with [`View::State`] (the default, as slinky's strength tests) an agent searches the
//! true game state, which includes the royale shrink sides drawn at reset. With
//! [`View::Request`] each seat is asked as the ladder asks it: the state becomes a `/move`
//! request ([`slinky_env::to_request_json`]), which the agent parses and plays from its seat
//! (the request does not show future shrinks), one search per seat, the safe move on error.

use std::time::Instant;

use serde_json::{Value, json};
use slinky_api::{Game, safe_move_json};
use slinky_env::{EnvConfig, Rng, random_legal, to_request_json};
use slinky_rules::MAX_SNAKES;
use slinky_search::{JaxNoise, Search};

use crate::outcome::{GameScore, Outcome, contested, score_game};
use crate::player::Player;
use crate::util::derive;

/// What an agent sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// The true game state (slinky's strength tests).
    State,
    /// A `/move` request per seat (the ladder).
    Request,
}

impl View {
    /// `"state"` or `"request"`.
    pub fn name(self) -> &'static str {
        match self {
            View::State => "state",
            View::Request => "request",
        }
    }

    /// Parse [`View::name`].
    pub fn from_name(name: &str) -> Option<View> {
        match name {
            "state" => Some(View::State),
            "request" => Some(View::Request),
            _ => None,
        }
    }
}

/// Everything a game needs besides its id.
pub struct Setup<'a> {
    /// The game.
    pub env: EnvConfig,
    /// A and B.
    pub players: [&'a Player; 2],
    /// The match seed.
    pub seed: u64,
    /// The harness's cut-off (the agents do not know it): a game still going at this turn is
    /// over, a draw for A if A is alive with another snake.
    pub max_turns: i32,
    /// What the agents see.
    pub view: View,
    /// End a game once it is decided between the agents (A dead, or A alone).
    pub early_stop: bool,
    /// [`View::State`]: search each seat of a multi-seat agent separately (its own noise), not
    /// once per agent.
    pub per_seat: bool,
}

/// One thread's search workspaces, per agent.
pub struct Workspace {
    trees: [Vec<Search>; 2],
}

impl Workspace {
    /// Workspaces for the agents of `setup`.
    pub fn new(setup: &Setup) -> Workspace {
        let make = |p: &Player| -> Vec<Search> {
            match &p.agent {
                Some(a) => (0..a.threads())
                    .map(|_| Search::with_memory_limit(a.memory_limit()))
                    .collect(),
                None => Vec::new(),
            }
        };
        Workspace {
            trees: [make(setup.players[0]), make(setup.players[1])],
        }
    }
}

/// One agent's moves in a game: the time and simulations of each search (one per turn, or one
/// per seat with [`View::Request`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MoveLog {
    /// Search wall time per move, in microseconds.
    pub us: Vec<u32>,
    /// Simulations per move (0 for `random_legal`).
    pub sims: Vec<u32>,
    /// Moves where the agent failed and played the fallback move.
    pub errors: u32,
}

/// The record of a finished game.
#[derive(Debug, Clone, PartialEq)]
pub struct GameRecord {
    /// The game id.
    pub game: u64,
    /// Seats in the game.
    pub num_snakes: usize,
    /// A's seat (`game % num_snakes`).
    pub a_seat: usize,
    /// Turns played (the final state's turn).
    pub turns: i32,
    /// Why the game ended: `"rules"`, `"decided"` (early stop) or `"max_turns"`.
    pub ended: &'static str,
    /// Final alive flags per seat.
    pub alive: Vec<bool>,
    /// Elimination turn per seat (0 while alive).
    pub elim_turn: Vec<i32>,
    /// Elimination cause per seat (`slinky_rules::Cause` codes).
    pub elim_cause: Vec<i8>,
    /// A's result.
    pub score: GameScore,
    /// A's and B's moves.
    pub moves: [MoveLog; 2],
    /// Wall time of the game in seconds.
    pub seconds: f64,
}

const ENV_STREAM: u64 = 0x454E_5649;
const AGENT_STREAM: u64 = 0x4147_454E;

/// The env generator of game `game`.
pub fn env_rng(seed: u64, game: u64) -> Rng {
    Rng::with_stream(derive(&[seed, ENV_STREAM]), game)
}

/// Play game `game`. Returns `None` if `abort()` turned true before the game ended (checked
/// before each turn).
pub fn play_game(
    setup: &Setup,
    ws: &mut Workspace,
    game: u64,
    abort: &dyn Fn() -> bool,
) -> Option<GameRecord> {
    let t0 = Instant::now();
    let env = &setup.env;
    let rules = env.rules();
    let n = rules.num_snakes();
    let a_seat = (game % n as u64) as usize;
    let mut rng = env_rng(setup.seed, game);
    let mut state = env.reset(&mut rng);
    let mut policy_rng =
        [0u64, 1].map(|p| Rng::with_stream(derive(&[setup.seed, AGENT_STREAM, p]), game));
    let mut moves: [MoveLog; 2] = Default::default();
    let game_id = format!("naz-match-{}-{game}", setup.seed);
    let owner = |seat: usize| usize::from(seat != a_seat);
    let ended = loop {
        if state.done {
            break "rules";
        }
        if state.turn >= setup.max_turns {
            break "max_turns";
        }
        if setup.early_stop && !contested(&state.alive[..n], a_seat) {
            break "decided";
        }
        if abort() {
            return None;
        }
        let mut actions = [0u8; MAX_SNAKES];
        for p in 0..2 {
            let seats: Vec<usize> = (0..n)
                .filter(|&s| owner(s) == p && state.alive[s])
                .collect();
            if seats.is_empty() {
                continue;
            }
            let player = setup.players[p];
            let log = &mut moves[p];
            let noise_seed = |extra: u64| {
                derive(&[
                    setup.seed,
                    AGENT_STREAM,
                    p as u64,
                    game,
                    state.turn as u64,
                    extra,
                ]) as u32
            };
            match (&player.agent, setup.view) {
                (None, _) => {
                    let t = Instant::now();
                    let a = random_legal(&mut policy_rng[p], rules, &state);
                    for &s in &seats {
                        actions[s] = a[s];
                    }
                    log.us
                        .push(t.elapsed().as_micros().min(u32::MAX as u128) as u32);
                    log.sims.push(0);
                }
                (Some(agent), View::State) => {
                    let g = Game {
                        config: *rules,
                        state,
                        mode: env.mode(),
                    };
                    // One search for all the agent's seats, or one per seat with its own noise.
                    // An agent holding one seat (A, or B in the duel) searches the same way
                    // in both modes, so duel games do not depend on the option.
                    let multi_seat = p == 1 && n > 2;
                    let groups: Vec<(u64, Vec<usize>)> = if setup.per_seat && multi_seat {
                        seats.iter().map(|&s| (1 + s as u64, vec![s])).collect()
                    } else {
                        vec![(0, seats.clone())]
                    };
                    for (extra, group) in groups {
                        let t = Instant::now();
                        let budget = agent.budget().budget_from(t);
                        let seed = noise_seed(extra);
                        let trees = &mut ws.trees[p];
                        let res = if trees.len() > 1 {
                            agent.search_game_threads(trees, &g, budget, seed)
                        } else {
                            agent.search_game(
                                &mut trees[0],
                                &g,
                                budget,
                                &mut JaxNoise::from_seed(seed),
                            )
                        };
                        let us = t.elapsed().as_micros().min(u32::MAX as u128) as u32;
                        match res {
                            Ok(m) => {
                                for &s in &group {
                                    actions[s] = m.search.action[s];
                                }
                                log.us.push(us);
                                log.sims
                                    .push(m.search.simulations.min(u32::MAX as usize) as u32);
                            }
                            Err(_) => {
                                log.errors += 1;
                                let a = random_legal(&mut policy_rng[p], rules, &state);
                                for &s in &group {
                                    actions[s] = a[s];
                                }
                            }
                        }
                    }
                }
                (Some(agent), View::Request) => {
                    for &s in &seats {
                        let Some(req) = to_request_json(env, &state, s, &game_id) else {
                            continue;
                        };
                        let body = req.to_string();
                        let t = Instant::now();
                        let res = agent.play_from(&body, Some(n), noise_seed(1 + s as u64), t);
                        let us = t.elapsed().as_micros().min(u32::MAX as u128) as u32;
                        match res {
                            Ok(m) => {
                                actions[s] = m.mv as u8;
                                log.us.push(us);
                                log.sims
                                    .push(m.search.simulations.min(u32::MAX as usize) as u32);
                            }
                            Err(_) => {
                                log.errors += 1;
                                actions[s] = safe_move_json(&body) as u8;
                            }
                        }
                    }
                }
            }
        }
        env.step(&mut rng, &mut state, &actions);
    };
    let score = score_game(&state.alive[..n], &state.elim_turn[..n], a_seat);
    Some(GameRecord {
        game,
        num_snakes: n,
        a_seat,
        turns: state.turn,
        ended,
        alive: state.alive[..n].to_vec(),
        elim_turn: state.elim_turn[..n].to_vec(),
        elim_cause: state.elim_cause[..n].to_vec(),
        score,
        moves,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

impl GameRecord {
    /// The record as a JSON object (one line of the games file).
    pub fn to_json(&self, matchup_id: &str, fingerprint: &str) -> Value {
        let log = |m: &MoveLog| json!({"us": m.us, "sims": m.sims, "errors": m.errors});
        json!({
            "matchup_id": matchup_id,
            "code_fingerprint": fingerprint,
            "game": self.game,
            "num_snakes": self.num_snakes,
            "a_seat": self.a_seat,
            "turns": self.turns,
            "ended": self.ended,
            "outcome": self.score.outcome.code(),
            "truncated": self.score.truncated,
            "share": self.score.share,
            "placement": self.score.placement,
            "alive": self.alive,
            "elim_turn": self.elim_turn,
            "elim_cause": self.elim_cause,
            "a": log(&self.moves[0]),
            "b": log(&self.moves[1]),
            "seconds": (self.seconds * 1e3).round() / 1e3,
        })
    }

    /// Read a line of the games file back (`None` if it is not a valid record).
    pub fn from_json(v: &Value) -> Option<GameRecord> {
        let u32s = |x: &Value| -> Option<Vec<u32>> {
            x.as_array()?
                .iter()
                .map(|e| e.as_u64().and_then(|e| u32::try_from(e).ok()))
                .collect()
        };
        let log = |x: &Value| -> Option<MoveLog> {
            Some(MoveLog {
                us: u32s(x.get("us")?)?,
                sims: u32s(x.get("sims")?)?,
                errors: u32::try_from(x.get("errors")?.as_u64()?).ok()?,
            })
        };
        let num_snakes = usize::try_from(v.get("num_snakes")?.as_u64()?).ok()?;
        let alive: Vec<bool> = v
            .get("alive")?
            .as_array()?
            .iter()
            .map(Value::as_bool)
            .collect::<Option<_>>()?;
        let elim_turn: Vec<i32> = v
            .get("elim_turn")?
            .as_array()?
            .iter()
            .map(|e| e.as_i64().and_then(|e| i32::try_from(e).ok()))
            .collect::<Option<_>>()?;
        let elim_cause: Vec<i8> = v
            .get("elim_cause")?
            .as_array()?
            .iter()
            .map(|e| e.as_i64().and_then(|e| i8::try_from(e).ok()))
            .collect::<Option<_>>()?;
        let a_seat = usize::try_from(v.get("a_seat")?.as_u64()?).ok()?;
        if !(2..=MAX_SNAKES).contains(&num_snakes)
            || alive.len() != num_snakes
            || elim_turn.len() != num_snakes
            || elim_cause.len() != num_snakes
            || a_seat >= num_snakes
        {
            return None;
        }
        let ended = match v.get("ended")?.as_str()? {
            "rules" => "rules",
            "decided" => "decided",
            "max_turns" => "max_turns",
            _ => return None,
        };
        // The score is recomputed from the final state; the stored fields must agree.
        let score = score_game(&alive, &elim_turn, a_seat);
        if Outcome::from_code(v.get("outcome")?.as_str()?)? != score.outcome {
            return None;
        }
        Some(GameRecord {
            game: v.get("game")?.as_u64()?,
            num_snakes,
            a_seat,
            turns: i32::try_from(v.get("turns")?.as_i64()?).ok()?,
            ended,
            alive,
            elim_turn,
            elim_cause,
            score,
            moves: [log(v.get("a")?)?, log(v.get("b")?)?],
            seconds: v.get("seconds")?.as_f64()?,
        })
    }
}
