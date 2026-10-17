//! Games from real positions: a Battlesnake `/move` request to an environment and a state, and a
//! state back to a request.
//!
//! [`from_request`] builds on `slinky_api::game_from_request_any_map` (the Python server's exact
//! conversion, plus the `empty` map under every ruleset: our snake in seat 0, dead seats padded in when the starting snake count is
//! known) and adds what a simulator needs beyond the search: the map (`empty` for the official
//! constrictor ladder), the food settings, and royale shrink sides for the shrinks the request
//! has not shown yet (the server leaves them unknown; here they are drawn uniformly, as at reset).

use serde_json::{Value, json};
use slinky_api::{Request, game_from_request_any_map, setting};
use slinky_rules::GameState;

use crate::rng::{EnvRandom, side_of};
use crate::{EnvConfig, EnvError, MapKind};

/// The environment and state of a request's game (`starting_snakes`: the snake count at
/// `/start`, see `slinky_api::game_from_request`). Royale sides not yet shown are drawn from
/// `rng`; everything else is deterministic.
///
/// The map is the request's (`standard` when missing). Unlike the Python server, which plays
/// the `empty` map only under constrictor, every ruleset is accepted on it, as the environment
/// simulates them (so [`to_request_json`] round-trips every [`EnvConfig`]). The env is built
/// with [`EnvConfig::for_position`]: a position on a board too small for a reset is accepted. `foodSpawnChance` and `minimumFood` are read from the ruleset settings with the
/// server's leniency (defaults 15 and 1).
pub fn from_request<R: EnvRandom + ?Sized>(
    req: &Request,
    starting_snakes: Option<usize>,
    rng: &mut R,
) -> Result<(EnvConfig, GameState), EnvError> {
    let game = game_from_request_any_map(req, starting_snakes)?;
    let rules = game.config;
    let g = req.root().get("game");
    let map_name = g
        .and_then(|g| g.get("map"))
        .and_then(|m| m.as_str())
        .unwrap_or("standard");
    // The conversion read the ruleset (the royale map implies the royale ruleset); the map name
    // picks the map the env plays it on.
    let map = match map_name {
        "royale" => MapKind::Royale,
        "empty" => MapKind::Empty,
        _ => MapKind::Standard,
    };
    let settings = g
        .and_then(|g| g.get("ruleset"))
        .and_then(|r| r.get("settings"))
        .filter(|s| s.truthy());
    let chance = setting(settings, "foodSpawnChance", 15)?;
    let minimum = setting(settings, "minimumFood", 1)?;
    // A position needs no room for a reset's random placement.
    let env = EnvConfig::for_position(rules, map)?.with_food(chance, minimum);
    let mut state = game.state;
    if env.has_sides() {
        for s in state.sides[..rules.num_sides()].iter_mut() {
            if !(0..4).contains(s) {
                *s = side_of(rng);
            }
        }
    }
    Ok((env, state))
}

/// The head-first `(x, y)` segments of a living seat, rebuilt from its countdown grid (cells in
/// decreasing countdown; a cell with countdown `c` above the next one's `c'` holds `c - c'`
/// stacked segments). Empty for a dead seat.
pub fn body_segments(env: &EnvConfig, state: &GameState, seat: usize) -> Vec<(i32, i32)> {
    let config = env.rules();
    if seat >= config.num_snakes() || !state.alive[seat] {
        return Vec::new();
    }
    let mut cells: Vec<(i16, usize)> = state.body[seat][..config.cells()]
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > 0)
        .map(|(c, &v)| (v, c))
        .collect();
    cells.sort_unstable_by(|a, b| b.cmp(a));
    let mut out = Vec::new();
    for (k, &(v, c)) in cells.iter().enumerate() {
        let next = cells.get(k + 1).map_or(0, |&(w, _)| w);
        for _ in 0..(v - next) {
            out.push(config.xy(c));
        }
    }
    out
}

fn point((x, y): (i32, i32)) -> Value {
    json!({"x": x, "y": y})
}

/// A `/move` request for the living seat `you` in `state` (`None` if that seat is not alive):
/// living snakes in seat order with ids `"seat<i>"`, the food, the hazards (a stacked hazard
/// listed once per layer), the ruleset, map and settings of `env`. What a referee sends a server,
/// and the inverse of [`from_request`] for the living snakes.
pub fn to_request_json(
    env: &EnvConfig,
    state: &GameState,
    you: usize,
    game_id: &str,
) -> Option<Value> {
    let config = env.rules();
    if you >= config.num_snakes() || !state.alive[you] {
        return None;
    }
    let snake = |seat: usize| {
        let body = body_segments(env, state, seat);
        let [hx, hy] = state.head[seat];
        json!({
            "id": format!("seat{seat}"),
            "name": format!("seat{seat}"),
            "health": state.health[seat],
            "body": body.iter().map(|&p| point(p)).collect::<Vec<_>>(),
            "head": point((hx, hy)),
            "length": body.len(),
            "latency": "0",
            "shout": "",
        })
    };
    let mut food = Vec::new();
    let mut hazards = Vec::new();
    for c in 0..config.cells() {
        if state.has_food(c) {
            food.push(point(config.xy(c)));
        }
        for _ in 0..state.hazard[c].max(0) {
            hazards.push(point(config.xy(c)));
        }
    }
    let snakes: Vec<Value> = (0..config.num_snakes())
        .filter(|&i| state.alive[i])
        .map(snake)
        .collect();
    Some(json!({
        "game": {
            "id": game_id,
            "ruleset": {
                "name": config.ruleset().name(),
                "version": "v1.2.3",
                "settings": {
                    "foodSpawnChance": env.food_spawn_chance(),
                    "minimumFood": env.minimum_food(),
                    "hazardDamagePerTurn": config.hazard_damage_per_turn(),
                    "royale": {"shrinkEveryNTurns": config.shrink_every_n_turns()},
                },
            },
            "map": env.map().name(),
            "timeout": 500,
            "source": "custom",
        },
        "turn": state.turn,
        "board": {
            "width": config.width(),
            "height": config.height(),
            "food": food,
            "hazards": hazards,
            "snakes": snakes,
        },
        "you": snake(you),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rng, random_legal};
    use slinky_rules::{Mode, royale_hazards};

    /// Play random games, convert each position to a request for some living seat and back, and
    /// check the living snakes, food, hazards, turn and (royale) the zone at this turn survive.
    fn round_trip(env: EnvConfig, games: u64) {
        let cfg = *env.rules();
        let n = cfg.num_snakes();
        let mut rng = Rng::new(11);
        let mut checked = 0;
        for g in 0..games {
            let mut s = env.reset(&mut rng);
            while !s.done {
                let you = (0..n)
                    .filter(|&i| s.alive[i])
                    .nth(g as usize % 2)
                    .unwrap_or(0);
                let body = to_request_json(&env, &s, you, "g").unwrap();
                let req = Request::from_value(&body).unwrap();
                let (env2, t) = from_request(&req, Some(n), &mut rng).unwrap();
                assert_eq!(env2, env);
                // Seat order: you first, then the other living seats in order.
                let order: Vec<usize> = std::iter::once(you)
                    .chain((0..n).filter(|&i| s.alive[i] && i != you))
                    .collect();
                for (k, &i) in order.iter().enumerate() {
                    assert_eq!(t.body[k], s.body[i]);
                    assert_eq!(t.head[k], s.head[i]);
                    assert_eq!((t.length[k], t.health[k]), (s.length[i], s.health[i]));
                    assert!(t.alive[k]);
                    if s.turn > 0 {
                        assert_eq!(t.last_move[k], s.last_move[i]);
                    }
                }
                for k in order.len()..n {
                    assert!(!t.alive[k]);
                }
                assert_eq!(
                    (t.food, t.hazard, t.turn, t.done),
                    (s.food, s.hazard, s.turn, s.done)
                );
                if env.has_sides() {
                    assert!(
                        t.sides[..cfg.num_sides()]
                            .iter()
                            .all(|v| (0..4).contains(v))
                    );
                    assert_eq!(
                        royale_hazards(&cfg, &t.sides, s.turn),
                        royale_hazards(&cfg, &s.sides, s.turn)
                    );
                }
                checked += 1;
                let a = random_legal(&mut rng, &cfg, &s);
                env.step(&mut rng, &mut s, &a);
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn request_round_trips() {
        round_trip(EnvConfig::ladder(Mode::Duel).unwrap(), 20);
        round_trip(EnvConfig::ladder(Mode::Standard).unwrap(), 10);
        round_trip(EnvConfig::ladder(Mode::Constrictor).unwrap(), 10);
        let royale = EnvConfig::ladder(Mode::Royale).unwrap().with_food(30, 2);
        round_trip(royale, 10);
    }

    #[test]
    fn request_settings_and_unknown_sides() {
        let body = json!({
            "game": {"id": "x", "map": "royale",
                     "ruleset": {"name": "royale", "settings": {"foodSpawnChance": "25",
                                 "minimumFood": 3, "royale": {"shrinkEveryNTurns": 10}}}},
            "turn": 12,
            "board": {"width": 11, "height": 11, "food": [], "hazards": [
                {"x": 0, "y": 0}, {"x": 0, "y": 1}, {"x": 0, "y": 2}, {"x": 0, "y": 3},
                {"x": 0, "y": 4}, {"x": 0, "y": 5}, {"x": 0, "y": 6}, {"x": 0, "y": 7},
                {"x": 0, "y": 8}, {"x": 0, "y": 9}, {"x": 0, "y": 10}],
              "snakes": [
                {"id": "a", "health": 90, "body": [{"x": 3, "y": 3}, {"x": 3, "y": 2}, {"x": 3, "y": 1}]},
                {"id": "b", "health": 80, "body": [{"x": 7, "y": 7}, {"x": 7, "y": 8}, {"x": 7, "y": 9}]}]},
            "you": {"id": "b", "health": 80, "body": [{"x": 7, "y": 7}, {"x": 7, "y": 8}, {"x": 7, "y": 9}]}
        });
        let req = Request::from_value(&body).unwrap();
        let mut rng = Rng::new(0);
        let (env, s) = from_request(&req, None, &mut rng).unwrap();
        assert_eq!((env.food_spawn_chance(), env.minimum_food()), (25, 3));
        assert_eq!(env.map(), MapKind::Royale);
        assert_eq!(env.rules().shrink_every_n_turns(), 10);
        assert_eq!(s.sides[0], 0); // the one shrink shown: the left column
        assert!(s.sides[1..21].iter().all(|v| (0..4).contains(v)));
        assert_eq!(s.head[0], [7, 7]);
        let mut empty = body.clone();
        empty["game"]["map"] = json!("empty");
        empty["game"]["ruleset"] = json!({"name": "constrictor"});
        empty["board"]["hazards"] = json!([]);
        let req = Request::from_value(&empty).unwrap();
        let (env, _) = from_request(&req, Some(4), &mut rng).unwrap();
        assert_eq!((env.map(), env.rules().num_snakes()), (MapKind::Empty, 4));
        assert_eq!(env.mode(), Mode::Constrictor);
        let mut wrapped = body;
        wrapped["game"]["ruleset"]["name"] = json!("wrapped");
        let req = Request::from_value(&wrapped).unwrap();
        assert!(from_request(&req, None, &mut rng).is_err());
    }
}
