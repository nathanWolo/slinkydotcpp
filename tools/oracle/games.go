package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/BattlesnakeOfficial/rules"
	"github.com/BattlesnakeOfficial/rules/maps"
)

type gamesConfig struct {
	Ruleset           string            `json:"ruleset"`
	Map               string            `json:"map"`
	Width             int               `json:"width"`
	Height            int               `json:"height"`
	Snakes            int               `json:"snakes"`
	Games             int               `json:"games"`
	Seed              int64             `json:"seed"`
	MaxTurns          int               `json:"max_turns"`
	FoodSpawnChance   int               `json:"food_spawn_chance"`
	MinimumFood       int               `json:"minimum_food"`
	HazardDamage      int               `json:"hazard_damage"`
	ShrinkEveryNTurns int               `json:"shrink_every_n_turns"`
	InvalidMoveProb   float64           `json:"invalid_move_prob"`
	PRandom           float64           `json:"p_random"`
	PAggressive       float64           `json:"p_aggressive"`
	PFood             float64           `json:"p_food"`
	PFoodAverse       float64           `json:"p_food_averse"`
	PCareful          float64           `json:"p_careful"`
	Solo              bool              `json:"solo"`      // derived: snakes == 1
	Settings          map[string]string `json:"settings"`  // derived: params passed to the engine
	GameSeed          int64             `json:"game_seed"` // derived: hash of (seed, game index)
}

type initialRecord struct {
	Kind   string      `json:"kind"`
	Game   int         `json:"game"`
	Config gamesConfig `json:"config"`
	State  *JSONState  `json:"state"`
}

type transitionRecord struct {
	Kind      string     `json:"kind"`
	Game      int        `json:"game"`
	Pre       *JSONState `json:"pre"`
	Moves     []string   `json:"moves"`
	PostRules *JSONState `json:"post_rules"`
	PostMap   *JSONState `json:"post_map"`
	GameOver  bool       `json:"game_over"`
}

func runGames(args []string) error {
	fs := flag.NewFlagSet("games", flag.ContinueOnError)
	var c gamesConfig
	fs.StringVar(&c.Ruleset, "ruleset", rules.GameTypeStandard, "ruleset name (standard, solo, royale, wrapped, constrictor, wrapped_constrictor)")
	fs.StringVar(&c.Map, "map", "standard", "game map ID (e.g. standard, royale, empty, ...)")
	fs.IntVar(&c.Width, "width", 11, "board width")
	fs.IntVar(&c.Height, "height", 11, "board height")
	fs.IntVar(&c.Snakes, "snakes", 2, "number of snakes (1 => solo game-over rule, like the CLI)")
	fs.IntVar(&c.Games, "games", 10, "number of games")
	fs.Int64Var(&c.Seed, "seed", 1, "base seed; game g uses a hash of (seed, g) for the engine and the policy")
	fs.IntVar(&c.MaxTurns, "max-turns", 500, "maximum number of transitions recorded per game")
	fs.IntVar(&c.FoodSpawnChance, "food-spawn-chance", 15, "engine foodSpawnChance")
	fs.IntVar(&c.MinimumFood, "minimum-food", 1, "engine minimumFood")
	fs.IntVar(&c.HazardDamage, "hazard-damage", 14, "engine damagePerTurn")
	fs.IntVar(&c.ShrinkEveryNTurns, "shrink-every-n-turns", 25, "engine shrinkEveryNTurns")
	fs.Float64Var(&c.InvalidMoveProb, "invalid-move-prob", 0, "probability a living snake sends an invalid move string")
	fs.Float64Var(&c.PRandom, "p-random", 0.03, "policy: probability of a uniformly random move")
	fs.Float64Var(&c.PAggressive, "p-aggressive", 0.3, "policy: probability of chasing the closest other head")
	fs.Float64Var(&c.PFood, "p-food", 0.3, "policy: probability of chasing the closest food")
	fs.Float64Var(&c.PFoodAverse, "p-food-averse", 0.15, "policy: per-snake probability of avoiding food all game")
	fs.Float64Var(&c.PCareful, "p-careful", 0.75, "policy: per-snake probability of avoiding dead ends and risky head-to-heads")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected arguments: %v", fs.Args())
	}
	if err := checkRulesetName(c.Ruleset); err != nil {
		return err
	}
	gameMap, err := maps.GetMap(c.Map)
	if err != nil {
		return fmt.Errorf("map %q: %w", c.Map, err)
	}

	c.Solo = c.Snakes < 2
	c.Settings = map[string]string{
		rules.ParamFoodSpawnChance:     fmt.Sprint(c.FoodSpawnChance),
		rules.ParamMinimumFood:         fmt.Sprint(c.MinimumFood),
		rules.ParamHazardDamagePerTurn: fmt.Sprint(c.HazardDamage),
		rules.ParamShrinkEveryNTurns:   fmt.Sprint(c.ShrinkEveryNTurns),
	}
	params := policyParams{
		PRandom:     c.PRandom,
		PAggressive: c.PAggressive,
		PFood:       c.PFood,
		PInvalid:    c.InvalidMoveProb,
		PFoodAverse: c.PFoodAverse,
		PCareful:    c.PCareful,
	}
	wrapped := c.Ruleset == rules.GameTypeWrapped || c.Ruleset == rules.GameTypeWrappedConstrictor

	out := bufio.NewWriterSize(os.Stdout, 1<<20)
	defer out.Flush()
	enc := json.NewEncoder(out)
	enc.SetEscapeHTML(false)

	for g := 0; g < c.Games; g++ {
		gc := c
		gc.GameSeed = gameSeed(c.Seed, g)
		if err := playGame(enc, g, gc, gameMap, params, wrapped); err != nil {
			return fmt.Errorf("game %d (seed %d): %w", g, gc.GameSeed, err)
		}
	}
	return out.Flush()
}

// playGame mirrors cli/commands/play.go: initializeBoardFromArgs followed by
// createNextBoardState in a loop, with moves from the built-in policy.
func playGame(enc *json.Encoder, g int, c gamesConfig, gameMap maps.GameMap, params policyParams, wrapped bool) error {
	ruleset := rules.NewRulesetBuilder().
		WithSeed(c.GameSeed).
		WithParams(c.Settings).
		WithSolo(c.Solo).
		NamedRuleset(c.Ruleset)
	settings := ruleset.Settings()

	ids := make([]string, c.Snakes)
	for i := range ids {
		ids[i] = fmt.Sprintf("s%d", i)
	}

	// initializeBoardFromArgs
	board, err := maps.SetupBoard(gameMap.ID(), settings, c.Width, c.Height, ids)
	if err != nil {
		return fmt.Errorf("SetupBoard: %w", err)
	}
	gameOver, board, err := ruleset.Execute(board, nil)
	if err != nil {
		return fmt.Errorf("initial Execute: %w", err)
	}
	if err := enc.Encode(initialRecord{Kind: "initial", Game: g, Config: c, State: stateToJSON(board)}); err != nil {
		return err
	}

	pol := newPolicy(c.GameSeed, params, wrapped, len(board.Snakes))

	for turns := 0; !gameOver && turns < c.MaxTurns; turns++ {
		// createNextBoardState
		pre, err := maps.PreUpdateBoard(gameMap, board, settings)
		if err != nil {
			return fmt.Errorf("PreUpdateBoard: %w", err)
		}
		preJSON := stateToJSON(pre)

		moveList := pol.moves(pre)
		// Like the CLI, only living snakes submit moves.
		var moves []rules.SnakeMove
		for i, s := range pre.Snakes {
			if s.EliminatedCause == rules.NotEliminated {
				moves = append(moves, rules.SnakeMove{ID: s.ID, Move: moveList[i]})
			}
		}

		var postRules *rules.BoardState
		gameOver, postRules, err = ruleset.Execute(pre, moves)
		if err != nil {
			return fmt.Errorf("Execute (turn %d): %w", pre.Turn, err)
		}
		postRulesJSON := stateToJSON(postRules)

		postMap, err := maps.PostUpdateBoard(gameMap, postRules, settings)
		if err != nil {
			return fmt.Errorf("PostUpdateBoard (turn %d): %w", pre.Turn, err)
		}
		postMap.Turn += 1

		rec := transitionRecord{
			Kind:      "transition",
			Game:      g,
			Pre:       preJSON,
			Moves:     moveList,
			PostRules: postRulesJSON,
			PostMap:   stateToJSON(postMap),
			GameOver:  gameOver,
		}
		if err := enc.Encode(rec); err != nil {
			return err
		}
		board = postMap
	}
	return nil
}

// gameSeed derives game g's seed by hashing (base seed, g) with splitmix64.
// Consecutive math/rand seeds give correlated streams, and the engine seeds
// turn T with gameSeed+T, so consecutive per-game seeds would make games share
// food rolls and skew start-position statistics. The result is positive, below
// 2^62 (so adding turn numbers can't overflow) and never 0 (which the engine
// treats as "unseeded").
func gameSeed(base int64, g int) int64 {
	z := uint64(base)*0x9E3779B97F4A7C15 + uint64(g) + 0x9E3779B97F4A7C15
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	z ^= z >> 31
	if s := int64(z >> 2); s != 0 {
		return s
	}
	return 1
}
