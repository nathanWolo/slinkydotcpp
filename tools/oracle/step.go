package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/BattlesnakeOfficial/rules"
)

type stepRequest struct {
	Ruleset  string     `json:"ruleset"`
	Solo     bool       `json:"solo"`
	Settings StringMap  `json:"settings"`
	Seed     *int64     `json:"seed"`
	State    *JSONState `json:"state"`
	// Moves[i] is the move for snake i. JSON null (or an absent key) passes a
	// nil move list to Execute, which is how the CLI initialises turn 0.
	Moves *[]string `json:"moves"`
}

type stepResponse struct {
	GameOver bool       `json:"game_over"`
	State    *JSONState `json:"state"`
	Error    string     `json:"error"`
}

// runStep implements `oracle step`: one JSON request per input line, one
// JSON response per output line, flushed after every line.
func runStep(args []string) error {
	if len(args) > 0 {
		return fmt.Errorf("step takes no arguments (reads JSON Lines on stdin)")
	}
	in := bufio.NewReaderSize(os.Stdin, 1<<20)
	out := bufio.NewWriterSize(os.Stdout, 1<<16)
	enc := json.NewEncoder(out)
	enc.SetEscapeHTML(false)

	for {
		line, err := in.ReadBytes('\n')
		if len(trimSpace(line)) > 0 {
			resp := handleStepLine(line)
			if encErr := enc.Encode(resp); encErr != nil {
				return encErr
			}
			if flErr := out.Flush(); flErr != nil {
				return flErr
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func trimSpace(b []byte) []byte {
	start, end := 0, len(b)
	for start < end && (b[start] == ' ' || b[start] == '\t' || b[start] == '\r' || b[start] == '\n') {
		start++
	}
	for end > start && (b[end-1] == ' ' || b[end-1] == '\t' || b[end-1] == '\r' || b[end-1] == '\n') {
		end--
	}
	return b[start:end]
}

func handleStepLine(line []byte) (resp stepResponse) {
	// The engine indexes slices freely (e.g. constrictor reads body[len-2]);
	// a malformed state must produce an error response, not kill the process.
	defer func() {
		if r := recover(); r != nil {
			resp = stepResponse{Error: fmt.Sprintf("engine panic: %v", r)}
		}
	}()

	var req stepRequest
	if err := json.Unmarshal(line, &req); err != nil {
		return stepResponse{Error: "bad request: " + err.Error()}
	}
	if err := checkRulesetName(req.Ruleset); err != nil {
		return stepResponse{Error: err.Error()}
	}
	if req.State == nil {
		return stepResponse{Error: "bad request: missing state"}
	}
	seed := int64(1)
	if req.Seed != nil {
		seed = *req.Seed
	}

	board := stateFromJSON(req.State)

	var moves []rules.SnakeMove
	if req.Moves != nil {
		ms := *req.Moves
		if len(ms) > len(board.Snakes) {
			return stepResponse{Error: fmt.Sprintf("bad request: %d moves for %d snakes", len(ms), len(board.Snakes))}
		}
		moves = make([]rules.SnakeMove, len(ms))
		for i, m := range ms {
			moves[i] = rules.SnakeMove{ID: board.Snakes[i].ID, Move: m}
		}
	}

	settings := map[string]string(req.Settings)
	if settings == nil {
		settings = map[string]string{}
	}
	ruleset := rules.NewRulesetBuilder().
		WithParams(settings).
		WithSeed(seed).
		WithSolo(req.Solo).
		NamedRuleset(req.Ruleset)

	gameOver, next, err := ruleset.Execute(board, moves)
	if err != nil {
		return stepResponse{GameOver: gameOver, State: nil, Error: err.Error()}
	}
	return stepResponse{GameOver: gameOver, State: stateToJSON(next)}
}
