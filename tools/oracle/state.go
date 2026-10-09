package main

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/BattlesnakeOfficial/rules"
)

// JSONState is the wire format shared with the Python test-suite. The field
// order (and names) are part of the contract; see README.md.
type JSONState struct {
	Width   int         `json:"width"`
	Height  int         `json:"height"`
	Turn    int         `json:"turn"`
	Snakes  []JSONSnake `json:"snakes"`
	Food    [][2]int    `json:"food"`
	Hazards [][2]int    `json:"hazards"`
}

type JSONSnake struct {
	ID               string   `json:"id"`
	Body             [][2]int `json:"body"`
	Health           int      `json:"health"`
	EliminatedCause  string   `json:"eliminated_cause"`
	EliminatedOnTurn int      `json:"eliminated_on_turn"`
	EliminatedBy     string   `json:"eliminated_by"`
}

// pointsToJSON deep-copies engine points into [x, y] pairs. Never returns nil,
// so empty lists serialise as [] rather than null.
func pointsToJSON(ps []rules.Point) [][2]int {
	out := make([][2]int, len(ps))
	for i, p := range ps {
		out[i] = [2]int{p.X, p.Y}
	}
	return out
}

func pointsFromJSON(ps [][2]int) []rules.Point {
	out := make([]rules.Point, len(ps))
	for i, p := range ps {
		out[i] = rules.Point{X: p[0], Y: p[1]}
	}
	return out
}

// stateToJSON returns a deep copy of the engine state in wire format.
// Snakes keep engine order; food and hazards keep engine order and duplicates.
func stateToJSON(b *rules.BoardState) *JSONState {
	s := &JSONState{
		Width:   b.Width,
		Height:  b.Height,
		Turn:    b.Turn,
		Snakes:  make([]JSONSnake, len(b.Snakes)),
		Food:    pointsToJSON(b.Food),
		Hazards: pointsToJSON(b.Hazards),
	}
	for i, sn := range b.Snakes {
		s.Snakes[i] = JSONSnake{
			ID:               sn.ID,
			Body:             pointsToJSON(sn.Body),
			Health:           sn.Health,
			EliminatedCause:  sn.EliminatedCause,
			EliminatedOnTurn: sn.EliminatedOnTurn,
			EliminatedBy:     sn.EliminatedBy,
		}
	}
	return s
}

// stateFromJSON builds a fresh engine BoardState from wire format.
func stateFromJSON(s *JSONState) *rules.BoardState {
	b := rules.NewBoardState(s.Width, s.Height)
	b.Turn = s.Turn
	b.Food = pointsFromJSON(s.Food)
	b.Hazards = pointsFromJSON(s.Hazards)
	b.Snakes = make([]rules.Snake, len(s.Snakes))
	for i, sn := range s.Snakes {
		b.Snakes[i] = rules.Snake{
			ID:               sn.ID,
			Body:             pointsFromJSON(sn.Body),
			Health:           sn.Health,
			EliminatedCause:  sn.EliminatedCause,
			EliminatedOnTurn: sn.EliminatedOnTurn,
			EliminatedBy:     sn.EliminatedBy,
		}
	}
	return b
}

// StringMap accepts a JSON object whose values are strings, numbers or
// booleans, and stores each value as the string the engine expects.
type StringMap map[string]string

func (m *StringMap) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	out := make(map[string]string, len(raw))
	for k, v := range raw {
		var str string
		if err := json.Unmarshal(v, &str); err == nil {
			out[k] = str
			continue
		}
		var num json.Number
		if err := json.Unmarshal(v, &num); err == nil {
			out[k] = num.String()
			continue
		}
		var b bool
		if err := json.Unmarshal(v, &b); err == nil {
			out[k] = strconv.FormatBool(b)
			continue
		}
		return fmt.Errorf("settings[%q]: expected string, number or bool, got %s", k, string(v))
	}
	*m = out
	return nil
}

var knownRulesets = map[string]bool{
	rules.GameTypeStandard:           true,
	rules.GameTypeSolo:               true,
	rules.GameTypeRoyale:             true,
	rules.GameTypeWrapped:            true,
	rules.GameTypeConstrictor:        true,
	rules.GameTypeWrappedConstrictor: true,
}

func checkRulesetName(name string) error {
	if !knownRulesets[name] {
		return fmt.Errorf("unknown ruleset %q (want one of standard, solo, royale, wrapped, constrictor, wrapped_constrictor)", name)
	}
	return nil
}
