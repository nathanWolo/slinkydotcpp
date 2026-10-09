package main

import (
	"math/rand"

	"github.com/BattlesnakeOfficial/rules"
)

// The built-in move policy. It is not meant to play well: it is meant to
// produce long-ish, varied games that hit every elimination cause and the
// rule edge cases (stacked tails, head-to-heads of 2+ snakes, wrapping,
// hazards, starvation, default moves for invalid move strings).
//
// Everything is driven by a single seeded math/rand stream, and snakes are
// processed in index order, so a (state sequence, seed) pair always yields the
// same moves.

type direction struct {
	name   string
	dx, dy int
}

var directions = [4]direction{
	{rules.MoveUp, 0, 1},
	{rules.MoveDown, 0, -1},
	{rules.MoveLeft, -1, 0},
	{rules.MoveRight, 1, 0},
}

// Invalid move strings sent with --invalid-move-prob. All of them make the
// engine fall back to its default move (continue from neck towards head).
var invalidMoves = [...]string{"", "bogus", "UP", "none"}

type policyParams struct {
	PRandom     float64 // uniformly random move (walls, necks, bodies)
	PAggressive float64 // greedy towards the closest other snake's head
	PFood       float64 // greedy towards the closest food
	PInvalid    float64 // emit an invalid move string
	// Fraction of snakes (drawn per snake per game) that never seek food and
	// avoid stepping on it when they can; these produce out-of-health deaths.
	PFoodAverse float64
	// Fraction of snakes (drawn per snake per game) that avoid dead ends
	// (flood fill) and squares a not-shorter enemy head could also reach.
	// Careful snakes make games long enough for starvation, royale hazards
	// and big constrictor bodies to matter.
	PCareful float64
}

type snakePersona struct {
	foodAverse bool
	careful    bool
}

type policy struct {
	rng     *rand.Rand
	params  policyParams
	wrapped bool
	persona []snakePersona
}

func newPolicy(seed int64, params policyParams, wrapped bool, numSnakes int) *policy {
	p := &policy{
		rng:     rand.New(rand.NewSource(seed)),
		params:  params,
		wrapped: wrapped,
		persona: make([]snakePersona, numSnakes),
	}
	for i := range p.persona {
		p.persona[i].foodAverse = p.rng.Float64() < params.PFoodAverse
		p.persona[i].careful = p.rng.Float64() < params.PCareful
	}
	return p
}

func (p *policy) step(b *rules.BoardState, from rules.Point, d direction) rules.Point {
	n := rules.Point{X: from.X + d.dx, Y: from.Y + d.dy}
	if p.wrapped {
		n.X = ((n.X % b.Width) + b.Width) % b.Width
		n.Y = ((n.Y % b.Height) + b.Height) % b.Height
	}
	return n
}

func (p *policy) dist(b *rules.BoardState, a, c rules.Point) int {
	dx, dy := absInt(a.X-c.X), absInt(a.Y-c.Y)
	if p.wrapped {
		if b.Width-dx < dx {
			dx = b.Width - dx
		}
		if b.Height-dy < dy {
			dy = b.Height - dy
		}
	}
	return dx + dy
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func onBoard(b *rules.BoardState, q rules.Point) bool {
	return q.X >= 0 && q.X < b.Width && q.Y >= 0 && q.Y < b.Height
}

// blockedNextTurn returns the squares that will still hold a body segment
// after every living snake moves: body[:len-1] for each snake. The tail
// square is included exactly when the last two segments are equal (the
// snake ate last turn, or is stacked / always growing in constrictor).
func blockedNextTurn(b *rules.BoardState) map[rules.Point]bool {
	blocked := map[rules.Point]bool{}
	for _, s := range b.Snakes {
		if s.EliminatedCause != rules.NotEliminated || len(s.Body) == 0 {
			continue
		}
		for _, q := range s.Body[:len(s.Body)-1] {
			blocked[rules.Point{X: q.X, Y: q.Y}] = true
		}
	}
	return blocked
}

// moves returns one move per snake (index order). Eliminated snakes get "".
func (p *policy) moves(b *rules.BoardState) []string {
	out := make([]string, len(b.Snakes))
	blocked := blockedNextTurn(b)
	food := map[rules.Point]bool{}
	for _, f := range b.Food {
		food[rules.Point{X: f.X, Y: f.Y}] = true
	}

	for i, s := range b.Snakes {
		if s.EliminatedCause != rules.NotEliminated || len(s.Body) == 0 {
			continue
		}
		out[i] = p.chooseMove(b, i, blocked, food)
	}
	return out
}

func (p *policy) randomMove() string {
	return directions[p.rng.Intn(4)].name
}

func (p *policy) chooseMove(b *rules.BoardState, idx int, blocked, food map[rules.Point]bool) string {
	if p.params.PInvalid > 0 && p.rng.Float64() < p.params.PInvalid {
		return invalidMoves[p.rng.Intn(len(invalidMoves))]
	}

	head := b.Snakes[idx].Body[0]
	head = rules.Point{X: head.X, Y: head.Y}
	averse := p.persona[idx].foodAverse

	// Safe moves: on the board (after wrapping) and not into a square that
	// will still be occupied by a body segment next turn.
	var safe []direction
	for _, d := range directions {
		n := p.step(b, head, d)
		if onBoard(b, n) && !blocked[n] {
			safe = append(safe, d)
		}
	}
	if averse {
		var noFood []direction
		for _, d := range safe {
			if !food[p.step(b, head, d)] {
				noFood = append(noFood, d)
			}
		}
		if len(noFood) > 0 {
			safe = noFood
		}
	}

	if p.persona[idx].careful && len(safe) > 0 {
		safe = p.carefulFilter(b, idx, safe, blocked)
	}

	r := p.rng.Float64()
	if r < p.params.PRandom || len(safe) == 0 {
		return p.randomMove()
	}
	r -= p.params.PRandom

	if r < p.params.PAggressive {
		// Greedy towards the closest other living snake's head.
		best, found := rules.Point{}, false
		bestD := 0
		for j, o := range b.Snakes {
			if j == idx || o.EliminatedCause != rules.NotEliminated || len(o.Body) == 0 {
				continue
			}
			oh := rules.Point{X: o.Body[0].X, Y: o.Body[0].Y}
			if d := p.dist(b, head, oh); !found || d < bestD {
				best, bestD, found = oh, d, true
			}
		}
		if found {
			return p.greedy(b, head, best, safe)
		}
		return safe[p.rng.Intn(len(safe))].name
	}
	r -= p.params.PAggressive

	if r < p.params.PFood && !averse {
		best, found := rules.Point{}, false
		bestD := 0
		for _, f := range b.Food {
			fp := rules.Point{X: f.X, Y: f.Y}
			if d := p.dist(b, head, fp); !found || d < bestD {
				best, bestD, found = fp, d, true
			}
		}
		if found {
			return p.greedy(b, head, best, safe)
		}
	}

	return safe[p.rng.Intn(len(safe))].name
}

// greedy picks, among the candidate moves, one that minimises the distance
// to target (ties broken uniformly at random).
func (p *policy) greedy(b *rules.BoardState, head, target rules.Point, cands []direction) string {
	var best []direction
	bestD := 0
	for _, d := range cands {
		dd := p.dist(b, p.step(b, head, d), target)
		if len(best) == 0 || dd < bestD {
			best, bestD = []direction{d}, dd
		} else if dd == bestD {
			best = append(best, d)
		}
	}
	return best[p.rng.Intn(len(best))].name
}

// carefulFilter keeps the candidate moves that do not lead into a region
// smaller than the snake (falling back to the roomiest ones), and among those
// prefers squares that no other living, not-shorter snake's head can reach.
func (p *policy) carefulFilter(b *rules.BoardState, idx int, cands []direction, blocked map[rules.Point]bool) []direction {
	me := b.Snakes[idx]
	head := rules.Point{X: me.Body[0].X, Y: me.Body[0].Y}
	need := len(me.Body)

	var roomy []direction
	bestArea, areas := -1, make([]int, len(cands))
	for k, d := range cands {
		areas[k] = p.floodArea(b, p.step(b, head, d), blocked, need)
		if areas[k] > bestArea {
			bestArea = areas[k]
		}
	}
	for k, d := range cands {
		if areas[k] >= need || areas[k] == bestArea {
			roomy = append(roomy, d)
		}
	}

	var calm []direction
	for _, d := range roomy {
		n := p.step(b, head, d)
		risky := false
		for j, o := range b.Snakes {
			if j == idx || o.EliminatedCause != rules.NotEliminated || len(o.Body) < need {
				continue
			}
			if p.dist(b, n, rules.Point{X: o.Body[0].X, Y: o.Body[0].Y}) == 1 {
				risky = true
				break
			}
		}
		if !risky {
			calm = append(calm, d)
		}
	}
	if len(calm) > 0 {
		return calm
	}
	return roomy
}

// floodArea counts squares reachable from start without crossing blocked
// squares, stopping once limit squares have been found.
func (p *policy) floodArea(b *rules.BoardState, start rules.Point, blocked map[rules.Point]bool, limit int) int {
	seen := map[rules.Point]bool{start: true}
	queue := []rules.Point{start}
	for len(queue) > 0 && len(seen) < limit {
		q := queue[0]
		queue = queue[1:]
		for _, d := range directions {
			n := p.step(b, q, d)
			if !onBoard(b, n) || blocked[n] || seen[n] {
				continue
			}
			seen[n] = true
			queue = append(queue, n)
		}
	}
	return len(seen)
}
