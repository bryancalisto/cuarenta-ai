// Package search is a Go port of the ISMCTS used by `src/ismcts/network_ismcts_1.py`, so
// the web backend can play with the SEARCH and not only with the raw policy.
//
// WHY THIS EXISTS AT ALL: the ONNX model the backend serves is a policy network. On its
// own it is worth about a coin flip against the rule-based heuristic — MEASURED on
// `checkpoints-return/best_heuristic.pt` over 200 games: 49.0% ± 3.5% with the bare policy,
// versus 77.0% for the same weights with the search (and the training run's own
// `teacher_edge` metric said the same thing all along: the search beats this policy on
// 67-85% of decisions). The strength of an AlphaZero-style agent lives in its search, so
// serving the net alone throws most of it away.
//
// This package is deliberately a faithful port, not a re-design: same PUCT, same
// determinization, same value semantics, same dense-reward accumulation in "return" mode.
// It is verified against the engine it lives in (`StepState` must reproduce `game.Env.step`
// step for step, see state_test.go) and against the Python search's *mechanics*
// (search_test.go pins what return mode and win mode must prefer).
package search

import (
	"fmt"
	"math/rand"

	"cua/internal/game"
)

// Reward terms, mirroring `src/ismcts/env.py`. They MUST match that file: the value head
// these numbers feed was trained against them, and a mismatch would make the search
// optimise a different objective than the one the policy imitates. `state_test.go` pins
// them, and `scripts/` generates a cross-language fixture that compares this reward with
// the Python environment's on real games.
const (
	CaidaPoints  = 2
	LimpiaPoints = 2
	WinBonus     = 20
)

// Clone returns a deep copy of a state.
//
// The search mutates states freely (it steps them millions of times per game), so nothing
// may be shared with the caller's state: `Hands` is a map, `Table`/`Deck`/`CardsCaptured`
// are slices, and `LastPlaced`/`Winner` are pointers.
func Clone(s *game.State) *game.State {
	if s == nil {
		return nil
	}
	out := &game.State{
		Hands:         make(map[int][]game.Card, len(s.Hands)),
		Table:         append([]game.Card(nil), s.Table...),
		Deck:          append([]game.Card(nil), s.Deck...),
		DecksConsumed: s.DecksConsumed,
		Scores:        s.Scores,
		CurrentPlayer: s.CurrentPlayer,
		Done:          s.Done,
		PlaysThisDeal: s.PlaysThisDeal,
	}
	for seat, hand := range s.Hands {
		out.Hands[seat] = append([]game.Card(nil), hand...)
	}
	for team := 0; team < 2; team++ {
		out.CardsCaptured[team] = append([]game.Card(nil), s.CardsCaptured[team]...)
	}
	if s.LastPlaced != nil {
		card := *s.LastPlaced
		out.LastPlaced = &card
	}
	if s.Winner != nil {
		winner := *s.Winner
		out.Winner = &winner
	}
	return out
}

// CartonPoints is the end-of-deal carton for one team: `len(captured) - 19`, plus 5 (and
// nudged to even) when positive, else 0. Mirrors `game.Env`'s deal-end block and
// `src/ismcts/env.py: _count_carton_points`.
func CartonPoints(captured []game.Card) int {
	points := len(captured) - 19
	if points <= 0 {
		return 0
	}
	points += 5
	if points%2 != 0 {
		points++
	}
	return points
}

// StepResult is what one transition produced: the DENSE reward paid to the acting team
// (0 in win-value-target mode, where the search ignores rewards), whether the game ended,
// and the acting seat.
type StepResult struct {
	Reward  float64
	Done    bool
	Player  int
	EndedBy string // "win" | "carton" | ""  — for tests and diagnostics
}

// StepState applies `action` to a COPY of `s`. It is the search's transition and must be
// exactly equivalent to `game.Env.step` on the same sequence of actions and RNG (see
// state_test.go, which runs both in lockstep).
//
// The reward follows `src/ismcts/env.py`:
//
//	+1 per captured card, +2 caída, +2 limpia, the carton when the deal ends here,
//	+WinBonus when the move ends the game in the acting team's favour
//
// The carton is paid UNCAPPED (the cap only applies to the score), matching the Python env.
func StepState(s *game.State, action game.Action, rng *rand.Rand) (*game.State, StepResult, error) {
	if s.Done {
		return nil, StepResult{}, fmt.Errorf("game is already over")
	}

	ns := Clone(s)
	player := ns.CurrentPlayer
	team := player % 2 // teamByPlayer in the game package: {0:0, 1:1, 2:0, 3:1}

	legal := legalActionsFromState(ns)
	if !actionExists(action, legal) {
		return nil, StepResult{}, fmt.Errorf("illegal action: %s", action.String())
	}

	result := StepResult{Player: player}

	// The caída/limpia flags come from the state BEFORE the move (`scorePlay` needs the
	// table as it was and LastPlaced as it was). `game.ActionFeatures` is the single source
	// of that logic — the same call the network's input is built from — so the reward and
	// the features cannot drift apart.
	feats := game.ActionFeatures(action, ns)
	caida := feats[game.ActionFeatureIsCaida] == 1
	limpia := feats[game.ActionFeatureIsLimpia] == 1

	ns.Hands[player] = removeCard(ns.Hands[player], action.Card)
	ns.PlaysThisDeal[player]++

	if len(action.Capture) > 0 {
		for _, captured := range canonicalCards(action.Capture) {
			ns.Table = removeCard(ns.Table, captured)
			ns.CardsCaptured[team] = append(ns.CardsCaptured[team], captured)
		}
		ns.CardsCaptured[team] = append(ns.CardsCaptured[team], action.Card)
		ns.LastPlaced = nil

		result.Reward += float64(len(action.Capture))
		if caida {
			result.Reward += CaidaPoints
		}
		if limpia {
			result.Reward += LimpiaPoints
		}
	} else {
		ns.Table = append(ns.Table, action.Card)
		card := action.Card
		ns.LastPlaced = &card
	}

	// The SAME bonuses are what the score gains (scorePlay: caída and limpia are each
	// worth 2 points, and at most one of them can apply).
	if caida || limpia {
		ns.Scores[team] += CaidaPoints
	}

	if ns.Scores[team] >= game.WinScore {
		ns.Done = true
		winner := team
		ns.Winner = &winner
		result.Reward += WinBonus
		result.Done = true
		result.EndedBy = "win"
		return ns, result, nil
	}

	ns.CurrentPlayer = (player + 1) % game.NumPlayers

	if allHandsEmpty(ns) {
		if len(ns.Deck) >= game.NumPlayers*game.HandSize {
			redeal(ns)
		} else {
			// Carton: scored for both teams (capped at MaxAchievable...), and PAID TO THE
			// ACTING TEAM as reward, uncapped.
			result.Reward += float64(CartonPoints(ns.CardsCaptured[team]))
			for t := 0; t < 2; t++ {
				carton := CartonPoints(ns.CardsCaptured[t])
				if carton == 0 {
					continue
				}
				ns.Scores[t] = min(ns.Scores[t]+carton, game.MaxAchievablePointsByNonCaidaActions)
			}

			// NOTE (a real divergence from the Python training env): the carton can win the
			// game, and this engine honours that, while `src/ismcts/env.py` checks the win
			// condition only after a MOVE, so it lets a game continue after a carton has
			// taken a team past 40. The backend plays THIS rule (it is the correct one), so
			// the search models it too.
			if ns.Scores[0] >= game.WinScore || ns.Scores[1] >= game.WinScore {
				ns.Done = true
				winner := 0
				if ns.Scores[1] > ns.Scores[0] {
					winner = 1
				}
				ns.Winner = &winner
				if winner == team {
					result.Reward += WinBonus
				}
				result.Done = true
				result.EndedBy = "carton"
				return ns, result, nil
			}

			newRound(ns, rng)
		}
	}

	return ns, result, nil
}

func allHandsEmpty(s *game.State) bool {
	for p := 0; p < game.NumPlayers; p++ {
		if len(s.Hands[p]) != 0 {
			return false
		}
	}
	return true
}

func redeal(s *game.State) {
	for p := 0; p < game.NumPlayers; p++ {
		hand := make([]game.Card, game.HandSize)
		copy(hand, s.Deck[:game.HandSize])
		s.Hands[p] = hand
		s.Deck = s.Deck[game.HandSize:]
	}
	s.PlaysThisDeal = [game.NumPlayers]int{}
}

func newRound(s *game.State, rng *rand.Rand) {
	newDeck := game.ShuffledDeck(rng)
	hands := make(map[int][]game.Card, game.NumPlayers)
	for p := 0; p < game.NumPlayers; p++ {
		hand := make([]game.Card, game.HandSize)
		copy(hand, newDeck[:game.HandSize])
		hands[p] = hand
		newDeck = newDeck[game.HandSize:]
	}
	s.Hands, s.Deck = hands, newDeck
	s.CardsCaptured = [2][]game.Card{}
	s.Table = nil
	s.LastPlaced = nil
	s.CurrentPlayer = (s.CurrentPlayer + 1) % game.NumPlayers
	s.PlaysThisDeal = [game.NumPlayers]int{}
}

// legalActionsFromState is `game.Env.legalActions` for a bare state (the game package's
// version is a method on Env, which owns the mutable state).
func legalActionsFromState(s *game.State) []game.Action {
	player := s.CurrentPlayer
	hand := s.Hands[player]
	actions := make([]game.Action, 0, len(hand))
	for _, card := range hand {
		captures := game.ValidCaptures(card, s.Table)
		if len(captures) > 0 {
			for _, capSet := range captures {
				actions = append(actions, game.Action{Card: card, Capture: canonicalCards(capSet)})
			}
			continue
		}
		actions = append(actions, game.Action{Card: card})
	}
	return actions
}

func canonicalCards(cards []game.Card) []game.Card {
	out := append([]game.Card(nil), cards...)
	sortCards(out)
	return out
}

func sortCards(cards []game.Card) {
	for i := 1; i < len(cards); i++ {
		for j := i; j > 0 && cards[j].Less(cards[j-1]); j-- {
			cards[j], cards[j-1] = cards[j-1], cards[j]
		}
	}
}

func removeCard(cards []game.Card, target game.Card) []game.Card {
	out := make([]game.Card, 0, len(cards))
	removed := false
	for _, card := range cards {
		if !removed && card == target {
			removed = true
			continue
		}
		out = append(out, card)
	}
	return out
}

func actionExists(action game.Action, legal []game.Action) bool {
	for _, candidate := range legal {
		if candidate.Card == action.Card && sameCapture(candidate.Capture, action.Capture) {
			return true
		}
	}
	return false
}

func sameCapture(a, b []game.Card) bool {
	if len(a) != len(b) {
		return false
	}
	sortedA, sortedB := canonicalCards(a), canonicalCards(b)
	for i := range sortedA {
		if sortedA[i] != sortedB[i] {
			return false
		}
	}
	return true
}

// LegalActions is the exported form, used by the tests and by the search's root.
func LegalActions(s *game.State) []game.Action {
	return legalActionsFromState(s)
}

// Validate reports whether the state is internally consistent (40 distinct cards across
// hands, table and deck... note that captured cards are NOT in the deck any more, so the
// invariant is: nothing appears twice anywhere).
func Validate(s *game.State) error {
	seen := map[game.Card]string{}
	add := func(card game.Card, where string) error {
		if prev, ok := seen[card]; ok {
			return fmt.Errorf("card %s appears twice: %s and %s", card, prev, where)
		}
		seen[card] = where
		return nil
	}
	for seat, hand := range s.Hands {
		for _, card := range hand {
			if err := add(card, fmt.Sprintf("hand[%d]", seat)); err != nil {
				return err
			}
		}
	}
	for _, card := range s.Table {
		if err := add(card, "table"); err != nil {
			return err
		}
	}
	for _, card := range s.Deck {
		if err := add(card, "deck"); err != nil {
			return err
		}
	}
	for team := 0; team < 2; team++ {
		for _, card := range s.CardsCaptured[team] {
			if err := add(card, fmt.Sprintf("captured[%d]", team)); err != nil {
				return err
			}
		}
	}
	return nil
}
