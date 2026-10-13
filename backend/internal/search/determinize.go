package search

import (
	"fmt"
	"math/rand"

	"cua/internal/game"
)

// Sample returns one plausible world from `rootSeat`'s point of view: the hidden hands are
// dealt out of the cards that seat cannot see.
//
// This is the ISMCTS "determinization" step, and it is the honest model of hidden
// information: the seat knows its own hand, the table and every capture (all public), and
// NOTHING about the other three hands or the order of the deck. It mirrors
// `src/ismcts/determinize.py: Determinizer.sample`.
//
// Note that the root seat's TEAMMATE is treated as an opponent here (its hand is sampled),
// which is what the Python implementation does too: for information-set purposes a
// teammate is another hidden hand.
func Sample(rootSeat int, s *game.State, rng *rand.Rand) (*game.State, error) {
	sampled := Clone(s)

	known := make(map[game.Card]bool, game.HandSize+len(s.Table)+
		len(s.CardsCaptured[0])+len(s.CardsCaptured[1]))
	for _, card := range s.Hands[rootSeat] {
		known[card] = true
	}
	for _, card := range s.Table {
		known[card] = true
	}
	for team := 0; team < 2; team++ {
		for _, card := range s.CardsCaptured[team] {
			known[card] = true
		}
	}

	pool := make([]game.Card, 0, len(game.BuildDeck())-len(known))
	for _, card := range game.BuildDeck() {
		if !known[card] {
			pool = append(pool, card)
		}
	}

	needed := 0
	sizes := [game.NumPlayers]int{}
	for p := 0; p < game.NumPlayers; p++ {
		if p == rootSeat {
			continue
		}
		// `plays_this_deal` counts the turns that seat has taken, so its hand holds
		// HandSize - plays cards. (The backend's own deal bookkeeping must agree, or the
		// pool cannot be split exactly; see the error below.)
		size := game.HandSize - s.PlaysThisDeal[p]
		if size < 0 {
			return nil, fmt.Errorf("seat %d has played %d of %d cards",
				p, s.PlaysThisDeal[p], game.HandSize)
		}
		if size != len(s.Hands[p]) {
			return nil, fmt.Errorf("plays_this_deal desync for seat %d: HandSize-plays=%d "+
				"but the hand holds %d cards", p, size, len(s.Hands[p]))
		}
		sizes[p] = size
		needed += size
	}
	needed += len(s.Deck)

	if needed != len(pool) {
		return nil, fmt.Errorf("determinization bookkeeping: need %d cards, %d unseen "+
			"(is the state consistent? see Validate)", needed, len(pool))
	}

	rng.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })

	idx := 0
	hands := make(map[int][]game.Card, game.NumPlayers)
	hands[rootSeat] = append([]game.Card(nil), s.Hands[rootSeat]...)
	for p := 0; p < game.NumPlayers; p++ {
		if p == rootSeat {
			continue
		}
		hand := make([]game.Card, sizes[p])
		copy(hand, pool[idx:idx+sizes[p]])
		hands[p] = hand
		idx += sizes[p]
	}
	sampled.Hands = hands
	sampled.Deck = append([]game.Card(nil), pool[idx:idx+len(s.Deck)]...)

	return sampled, nil
}

// PoolSize is how many cards are hidden from `rootSeat` — used by the tests and by the
// diagnostics printout.
func PoolSize(rootSeat int, s *game.State) int {
	known := map[game.Card]bool{}
	for _, card := range s.Hands[rootSeat] {
		known[card] = true
	}
	for _, card := range s.Table {
		known[card] = true
	}
	for team := 0; team < 2; team++ {
		for _, card := range s.CardsCaptured[team] {
			known[card] = true
		}
	}
	count := 0
	for _, card := range game.BuildDeck() {
		if !known[card] {
			count++
		}
	}
	return count
}
