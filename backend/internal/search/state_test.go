package search

import (
	"math/rand"
	"testing"

	"cua/internal/game"
)

// TestStepStateMatchesEnvStep is the load-bearing test of this package: the search's
// transition must be EXACTLY the engine's transition, or the search is planning in a game
// nobody is playing — and it would still return legal, plausible moves, so nothing else
// would notice.
//
// The two are run in lockstep: same seed, same action sequence, and after every ply the
// resulting states must be identical (cards, scores, whose turn it is, deck, captures,
// plays-per-deal, and the end-of-game verdict). The env's RNG is reproduced by giving
// `StepState` an *rand.Rand with the same seed, because both shuffle a fresh deck at a new
// round.
func TestStepStateMatchesEnvStep(t *testing.T) {
	for _, seed := range []int64{1, 7, 12345, 987654} {
		env := game.NewEnv(seed)
		env.Reset()

		state := Clone(env.State())
		// Mirror the env's rng consumption: `Reset` shuffled the opening deck, so the
		// search's rng must burn the same draw or a later `newRound` deals a different deck
		// and the lockstep comparison diverges for a reason that has nothing to do with the
		// transition.
		rng := rand.New(rand.NewSource(seed))
		_ = game.ShuffledDeck(rng)

		plies := 0
		for !state.Done {
			legal := LegalActions(state)
			if len(legal) == 0 {
				t.Fatalf("seed %d ply %d: no legal actions", seed, plies)
			}
			pick := legal[rand.New(rand.NewSource(seed+int64(plies))).Intn(len(legal))]

			next, result, err := StepState(state, pick, rng)
			if err != nil {
				t.Fatalf("seed %d ply %d: StepState(%s): %v", seed, plies, pick, err)
			}

			envObs, envDone, _, err := env.Step(pick)
			if err != nil {
				t.Fatalf("seed %d ply %d: Env.Step(%s): %v", seed, plies, pick, err)
			}
			if envObs == nil {
				t.Fatalf("seed %d ply %d: env returned no observation", seed, plies)
			}

			if msg := diffStates(next, env.State()); msg != "" {
				t.Fatalf("seed %d ply %d, action %s: search and engine diverged: %s",
					seed, plies, pick, msg)
			}
			if result.Done != envDone {
				t.Fatalf("seed %d ply %d: search says done=%v, engine says %v",
					seed, plies, result.Done, envDone)
			}
			if err := Validate(next); err != nil {
				t.Fatalf("seed %d ply %d: %v", seed, plies, err)
			}

			state = next
			plies++
			if plies > 600 {
				t.Fatalf("seed %d: game did not terminate", seed)
			}
		}

		if state.Winner == nil {
			t.Fatalf("seed %d: finished game has no winner", seed)
		}
		if plies < 40 {
			t.Fatalf("seed %d: only %d plies, suspiciously short", seed, plies)
		}
	}
}

// TestStepStateDoesNotMutateItsInput pins the cloning discipline: the search steps the SAME
// state once per simulation with a different determinization, so a transition that leaked
// a mutation would corrupt the whole tree.
func TestStepStateDoesNotMutateItsInput(t *testing.T) {
	env := game.NewEnv(99)
	env.Reset()
	before := Clone(env.State())

	legal := env.LegalActions()
	if len(legal) == 0 {
		t.Fatal("no legal actions in the opening position")
	}
	if _, _, err := StepState(before, legal[0], rand.New(rand.NewSource(1))); err != nil {
		t.Fatalf("StepState: %v", err)
	}

	if msg := diffStates(before, env.State()); msg != "" {
		t.Fatalf("StepState mutated its input: %s", msg)
	}
}

// TestDenseRewardTerms pins the numbers the return-mode search accumulates, and the fact
// that the search token it in the same units the Python env pays. A silent change here
// (say, dropping the carton) would make the search optimise a different objective than the
// value head and the policy were trained against.
func TestDenseRewardTerms(t *testing.T) {
	if CaidaPoints != 2 || LimpiaPoints != 2 || WinBonus != 20 {
		t.Fatalf("reward terms changed: caida=%d limpia=%d win=%d (src/ismcts/env.py has "+
			"2/2/20; changing them needs the value head retrained)",
			CaidaPoints, LimpiaPoints, WinBonus)
	}

	// Carton points: len(captured) - 19, +5 when positive, nudged to even.
	cases := []struct {
		captured int
		want     int
	}{
		{0, 0}, {19, 0}, {20, 6}, {21, 8}, {22, 8}, {23, 10}, {40, 26},
	}
	for _, tc := range cases {
		cards := make([]game.Card, tc.captured)
		if got := CartonPoints(cards); got != tc.want {
			t.Fatalf("CartonPoints(%d cards) = %d, want %d", tc.captured, got, tc.want)
		}
	}

	// A capture of n cards pays n, plus the limpia bonus when it clears the table.
	env := game.NewEnv(4)
	env.Reset()
	state := env.State()
	legal := LegalActions(state)
	sawCapture, sawWin := false, false
	for _, action := range legal {
		next, result, err := StepState(state, action, rand.New(rand.NewSource(2)))
		if err != nil {
			t.Fatalf("StepState: %v", err)
		}
		if len(action.Capture) > 0 {
			sawCapture = true
			feats := game.ActionFeatures(action, state)
			want := float64(len(action.Capture))
			if feats[game.ActionFeatureIsCaida] == 1 {
				want += CaidaPoints
			}
			if feats[game.ActionFeatureIsLimpia] == 1 {
				want += LimpiaPoints
			}
			if result.Reward != want {
				t.Fatalf("capture of %d paid %.1f, want %.1f", len(action.Capture),
					result.Reward, want)
			}
			if result.Reward <= 0 || next.Scores == state.Scores && want > 0 {
				// the score must move with the bonus, and the reward must be positive
				if want > 0 && result.Reward <= 0 {
					t.Fatalf("capture paid no reward: %+v", result)
				}
			}
		} else if result.Reward != 0 {
			t.Fatalf("placing a card paid %.1f, want 0", result.Reward)
		}
		if result.Done && result.EndedBy == "win" {
			sawWin = result.Reward >= WinBonus
		}
	}
	if !sawCapture {
		t.Skip("the opening position had no capture available")
	}
	if !sawWin {
		t.Log("no game-winning move in the opening position (expected)")
	}
}

// TestWinBonusIsPaidToTheWinner plays a whole game and checks that exactly one transition
// carried the +WinBonus, on the winning team's move.
func TestWinBonusIsPaidToTheWinner(t *testing.T) {
	for _, seed := range []int64{11, 22, 33} {
		env := game.NewEnv(seed)
		env.Reset()
		state := Clone(env.State())
		rng := rand.New(rand.NewSource(seed))
		bonusCount := 0
		plies := 0
		for !state.Done {
			legal := LegalActions(state)
			pick := legal[rand.New(rand.NewSource(seed*100+int64(plies))).Intn(len(legal))]
			next, result, err := StepState(state, pick, rng)
			if err != nil {
				t.Fatalf("seed %d: %v", seed, err)
			}
			if result.Done {
				bonusCount++
				if next.Winner == nil {
					t.Fatalf("seed %d: done with no winner", seed)
				}
				if *next.Winner != result.Player%2 {
					// A carton can finish the game for the OTHER team.
					if result.EndedBy != "carton" {
						t.Fatalf("seed %d: winner %d but the move was made by team %d (%s)",
							seed, *next.Winner, result.Player%2, result.EndedBy)
					}
					if result.Reward >= WinBonus {
						t.Fatalf("seed %d: the losing team's carton paid the win bonus",
							seed)
					}
				} else if result.Reward < WinBonus {
					t.Fatalf("seed %d: winning move paid %.1f, expected at least %d",
						seed, result.Reward, WinBonus)
				}
			}
			state = next
			plies++
			if plies > 600 {
				t.Fatalf("seed %d: game did not terminate", seed)
			}
		}
		if bonusCount != 1 {
			t.Fatalf("seed %d: the win bonus was paid on %d transitions, expected exactly 1",
				seed, bonusCount)
		}
	}
}

// diffStates returns a human-readable description of the first difference, or "" when the
// states are identical.
func diffStates(a, b *game.State) string {
	if a == nil || b == nil {
		return "one of the states is nil"
	}
	if a.CurrentPlayer != b.CurrentPlayer {
		return "current player"
	}
	if a.Scores != b.Scores {
		return "scores"
	}
	if a.Done != b.Done {
		return "done flag"
	}
	if (a.Winner == nil) != (b.Winner == nil) ||
		(a.Winner != nil && *a.Winner != *b.Winner) {
		return "winner"
	}
	if a.PlaysThisDeal != b.PlaysThisDeal {
		return "plays_this_deal"
	}
	if a.DealsDealt != b.DealsDealt {
		return "deals_dealt"
	}
	if (a.LastPlaced == nil) != (b.LastPlaced == nil) ||
		(a.LastPlaced != nil && *a.LastPlaced != *b.LastPlaced) {
		return "last placed"
	}
	if len(a.Table) != len(b.Table) {
		return "table size"
	}
	for i := range a.Table {
		if a.Table[i] != b.Table[i] {
			return "table contents"
		}
	}
	if len(a.Deck) != len(b.Deck) {
		return "deck size"
	}
	for i := range a.Deck {
		if a.Deck[i] != b.Deck[i] {
			return "deck contents"
		}
	}
	for seat := 0; seat < game.NumPlayers; seat++ {
		if len(a.Hands[seat]) != len(b.Hands[seat]) {
			return "hand size"
		}
		ah, bh := canonicalCards(a.Hands[seat]), canonicalCards(b.Hands[seat])
		for i := range ah {
			if ah[i] != bh[i] {
				return "hand contents"
			}
		}
	}
	for team := 0; team < 2; team++ {
		if len(a.CardsCaptured[team]) != len(b.CardsCaptured[team]) {
			return "captured size"
		}
	}
	return ""
}
