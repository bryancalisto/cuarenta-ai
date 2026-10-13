package search

import (
	"math"
	"math/rand"
	"testing"

	"cua/internal/game"
)

// fakeNet is a stub evaluator: uniform policy every time and a constant value. It is what
// makes these tests deterministic and about the SEARCH, not about a particular checkpoint.
type fakeNet struct {
	value float64
	steps int
	calls int
	// priorBias lets a test favour a slot (index into the legal list) without a real net.
	priorBias []float64
}

func (f *fakeNet) Steps() int {
	if f.steps < 1 {
		return 1
	}
	return f.steps
}

func (f *fakeNet) Forward(obsSeq, actSeq, actMask, feats, mask []float32, steps int) ([]float32, float32, error) {
	f.calls++
	logits := make([]float32, game.MaxActionSlots)
	n := 0
	for i := 0; i < game.MaxActionSlots; i++ {
		if mask[i] == 1 {
			n++
		}
	}
	for i := 0; i < n; i++ {
		bias := 0.0
		if i < len(f.priorBias) {
			bias = f.priorBias[i]
		}
		logits[i] = float32(bias)
	}
	for i := n; i < game.MaxActionSlots; i++ {
		logits[i] = -1e9 // the model masks illegal slots; pretend this stub does too
	}
	return logits, float32(f.value), nil
}

// capturePosition builds a position whose legal actions differ ONLY in how much reward they
// collect: playing the 7 captures two cards (a limpia, so +4 in return mode), the 5 captures
// one (+1), the figures place a card for 0.
func capturePosition(t *testing.T) *game.State {
	t.Helper()
	env := game.NewEnv(0)
	env.Reset()
	state := Clone(env.State())

	// A state the search can own: seat 0 to move, a small table, one card per action.
	table := []game.Card{{Rank: game.Dos, Suit: game.Brillos}, {Rank: game.Cinco, Suit: game.Treboles}}
	hand := []game.Card{
		{Rank: game.Siete, Suit: game.CorazonesRojos}, // sums 2+5 -> captures both
		{Rank: game.Cinco, Suit: game.CorazonesRojos}, // matches the 5 -> captures one
		{Rank: game.King, Suit: game.CorazonesNegros},
		{Rank: game.Queen, Suit: game.Brillos},
		{Rank: game.Jack, Suit: game.Treboles},
	}
	state.Hands = map[int][]game.Card{
		0: hand,
		1: {{Rank: game.As, Suit: game.Brillos}, {Rank: game.Tres, Suit: game.Treboles},
			{Rank: game.Cuatro, Suit: game.CorazonesRojos}, {Rank: game.Seis, Suit: game.CorazonesNegros},
			{Rank: game.King, Suit: game.Treboles}},
		2: {{Rank: game.As, Suit: game.CorazonesRojos}, {Rank: game.Dos, Suit: game.Treboles},
			{Rank: game.Cuatro, Suit: game.Brillos}, {Rank: game.Seis, Suit: game.Brillos},
			{Rank: game.Queen, Suit: game.CorazonesRojos}},
		3: {{Rank: game.As, Suit: game.CorazonesNegros}, {Rank: game.Tres, Suit: game.CorazonesRojos},
			{Rank: game.Cuatro, Suit: game.CorazonesNegros}, {Rank: game.King, Suit: game.CorazonesRojos},
			{Rank: game.Queen, Suit: game.Treboles}},
	}
	state.Table = table
	state.LastPlaced = nil
	state.CurrentPlayer = 0
	state.Scores = [2]int{}
	state.CardsCaptured = [2][]game.Card{}
	state.PlaysThisDeal = [game.NumPlayers]int{}
	// The rest of the deck, so the determinizer's bookkeeping adds up: everything that is
	// not in a hand or on the table must be in the deck.
	used := map[game.Card]bool{}
	for _, hand := range state.Hands {
		for _, card := range hand {
			used[card] = true
		}
	}
	for _, card := range table {
		used[card] = true
	}
	state.Deck = nil
	for _, card := range game.BuildDeck() {
		if !used[card] {
			state.Deck = append(state.Deck, card)
		}
	}
	if err := Validate(state); err != nil {
		t.Fatalf("crafted position is inconsistent: %v", err)
	}
	return state
}

// TestReturnModeSearchPrefersTheBiggerCapture is the Go twin of the Python
// `test_return_mode_search_prefers_the_bigger_capture`: with a value head that knows
// nothing (constant value), the only signal left is the dense reward the transition pays.
// In return mode the search must chase it; in win mode it must not even see it.
func TestReturnModeSearchPrefersTheBiggerCapture(t *testing.T) {
	for _, tc := range []struct {
		target  string
		wantBig bool
	}{
		{ValueTargetReturn, true},
		{ValueTargetWin, false},
	} {
		state := capturePosition(t)
		legal := LegalActions(state)
		best := 0
		for i, action := range legal {
			if len(action.Capture) > len(legal[best].Capture) {
				best = i
			}
		}
		if len(legal[best].Capture) != 2 {
			t.Fatalf("the crafted position should offer a 2-card capture, got %d",
				len(legal[best].Capture))
		}

		cfg := DefaultConfig()
		cfg.Iterations = 200
		cfg.CPuct = 0.1 // this test is about the BACKUP, not about exploration
		cfg.ValueTarget = tc.target
		cfg.ValueScale = 30
		net := &fakeNet{value: 0}
		s, err := New(state, 0, net, cfg, rand.New(rand.NewSource(7)))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		chosen, stats, err := s.Decide(state, nil)
		if err != nil {
			t.Fatalf("Decide: %v", err)
		}
		if stats.Simulations != cfg.Iterations {
			t.Fatalf("ran %d simulations, expected %d", stats.Simulations, cfg.Iterations)
		}
		// Compare the big capture's VISIT SHARE, not the single chosen action: in win mode
		// the reward is invisible, so the visits spread over the five legal actions and the
		// argmax is a coin toss between them; in return mode the reward concentrates them.
		share := 0.0
		total := 0
		for i, count := range stats.Visits {
			total += count
			if sameAction(stats.Actions[i], legal[best]) {
				share = float64(count)
			}
		}
		if total == 0 {
			t.Fatalf("no visits recorded")
		}
		share /= float64(total)
		if tc.wantBig && share < 0.7 {
			t.Fatalf("return mode should chase the 2-card capture (+4 reward); it took it "+
				"%.0f%% of the time (visits %v)", share*100, stats.Visits)
		}
		if !tc.wantBig && share > 0.7 {
			t.Fatalf("win mode must ignore the reward; the 2-card capture got %.0f%% of the "+
				"visits (visits %v)", share*100, stats.Visits)
		}
		t.Logf("value target %-6s -> %s (captures %d, share %.0f%%, root value %.3f)",
			tc.target, chosen, len(chosen.Capture), share*100, stats.RootValue)
	}
}

// TestSearchWithNoSimulationsFollowsThePolicy pins the interpretation of the budget: one
// simulation is the smallest search, and with a peaked prior the search must agree with the
// policy — otherwise the search would be overriding a good prior with noise.
func TestOneSimulationFollowsThePolicy(t *testing.T) {
	state := capturePosition(t)
	legal := LegalActions(state)

	cfg := DefaultConfig()
	cfg.Iterations = 1
	cfg.CPuct = 1.5
	cfg.ValueTarget = ValueTargetReturn
	cfg.ValueScale = 30
	// Strongly favour the LAST legal action.
	bias := make([]float64, len(legal))
	for i := range bias {
		bias[i] = -5
	}
	bias[len(legal)-1] = 5

	net := &fakeNet{value: 0.2, priorBias: bias}
	s, err := New(state, 0, net, cfg, rand.New(rand.NewSource(3)))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	chosen, _, err := s.Decide(state, nil)
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if !sameAction(chosen, legal[len(legal)-1]) {
		t.Fatalf("with 1 simulation and a peaked prior the search chose %s, want %s",
			chosen, legal[len(legal)-1])
	}
}

// TestSearchNeverMutatesTheRealState: the search steps SAMPLED worlds, and every one of them
// starts as a clone of the caller's state. A leak here would corrupt the live game.
func TestSearchNeverMutatesTheRealState(t *testing.T) {
	env := game.NewEnv(5)
	env.Reset()
	state := Clone(env.State())
	before := Clone(state)

	cfg := DefaultConfig()
	cfg.Iterations = 50
	cfg.ValueTarget = ValueTargetWin
	s, err := New(state, state.CurrentPlayer, &fakeNet{}, cfg, rand.New(rand.NewSource(11)))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := s.Decide(state, nil); err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if msg := diffStates(state, before); msg != "" {
		t.Fatalf("the search mutated the caller's state: %s", msg)
	}
}

// TestSearchPlaysWholeGames runs the search for every seat of complete games: it must always
// return a legal action, never panic, and the engine must accept every choice.
func TestSearchPlaysWholeGames(t *testing.T) {
	for _, seed := range []int64{1, 2} {
		for _, target := range []string{ValueTargetWin, ValueTargetReturn} {
			env := game.NewEnv(seed)
			env.Reset()
			plies := 0
			for !env.State().Done {
				state := env.State()
				seat := state.CurrentPlayer
				cfg := DefaultConfig()
				cfg.Iterations = 8
				cfg.ValueTarget = target
				cfg.ValueScale = 30
				s, err := New(state, seat, &fakeNet{value: 0.1}, cfg,
					rand.New(rand.NewSource(seed*100+int64(plies))))
				if err != nil {
					t.Fatalf("seed %d ply %d: New: %v", seed, plies, err)
				}
				action, stats, err := s.Decide(state, nil)
				if err != nil {
					t.Fatalf("seed %d ply %d: Decide: %v", seed, plies, err)
				}
				if stats.Evaluations == 0 {
					t.Fatalf("seed %d ply %d: no evaluations at all", seed, plies)
				}
				if _, _, _, err := env.Step(action); err != nil {
					t.Fatalf("seed %d ply %d: the engine rejected the search's action %s: %v",
						seed, plies, action, err)
				}
				plies++
				if plies > 600 {
					t.Fatalf("seed %d: game did not terminate", seed)
				}
			}
			if plies < 40 {
				t.Fatalf("seed %d (%s): only %d plies", seed, target, plies)
			}
		}
	}
}

// TestDeterminizationRespectsTheInformationSet: the sampled world may only differ from the
// real one in the cards the root seat cannot see.
func TestDeterminizationRespectsTheInformationSet(t *testing.T) {
	for _, seed := range []int64{1, 2, 3} {
		env := game.NewEnv(seed)
		env.Reset()
		for ply := 0; ply < 12 && !env.State().Done; ply++ {
			state := env.State()
			root := state.CurrentPlayer
			sampled, err := Sample(root, state, rand.New(rand.NewSource(seed+int64(ply))))
			if err != nil {
				t.Fatalf("seed %d ply %d: Sample: %v", seed, ply, err)
			}
			if err := Validate(sampled); err != nil {
				t.Fatalf("seed %d ply %d: %v", seed, ply, err)
			}
			if len(sampled.Deck) != len(state.Deck) {
				t.Fatalf("deck size changed: %d -> %d", len(state.Deck), len(sampled.Deck))
			}
			if len(sampled.Hands[root]) != len(state.Hands[root]) {
				t.Fatalf("the root seat's hand size changed")
			}
			// The root seat's OWN hand must be untouched, card for card.
			for i, card := range state.Hands[root] {
				if sampled.Hands[root][i] != card {
					t.Fatalf("seed %d ply %d: the root seat's hand was re-dealt", seed, ply)
				}
			}
			// Public information must be identical.
			if msg := diffStates(state, sampled); msg == "" {
				t.Fatalf("seed %d ply %d: the sampled world is identical to the real one "+
					"(expected the hidden hands to differ at least sometimes)", seed, ply)
			}
			if sampled.CurrentPlayer != state.CurrentPlayer || sampled.Scores != state.Scores ||
				len(sampled.Table) != len(state.Table) ||
				len(sampled.CardsCaptured[0]) != len(state.CardsCaptured[0]) {
				t.Fatalf("seed %d ply %d: sampling changed public state", seed, ply)
			}
			// Same seed -> same world.
			again, err := Sample(root, state, rand.New(rand.NewSource(seed+int64(ply))))
			if err != nil {
				t.Fatalf("Sample: %v", err)
			}
			for seat := 0; seat < game.NumPlayers; seat++ {
				if len(again.Hands[seat]) != len(sampled.Hands[seat]) {
					t.Fatalf("sampling is not reproducible for seat %d", seat)
				}
				for i := range again.Hands[seat] {
					if again.Hands[seat][i] != sampled.Hands[seat][i] {
						t.Fatalf("sampling is not reproducible for seat %d", seat)
					}
				}
			}
			env.Step(env.LegalActions()[0])
		}
	}
}

// TestRootHistoryIsUsedWhenTheModelHasAWindow: with a T=1 model the root sees only the
// pending step; with a longer window the history is packed in front of it.
func TestRootHistoryIsUsedWhenTheModelHasAWindow(t *testing.T) {
	state := capturePosition(t)
	history := make([]HistoryStep, 0, 3)
	for i := 0; i < 3; i++ {
		obs := make([]float32, game.ObsSizeISMCTS)
		feats := make([]float32, game.ActionFeatureDim)
		obs[0] = float32(i + 1)
		feats[0] = float32(i + 1)
		history = append(history, HistoryStep{Obs: obs, Feats: feats})
	}

	cfg := DefaultConfig()
	cfg.Iterations = 1
	cfg.ValueTarget = ValueTargetWin

	// T = 1: the observation sequence is just the pending step.
	net1 := &fakeNet{steps: 1}
	s1, err := New(state, 0, net1, cfg, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := s1.Decide(state, history); err != nil {
		t.Fatalf("Decide: %v", err)
	}

	// T = 5: two steps of padding + three history steps + the pending step.
	net5 := &windowNet{steps: 5}
	s5, err := New(state, 0, net5, cfg, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := s5.Decide(state, history); err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if net5.seenSteps != 5 {
		t.Fatalf("the evaluator was given %d steps, expected 5", net5.seenSteps)
	}
	if net5.maskedSteps != 3 {
		t.Fatalf("%d history steps were marked real, expected 3", net5.maskedSteps)
	}
	// Window layout with steps=5 and 3 history steps: [padding(1), hist(1), hist(2), hist(3),
	// pending]. The history markers are the values the test put in obs[0].
	markers := make([]float32, 0, 5)
	for i := 0; i < 5; i++ {
		markers = append(markers, net5.window[i*game.ObsSizeISMCTS])
	}
	want := []float32{0, 1, 2, 3, 0}
	for i := range want {
		if markers[i] != want[i] {
			t.Fatalf("window markers %v, want %v (padding at the front, pending step LAST)",
				markers, want)
		}
	}
}

// windowNet records the window it was handed.
type windowNet struct {
	steps       int
	seenSteps   int
	maskedSteps int
	window      []float32 // the ROOT call's observation window, flattened
}

func (w *windowNet) Steps() int { return w.steps }

func (w *windowNet) Forward(obsSeq, actSeq, actMask, feats, mask []float32, steps int) ([]float32, float32, error) {
	if w.seenSteps == 0 {
		// The ROOT call is the first one; later (interior) evaluations are single steps by
		// design, so only the first call tells us what the root window was.
		w.seenSteps = steps
		w.maskedSteps = 0
		for _, m := range actMask[:steps] {
			if m == 1 {
				w.maskedSteps++
			}
		}
		// obsSeq is [steps][208]; keep the whole window so the test can check its layout.
		w.window = append([]float32(nil), obsSeq[:steps*game.ObsSizeISMCTS]...)
	}
	logits := make([]float32, game.MaxActionSlots)
	for i := range logits {
		if mask[i] == 1 {
			logits[i] = 0
		} else {
			logits[i] = -1e9
		}
	}
	return logits, 0, nil
}

// TestSoftmaxPrefixIsNormalised guards the prior computation the search feeds PUCT.
func TestSoftmaxPrefixIsNormalised(t *testing.T) {
	logits := []float32{1, 2, 3, -1e9, -1e9}
	priors := softmaxPrefix(logits, 3)
	sum := 0.0
	for _, p := range priors {
		if p < 0 || math.IsNaN(p) {
			t.Fatalf("bad prior %v", p)
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("priors sum to %v, expected 1", sum)
	}
	if !(priors[2] > priors[1] && priors[1] > priors[0]) {
		t.Fatalf("softmax lost the ordering: %v", priors)
	}
}
