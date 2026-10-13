package game

import (
	"testing"
)

func TestLegalActionsForCard_MandatoryCapture(t *testing.T) {
	// Setup a table with a card that can be captured
	table := []Card{
		{Rank: Cinco, Suit: Brillos},
	}

	env := &Env{
		state: &State{
			Table: table,
			Hands: map[int][]Card{
				0: {{Rank: Cinco, Suit: CorazonesRojos}},
			},
			CurrentPlayer: 0,
		},
	}

	played := Card{Rank: Cinco, Suit: CorazonesRojos}
	actions := env.LegalActionsForCard(played)

	// Should only have one action: play 5 and capture 5
	// Before the fix, it would also have "play 5 and place"

	if len(actions) != 1 {
		t.Errorf("Expected 1 legal action when capture is available, got %d", len(actions))
		for i, a := range actions {
			t.Logf("Action %d: %s", i, a.String())
		}
	}

	if len(actions[0].Capture) == 0 {
		t.Errorf("Expected the legal action to be a capture, but it was a place")
	}
}

func TestLegalActionsForCard_NoCapture(t *testing.T) {
	// Setup a table with no matching cards
	table := []Card{
		{Rank: Siete, Suit: Brillos},
	}

	env := &Env{
		state: &State{
			Table: table,
			Hands: map[int][]Card{
				0: {{Rank: Cinco, Suit: CorazonesRojos}},
			},
			CurrentPlayer: 0,
		},
	}

	played := Card{Rank: Cinco, Suit: CorazonesRojos}
	actions := env.LegalActionsForCard(played)

	// Should only have one action: play 5 and place
	if len(actions) != 1 {
		t.Errorf("Expected 1 legal action when no capture is available, got %d", len(actions))
	}

	if len(actions[0].Capture) != 0 {
		t.Errorf("Expected the legal action to be a place, but it was a capture")
	}
}

func TestCaptureCount(t *testing.T) {
	// Setup a table with a card that can be captured
	table := []Card{
		{Rank: Cinco, Suit: Brillos},
	}

	env := &Env{
		state: &State{
			Table: table,
			Hands: map[int][]Card{
				0: {{Rank: Cinco, Suit: CorazonesRojos}},
				1: {{Rank: Siete, Suit: Treboles}},
				2: {{Rank: Siete, Suit: Brillos}},
				3: {{Rank: Siete, Suit: CorazonesNegros}},
			},
			CurrentPlayer: 0,
			Scores:        [2]int{0, 0},
			CardsCaptured: [2][]Card{},
		},
	}

	action := Action{
		Card:    Card{Rank: Cinco, Suit: CorazonesRojos},
		Capture: []Card{{Rank: Cinco, Suit: Brillos}},
	}

	_, _, _, err := env.Step(action)
	if err != nil {
		t.Fatalf("Step failed: %v", err)
	}

	capturedCount := len(env.state.CardsCaptured[0])
	if capturedCount != 2 {
		t.Errorf("Expected 2 cards captured (played + table card), got %d", capturedCount)
	}
}

// TestDealCounterAndRoundEndEvent pins the two events the web client animates.
//
// `DealsDealt` counts deals so the client can animate a deal whenever it changes (the
// opening deal is 1, a redeal or a new round bumps it), and a deal-ending carton publishes
// a `game.RoundEnd` in the step's info: the piles as they were counted, the cards still on
// the table, and the points each team was paid. The client shows the counting from that
// payload and the state clears the piles in the same step, so this is the last moment the
// information exists.
func TestDealCounterAndRoundEndEvent(t *testing.T) {
	env := NewEnv(1234)
	env.Reset()
	if got := env.State().DealsDealt; got != 1 {
		t.Fatalf("the opening deal should be deal 1, got %d", got)
	}

	seenDeal := map[int]bool{1: true}
	var withRoundEnd int
	plies := 0
	for !env.State().Done {
		before := env.State().DealsDealt
		scoresBefore := env.State().Scores
		legal := env.LegalActions()
		if len(legal) == 0 {
			t.Fatalf("ply %d: no legal actions", plies)
		}
		_, _, info, err := env.Step(legal[plies%len(legal)])
		if err != nil {
			t.Fatalf("ply %d: %v", plies, err)
		}

		after := env.State().DealsDealt
		if after < before || after > before+1 {
			t.Fatalf("ply %d: deals went %d -> %d", plies, before, after)
		}
		if after == before+1 {
			if seenDeal[after] {
				t.Fatalf("deal %d was reported twice", after)
			}
			seenDeal[after] = true

			// A deal boundary is either a plain redeal (deck left) or a carton.
			if raw, ok := info["roundEnd"]; ok {
				event, ok := raw.(RoundEnd)
				if !ok {
					t.Fatalf("ply %d: roundEnd has type %T", plies, raw)
				}
				withRoundEnd++
				// The carton is paid IN the score of this same step, which is what lets the web UI
				// hold those points back until it has shown the counting (it subtracts them again
				// while the overlay is up). If the payment ever moved to a later step, the board
				// would be subtracting points nobody had added. The step also pays the move's own
				// points, so this is a lower bound rather than an equality.
				scoresAfter := env.State().Scores
				for team := 0; team < 2; team++ {
					if paid := scoresAfter[team] - scoresBefore[team]; paid < event.Carton[team] {
						t.Fatalf("ply %d: team %d scored %d this step, less than the carton of %d: the carton "+
							"must be in the score of the step that reports it", plies, team, paid, event.Carton[team])
					}
				}
				for team := 0; team < 2; team++ {
					if event.Counted[team] != len(event.Captured[team]) {
						t.Fatalf("ply %d: counted %d but sent %d cards for team %d", plies,
							event.Counted[team], len(event.Captured[team]), team)
					}
					want := 0
					if event.Counted[team] > 19 {
						want = event.Counted[team] - 19 + 5
						if want%2 != 0 {
							want++
						}
					}
					if event.Carton[team] != want {
						t.Fatalf("ply %d: carton for team %d is %d, expected %d (%d cards)",
							plies, team, event.Carton[team], want, event.Counted[team])
					}
				}
			}
		}
		plies++
		if plies > 600 {
			t.Fatalf("game did not terminate")
		}
	}
	if withRoundEnd == 0 {
		t.Fatalf("no deal-ending carton event in a whole game")
	}
	t.Logf("deals: %d, carton events: %d, plies: %d", len(seenDeal), withRoundEnd, plies)
}
