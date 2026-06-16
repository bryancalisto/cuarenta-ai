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
