package cua

import (
	"cua/internal/game"
	"testing"
)

// TestSameRankCaptureAppliesThroughTheAPI: the same thing end to end, through the handler a client
// actually talks to, with the card indices the frontend sends.
func TestSameRankCaptureAppliesThroughTheAPI(t *testing.T) {
	fake := &fakeInference{}
	handler, gameObj, session := humanTurnFixture(t, fake)

	state := session.env.State()
	// The fixture gives the human (seat 1) the Cinco de Corazones Rojos against two fives; make it a
	// Jack against a Jack, which is the reported case.
	played := game.Card{Rank: game.Jack, Suit: game.CorazonesRojos}
	onTable := game.Card{Rank: game.Jack, Suit: game.Treboles}
	state.Hands[1] = []game.Card{played}
	state.Table = []game.Card{onTable}
	state.LastPlaced = nil

	snapshot, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, "creator-1",
		PlayGameDTO{CardIndex: played.Index()})
	if err != nil {
		t.Fatalf("playing a %s onto a %s failed: %v", played, onTable, err)
	}
	if snapshot.AwaitingCapture {
		t.Fatalf("a single same-rank capture must not ask the client to choose: %+v", snapshot.CaptureOptions)
	}

	// Both cards are gone from the table and both are in team 1's cartón (seat 1 is team 1).
	if len(state.Table) != 0 {
		t.Errorf("the table should be empty after the capture, it holds %v", cardIndices(state.Table))
	}
	captured := cardIndices(state.CardsCaptured[1])
	if len(captured) != 2 {
		t.Fatalf("the cartón should hold the played card and the captured one, it holds %v", captured)
	}
}
