package cua

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"cua/internal/game"
	"cua/internal/models"
	"cua/internal/search"

	"github.com/benbjohnson/clock"
)

// fakeInference stands in for the ONNX service: it records exactly what the AI path and
// the human-turn path handed it and answers with a fixed slot, so the wiring (tensor
// shapes, the slot -> action mapping, per-turn recurrent advances, state clearing) can be
// tested without a model or a runtime.

type inferenceCall struct {
	gameID  string
	obsLen  int
	featLen int
	maskLen int
	nLegal  int
	// seat is the acting player the observation was built for: the `current_player`
	// scalar of the flattened observation.
	seat float32
	// obs is a copy of the flattened observation, so a test can check WHICH decision
	// point a turn was fed from.
	obs []float32
}

type fakeInference struct {
	slot  int
	value float32
	err   error
	// advErr only fails AdvanceRecurrentState, so a test can check that a human's move
	// still goes through when the net cannot run.
	advErr error

	obsLen  int
	featLen int
	maskLen int
	nLegal  int

	calls    []string // "predict"/"advance", in call order
	advances []inferenceCall
	cleared  []string

	// search, when set, is handed to the AI path instead of falling back to the policy;
	// searchErr forces the fallback. history records what the AI path appended.
	search       search.ModelEvaluator
	searchErr    error
	history      map[string][]search.HistoryStep
	historySeats []int
}

func (f *fakeInference) StartEngine() error    { return nil }
func (f *fakeInference) ShutdownEngine() error { return nil }

func (f *fakeInference) PredictActionSlot(gameID string, obs []float32, feats []float32, mask []float32, nLegal int) (int, float32, error) {
	f.calls = append(f.calls, "predict")
	f.obsLen, f.featLen, f.maskLen, f.nLegal = len(obs), len(feats), len(mask), nLegal
	if f.err != nil {
		return 0, 0, f.err
	}
	return f.slot, f.value, nil
}

func (f *fakeInference) AdvanceRecurrentState(gameID string, obs []float32, feats []float32, mask []float32, nLegal int) error {
	f.calls = append(f.calls, "advance")

	seat := float32(-1)
	if len(obs) > game.ObsOffsetCurrentPlayer {
		seat = obs[game.ObsOffsetCurrentPlayer]
	}
	f.advances = append(f.advances, inferenceCall{
		gameID:  gameID,
		obsLen:  len(obs),
		featLen: len(feats),
		maskLen: len(mask),
		nLegal:  nLegal,
		seat:    seat,
		obs:     append([]float32(nil), obs...),
	})

	return f.advErr
}

func (f *fakeInference) ClearGameState(gameID string) {
	f.cleared = append(f.cleared, gameID)
}

// SearchEvaluator reports that no search is available, which is the fallback path: the AI
// path must then use PredictActionSlot exactly as before. (A test that wants the search can
// set fakeSearch and get a stub evaluator instead.)
func (f *fakeInference) SearchEvaluator() (search.ModelEvaluator, error) {
	if f.searchErr != nil {
		return nil, f.searchErr
	}
	if f.search == nil {
		return nil, errors.New("no search in this fake")
	}
	return f.search, nil
}

// RecordHistory / History keep the per-seat window the real service keeps, so a test can
// assert what the search would have been given.
func (f *fakeInference) RecordHistory(gameID string, seat int, obs, actionFeatures []float32) {
	if f.history == nil {
		f.history = map[string][]search.HistoryStep{}
	}
	f.history[gameID] = append(f.history[gameID], search.HistoryStep{
		Obs:   append([]float32(nil), obs...),
		Feats: append([]float32(nil), actionFeatures...),
	})
	f.historySeats = append(f.historySeats, seat)
}

func (f *fakeInference) History(gameID string, seat int) []search.HistoryStep {
	return f.history[gameID]
}

// gameFixture starts a running game with three human players (seats 1, 2 and 3) and one
// AI seat (seat 0), then hands `moveSeat` a position with exactly two legal actions: the
// Cinco de Corazones Rojos can capture either of the two fives on the table.
//
// Seats 2 and 3 being human keeps playAITurns from walking past the AI seat, so a test
// that plays a human turn stays fast and deterministic.
func gameFixture(t *testing.T, service InferenceService, moveSeat int) (*MainGameplay, models.Game, *gameSession) {
	t.Helper()

	clockMock := clock.NewMock()
	clockMock.Set(time.Date(2026, 6, 16, 10, 0, 0, 0, time.UTC))
	DB := NewMemDatabase(context.Background(), clockMock)

	// HumanPlayer.Index is 1-based and the seat is Index-1 (see humanSeats); the API also
	// validates the pairing (Index 1|3 -> team 1, Index 2|4 -> team 2). So creator-1 sits
	// in seat 1 with team 2, the two extra humans take seats 2 and 3, and seat 0 is left
	// as the only AI seat.
	gameObj, err := DB.CreateGame("creator-1", models.HumanPlayer{UserId: "creator-1", Index: 2, Team: 2})
	if err != nil {
		t.Fatalf("CreateGame failed: %v", err)
	}
	for _, player := range []models.HumanPlayer{
		{UserId: "human-2", Index: 3, Team: 1}, // seat 2
		{UserId: "human-3", Index: 4, Team: 2}, // seat 3
	} {
		if _, err := DB.JoinGame(gameObj.Id, player); err != nil {
			t.Fatalf("JoinGame(%s) failed: %v", player.UserId, err)
		}
	}
	gameObj, err = DB.StartGame(gameObj.Id, "creator-1")
	if err != nil {
		t.Fatalf("StartGame failed: %v", err)
	}

	handler := NewMainGameplay(DB, clockMock, service)
	session, _, err := handler.prepareSession(gameObj)
	if err != nil {
		t.Fatalf("prepareSession failed: %v", err)
	}
	t.Cleanup(func() { handler.deleteSession(gameObj.Id) })

	state := session.env.State()
	state.CurrentPlayer = moveSeat
	state.Hands[moveSeat] = []game.Card{{Rank: game.Cinco, Suit: game.CorazonesRojos}} // index 14
	state.Table = []game.Card{
		{Rank: game.Cinco, Suit: game.Brillos},  // index 4
		{Rank: game.Cinco, Suit: game.Treboles}, // index 34
	}
	state.CardsCaptured = [2][]game.Card{}
	state.LastPlaced = nil
	state.Scores = [2]int{0, 0}

	return handler, gameObj, session
}

// aiTurnFixture is the fixture with the position given to the AI seat (0).
func aiTurnFixture(t *testing.T, service InferenceService) (*MainGameplay, models.Game, *gameSession) {
	t.Helper()
	handler, gameObj, session := gameFixture(t, service, 0)
	if _, isHuman := session.humanSeats[0]; isHuman {
		t.Fatal("seat 0 must be an AI seat in this fixture")
	}
	return handler, gameObj, session
}

// humanTurnFixture is the fixture with the position given to the human seat (1),
// played by "creator-1".
func humanTurnFixture(t *testing.T, service InferenceService) (*MainGameplay, models.Game, *gameSession) {
	t.Helper()
	handler, gameObj, session := gameFixture(t, service, 1)
	if _, isHuman := session.humanSeats[1]; !isHuman {
		t.Fatal("seat 1 must be a human seat in this fixture")
	}
	return handler, gameObj, session
}

func TestPlayAITurnPlaysThePredictedSlot(t *testing.T) {
	fake := &fakeInference{slot: 1}
	handler, gameObj, session := aiTurnFixture(t, fake)

	// Starting a game must drop any recurrent state left from a previous game.
	if !slices.Contains(fake.cleared, gameObj.Id) {
		t.Errorf("prepareSession did not clear the game's LSTM state (cleared=%v)", fake.cleared)
	}

	legal := session.env.LegalActions()
	if len(legal) != 2 {
		t.Fatalf("expected 2 legal actions in the fixture, got %d", len(legal))
	}

	if err := handler.playAITurn(gameObj.Id); err != nil {
		t.Fatalf("playAITurn failed: %v", err)
	}

	// The engine received the three tensors of the policy contract, sized as the model
	// declares them, for exactly the number of legal actions.
	if fake.obsLen != game.ObsSizeISMCTS {
		t.Errorf("observation was %d floats, expected %d", fake.obsLen, game.ObsSizeISMCTS)
	}
	if fake.featLen != game.MaxActionSlots*game.ActionFeatureDim {
		t.Errorf("action features were %d floats, expected %d", fake.featLen, game.MaxActionSlots*game.ActionFeatureDim)
	}
	if fake.maskLen != game.MaxActionSlots {
		t.Errorf("action mask was %d floats, expected %d", fake.maskLen, game.MaxActionSlots)
	}
	if fake.nLegal != len(legal) {
		t.Errorf("nLegal was %d, expected %d", fake.nLegal, len(legal))
	}

	// Slot 1 of the list it was given is the action that was played: the mapping the
	// model's answer is interpreted with is `legal[slot]`.
	updated, ok := handler.db.GetGame(gameObj.Id)
	if !ok {
		t.Fatal("game is missing from the database after the AI turn")
	}
	if updated.Move == nil {
		t.Fatal("the AI turn published no move")
	}
	if updated.Move.PlayerIndex != 0 {
		t.Errorf("move was attributed to seat %d, expected 0", updated.Move.PlayerIndex)
	}
	if updated.Move.CardIndex != legal[1].Card.Index() {
		t.Errorf("played card %d, expected %d (legal[1] = %s)", updated.Move.CardIndex, legal[1].Card.Index(), legal[1])
	}
	if !slices.Equal(updated.Move.CaptureIndices, cardIndices(legal[1].Capture)) {
		t.Errorf("captured %v, expected %v", updated.Move.CaptureIndices, cardIndices(legal[1].Capture))
	}

	// The rules were applied for that action (the cinco is now in team 0's pile).
	state := session.env.State()
	if len(state.Hands[0]) != 0 {
		t.Errorf("the played card is still in the AI hand: %v", state.Hands[0])
	}
	if len(state.CardsCaptured[0]) != 2 {
		t.Errorf("team 0 captured %d cards, expected 2 (the table card and the played card)", len(state.CardsCaptured[0]))
	}
	if state.CurrentPlayer != 1 {
		t.Errorf("turn did not advance: current player is %d", state.CurrentPlayer)
	}
}

func TestPlayAITurnFallsBackToTheFirstLegalAction(t *testing.T) {
	fake := &fakeInference{err: errors.New("onnx engine is not initialized")}
	handler, gameObj, session := aiTurnFixture(t, fake)

	legal := session.env.LegalActions()
	if len(legal) != 2 {
		t.Fatalf("expected 2 legal actions in the fixture, got %d", len(legal))
	}

	if err := handler.playAITurn(gameObj.Id); err != nil {
		t.Fatalf("playAITurn failed: %v", err)
	}

	updated, _ := handler.db.GetGame(gameObj.Id)
	if updated.Move == nil {
		t.Fatal("the AI turn published no move")
	}
	if updated.Move.CardIndex != legal[0].Card.Index() || !slices.Equal(updated.Move.CaptureIndices, cardIndices(legal[0].Capture)) {
		t.Errorf("fallback played card %d captures %v, expected legal[0] = %s",
			updated.Move.CardIndex, updated.Move.CaptureIndices, legal[0])
	}
}

func TestPlayAITurnClearsTheLSTMStateWhenTheGameEnds(t *testing.T) {
	fake := &fakeInference{slot: 0}
	handler, gameObj, session := aiTurnFixture(t, fake)

	// A caída at 38 points wins the game immediately.
	state := session.env.State()
	lastPlaced := state.Table[0]
	state.Table = []game.Card{lastPlaced}
	state.LastPlaced = &lastPlaced
	state.Scores = [2]int{38, 0}

	fake.cleared = nil
	if err := handler.playAITurn(gameObj.Id); err != nil {
		t.Fatalf("playAITurn failed: %v", err)
	}

	updated, _ := handler.db.GetGame(gameObj.Id)
	if !updated.Done || updated.Winner == nil || *updated.Winner != 0 {
		t.Fatalf("expected the game to be won by team 0, got done=%v winner=%v", updated.Done, updated.Winner)
	}
	if !slices.Contains(fake.cleared, gameObj.Id) {
		t.Errorf("the finished game's LSTM state was not cleared (cleared=%v)", fake.cleared)
	}
}

func TestPlayAITurnFallsBackWhenTheEngineReturnsAnImpossibleSlot(t *testing.T) {
	fake := &fakeInference{slot: 7}
	handler, gameObj, session := aiTurnFixture(t, fake)

	legal := session.env.LegalActions()
	if len(legal) != 2 {
		t.Fatalf("expected 2 legal actions in the fixture, got %d", len(legal))
	}

	if err := handler.playAITurn(gameObj.Id); err != nil {
		t.Fatalf("playAITurn failed: %v", err)
	}

	updated, _ := handler.db.GetGame(gameObj.Id)
	if updated.Move == nil {
		t.Fatal("the AI turn published no move")
	}
	if updated.Move.CardIndex != legal[0].Card.Index() || !slices.Equal(updated.Move.CaptureIndices, cardIndices(legal[0].Capture)) {
		t.Errorf("an out-of-range slot played card %d captures %v, expected legal[0] = %s",
			updated.Move.CardIndex, updated.Move.CaptureIndices, legal[0])
	}
}

// ---------------------------------------------------------------------------
// Human turns and the recurrent state
// ---------------------------------------------------------------------------

// TestHumanTurnAdvancesTheRecurrentState pins the per-turn contract: a human's turn is
// fed to the policy exactly like an AI turn — from the ACTING player's point of view, at
// the decision point, before the action is applied — because the network was trained with
// its memory advanced on every turn of the match (src/ppo/trainer.py).
func TestHumanTurnAdvancesTheRecurrentState(t *testing.T) {
	fake := &fakeInference{}
	handler, gameObj, session := humanTurnFixture(t, fake)

	// The Cinco de Corazones Rojos (14) captures the Cinco de Tréboles (34) on the table.
	dto := PlayGameDTO{CardIndex: 14, CaptureIndices: []int{34}}
	if _, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, "creator-1", dto); err != nil {
		t.Fatalf("playHumanTurnAndAdvanceAI failed: %v", err)
	}

	if len(fake.advances) != 1 {
		t.Fatalf("expected exactly 1 recurrent advance for the human turn, got %d", len(fake.advances))
	}
	advance := fake.advances[0]
	if advance.gameID != gameObj.Id {
		t.Errorf("the advance was recorded for game %q, expected %q", advance.gameID, gameObj.Id)
	}
	if advance.seat != 1 {
		t.Errorf("the turn was fed as current_player=%v, expected the acting human seat 1", advance.seat)
	}
	if advance.obsLen != game.ObsSizeISMCTS {
		t.Errorf("the advance was fed %d observation floats, expected %d", advance.obsLen, game.ObsSizeISMCTS)
	}
	if advance.featLen != game.MaxActionSlots*game.ActionFeatureDim {
		t.Errorf("the advance was fed %d feature floats, expected %d", advance.featLen, game.MaxActionSlots*game.ActionFeatureDim)
	}
	if advance.maskLen != game.MaxActionSlots {
		t.Errorf("the advance was fed a %d-float mask, expected %d", advance.maskLen, game.MaxActionSlots)
	}
	if advance.nLegal != 2 {
		t.Errorf("nLegal = %d, expected the human's 2 legal actions", advance.nLegal)
	}

	// The state was advanced from the position BEFORE the move: the observation still has
	// the played card in the human's hand.
	if advance.obs[game.ObsOffsetMyHand+14] != 1 {
		t.Error("the human turn was fed a position where the played card had already left the hand")
	}

	// The turn was fed, not decided: the network's slot was not requested for it.
	if !slices.Equal(fake.calls, []string{"advance"}) {
		t.Errorf("inference calls were %v, expected only the recurrent advance for a human turn", fake.calls)
	}

	// And the action was applied afterwards.
	updated, _ := handler.db.GetGame(gameObj.Id)
	if updated.Move == nil {
		t.Fatal("the human turn published no move")
	}
	if updated.Move.PlayerIndex != 1 || updated.Move.CardIndex != 14 || !slices.Equal(updated.Move.CaptureIndices, []int{34}) {
		t.Errorf("move was %+v, expected seat 1 to play card 14 capturing [34]", *updated.Move)
	}
	state := session.env.State()
	if len(state.Hands[1]) != 0 {
		t.Errorf("the played card is still in the human's hand: %v", state.Hands[1])
	}
	if len(state.CardsCaptured[1]) != 2 {
		t.Errorf("team 1 captured %d cards, expected 2", len(state.CardsCaptured[1]))
	}
}

// TestHumanTurnIsFedBeforeTheNextAIDecision pins the ordering: the human's turn reaches
// the policy's memory before the next AI decision reads it.
func TestHumanTurnIsFedBeforeTheNextAIDecision(t *testing.T) {
	fake := &fakeInference{slot: 0}
	handler, gameObj, _ := humanTurnFixture(t, fake)

	if _, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, "creator-1", PlayGameDTO{CardIndex: 14, CaptureIndices: []int{34}}); err != nil {
		t.Fatalf("playHumanTurnAndAdvanceAI failed: %v", err)
	}

	// playAITurns stops at the next human seat (seat 2), so drive one AI turn by hand.
	if err := handler.playAITurn(gameObj.Id); err != nil {
		t.Fatalf("playAITurn failed: %v", err)
	}

	if !slices.Equal(fake.calls, []string{"advance", "predict"}) {
		t.Errorf("inference call order was %v, expected the human turn to be fed before the AI decision", fake.calls)
	}
}

// TestRejectedHumanTurnDoesNotAdvanceTheRecurrentState covers the other half of the
// ordering rule: a turn that never happened must not enter the policy's memory.
func TestRejectedHumanTurnDoesNotAdvanceTheRecurrentState(t *testing.T) {
	t.Run("a card that is not in the player's hand", func(t *testing.T) {
		fake := &fakeInference{}
		handler, gameObj, _ := humanTurnFixture(t, fake)

		// The Siete de Brillos (6) captures nothing and is not held by the human.
		_, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, "creator-1", PlayGameDTO{CardIndex: 6})
		if err != ErrIllegalAction {
			t.Fatalf("expected ErrIllegalAction, got %v", err)
		}
		if len(fake.advances) != 0 {
			t.Errorf("a rejected turn was fed to the policy %d time(s)", len(fake.advances))
		}
	})

	t.Run("a capture-able card played without choosing a capture", func(t *testing.T) {
		fake := &fakeInference{}
		handler, gameObj, _ := humanTurnFixture(t, fake)

		snapshot, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, "creator-1", PlayGameDTO{CardIndex: 14})
		if err != nil {
			t.Fatalf("expected the capture prompt, got %v", err)
		}
		if !snapshot.AwaitingCapture {
			t.Error("expected an AwaitingCapture snapshot: the choice has not been made yet, so it is not a turn")
		}
		if len(fake.advances) != 0 {
			t.Errorf("the capture prompt fed the policy %d time(s), expected none", len(fake.advances))
		}
	})

	t.Run("someone else's turn", func(t *testing.T) {
		fake := &fakeInference{}
		handler, gameObj, _ := humanTurnFixture(t, fake)

		_, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, "human-2", PlayGameDTO{CardIndex: 14, CaptureIndices: []int{34}})
		if err != ErrForbiddenAction {
			t.Fatalf("expected ErrForbiddenAction, got %v", err)
		}
		if len(fake.advances) != 0 {
			t.Errorf("another player's rejected attempt fed the policy %d time(s)", len(fake.advances))
		}
	})
}

// TestHumanTurnSurvivesAnUnavailableNet: a human's move must never fail because the
// engine could not run.
func TestHumanTurnSurvivesAnUnavailableNet(t *testing.T) {
	unavailable := errors.New("onnx engine is not initialized")
	fake := &fakeInference{err: unavailable, advErr: unavailable}
	handler, gameObj, _ := humanTurnFixture(t, fake)

	updated, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, "creator-1", PlayGameDTO{CardIndex: 14, CaptureIndices: []int{34}})
	if err != nil {
		t.Fatalf("the human's move must not fail when inference is unavailable: %v", err)
	}
	if len(fake.advances) != 1 {
		t.Errorf("the recurrent advance was attempted %d time(s), expected 1", len(fake.advances))
	}
	if updated.Move == nil || updated.Move.CardIndex != 14 {
		t.Errorf("the human's move was not applied: %+v", updated.Move)
	}
}

// TestRecurrentStateIsDroppedWhenTheGameGoesAway: a finished or cancelled game must not
// leave its memory behind (lstmStates would otherwise grow without bound).
func TestRecurrentStateIsDroppedWhenTheGameGoesAway(t *testing.T) {
	t.Run("a game that ends on a human turn", func(t *testing.T) {
		fake := &fakeInference{}
		handler, gameObj, session := humanTurnFixture(t, fake)

		// Seat 1 plays for team 1; a caída at 38 points wins the game.
		state := session.env.State()
		lastPlaced := state.Table[0]
		state.Table = []game.Card{lastPlaced}
		state.LastPlaced = &lastPlaced
		state.Scores = [2]int{0, 38}

		fake.cleared = nil
		updated, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, "creator-1", PlayGameDTO{CardIndex: 14, CaptureIndices: []int{4}})
		if err != nil {
			t.Fatalf("playHumanTurnAndAdvanceAI failed: %v", err)
		}
		if !updated.Done {
			t.Fatal("expected the caída to end the game")
		}
		if !slices.Contains(fake.cleared, gameObj.Id) {
			t.Errorf("the finished game's recurrent state was not cleared (cleared=%v)", fake.cleared)
		}
	})

	t.Run("a deleted session", func(t *testing.T) {
		fake := &fakeInference{}
		handler, gameObj, _ := humanTurnFixture(t, fake)

		fake.cleared = nil
		handler.deleteSession(gameObj.Id)

		if !slices.Contains(fake.cleared, gameObj.Id) {
			t.Errorf("deleting a session left the recurrent state behind (cleared=%v)", fake.cleared)
		}
		if _, ok := getSession(gameObj.Id); ok {
			t.Error("the session is still registered after deleteSession")
		}
	})
}

// TestAITurnsWaitOutTheDealAnimation: the web UI animates a deal (five cards to one seat, then
// the next), so the first AI move of a deal is held back until that animation is over --
// otherwise a card flies across the table while the cards are still being dealt. The wait is
// paid ONCE per deal: `playAITurns` runs after every human move and snapshot request, and
// waiting every time would stall the game instead of pacing it.
//
// The pauses go through `handler.pause`, so this asserts the pacing without sleeping it out.
func TestAITurnsWaitOutTheDealAnimation(t *testing.T) {
	const graceMs = 6500

	// The fixture hands the AI seat a single card, so every call needs the position re-armed
	// (otherwise the second turn has no legal action at all).
	armAITurn := func(session *gameSession) {
		state := session.env.State()
		state.CurrentPlayer = 0
		state.Hands[0] = []game.Card{{Rank: game.Cinco, Suit: game.CorazonesRojos}}
		state.Table = []game.Card{
			{Rank: game.Cinco, Suit: game.Brillos},
			{Rank: game.Cinco, Suit: game.Treboles},
		}
	}

	newPacedHandler := func(t *testing.T) (*MainGameplay, models.Game, *gameSession, *[]time.Duration) {
		t.Helper()
		fake := &fakeInference{slot: 1}
		handler, gameObj, session := aiTurnFixture(t, fake)
		handler.dealGraceMs = graceMs
		pauses := &[]time.Duration{}
		handler.pause = func(d time.Duration) { *pauses = append(*pauses, d) }
		return handler, gameObj, session, pauses
	}

	// assertThinkingPause checks a pause is one of the "AI thinks" beats (1.5-3 s).
	assertThinkingPause := func(t *testing.T, pause time.Duration) {
		t.Helper()
		if pause < 1500*time.Millisecond || pause > 3000*time.Millisecond {
			t.Errorf("thinking pause was %v, expected it in [1.5s, 3s]", pause)
		}
	}

	t.Run("the first move of a deal waits for the animation", func(t *testing.T) {
		handler, gameObj, session, pauses := newPacedHandler(t)
		armAITurn(session)

		if err := handler.playAITurns(gameObj.Id); err != nil {
			t.Fatalf("playAITurns failed: %v", err)
		}

		if len(*pauses) != 2 {
			t.Fatalf("expected a deal wait plus a thinking pause, got %v", *pauses)
		}
		// Before anything else: the deal animation, then the beat.
		if (*pauses)[0] != graceMs*time.Millisecond {
			t.Errorf("first pause was %v, expected the %d ms deal grace", (*pauses)[0], graceMs)
		}
		assertThinkingPause(t, (*pauses)[1])
	})

	t.Run("later calls in the same deal do not wait again", func(t *testing.T) {
		handler, gameObj, session, pauses := newPacedHandler(t)
		armAITurn(session)

		if err := handler.playAITurns(gameObj.Id); err != nil {
			t.Fatalf("playAITurns failed: %v", err)
		}

		// Same deal, AI on turn again (the loop stops at the human seat otherwise).
		*pauses = nil
		deal := session.env.State().DealsDealt
		armAITurn(session)

		if err := handler.playAITurns(gameObj.Id); err != nil {
			t.Fatalf("playAITurns failed: %v", err)
		}

		if session.env.State().DealsDealt != deal {
			t.Fatalf("this test needs the deal to be unchanged, it went %d -> %d", deal, session.env.State().DealsDealt)
		}
		if len(*pauses) != 1 {
			t.Fatalf("expected only the thinking pause once the deal has been waited out, got %v", *pauses)
		}
		assertThinkingPause(t, (*pauses)[0])
	})

	t.Run("a new deal waits again", func(t *testing.T) {
		handler, gameObj, session, pauses := newPacedHandler(t)
		armAITurn(session)

		if err := handler.playAITurns(gameObj.Id); err != nil {
			t.Fatalf("playAITurns failed: %v", err)
		}

		// A redeal or the deal after a cartón: a fresh deal to animate.
		*pauses = nil
		armAITurn(session)
		session.env.State().DealsDealt++

		if err := handler.playAITurns(gameObj.Id); err != nil {
			t.Fatalf("playAITurns failed: %v", err)
		}

		if len(*pauses) != 2 || (*pauses)[0] != graceMs*time.Millisecond {
			t.Errorf("expected the deal grace to be paid again for the new deal, got %v", *pauses)
		}
	})

	t.Run("only the animation still running is waited for", func(t *testing.T) {
		handler, gameObj, session, pauses := newPacedHandler(t)

		// The deal was dealt 2 s ago: the cards have been flying for 2 s of the 6.5 s animation,
		// so the remaining 4.5 s is what holds the AI back -- not the whole grace.
		session.Mu.Lock()
		session.dealDealtAt = handler.clock.Now().Add(-2 * time.Second)
		session.Mu.Unlock()
		armAITurn(session)

		if err := handler.playAITurns(gameObj.Id); err != nil {
			t.Fatalf("playAITurns failed: %v", err)
		}

		if len(*pauses) != 2 {
			t.Fatalf("expected a shortened deal wait plus a thinking pause, got %v", *pauses)
		}
		if expected := 4500 * time.Millisecond; (*pauses)[0] != expected {
			t.Errorf("deal wait was %v, expected the remaining %v of the animation", (*pauses)[0], expected)
		}
	})

	t.Run("a deal dealt long ago does not wait at all", func(t *testing.T) {
		handler, gameObj, session, pauses := newPacedHandler(t)

		// The human was on turn first and played 30 s later: the animation is long over, so the
		// AI must not sit through it again.
		session.Mu.Lock()
		session.dealDealtAt = handler.clock.Now().Add(-30 * time.Second)
		session.Mu.Unlock()
		armAITurn(session)

		if err := handler.playAITurns(gameObj.Id); err != nil {
			t.Fatalf("playAITurns failed: %v", err)
		}

		if len(*pauses) != 1 {
			t.Fatalf("expected only the thinking pause for a long-finished deal, got %v", *pauses)
		}
		assertThinkingPause(t, (*pauses)[0])
	})

	t.Run("a zero grace disables the wait", func(t *testing.T) {
		handler, gameObj, session, pauses := newPacedHandler(t)
		handler.dealGraceMs = 0
		armAITurn(session)

		if err := handler.playAITurns(gameObj.Id); err != nil {
			t.Fatalf("playAITurns failed: %v", err)
		}

		if len(*pauses) != 1 {
			t.Errorf("expected only the thinking pause with the grace disabled, got %v", *pauses)
		}
		assertThinkingPause(t, (*pauses)[0])
	})
}

// TestADealStampsWhenItWasDealt closes the loop on the deal grace: the AI waits out the client's
// deal animation, and it can only wait out "what is left of it" because the deal is stamped at
// the moment it happens (inside the move that dealt it). This plays the last card of a deal with
// a full deck left, which is the redeal the UI animates.
func TestADealStampsWhenItWasDealt(t *testing.T) {
	fake := &fakeInference{}
	handler, gameObj, session := humanTurnFixture(t, fake)

	state := session.env.State()
	for p := 0; p < 4; p++ {
		if p != 1 {
			state.Hands[p] = nil // everyone else is out of cards
		}
	}
	if len(state.Deck) < 4*game.HandSize {
		t.Fatalf("the fixture needs a full deck for the redeal, it has %d cards", len(state.Deck))
	}
	dealBefore := state.DealsDealt
	if !session.dealDealtAt.IsZero() {
		t.Fatal("the fixture should start with no deal stamped")
	}

	// The human plays their last card (a capture), which empties every hand.
	if _, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, "creator-1", PlayGameDTO{CardIndex: 14, CaptureIndices: []int{4}}); err != nil {
		t.Fatalf("playHumanTurnAndAdvanceAI failed: %v", err)
	}

	if got := session.env.State().DealsDealt; got != dealBefore+1 {
		t.Fatalf("expected the last cards to trigger a redeal (%d -> %d), got %d", dealBefore, dealBefore+1, got)
	}

	session.Mu.Lock()
	stamped := session.dealDealtAt
	session.Mu.Unlock()

	if stamped.IsZero() {
		t.Fatal("a deal was handed out but nothing stamped when: the AI grace would then always wait the full animation")
	}
	if !stamped.Equal(handler.clock.Now()) {
		t.Errorf("the deal was stamped at %v, expected the current time %v", stamped, handler.clock.Now())
	}
}
