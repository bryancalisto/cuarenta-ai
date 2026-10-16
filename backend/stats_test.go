package cua

import (
	"cua/internal/game"
	"cua/internal/models"
	"cua/internal/stats"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The statistics database is what the admin dashboard reads, and it is written from inside the
// game loop — the one place where a mistake is invisible until someone looks at a chart weeks
// later. These tests play real games through the real handlers and then ask the store what it
// recorded, so the numbers the dashboard shows are the numbers the game produced.

// statsTestStore opens a statistics database in a temp directory and returns the store plus the
// recorder to hand to the gameplay service.
func statsTestStore(t *testing.T) (*stats.Store, *stats.Recorder) {
	t.Helper()

	store, err := stats.Open(filepath.Join(t.TempDir(), "statistics.db"))
	if err != nil {
		t.Fatalf("could not open the statistics database: %v", err)
	}
	t.Cleanup(func() { store.Close() })

	return store, stats.NewRecorder(store)
}

// statsFixture builds the AI-turn fixture with statistics recording switched on.
func statsFixture(t *testing.T, recorder *stats.Recorder) (*MainGameplay, models.Game, *gameSession) {
	t.Helper()

	fake := &fakeInference{slot: 1}
	handler, gameObj, session := aiTurnFixture(t, fake)
	handler.stats = recorder
	// No real sleeping: the AI's pacing (its thinking beat and the deal grace) is asserted in
	// gameplay_ai_test.go, and a recording test should not wait minutes for it.
	handler.pause = func(time.Duration) {}

	// The fixture crafts a position by hand, which only makes sense on top of a fixed deal: rebuild
	// the engine from the seed that is about to be RECORDED, or the recorded seed would describe a
	// different deal than the one the game is playing (and the replay would refuse the game).
	session.seed = 4242
	session.env = game.NewEnv(session.seed)
	session.env.Reset()

	// The start row is written by startRunningGame in production; the fixture builds a running game
	// directly, so call the recorder the same way that path does.
	seats := make([]int, 0, len(session.humanSeats))
	for seat := range session.humanSeats {
		seats = append(seats, seat)
	}
	recorder.StartGame(gameObj.Id, time.Now(), session.seed, seats, 128, 6500, "test-model.onnx")

	return handler, gameObj, session
}

// armAITurn puts the AI seat back in a position with something legal to play, so a test can play
// several AI moves in a row.
func armAITurnForStats(session *gameSession) {
	state := session.env.State()
	state.CurrentPlayer = 0
	state.Hands[0] = []game.Card{{Rank: game.Cinco, Suit: game.CorazonesRojos}}
	state.Table = []game.Card{
		{Rank: game.Cinco, Suit: game.Brillos},
		{Rank: game.Cinco, Suit: game.Treboles},
	}
}

func TestGameplayRecordsPlies(t *testing.T) {
	store, recorder := statsTestStore(t)
	handler, gameObj, session := statsFixture(t, recorder)

	armAITurnForStats(session)
	if err := handler.playAITurn(gameObj.Id); err != nil {
		t.Fatalf("playAITurn failed: %v", err)
	}

	plies, err := store.Plies(gameObj.Id)
	if err != nil {
		t.Fatalf("could not read the recorded plies: %v", err)
	}
	if len(plies) != 1 {
		t.Fatalf("expected 1 recorded ply, got %d", len(plies))
	}

	ply := plies[0]
	if ply.Ply != 1 || ply.Seat != 0 {
		t.Errorf("first ply should be 1 and belong to seat 0, got ply %d seat %d", ply.Ply, ply.Seat)
	}
	if ply.Kind != "ai" {
		t.Errorf("seat 0 is an AI seat, the ply was recorded as %q", ply.Kind)
	}
	if ply.Card != 14 {
		t.Errorf("the fixture plays the Cinco de Corazones Rojos (14), recorded %d", ply.Card)
	}
	// The fixture offers exactly two legal actions: taking the five of brillos (4) or the five of
	// tréboles (34). The fake inference picks slot 1, i.e. the second one.
	if len(ply.Captures) != 1 || ply.Captures[0] != 34 {
		t.Errorf("expected the move to capture the five of tréboles (34), recorded %v", ply.Captures)
	}
	if ply.Caida || ply.Limpia {
		t.Errorf("the fixture's move is neither a caída nor a limpia, recorded caida=%v limpia=%v", ply.Caida, ply.Limpia)
	}
}

func TestGameplayRecordsTheGameShape(t *testing.T) {
	store, recorder := statsTestStore(t)
	handler, gameObj, session := statsFixture(t, recorder)

	armAITurnForStats(session)
	if err := handler.playAITurn(gameObj.Id); err != nil {
		t.Fatalf("playAITurn failed: %v", err)
	}

	row, found, err := store.GameRow(gameObj.Id)
	if err != nil || !found {
		t.Fatalf("could not read the recorded game (found=%v, err=%v)", found, err)
	}

	if row.Plies != 1 {
		t.Errorf("expected the game to have 1 ply recorded, got %d", row.Plies)
	}
	if len(row.Seats) != 4 {
		t.Fatalf("expected 4 seats, got %d", len(row.Seats))
	}
	// The fixture makes seats 1, 2 and 3 human and leaves seat 0 to the AI.
	wantKinds := []string{"ai", "human", "human", "human"}
	for i, want := range wantKinds {
		if row.Seats[i].Kind != want {
			t.Errorf("seat %d should be %q, recorded %q", i, want, row.Seats[i].Kind)
		}
	}
	if row.Seats[0].Team != 0 || row.Seats[2].Team != 0 || row.Seats[1].Team != 1 {
		t.Errorf("teams are wrong: %+v", row.Seats)
	}
	if row.Status != "running" {
		t.Errorf("a game in progress should be recorded as running, got %q", row.Status)
	}
}

// TestStatisticsSurviveFinishedGamesAndAggregate: play a game to the end and check that the
// dashboard's queries see it — totals, seat win rates, and the "1 (H, AI)" label of the history.
func TestStatisticsAggregateAFinishedGame(t *testing.T) {
	store, recorder := statsTestStore(t)
	handler, gameObj, session := statsFixture(t, recorder)

	armAITurnForStats(session)
	if err := handler.playAITurn(gameObj.Id); err != nil {
		t.Fatalf("playAITurn failed: %v", err)
	}

	// Rather than playing a whole game (hundreds of moves through the fixtures), close this one
	// the way a real win does: the game ends, and the recorder is told the result.
	session.Mu.Lock()
	session.env.State().Scores = [2]int{40, 12}
	winner := 0
	session.env.State().Winner = &winner
	session.env.State().Done = true
	handler.recordGameEnd(gameObj.Id, session, 90*time.Second)
	session.Mu.Unlock()

	overview, err := store.Overview(10)
	if err != nil {
		t.Fatalf("overview failed: %v", err)
	}
	if overview.Totals.Games != 1 || overview.Totals.Finished != 1 {
		t.Errorf("expected 1 game and 1 finished, got %d/%d", overview.Totals.Games, overview.Totals.Finished)
	}
	if overview.Totals.Plies != 1 {
		t.Errorf("expected 1 ply in the totals, got %d", overview.Totals.Plies)
	}
	if overview.Totals.AvgDurationSeconds != 90 {
		t.Errorf("expected the 90s duration to be recorded, got %v", overview.Totals.AvgDurationSeconds)
	}

	// Seat win rates: team 0 (seats 0 and 2) won, so the AI seat won and the human seat in seat 2
	// won with it, while the humans in seats 1 and 3 lost.
	byKind := map[string]stats.KindWinRate{}
	for _, rate := range overview.BySeatKind {
		byKind[rate.Kind] = rate
	}
	if got := byKind["ai"]; got.Seats != 1 || got.Wins != 1 || got.WinRate != 1 {
		t.Errorf("AI seats: expected 1 seat and 1 win, got %+v", got)
	}
	if got := byKind["human"]; got.Seats != 3 || got.Wins != 1 {
		t.Errorf("human seats: expected 3 seats and 1 win, got %+v", got)
	}

	if len(overview.RecentGames) != 1 {
		t.Fatalf("expected the game in the recent list, got %d", len(overview.RecentGames))
	}
	row := overview.RecentGames[0]
	if row.Winner == nil || row.Loser == nil {
		t.Fatalf("a finished game needs a winner and a loser, got %+v / %+v", row.Winner, row.Loser)
	}
	// Team 1 is seats 0 and 2: an AI and a human.
	if row.Winner.Team != 1 || row.Winner.Label != "1 (AI, H)" {
		t.Errorf("expected the winner to be labelled `1 (AI, H)`, got %q (team %d)", row.Winner.Label, row.Winner.Team)
	}
	if !row.Winner.Mixed {
		t.Error("a team with an AI and a human is mixed")
	}
	if row.Loser.Team != 2 || row.Loser.Label != "2 (H, H)" {
		t.Errorf("expected the loser to be labelled `2 (H, H)`, got %q", row.Loser.Label)
	}
	if row.ScoreTeam1 != 40 || row.ScoreTeam2 != 12 {
		t.Errorf("scores are wrong: %d-%d", row.ScoreTeam1, row.ScoreTeam2)
	}

	// The composition table must credit both teams of the game, and this game is not a clean
	// human-vs-AI one, so the headline comparison stays empty rather than counting it.
	if len(overview.ByComposition) != 2 {
		t.Errorf("expected 2 composition rows (one per team), got %d", len(overview.ByComposition))
	}
	if overview.AiVsHuman.Games != 0 {
		t.Errorf("a game with humans on both teams is not a human-vs-AI game, got %d", overview.AiVsHuman.Games)
	}
	if len(overview.BySearchBudget) != 1 || overview.BySearchBudget[0].Iterations != 128 {
		t.Errorf("expected the AI seats to be grouped under 128 simulations, got %+v", overview.BySearchBudget)
	}
}

// TestCanceledGamesAreClosedOut: a cancelled game must not sit in the dashboard as "running"
// forever, which is what it looked like before the session teardown closed it.
func TestCanceledGamesAreClosedOut(t *testing.T) {
	store, recorder := statsTestStore(t)
	_, gameObj, _ := statsFixture(t, recorder)

	if err := store.CancelIfRunning(gameObj.Id); err != nil {
		t.Fatalf("CancelIfRunning failed: %v", err)
	}

	row, found, err := store.GameRow(gameObj.Id)
	if err != nil || !found {
		t.Fatalf("could not read the game (found=%v err=%v)", found, err)
	}
	if row.Status != "canceled" {
		t.Errorf("expected the game to be canceled, got %q", row.Status)
	}
	if row.EndedAt == nil {
		t.Error("a closed game needs an end time")
	}
	if row.Winner != nil {
		t.Error("a canceled game has no winner, and the dashboard shows a dash instead")
	}
}

// TestCancelIfRunningLeavesFinishedGamesAlone is the reason it is not simply "cancel": a session
// teardown races the end of a game, and the result must survive it.
func TestCancelIfRunningLeavesFinishedGamesAlone(t *testing.T) {
	store, recorder := statsTestStore(t)
	handler, gameObj, session := statsFixture(t, recorder)

	session.Mu.Lock()
	winner := 1
	session.env.State().Winner = &winner
	handler.recordGameEnd(gameObj.Id, session, time.Minute)
	session.Mu.Unlock()

	if err := store.CancelIfRunning(gameObj.Id); err != nil {
		t.Fatalf("CancelIfRunning failed: %v", err)
	}

	row, found, err := store.GameRow(gameObj.Id)
	if err != nil || !found {
		t.Fatalf("could not read the game (found=%v err=%v)", found, err)
	}
	if row.Status != "finished" {
		t.Errorf("a finished game must keep its result, got %q", row.Status)
	}
	if row.Winner == nil || row.Winner.Team != 2 {
		t.Errorf("the winner was lost: %+v", row.Winner)
	}
}

// TestStatisticsDisabledIsHarmless: a server with no statistics database (no file, or a broken
// one) must keep playing. This is what the nil recorder has to survive.
func TestStatisticsDisabledIsHarmless(t *testing.T) {
	fake := &fakeInference{slot: 1}
	handler, gameObj, session := aiTurnFixture(t, fake)
	handler.stats = nil // no recorder at all

	armAITurnForStats(session)
	if err := handler.playAITurn(gameObj.Id); err != nil {
		t.Fatalf("a game must play without statistics: %v", err)
	}
	handler.recordGameStart(gameObj)
}

// TestStatsDatabaseIsCreatedWhereConfigured: the environment decides the file, and the dashboard
// is only as useful as the path it is actually writing to.
func TestStatsDatabaseIsCreatedWhereConfigured(t *testing.T) {
	path := filepath.Join(t.TempDir(), "statistics.db")
	store, err := stats.Open(path)
	if err != nil {
		t.Fatalf("could not open %s: %v", path, err)
	}
	defer store.Close()

	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected the database file to exist: %v", err)
	}

	// Re-opening an existing database must not fail on the schema.
	again, err := stats.Open(path)
	if err != nil {
		t.Fatalf("re-opening an existing statistics database failed: %v", err)
	}
	defer again.Close()
}

// statsCleanFixture is the recording fixture WITHOUT the hand-crafted position: the game is the one
// its seed deals. The replay tests need that, because a replay is rebuilt from the seed plus the
// recorded moves — a position that was set up by hand is, by definition, not reproducible that way.
func statsCleanFixture(t *testing.T, recorder *stats.Recorder) (*MainGameplay, models.Game, *gameSession) {
	t.Helper()

	fake := &fakeInference{slot: 1}
	handler, gameObj, session := aiTurnFixture(t, fake)
	handler.stats = recorder
	// No real sleeping: the AI's pacing (its thinking beat and the deal grace) is asserted in
	// gameplay_ai_test.go, and a recording test should not wait minutes for it.
	handler.pause = func(time.Duration) {}

	// A fixed seed makes the dealt game deterministic, so a failure here is reproducible.
	session.seed = 20260616
	session.env = game.NewEnv(session.seed)
	session.env.Reset()

	seats := make([]int, 0, len(session.humanSeats))
	for seat := range session.humanSeats {
		seats = append(seats, seat)
	}
	recorder.StartGame(gameObj.Id, time.Now(), session.seed, seats, 128, 6500, "test-model.onnx")

	return handler, gameObj, session
}

// playRecordedGame plays a real game: every move is chosen from the engine's own legal actions and
// applied through the real handlers, so every ply is recorded exactly as production records it. It
// plays until the game ends or `maxPlies` plies have been recorded, and returns how many were —
// counted from the session, because one human turn can drag several AI turns behind it.
func playRecordedGame(t *testing.T, handler *MainGameplay, gameObj models.Game, session *gameSession, maxPlies int) int {
	t.Helper()

	for session.stepCount < maxPlies {
		state := session.env.State()
		if state == nil || state.Done {
			break
		}

		seat := state.CurrentPlayer
		legal := session.env.LegalActions()
		if len(legal) == 0 {
			t.Fatalf("ply %d: seat %d has no legal action", session.stepCount+1, seat)
		}
		action := legal[0]

		if userID, isHuman := session.humanSeats[seat]; isHuman {
			dto := PlayGameDTO{CardIndex: action.Card.Index(), CaptureIndices: cardIndices(action.Capture)}
			if _, err := handler.playHumanTurnAndAdvanceAI(gameObj.Id, userID, dto); err != nil {
				t.Fatalf("ply %d (human seat %d): %v", session.stepCount+1, seat, err)
			}
		} else if err := handler.playAITurn(gameObj.Id); err != nil {
			t.Fatalf("ply %d (AI seat %d): %v", session.stepCount+1, seat, err)
		}
	}
	return session.stepCount
}
