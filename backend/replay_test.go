package cua

import (
	"cua/internal/game"
	"cua/internal/stats"
	"testing"
	"time"
)

// A replay is only worth having if it reproduces the game it claims to. The whole design rests on
// one promise: the same seed plus the same moves gives the same game, so the statistics database
// can store a few small rows instead of a snapshot of the world per ply. These tests hold that
// promise to account against the engine itself.

// playFixtureGame plays `moves` plies of a genuine seeded game, recording every ply, and returns
// the store, the game id, the engine's final state and the recorded plies.
func playFixtureGame(t *testing.T, moves int) (*stats.Store, string, *game.Env, []stats.RecordedPly) {
	t.Helper()

	store, recorder := statsTestStore(t)
	handler, gameObj, session := statsCleanFixture(t, recorder)

	played := playRecordedGame(t, handler, gameObj, session, moves)
	if played != moves {
		t.Fatalf("expected to play %d plies, played %d", moves, played)
	}

	plies, err := store.Plies(gameObj.Id)
	if err != nil {
		t.Fatalf("could not read the recorded plies: %v", err)
	}
	if len(plies) != moves {
		t.Fatalf("expected %d recorded plies, got %d", moves, len(plies))
	}
	return store, gameObj.Id, session.env, plies
}

// TestReplayReproducesTheGame is the core guarantee: the rebuilt frames must match, ply by ply,
// what the game actually did — the table, the hands, the captured piles and the scores.
func TestReplayReproducesTheGame(t *testing.T) {
	store, gameID, env, plies := playFixtureGame(t, 3)

	row, found, err := store.GameRow(gameID)
	if err != nil || !found {
		t.Fatalf("could not read the game row (found=%v err=%v)", found, err)
	}

	replay, err := buildReplay(store, gameID, row)
	if err != nil {
		t.Fatalf("buildReplay failed: %v", err)
	}

	if len(replay.Frames) != len(plies)+1 {
		t.Fatalf("a replay of %d plies needs %d frames (the opening deal included), got %d",
			len(plies), len(plies)+1, len(replay.Frames))
	}

	// Frame 0 is the opening deal, before anybody moved.
	opening := replay.Frames[0]
	if opening.Ply != 0 || opening.Seat != nil || opening.Move != nil {
		t.Errorf("frame 0 must be the opening position with no move, got %+v", opening)
	}
	if total := len(opening.Hands[0]) + len(opening.Hands[1]) + len(opening.Hands[2]) + len(opening.Hands[3]); total != 20 {
		t.Errorf("the opening deal must show 20 cards in four hands, got %d", total)
	}
	if opening.DeckRemaining != 20 {
		t.Errorf("a 40-card deck with four 5-card hands leaves 20, got %d", opening.DeckRemaining)
	}

	// Every later frame has to agree with the engine's own state after the same moves.
	rebuilt := game.NewEnv(20260616)
	rebuilt.Reset()
	for index, ply := range plies {
		action, err := actionFromRecorded(ply)
		if err != nil {
			t.Fatalf("ply %d: %v", ply.Ply, err)
		}
		if _, _, _, err := rebuilt.Step(action); err != nil {
			t.Fatalf("ply %d could not be replayed: %v", ply.Ply, err)
		}

		frame := replay.Frames[index+1]
		if frame.Ply != ply.Ply {
			t.Errorf("frame %d reports ply %d", index+1, frame.Ply)
		}
		if frame.Seat == nil || *frame.Seat != ply.Seat {
			t.Errorf("ply %d: frame seat %v, recorded seat %d", ply.Ply, frame.Seat, ply.Seat)
		}
		if frame.Kind == nil || *frame.Kind != ply.Kind {
			t.Errorf("ply %d: frame kind %v, recorded kind %q", ply.Ply, frame.Kind, ply.Kind)
		}
		if frame.Move == nil || frame.Move.Card != ply.Card {
			t.Errorf("ply %d: frame move %+v, recorded card %d", ply.Ply, frame.Move, ply.Card)
		}

		wantState := rebuilt.State()
		if !equalInts(frame.Scores, []int{wantState.Scores[0], wantState.Scores[1]}) {
			t.Errorf("ply %d: scores %v, engine says %v", ply.Ply, frame.Scores, wantState.Scores)
		}
		if !equalInts(frame.Table, cardIndices(wantState.Table)) {
			t.Errorf("ply %d: table %v, engine says %v", ply.Ply, frame.Table, cardIndices(wantState.Table))
		}
		for seat := 0; seat < 4; seat++ {
			if !equalInts(frame.Hands[seat], cardIndices(wantState.Hands[seat])) {
				t.Errorf("ply %d: seat %d hand %v, engine says %v", ply.Ply, seat, frame.Hands[seat], cardIndices(wantState.Hands[seat]))
			}
		}
		for team := 0; team < 2; team++ {
			if !equalInts(frame.Captured[team], cardIndices(wantState.CardsCaptured[team])) {
				t.Errorf("ply %d: team %d cartón %v, engine says %v", ply.Ply, team, frame.Captured[team], cardIndices(wantState.CardsCaptured[team]))
			}
		}
		if frame.DeckRemaining != len(wantState.Deck) {
			t.Errorf("ply %d: deck %d, engine says %d", ply.Ply, frame.DeckRemaining, len(wantState.Deck))
		}
		if frame.CurrentPlayer != wantState.CurrentPlayer {
			t.Errorf("ply %d: current player %d, engine says %d", ply.Ply, frame.CurrentPlayer, wantState.CurrentPlayer)
		}
	}

	// And the final frame must be where the game actually stood.
	final := replay.Frames[len(replay.Frames)-1]
	if !equalInts(final.Table, cardIndices(env.State().Table)) {
		t.Errorf("the last frame's table does not match the game: %v vs %v", final.Table, cardIndices(env.State().Table))
	}
}

// TestReplayRefusesAGameItCannotRebuild: if the rules change under a recording, a stored move can
// become illegal. Showing a plausible-looking wrong game is worse than showing none.
func TestReplayRefusesAGameItCannotRebuild(t *testing.T) {
	store, gameID, _, _ := playFixtureGame(t, 2)

	// A game whose recorded seed does not match any deal will not accept the stored moves.
	if err := store.StartGame(stats.GameStart{
		ID:               gameID,
		CreatedAt:        time.Now(),
		Seed:             99,
		HumanSeats:       []int{0},
		SearchIterations: 128,
		Model:            "test.onnx",
	}); err != nil {
		t.Fatalf("could not rewrite the game's seed: %v", err)
	}

	row, found, err := store.GameRow(gameID)
	if err != nil || !found {
		t.Fatalf("could not read the game (found=%v err=%v)", found, err)
	}

	if _, err := buildReplay(store, gameID, row); err == nil {
		t.Fatal("expected the replay to be refused when the moves no longer fit the deal")
	}
}

// TestReplayOfAnUnknownGameIsRefused: the dashboard asks by id, and a bad id must be a clean
// refusal rather than an empty replay that looks like a game with no moves.
func TestReplayOfAnUnknownGameIsRefused(t *testing.T) {
	store, _ := statsTestStore(t)

	row, found, err := store.GameRow("NOPE")
	if err != nil {
		t.Fatalf("reading a missing game should not error: %v", err)
	}
	if found {
		t.Fatal("a game that was never recorded must not be found")
	}
	if _, err := buildReplay(store, "NOPE", row); err == nil {
		t.Fatal("expected the replay of an unknown game to be refused")
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
