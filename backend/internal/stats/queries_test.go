package stats

import (
	"path/filepath"
	"testing"
	"time"
)

// The dashboard reads these queries directly, and a brand-new installation is the case that breaks
// charts in ways nobody notices until a user opens the page: empty aggregates come back as NULL,
// averages divide by zero, and a "no data" chart either explains itself or looks broken.

func testStore(t *testing.T) *Store {
	t.Helper()
	store, err := Open(filepath.Join(t.TempDir(), "statistics.db"))
	if err != nil {
		t.Fatalf("could not open the statistics database: %v", err)
	}
	t.Cleanup(func() { store.Close() })
	return store
}

// startGame is the compact way to record a game these tests care about.
func startGame(t *testing.T, store *Store, id string, humanSeats []int, iterations int) {
	t.Helper()
	if err := store.StartGame(GameStart{
		ID:               id,
		CreatedAt:        time.Now(),
		Seed:             7,
		HumanSeats:       humanSeats,
		SearchIterations: iterations,
		Model:            "test.onnx",
	}); err != nil {
		t.Fatalf("StartGame(%s) failed: %v", id, err)
	}
}

func TestEmptyDatabaseAnswersWithoutNulls(t *testing.T) {
	store := testStore(t)

	overview, err := store.Overview(10)
	if err != nil {
		t.Fatalf("Overview on an empty database failed: %v", err)
	}
	if overview.Totals.Games != 0 || overview.Totals.Plies != 0 {
		t.Errorf("an empty database has zero of everything, got %+v", overview.Totals)
	}
	if overview.Totals.AvgPliesPerGame != 0 || overview.Totals.AvgAiDecisionMs != 0 {
		t.Errorf("averages over nothing must be 0, not NaN: %+v", overview.Totals)
	}
	// Both seat kinds are always present, so the chart always has its two bars.
	if len(overview.BySeatKind) != 2 {
		t.Fatalf("expected a human and an AI bar even with no games, got %d", len(overview.BySeatKind))
	}
	if overview.BySeatKind[0].Kind != "human" || overview.BySeatKind[1].Kind != "ai" {
		t.Errorf("seat kinds must come out in a stable order, got %+v", overview.BySeatKind)
	}
	if overview.AiVsHuman.Games != 0 || overview.AiVsHuman.AiTeamWinRate != 0 {
		t.Errorf("win rates over no games are 0, got %+v", overview.AiVsHuman)
	}
	if overview.ByComposition == nil || len(overview.ByComposition) != 0 {
		t.Errorf("expected an empty (not nil) composition list so JSON carries [], got %+v", overview.ByComposition)
	}

	page, err := store.GamesPage(0, 50)
	if err != nil {
		t.Fatalf("GamesPage on an empty database failed: %v", err)
	}
	if page.Total != 0 || len(page.Games) != 0 {
		t.Errorf("expected an empty page, got %+v", page)
	}
}

func TestTimeseriesFillsEveryDayInTheWindow(t *testing.T) {
	store := testStore(t)
	startGame(t, store, "AB12", []int{0, 1}, 128)

	series, err := store.Timeseries(7)
	if err != nil {
		t.Fatalf("Timeseries failed: %v", err)
	}
	if len(series.Points) != 7 {
		t.Fatalf("expected 7 days, got %d", len(series.Points))
	}

	// Chronological, one per day, no gaps: a chart with holes in it lies about activity.
	for i := 1; i < len(series.Points); i++ {
		previous, err := time.Parse("2006-01-02", series.Points[i-1].Day)
		if err != nil {
			t.Fatalf("bad day %q: %v", series.Points[i-1].Day, err)
		}
		current, err := time.Parse("2006-01-02", series.Points[i].Day)
		if err != nil {
			t.Fatalf("bad day %q: %v", series.Points[i].Day, err)
		}
		if current.Sub(previous) != 24*time.Hour {
			t.Errorf("days must be consecutive: %s then %s", series.Points[i-1].Day, series.Points[i].Day)
		}
	}

	// Today carries the game; the other days are zeroes with no win rate at all (null, not 0%).
	today := time.Now().UTC().Format("2006-01-02")
	var foundToday bool
	for _, point := range series.Points {
		if point.Day == today {
			foundToday = true
			if point.Games != 1 {
				t.Errorf("today should hold the game, got %d", point.Games)
			}
			// The game has not finished, so there is no win rate to draw: null is honest.
			if point.AiTeamWinRate != nil || point.HumanTeamWinRate != nil {
				t.Errorf("a day with no finished game has no win rate, got %v/%v", point.AiTeamWinRate, point.HumanTeamWinRate)
			}
			continue
		}
		if point.Games != 0 || point.Plies != 0 {
			t.Errorf("%s has no games, got %+v", point.Day, point)
		}
	}
	if !foundToday {
		t.Error("the window must include today")
	}
}

// TestTeamLabelMarksHumansAndAI is the column the dashboard's history table leans on: the label has
// to say which seats were people and which were the bot, in seat order.
func TestTeamLabelMarksHumansAndAI(t *testing.T) {
	cases := []struct {
		kinds     []string
		teamIndex int
		label     string
		mixed     bool
	}{
		{[]string{"human", "human", "human", "human"}, 0, "1 (H, H)", false},
		{[]string{"ai", "ai", "ai", "ai"}, 0, "1 (AI, AI)", false},
		{[]string{"human", "ai", "ai", "ai"}, 0, "1 (H, AI)", true},
		{[]string{"human", "ai", "ai", "ai"}, 1, "2 (AI, AI)", false},
		{[]string{"ai", "human", "human", "ai"}, 1, "2 (H, AI)", true},
	}
	for _, testCase := range cases {
		tag := teamTag(testCase.teamIndex, testCase.kinds)
		if tag == nil {
			t.Fatalf("no tag for team %d of %v", testCase.teamIndex, testCase.kinds)
		}
		if tag.Label != testCase.label {
			t.Errorf("team %d of %v: expected %q, got %q", testCase.teamIndex, testCase.kinds, testCase.label, tag.Label)
		}
		if tag.Mixed != testCase.mixed {
			t.Errorf("team %d of %v: mixed should be %v", testCase.teamIndex, testCase.kinds, testCase.mixed)
		}
		if tag.Team != testCase.teamIndex+1 {
			t.Errorf("teams are numbered from 1 in the dashboard, got %d", tag.Team)
		}
	}
}

// TestHumanVersusAiOnlyCountsCleanGames: the headline number is only meaningful when one team is
// all people and the other is all bots. A mixed game must not be quietly counted as a human win.
func TestHumanVersusAiOnlyCountsCleanGames(t *testing.T) {
	store := testStore(t)

	// A clean 2-human table: seats 0 and 2 (team 0) are people, seats 1 and 3 are bots.
	startGame(t, store, "CLEAN", []int{0, 2}, 400)
	finishGame(t, store, "CLEAN", 2 /* winner seat -> team 1? */, 40, 30)

	overview, err := store.Overview(10)
	if err != nil {
		t.Fatalf("Overview failed: %v", err)
	}
	if overview.AiVsHuman.Games != 1 {
		t.Fatalf("a clean humans-vs-AI game must be counted, got %d", overview.AiVsHuman.Games)
	}
	// finishGame's winnerTeam argument is a team index: 0 is the human team here.
	if overview.AiVsHuman.AiTeamWins != 1 || overview.AiVsHuman.HumanTeamWins != 0 {
		t.Errorf("expected the AI team to have won, got %+v", overview.AiVsHuman)
	}
	if overview.AiVsHuman.AiTeamWinRate != 1 || overview.AiVsHuman.HumanTeamWinRate != 0 {
		t.Errorf("expected 100%% / 0%%, got %+v", overview.AiVsHuman)
	}

	// Mixing a human into the AI team takes the game out of the headline comparison.
	startGame(t, store, "MIXED", []int{0, 3}, 400)
	finishGame(t, store, "MIXED", 0, 40, 25)

	overview, err = store.Overview(10)
	if err != nil {
		t.Fatalf("Overview failed: %v", err)
	}
	if overview.AiVsHuman.Games != 1 {
		t.Errorf("a game with a human on each side is not humans-vs-AI, got %d games", overview.AiVsHuman.Games)
	}
}

// finishGame closes a game with a winner, the way a real game ends.
func finishGame(t *testing.T, store *Store, id string, winnerTeam, score0, score1 int) {
	t.Helper()
	if err := store.FinishGame(id, GameResult{
		Status:          "finished",
		ScoreTeam0:      score0,
		ScoreTeam1:      score1,
		WinnerTeam:      winnerTeam,
		Plies:           30,
		Deals:           3,
		DurationSeconds: 600,
	}); err != nil {
		t.Fatalf("FinishGame(%s) failed: %v", id, err)
	}
}
