package stats

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------------------------
// Wire types. These are what the dashboard consumes (see web-ui/src/lib/admin-api.ts): the JSON
// field names here ARE the contract, so they are written out explicitly rather than left to
// default capitalisation.
// ---------------------------------------------------------------------------------------------

// Seat is one seat of a game: who sat there, on which team.
type Seat struct {
	Seat int    `json:"seat"`
	Team int    `json:"team"` // 0 or 1
	Kind string `json:"kind"` // human | ai
}

// TeamTag labels a team for the match history, e.g. team 1 with a human in seat 0 and an AI in
// seat 2 becomes `{"team":1,"label":"1 (H, AI)","seats":["human","ai"],"mixed":true}`.
type TeamTag struct {
	Team  int      `json:"team"` // 1 or 2, as a player would say it
	Label string   `json:"label"`
	Seats []string `json:"seats"`
	Mixed bool     `json:"mixed"`
}

// GameRow is one row of the match history.
type GameRow struct {
	ID              string     `json:"id"`
	StartedAt       *time.Time `json:"startedAt"`
	EndedAt         *time.Time `json:"endedAt"`
	Status          string     `json:"status"`
	Plies           int        `json:"plies"`
	Deals           int        `json:"deals"`
	DurationSeconds int        `json:"durationSeconds"`
	ScoreTeam1      int        `json:"scoreTeam1"` // display team numbers: 1 is team index 0
	ScoreTeam2      int        `json:"scoreTeam2"`
	Seats           []Seat     `json:"seats"`
	Winner          *TeamTag   `json:"winner"`
	Loser           *TeamTag   `json:"loser"`
}

// Totals is the headline block of the dashboard.
type Totals struct {
	Games              int     `json:"games"`
	Finished           int     `json:"finished"`
	Canceled           int     `json:"canceled"`
	Running            int     `json:"running"`
	Plies              int     `json:"plies"`
	Humans             int     `json:"humans"`
	AIs                int     `json:"ais"`
	AvgPliesPerGame    float64 `json:"avgPliesPerGame"`
	AvgDurationSeconds float64 `json:"avgDurationSeconds"`
	AvgAiDecisionMs    float64 `json:"avgAiDecisionMs"`
	Captures           int     `json:"captures"`
	Caidas             int     `json:"caidas"`
	Limpias            int     `json:"limpias"`
}

// KindWinRate is the win rate of one kind of seat (human seats or AI seats).
type KindWinRate struct {
	Kind    string  `json:"kind"`
	Seats   int     `json:"seats"`
	Wins    int     `json:"wins"`
	WinRate float64 `json:"winRate"`
}

// CompositionWinRate is the win rate of one team composition against another, e.g. "H+AI" against
// "AI+AI". This is the comparison a mixed table of humans and AI needs: "the AI wins 60%" is much
// less useful than "the AI wins 60% when it is paired with a human and 72% against one".
type CompositionWinRate struct {
	Composition string  `json:"composition"`
	Opponent    string  `json:"opponent"`
	Games       int     `json:"games"`
	Wins        int     `json:"wins"`
	WinRate     float64 `json:"winRate"`
}

// AiVsHuman is the headline comparison: games where exactly one team was made up of humans only
// and the other of AI seats only, i.e. the games that actually measure the model against people.
type AiVsHuman struct {
	Games            int     `json:"games"`
	HumanTeamWins    int     `json:"humanTeamWins"`
	AiTeamWins       int     `json:"aiTeamWins"`
	HumanTeamWinRate float64 `json:"humanTeamWinRate"`
	AiTeamWinRate    float64 `json:"aiTeamWinRate"`
}

// BudgetWinRate is the AI seats' win rate grouped by the search budget they played with.
type BudgetWinRate struct {
	Iterations int     `json:"iterations"`
	Games      int     `json:"games"`
	AiTeams    int     `json:"aiTeams"`
	AiWins     int     `json:"aiWins"`
	WinRate    float64 `json:"winRate"`
}

// Overview is the whole dashboard payload for the top of the page.
type Overview struct {
	GeneratedAt    time.Time            `json:"generatedAt"`
	Totals         Totals               `json:"totals"`
	BySeatKind     []KindWinRate        `json:"bySeatKind"`
	ByComposition  []CompositionWinRate `json:"byComposition"`
	AiVsHuman      AiVsHuman            `json:"aiVsHuman"`
	BySearchBudget []BudgetWinRate      `json:"bySearchBudget"`
	RecentGames    []GameRow            `json:"recentGames"`
}

// TimeseriesPoint is one day of activity.
type TimeseriesPoint struct {
	Day              string   `json:"day"` // YYYY-MM-DD
	Games            int      `json:"games"`
	Plies            int      `json:"plies"`
	AvgPlies         float64  `json:"avgPlies"`
	AiTeamWinRate    *float64 `json:"aiTeamWinRate"`
	HumanTeamWinRate *float64 `json:"humanTeamWinRate"`
}

// Timeseries is the chart data for games/plies/win rates over time.
type Timeseries struct {
	Days   int               `json:"days"`
	Points []TimeseriesPoint `json:"points"`
}

// GamesPage is a page of the match history.
type GamesPage struct {
	Games  []GameRow `json:"games"`
	Total  int       `json:"total"`
	Limit  int       `json:"limit"`
	Offset int       `json:"offset"`
}

// RecordedPly is a stored move, used to rebuild a replay.
type RecordedPly struct {
	Ply        int
	Seat       int
	Kind       string
	Card       int
	Captures   []int
	Points     int
	Caida      bool
	Limpia     bool
	Score0     int
	Score1     int
	TableCards int
	DealNo     int
	DecisionMs int
}

// GameMeta is the part of a game the replay needs: the seed to rebuild the deal, and who sat where.
type GameMeta struct {
	ID          string
	Seed        int64
	HumanSeats  []int
	Status      string
	SearchIters int
	Model       string
}

func ratio(wins, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(wins) / float64(total)
}

func csvToInts(value string) []int {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err == nil {
			out = append(out, n)
		}
	}
	return out
}

// ---------------------------------------------------------------------------------------------
// Row assembly
// ---------------------------------------------------------------------------------------------

// seatKinds reads the four seats of a game, in seat order. Seats missing from the table (an
// unfinished write, or a game recorded before the seat table existed) default to "ai", which is
// what a game with no humans would be anyway.
func (s *Store) seatKinds(gameID string) ([]string, error) {
	kinds := []string{"ai", "ai", "ai", "ai"}
	rows, err := s.db.QueryContext(context.Background(),
		`SELECT seat, kind FROM game_seats WHERE game_id = ?`, gameID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var seat int
		var kind string
		if err := rows.Scan(&seat, &kind); err != nil {
			return nil, err
		}
		if seat >= 0 && seat < 4 {
			kinds[seat] = kind
		}
	}
	return kinds, rows.Err()
}

// teamTag builds the "1 (H, AI)" label: the team number as a player says it, then each seat of
// that team in seat order, marked H for a human and AI for the bot.
func teamTag(teamIndex int, kinds []string) *TeamTag {
	seats := make([]string, 0, 2)
	labels := make([]string, 0, 2)
	humans, ais := 0, 0
	for seat := 0; seat < 4; seat++ {
		if teamOfSeat(seat) != teamIndex {
			continue
		}
		kind := kinds[seat]
		seats = append(seats, kind)
		if kind == "human" {
			labels = append(labels, "H")
			humans++
		} else {
			labels = append(labels, "AI")
			ais++
		}
	}
	if len(labels) == 0 {
		return nil
	}
	return &TeamTag{
		Team:  teamIndex + 1,
		Label: fmt.Sprintf("%d (%s)", teamIndex+1, strings.Join(labels, ", ")),
		Seats: seats,
		Mixed: humans > 0 && ais > 0,
	}
}

func (s *Store) gameRow(gameID string) (GameRow, bool, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT id, created_at, started_at, ended_at, status, plies, deals, duration_seconds,
		       score_team_0, score_team_1, winner_team
		FROM games WHERE id = ?`, gameID)
	if err != nil {
		return GameRow{}, false, err
	}
	defer rows.Close()
	if !rows.Next() {
		return GameRow{}, false, rows.Err()
	}
	row, winnerTeam, err := scanGameRow(rows)
	if err != nil {
		return GameRow{}, false, err
	}
	kinds, err := s.seatKinds(gameID)
	if err != nil {
		return GameRow{}, false, err
	}
	row.Seats = seatsOf(kinds)
	if row.Status == "finished" && winnerTeam >= 0 && winnerTeam <= 1 {
		row.Winner = teamTag(winnerTeam, kinds)
		row.Loser = teamTag(1-winnerTeam, kinds)
	}
	return row, true, nil
}

func seatsOf(kinds []string) []Seat {
	seats := make([]Seat, 0, 4)
	for seat, kind := range kinds {
		seats = append(seats, Seat{Seat: seat, Team: teamOfSeat(seat), Kind: kind})
	}
	return seats
}

type rowScanner interface {
	Scan(dest ...any) error
}

// scanGameRow reads one games row. The winner's team index comes back separately instead of as a
// field on GameRow: it is an internal detail, and a hidden field on a wire type is how something
// ends up serialised by accident.
func scanGameRow(scanner rowScanner) (GameRow, int, error) {
	var (
		row        GameRow
		createdAt  string
		startedAt  *string
		endedAt    *string
		winnerTeam int
	)
	if err := scanner.Scan(
		&row.ID, &createdAt, &startedAt, &endedAt, &row.Status, &row.Plies, &row.Deals,
		&row.DurationSeconds, &row.ScoreTeam1, &row.ScoreTeam2, &winnerTeam,
	); err != nil {
		return GameRow{}, 0, err
	}
	row.StartedAt = parseTime(startedAt)
	row.EndedAt = parseTime(endedAt)
	return row, winnerTeam, nil
}

func parseTime(value *string) *time.Time {
	if value == nil || *value == "" {
		return nil
	}
	parsed, err := time.Parse(time.RFC3339, *value)
	if err != nil {
		return nil
	}
	return &parsed
}

// ---------------------------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------------------------

const winnerJoin = `
	FROM games g
	JOIN game_seats gs ON gs.game_id = g.id
	WHERE g.status = 'finished' AND g.winner_team >= 0`

// Overview assembles everything the dashboard shows above the fold.
func (s *Store) Overview(recent int) (Overview, error) {
	overview := Overview{GeneratedAt: time.Now().UTC()}

	totals, err := s.totals()
	if err != nil {
		return overview, err
	}
	overview.Totals = totals

	if overview.BySeatKind, err = s.bySeatKind(); err != nil {
		return overview, err
	}
	if overview.ByComposition, overview.AiVsHuman, err = s.byComposition(); err != nil {
		return overview, err
	}
	if overview.BySearchBudget, err = s.bySearchBudget(); err != nil {
		return overview, err
	}
	page, err := s.games(0, recent)
	if err != nil {
		return overview, err
	}
	overview.RecentGames = page.Games
	return overview, nil
}

func (s *Store) totals() (Totals, error) {
	var totals Totals
	err := s.db.QueryRowContext(context.Background(), `
		SELECT
			COUNT(*),
			COALESCE(SUM(CASE WHEN status = 'finished' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status = 'canceled' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(plies), 0),
			COALESCE(SUM(human_count), 0),
			COALESCE(SUM(ai_count), 0),
			COALESCE(AVG(CASE WHEN status = 'finished' THEN plies END), 0),
			COALESCE(AVG(CASE WHEN status = 'finished' THEN duration_seconds END), 0)
		FROM games`).Scan(
		&totals.Games, &totals.Finished, &totals.Canceled, &totals.Running, &totals.Plies,
		&totals.Humans, &totals.AIs, &totals.AvgPliesPerGame, &totals.AvgDurationSeconds,
	)
	if err != nil {
		return totals, fmt.Errorf("totals: %w", err)
	}

	// Captures are stored as a csv, so counting them means counting separators: an empty csv is
	// zero cards, otherwise it is commas + 1.
	err = s.db.QueryRowContext(context.Background(), `
		SELECT
			COALESCE(SUM(CASE WHEN captures = '' THEN 0
			                  ELSE 1 + LENGTH(captures) - LENGTH(REPLACE(captures, ',', '')) END), 0),
			COALESCE(SUM(caida), 0),
			COALESCE(SUM(limpia), 0),
			COALESCE(AVG(CASE WHEN kind = 'ai' THEN decision_ms END), 0)
		FROM plies`).Scan(&totals.Captures, &totals.Caidas, &totals.Limpias, &totals.AvgAiDecisionMs)
	if err != nil {
		return totals, fmt.Errorf("ply totals: %w", err)
	}
	return totals, nil
}

func (s *Store) bySeatKind() ([]KindWinRate, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT gs.kind,
		       COUNT(*),
		       SUM(CASE WHEN g.winner_team = gs.team THEN 1 ELSE 0 END)
		`+winnerJoin+`
		GROUP BY gs.kind`)
	if err != nil {
		return nil, fmt.Errorf("by seat kind: %w", err)
	}
	defer rows.Close()

	rates := map[string]KindWinRate{}
	for rows.Next() {
		var rate KindWinRate
		if err := rows.Scan(&rate.Kind, &rate.Seats, &rate.Wins); err != nil {
			return nil, err
		}
		rate.WinRate = ratio(rate.Wins, rate.Seats)
		rates[rate.Kind] = rate
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Always both kinds, in a fixed order, so the chart has its two bars even before any game of
	// one kind has finished.
	out := make([]KindWinRate, 0, 2)
	for _, kind := range []string{"human", "ai"} {
		if rate, ok := rates[kind]; ok {
			out = append(out, rate)
			continue
		}
		out = append(out, KindWinRate{Kind: kind})
	}
	return out, nil
}

type compositionKey struct {
	composition string
	opponent    string
}

func compositionOf(kinds []string, teamIndex int) string {
	parts := make([]string, 0, 2)
	for seat := 0; seat < 4; seat++ {
		if teamOfSeat(seat) != teamIndex {
			continue
		}
		if kinds[seat] == "human" {
			parts = append(parts, "H")
		} else {
			parts = append(parts, "AI")
		}
	}
	return strings.Join(parts, "+")
}

// byComposition answers "who wins when the teams are made up like this", and boils the same data
// down to the one comparison the project actually cares about: humans against the model.
func (s *Store) byComposition() ([]CompositionWinRate, AiVsHuman, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT g.id, g.winner_team, s0.kind, s1.kind, s2.kind, s3.kind
		FROM games g
		JOIN game_seats s0 ON s0.game_id = g.id AND s0.seat = 0
		JOIN game_seats s1 ON s1.game_id = g.id AND s1.seat = 1
		JOIN game_seats s2 ON s2.game_id = g.id AND s2.seat = 2
		JOIN game_seats s3 ON s3.game_id = g.id AND s3.seat = 3
		WHERE g.status = 'finished' AND g.winner_team >= 0`)
	if err != nil {
		return nil, AiVsHuman{}, fmt.Errorf("by composition: %w", err)
	}
	defer rows.Close()

	type bucket struct{ games, wins int }
	buckets := map[compositionKey]*bucket{}
	order := []compositionKey{}
	var aiVsHuman AiVsHuman

	for rows.Next() {
		var (
			id     string
			winner int
			kind0  string
			kind1  string
			kind2  string
			kind3  string
		)
		if err := rows.Scan(&id, &winner, &kind0, &kind1, &kind2, &kind3); err != nil {
			return nil, AiVsHuman{}, err
		}
		kinds := []string{kind0, kind1, kind2, kind3}

		for teamIndex := 0; teamIndex < 2; teamIndex++ {
			key := compositionKey{
				composition: compositionOf(kinds, teamIndex),
				opponent:    compositionOf(kinds, 1-teamIndex),
			}
			if _, ok := buckets[key]; !ok {
				buckets[key] = &bucket{}
				order = append(order, key)
			}
			buckets[key].games++
			if winner == teamIndex {
				buckets[key].wins++
			}
		}

		// The clean measurement: one team all human, the other all AI.
		team0Human := kind0 == "human" && kind2 == "human"
		team0AI := kind0 == "ai" && kind2 == "ai"
		team1Human := kind1 == "human" && kind3 == "human"
		team1AI := kind1 == "ai" && kind3 == "ai"
		switch {
		case team0Human && team1AI:
			aiVsHuman.Games++
			if winner == 0 {
				aiVsHuman.HumanTeamWins++
			} else {
				aiVsHuman.AiTeamWins++
			}
		case team1Human && team0AI:
			aiVsHuman.Games++
			if winner == 1 {
				aiVsHuman.HumanTeamWins++
			} else {
				aiVsHuman.AiTeamWins++
			}
		}
	}
	if err := rows.Err(); err != nil {
		return nil, AiVsHuman{}, err
	}

	out := make([]CompositionWinRate, 0, len(order))
	for _, key := range order {
		value := buckets[key]
		out = append(out, CompositionWinRate{
			Composition: key.composition,
			Opponent:    key.opponent,
			Games:       value.games,
			Wins:        value.wins,
			WinRate:     ratio(value.wins, value.games),
		})
	}
	// Most played first: a bar chart ordered by sample size says how much to trust each bar.
	sort.SliceStable(out, func(i, j int) bool { return out[i].Games > out[j].Games })

	aiVsHuman.HumanTeamWinRate = ratio(aiVsHuman.HumanTeamWins, aiVsHuman.Games)
	aiVsHuman.AiTeamWinRate = ratio(aiVsHuman.AiTeamWins, aiVsHuman.Games)
	return out, aiVsHuman, nil
}

func (s *Store) bySearchBudget() ([]BudgetWinRate, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT g.search_iterations,
		       COUNT(DISTINCT g.id),
		       COUNT(gs.seat),
		       SUM(CASE WHEN g.winner_team = gs.team THEN 1 ELSE 0 END)
		FROM games g
		JOIN game_seats gs ON gs.game_id = g.id AND gs.kind = 'ai'
		WHERE g.status = 'finished' AND g.winner_team >= 0
		GROUP BY g.search_iterations
		ORDER BY g.search_iterations`)
	if err != nil {
		return nil, fmt.Errorf("by search budget: %w", err)
	}
	defer rows.Close()

	// Initialised, not nil: a nil slice marshals to `null`, and the dashboard charts iterate it.
	out := []BudgetWinRate{}
	for rows.Next() {
		var rate BudgetWinRate
		if err := rows.Scan(&rate.Iterations, &rate.Games, &rate.AiTeams, &rate.AiWins); err != nil {
			return nil, err
		}
		rate.WinRate = ratio(rate.AiWins, rate.AiTeams)
		out = append(out, rate)
	}
	return out, rows.Err()
}

// games lists the match history, newest first.
func (s *Store) games(offset, limit int) (GamesPage, error) {
	page := GamesPage{Limit: limit, Offset: offset, Games: []GameRow{}}

	if err := s.db.QueryRowContext(context.Background(),
		`SELECT COUNT(*) FROM games`).Scan(&page.Total); err != nil {
		return page, fmt.Errorf("count games: %w", err)
	}

	rows, err := s.db.QueryContext(context.Background(), `
		SELECT g.id, g.created_at, g.started_at, g.ended_at, g.status, g.plies, g.deals,
		       g.duration_seconds, g.score_team_0, g.score_team_1, g.winner_team,
		       s0.kind, s1.kind, s2.kind, s3.kind
		FROM games g
		LEFT JOIN game_seats s0 ON s0.game_id = g.id AND s0.seat = 0
		LEFT JOIN game_seats s1 ON s1.game_id = g.id AND s1.seat = 1
		LEFT JOIN game_seats s2 ON s2.game_id = g.id AND s2.seat = 2
		LEFT JOIN game_seats s3 ON s3.game_id = g.id AND s3.seat = 3
		ORDER BY COALESCE(g.started_at, g.created_at) DESC, g.id DESC
		LIMIT ? OFFSET ?`, limit, offset)
	if err != nil {
		return page, fmt.Errorf("list games: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var (
			row        GameRow
			createdAt  string
			startedAt  *string
			endedAt    *string
			winnerTeam int
			kind0      *string
			kind1      *string
			kind2      *string
			kind3      *string
		)
		if err := rows.Scan(
			&row.ID, &createdAt, &startedAt, &endedAt, &row.Status, &row.Plies, &row.Deals,
			&row.DurationSeconds, &row.ScoreTeam1, &row.ScoreTeam2, &winnerTeam,
			&kind0, &kind1, &kind2, &kind3,
		); err != nil {
			return page, err
		}
		row.StartedAt = parseTime(startedAt)
		row.EndedAt = parseTime(endedAt)
		kinds := []string{kindsOr(kind0), kindsOr(kind1), kindsOr(kind2), kindsOr(kind3)}
		row.Seats = seatsOf(kinds)
		if row.Status == "finished" && winnerTeam >= 0 && winnerTeam <= 1 {
			row.Winner = teamTag(winnerTeam, kinds)
			row.Loser = teamTag(1-winnerTeam, kinds)
		}
		page.Games = append(page.Games, row)
	}
	return page, rows.Err()
}

func kindsOr(value *string) string {
	if value == nil || *value == "" {
		return "ai"
	}
	return *value
}

// GamesPage returns a page of the match history.
func (s *Store) GamesPage(offset, limit int) (GamesPage, error) {
	if limit <= 0 || limit > 500 {
		limit = 50
	}
	if offset < 0 {
		offset = 0
	}
	return s.games(offset, limit)
}

// GameRow returns one game by id.
func (s *Store) GameRow(gameID string) (GameRow, bool, error) {
	return s.gameRow(gameID)
}

// Timeseries returns one point per day for the last `days` days, including the days with no games
// (a chart with gaps lies about activity), with the win rates of human and AI seats per day.
func (s *Store) Timeseries(days int) (Timeseries, error) {
	if days <= 0 || days > 365 {
		days = 30
	}
	out := Timeseries{Days: days, Points: []TimeseriesPoint{}}
	today := time.Now().UTC().Truncate(24 * time.Hour)
	from := today.AddDate(0, 0, -(days - 1))

	// One bucket per day in the window, in order, and a lookup by day name. The pointers are into
	// that slice, so it is built completely BEFORE anything points into it.
	order := make([]string, 0, days)
	perDay := make(map[string]*TimeseriesPoint, days)
	for i := 0; i < days; i++ {
		day := from.AddDate(0, 0, i).Format("2006-01-02")
		out.Points = append(out.Points, TimeseriesPoint{Day: day})
		order = append(order, day)
	}
	for i := range out.Points {
		perDay[out.Points[i].Day] = &out.Points[i]
	}

	rows, err := s.db.QueryContext(context.Background(), `
		SELECT substr(COALESCE(started_at, created_at), 1, 10) AS day,
		       COUNT(*), COALESCE(SUM(plies), 0), COALESCE(AVG(plies), 0)
		FROM games
		WHERE substr(COALESCE(started_at, created_at), 1, 10) >= ?
		GROUP BY day`, from.Format("2006-01-02"))
	if err != nil {
		return out, fmt.Errorf("timeseries games: %w", err)
	}
	for rows.Next() {
		var day string
		var point TimeseriesPoint
		if err := rows.Scan(&day, &point.Games, &point.Plies, &point.AvgPlies); err != nil {
			rows.Close()
			return out, err
		}
		if target, ok := perDay[day]; ok {
			target.Games, target.Plies, target.AvgPlies = point.Games, point.Plies, point.AvgPlies
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return out, err
	}

	// Win rate per day, per seat kind.
	rows, err = s.db.QueryContext(context.Background(), `
		SELECT substr(COALESCE(g.started_at, g.created_at), 1, 10) AS day, gs.kind,
		       COUNT(*), SUM(CASE WHEN g.winner_team = gs.team THEN 1 ELSE 0 END)
		FROM games g
		JOIN game_seats gs ON gs.game_id = g.id
		WHERE g.status = 'finished' AND g.winner_team >= 0
		  AND substr(COALESCE(g.started_at, g.created_at), 1, 10) >= ?
		GROUP BY day, gs.kind`, from.Format("2006-01-02"))
	if err != nil {
		return out, fmt.Errorf("timeseries win rates: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var (
			day   string
			kind  string
			seats int
			wins  int
		)
		if err := rows.Scan(&day, &kind, &seats, &wins); err != nil {
			return out, err
		}
		target, ok := perDay[day]
		if !ok || seats == 0 {
			continue
		}
		rate := ratio(wins, seats)
		if kind == "ai" {
			target.AiTeamWinRate = &rate
		} else {
			target.HumanTeamWinRate = &rate
		}
	}
	if err := rows.Err(); err != nil {
		return out, err
	}

	// `order` is only used to keep the promise that the points come out day by day; the slice is
	// already in that order, so this is a cheap guard against a future refactor breaking it.
	if len(order) != len(out.Points) {
		sort.Strings(order)
		out.Points = out.Points[:0]
		for _, day := range order {
			out.Points = append(out.Points, *perDay[day])
		}
	}
	return out, nil
}

// GameMeta returns the seed and seat layout of a game, for the replay builder.
func (s *Store) GameMeta(gameID string) (GameMeta, bool, error) {
	var (
		meta       GameMeta
		humanSeats string
	)
	err := s.db.QueryRowContext(context.Background(), `
		SELECT id, seed, human_seats, status, search_iterations, model
		FROM games WHERE id = ?`, gameID).Scan(
		&meta.ID, &meta.Seed, &humanSeats, &meta.Status, &meta.SearchIters, &meta.Model,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return GameMeta{}, false, nil // unknown game: the caller answers 404, not 500
	}
	if err != nil {
		return GameMeta{}, false, fmt.Errorf("read game meta %s: %w", gameID, err)
	}
	meta.HumanSeats = csvToInts(humanSeats)
	return meta, true, nil
}

// Plies returns the moves of a game in play order.
func (s *Store) Plies(gameID string) ([]RecordedPly, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT ply, seat, kind, card, captures, points, caida, limpia, score_0, score_1,
		       table_cards, deal_no, decision_ms
		FROM plies WHERE game_id = ? ORDER BY ply`, gameID)
	if err != nil {
		return nil, fmt.Errorf("read plies of %s: %w", gameID, err)
	}
	defer rows.Close()

	out := []RecordedPly{}
	for rows.Next() {
		var (
			ply      RecordedPly
			captures string
			caida    int
			limpia   int
		)
		if err := rows.Scan(
			&ply.Ply, &ply.Seat, &ply.Kind, &ply.Card, &captures, &ply.Points, &caida, &limpia,
			&ply.Score0, &ply.Score1, &ply.TableCards, &ply.DealNo, &ply.DecisionMs,
		); err != nil {
			return nil, err
		}
		ply.Captures = csvToInts(captures)
		ply.Caida = caida == 1
		ply.Limpia = limpia == 1
		out = append(out, ply)
	}
	return out, rows.Err()
}
