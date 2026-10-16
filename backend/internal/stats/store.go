// Package stats records what happened in played games into a SQLite database
// (`statistics.db`) and answers questions about it for the admin dashboard.
//
// WHY A SECOND DATABASE: the game store is an in-memory map serialised with gob (backend/db.go),
// which is a poor fit for the dashboard's aggregate queries ("win rate of the AI seats by search
// budget over the last 30 days") and a poor place to put them: a statistics bug must not be able
// to cost a game its state. So games are *also* written here, in rows that are easy to aggregate
// and cheap to keep forever.
//
// WHAT IS RECORDED, and why each thing is there:
//
//   - one `games` row per game: the seed and the seat kinds (so the replay can be rebuilt), the
//     search budget and model the AI played with (so strength can be compared between settings),
//     and the final scores, winner and ply count;
//   - one `plies` row per move: who moved, whether that seat was human or AI, what card was
//     played, what it captured, the points and the bonuses, the score after, and how long the
//     decision took (AI thinking time is the one latency number worth watching).
//
// Everything is written synchronously on the game's own goroutine. That is deliberate: the
// statistics are worth less than the game, so a failure to record is logged and swallowed rather
// than propagated (see `Recorder`), and the writes are small enough (one row per move) that they
// never show up in the move latency.
package stats

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	// Pure-Go SQLite (no cgo): the backend is cross-compiled and deployed as a single binary, and
	// a cgo dependency would tie it to a C toolchain on the build machine.
	_ "modernc.org/sqlite"
)

// schemaVersion is stored in the meta table so a future change can migrate instead of guessing.
const schemaVersion = 1

const schema = `
CREATE TABLE IF NOT EXISTS meta (
	key   TEXT PRIMARY KEY,
	value TEXT NOT NULL
);

-- One row per game: the seed, the human seats, and the plies are everything the replay needs.
CREATE TABLE IF NOT EXISTS games (
	id                TEXT PRIMARY KEY,
	created_at        TEXT NOT NULL,
	started_at        TEXT,
	ended_at          TEXT,
	status            TEXT NOT NULL,          -- running | finished | canceled
	seed              INTEGER NOT NULL,
	human_seats       TEXT NOT NULL,          -- csv of seat indices, e.g. "0,2"
	human_count       INTEGER NOT NULL,
	ai_count          INTEGER NOT NULL,
	search_iterations INTEGER NOT NULL,       -- AI_SEARCH_ITERATIONS the AI seats played with
	deal_grace_ms     INTEGER NOT NULL,       -- AI_DEAL_GRACE_MS, so pacing changes are visible
	model             TEXT NOT NULL,          -- ONNX model file the AI seats used
	score_team_0      INTEGER NOT NULL DEFAULT 0,
	score_team_1      INTEGER NOT NULL DEFAULT 0,
	winner_team       INTEGER NOT NULL DEFAULT -1,   -- -1 when there is no winner
	plies             INTEGER NOT NULL DEFAULT 0,
	deals             INTEGER NOT NULL DEFAULT 0,
	duration_seconds  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS games_started_at ON games (started_at);

-- One row per seat per game. This is what makes every "win rate by ..." question in the dashboard
-- a plain join and GROUP BY instead of parsing the csv above in SQL: the four rows also make the
-- H/AI labelling of the match history a lookup rather than a guess.
CREATE TABLE IF NOT EXISTS game_seats (
	game_id TEXT NOT NULL REFERENCES games (id) ON DELETE CASCADE,
	seat    INTEGER NOT NULL,
	team    INTEGER NOT NULL,   -- 0 or 1: seat % 2, i.e. seats 0,2 and 1,3
	kind    TEXT NOT NULL,      -- human | ai
	PRIMARY KEY (game_id, seat)
);

-- One row per move ("ply"), in play order.
CREATE TABLE IF NOT EXISTS plies (
	game_id       TEXT NOT NULL REFERENCES games (id) ON DELETE CASCADE,
	ply           INTEGER NOT NULL,           -- 1-based, global across deals
	seat          INTEGER NOT NULL,
	team          INTEGER NOT NULL,           -- 0 or 1
	kind          TEXT NOT NULL,              -- human | ai
	card          INTEGER NOT NULL,           -- card index, suit*10 + rank position
	captures      TEXT NOT NULL,              -- csv of captured card indices, "" when it only places
	points        INTEGER NOT NULL,           -- game points the move scored
	caida         INTEGER NOT NULL,
	limpia        INTEGER NOT NULL,
	score_0       INTEGER NOT NULL,           -- scores after the move
	score_1       INTEGER NOT NULL,
	table_cards   INTEGER NOT NULL,           -- cards left on the table after it
	deal_no       INTEGER NOT NULL,           -- DealsDealt at that point
	decision_ms   INTEGER NOT NULL,           -- how long the decision took (AI thinking time)
	at            TEXT NOT NULL,
	PRIMARY KEY (game_id, ply)
);
`

// GameStart is everything known when a game starts.
type GameStart struct {
	ID               string
	CreatedAt        time.Time
	Seed             int64
	HumanSeats       []int
	SearchIterations int
	DealGraceMs      int
	Model            string
}

// Ply is one move, as played.
type Ply struct {
	GameID     string
	Ply        int
	Seat       int
	Kind       string // "human" | "ai"
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
	At         time.Time
}

// GameResult is what a finished (or cancelled) game ends with.
type GameResult struct {
	Status          string // "finished" | "canceled"
	ScoreTeam0      int
	ScoreTeam1      int
	WinnerTeam      int // -1 when there is none
	Plies           int
	Deals           int
	DurationSeconds int
}

// Store is the SQLite-backed recorder. All methods are safe for concurrent use: several games
// record at once, from different goroutines.
type Store struct {
	db *sql.DB
	mu sync.Mutex // serialises writes; the volumes here are tiny and SQLite hates write storms
}

// Open opens (creating if needed) the statistics database at path and applies the schema.
func Open(path string) (*Store, error) {
	// `_pragma=busy_timeout(5000)` keeps two writers from failing instantly under contention, and
	// `journal_mode(WAL)` lets the dashboard read while a game is being recorded.
	dsn := fmt.Sprintf("file:%s?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=foreign_keys(1)", path)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open statistics db %q: %w", path, err)
	}
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping statistics db %q: %w", path, err)
	}
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("create statistics schema: %w", err)
	}
	if _, err := db.Exec(
		`INSERT INTO meta (key, value) VALUES ('schema_version', ?)
		 ON CONFLICT (key) DO UPDATE SET value = excluded.value`,
		fmt.Sprint(schemaVersion),
	); err != nil {
		db.Close()
		return nil, fmt.Errorf("record schema version: %w", err)
	}
	return &Store{db: db}, nil
}

// Close releases the database.
func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

// DB exposes the handle for the read queries in queries.go.
func (s *Store) DB() *sql.DB { return s.db }

// Enabled reports whether recording is possible at all, so callers can skip the work when the
// dashboard is not configured.
func (s *Store) Enabled() bool { return s != nil && s.db != nil }

func csvInts(values []int) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = fmt.Sprint(value)
	}
	return strings.Join(parts, ",")
}

func boolToInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

// StartGame records the game before the first move, so a game that is cancelled or crashes still
// shows up in the history.
func (s *Store) StartGame(start GameStart) error {
	if !s.Enabled() {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	// The game and its four seats are written together: a game row without seats would silently
	// drop out of every win-rate query (they all join game_seats).
	tx, err := s.db.BeginTx(context.Background(), nil)
	if err != nil {
		return fmt.Errorf("begin game start tx: %w", err)
	}
	defer tx.Rollback()

	if _, err := tx.ExecContext(context.Background(), `
		INSERT INTO games (
			id, created_at, started_at, status, seed, human_seats, human_count, ai_count,
			search_iterations, deal_grace_ms, model
		) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT (id) DO UPDATE SET
			started_at = excluded.started_at,
			status = 'running',
			seed = excluded.seed,
			human_seats = excluded.human_seats,
			human_count = excluded.human_count,
			ai_count = excluded.ai_count,
			search_iterations = excluded.search_iterations,
			deal_grace_ms = excluded.deal_grace_ms,
			model = excluded.model`,
		start.ID,
		start.CreatedAt.UTC().Format(time.RFC3339),
		start.CreatedAt.UTC().Format(time.RFC3339),
		start.Seed,
		csvInts(start.HumanSeats),
		len(start.HumanSeats),
		// Four seats, and the ones that are not human are played by the AI.
		4-len(start.HumanSeats),
		start.SearchIterations,
		start.DealGraceMs,
		start.Model,
	); err != nil {
		return fmt.Errorf("record game start %s: %w", start.ID, err)
	}

	humans := make(map[int]bool, len(start.HumanSeats))
	for _, seat := range start.HumanSeats {
		humans[seat] = true
	}
	for seat := 0; seat < 4; seat++ {
		kind := "ai"
		if humans[seat] {
			kind = "human"
		}
		if _, err := tx.ExecContext(context.Background(),
			`INSERT INTO game_seats (game_id, seat, team, kind) VALUES (?, ?, ?, ?)
			 ON CONFLICT (game_id, seat) DO UPDATE SET team = excluded.team, kind = excluded.kind`,
			start.ID, seat, teamOfSeat(seat), kind,
		); err != nil {
			return fmt.Errorf("record seat %d of game %s: %w", seat, start.ID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit game start %s: %w", start.ID, err)
	}
	return nil
}

// RecordPly appends one move and keeps the game's running totals in step. Both happen in one
// transaction so the history never shows a ply count that disagrees with the plies table.
func (s *Store) RecordPly(ply Ply) error {
	if !s.Enabled() {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.BeginTx(context.Background(), nil)
	if err != nil {
		return fmt.Errorf("begin ply tx: %w", err)
	}
	defer tx.Rollback()

	if _, err := tx.ExecContext(context.Background(), `
		INSERT INTO plies (
			game_id, ply, seat, team, kind, card, captures, points, caida, limpia,
			score_0, score_1, table_cards, deal_no, decision_ms, at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT (game_id, ply) DO NOTHING`,
		ply.GameID, ply.Ply, ply.Seat, teamOfSeat(ply.Seat), ply.Kind, ply.Card,
		csvInts(ply.Captures), ply.Points, boolToInt(ply.Caida), boolToInt(ply.Limpia),
		ply.Score0, ply.Score1, ply.TableCards, ply.DealNo, ply.DecisionMs,
		ply.At.UTC().Format(time.RFC3339Nano),
	); err != nil {
		return fmt.Errorf("insert ply %d of game %s: %w", ply.Ply, ply.GameID, err)
	}

	if _, err := tx.ExecContext(context.Background(), `
		UPDATE games
		SET plies = MAX(plies, ?),
		    deals = MAX(deals, ?),
		    score_team_0 = ?, score_team_1 = ?
		WHERE id = ?`,
		ply.Ply, ply.DealNo, ply.Score0, ply.Score1, ply.GameID,
	); err != nil {
		return fmt.Errorf("update totals for game %s: %w", ply.GameID, err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit ply %d of game %s: %w", ply.Ply, ply.GameID, err)
	}
	return nil
}

// FinishGame closes a game out. Cancelled games come here too, so the history can tell them apart.
func (s *Store) FinishGame(gameID string, result GameResult) error {
	if !s.Enabled() {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(context.Background(), `
		UPDATE games
		SET status = ?, ended_at = ?, score_team_0 = ?, score_team_1 = ?, winner_team = ?,
		    plies = MAX(plies, ?), deals = MAX(deals, ?), duration_seconds = ?
		WHERE id = ?`,
		result.Status,
		time.Now().UTC().Format(time.RFC3339),
		result.ScoreTeam0, result.ScoreTeam1, result.WinnerTeam,
		result.Plies, result.Deals, result.DurationSeconds,
		gameID,
	)
	if err != nil {
		return fmt.Errorf("finish game %s: %w", gameID, err)
	}
	return nil
}

// teamOfSeat is the game's team layout: seats 0 and 2 are team 0, seats 1 and 3 are team 1.
func teamOfSeat(seat int) int { return seat % 2 }

// Recorder is the thin wrapper the gameplay code talks to. It exists so that the game can keep
// playing when the statistics database is missing, locked or broken: every failure is logged and
// dropped, because losing a statistic is survivable and losing a game is not.
type Recorder struct {
	store *Store
}

// NewRecorder wraps a store (which may be nil, meaning "recording is off").
func NewRecorder(store *Store) *Recorder { return &Recorder{store: store} }

func (r *Recorder) warn(action string, err error) {
	if err != nil {
		log.Printf("statistics: could not %s: %v", action, err)
	}
}

// StartGame records a game starting. `humanSeats` is the set of seats played by people.
func (r *Recorder) StartGame(gameID string, created time.Time, seed int64, humanSeats []int, searchIterations, dealGraceMs int, model string) {
	if r == nil || !r.store.Enabled() {
		return
	}
	r.warn("record the start of game "+gameID, r.store.StartGame(GameStart{
		ID:               gameID,
		CreatedAt:        created,
		Seed:             seed,
		HumanSeats:       humanSeats,
		SearchIterations: searchIterations,
		DealGraceMs:      dealGraceMs,
		Model:            model,
	}))
}

// RecordPly records one move.
func (r *Recorder) RecordPly(ply Ply) {
	if r == nil || !r.store.Enabled() {
		return
	}
	if ply.At.IsZero() {
		ply.At = time.Now()
	}
	r.warn(fmt.Sprintf("record ply %d of game %s", ply.Ply, ply.GameID), r.store.RecordPly(ply))
}

// FinishGame records the end of a game (finished or cancelled).
func (r *Recorder) FinishGame(gameID string, result GameResult) {
	if r == nil || !r.store.Enabled() {
		return
	}
	r.warn("record the end of game "+gameID, r.store.FinishGame(gameID, result))
}

// CancelIfRunning marks a game closed ONLY while it still counts as running. It is what a
// session teardown and a server shutdown call: those must not overwrite the result of a game that
// already finished, and they must not leave rows behind that look live forever.
func (s *Store) CancelIfRunning(gameID string) error {
	if !s.Enabled() {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(context.Background(), `
		UPDATE games SET status = 'canceled', ended_at = ?
		WHERE id = ? AND status = 'running'`,
		time.Now().UTC().Format(time.RFC3339), gameID)
	if err != nil {
		return fmt.Errorf("cancel stats for game %s: %w", gameID, err)
	}
	return nil
}

// RunningGameIDs lists the games the statistics database still believes are being played, so a
// shutdown can close them out (see MainGameplay.cancelRunningGames).
func (s *Store) RunningGameIDs() ([]string, error) {
	if !s.Enabled() {
		return nil, nil
	}
	rows, err := s.db.QueryContext(context.Background(), `SELECT id FROM games WHERE status = 'running'`)
	if err != nil {
		return nil, fmt.Errorf("list running games: %w", err)
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// CancelRunning closes every game the store still has as running.
func (r *Recorder) CancelRunning() {
	if r == nil || !r.store.Enabled() {
		return
	}
	ids, err := r.store.RunningGameIDs()
	if err != nil {
		r.warn("list the running games", err)
		return
	}
	for _, id := range ids {
		r.warn("cancel stats for game "+id, r.store.CancelIfRunning(id))
	}
}

// CancelIfRunning is the recorder's version: failures are logged, never returned.
func (r *Recorder) CancelIfRunning(gameID string) {
	if r == nil || !r.store.Enabled() {
		return
	}
	r.warn("cancel stats for game "+gameID, r.store.CancelIfRunning(gameID))
}
