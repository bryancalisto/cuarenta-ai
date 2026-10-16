package cua

import (
	"cua/internal/auth"
	"cua/internal/game"
	"cua/internal/stats"
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
)

// ---------------------------------------------------------------------------------------------
// Admin dashboard: authentication
//
// The dashboard is not part of the game. Its account is not a user (it never touches the game
// database), its credential comes from the environment (ADMIN_USERNAME/ADMIN_PASSWORD), and its
// token carries `role: "admin"`, so a player token — which is otherwise perfectly valid — is
// refused by every route below. Both halves are needed: the UI hides the page and the API refuses
// the request. A hidden button is not a permission.
// ---------------------------------------------------------------------------------------------

type AdminSignInDTO struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// AdminSignInCtl exchanges the configured admin credentials for a short-lived admin token.
func (s *Server) AdminSignInCtl(c *gin.Context) {
	if _, configured := auth.AdminUsername(); !configured {
		// No admin configured means no admin: refuse loudly instead of accepting anything.
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "The admin dashboard is not configured on this server"})
		return
	}

	var dto AdminSignInDTO
	if err := bindJSONDTO(c, &dto); err != nil {
		return
	}

	if !auth.CheckAdminCredentials(dto.Username, dto.Password) {
		// One message for a wrong user and a wrong password alike: which half was wrong is not
		// something an unauthenticated caller gets to learn.
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid admin credentials"})
		return
	}

	username, _ := auth.AdminUsername()
	token, err := s.AuthService.GenerateAdminJWT(username)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not generate the admin token"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"accessToken": token,
		"admin":       gin.H{"username": username},
	})
}

// ---------------------------------------------------------------------------------------------
// Admin dashboard: statistics
// ---------------------------------------------------------------------------------------------

// statsStore returns the statistics database, or reports that there is none. A server without it
// answers 503 instead of pretending the dashboard is empty: "no statistics configured" and "no
// games played yet" look identical in a chart, and confusing them wastes an afternoon.
func (s *Server) statsStore(c *gin.Context) (*stats.Store, bool) {
	if s.Stats == nil || !s.Stats.Enabled() {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Statistics are not available on this server"})
		return nil, false
	}
	return s.Stats, true
}

func (s *Server) AdminOverviewCtl(c *gin.Context) {
	store, ok := s.statsStore(c)
	if !ok {
		return
	}

	overview, err := store.Overview(10)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not read the statistics"})
		return
	}
	c.JSON(http.StatusOK, overview)
}

func (s *Server) AdminGamesCtl(c *gin.Context) {
	store, ok := s.statsStore(c)
	if !ok {
		return
	}

	page, err := store.GamesPage(queryInt(c, "offset", 0), queryInt(c, "limit", 50))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not read the match history"})
		return
	}
	c.JSON(http.StatusOK, page)
}

func (s *Server) AdminTimeseriesCtl(c *gin.Context) {
	store, ok := s.statsStore(c)
	if !ok {
		return
	}

	series, err := store.Timeseries(queryInt(c, "days", 30))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not read the statistics"})
		return
	}
	c.JSON(http.StatusOK, series)
}

// AdminReplayCtl returns one game move by move, for the dashboard's replay viewer.
func (s *Server) AdminReplayCtl(c *gin.Context) {
	store, ok := s.statsStore(c)
	if !ok {
		return
	}

	gameID := normalizeGameId(c.Param("id"))
	row, found, err := store.GameRow(gameID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not read the game"})
		return
	}
	if !found {
		c.JSON(http.StatusNotFound, gin.H{"error": "Unknown game"})
		return
	}

	replay, err := buildReplay(store, gameID, row)
	if err != nil {
		// A game that cannot be rebuilt is still a game in the history: the row stays, and only
		// its replay is unavailable, with the reason spelled out.
		c.JSON(http.StatusConflict, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, replay)
}

func queryInt(c *gin.Context, name string, fallback int) int {
	raw := strings.TrimSpace(c.Query(name))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

// ---------------------------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------------------------

// ReplayMove is one move of a replay, as the dashboard shows it.
type ReplayMove struct {
	Card     int   `json:"card"`
	Captures []int `json:"captures"`
	Points   int   `json:"points"`
	Caida    bool  `json:"caida"`
	Limpia   bool  `json:"limpia"`
}

// ReplayFrame is the position AFTER a move (or the opening deal, for ply 0). Every seat's hand is
// included on purpose: seeing what everybody held is the point of watching a game back — and in a
// game the AI won, the interesting part is what it could NOT see.
type ReplayFrame struct {
	Ply           int         `json:"ply"`
	Seat          *int        `json:"seat"`
	Kind          *string     `json:"kind"`
	Move          *ReplayMove `json:"move"`
	Scores        []int       `json:"scores"`
	Table         []int       `json:"table"`
	Hands         [][]int     `json:"hands"`
	Captured      [][]int     `json:"captured"`
	LastPlaced    *int        `json:"lastPlaced"`
	DeckRemaining int         `json:"deckRemaining"`
	CurrentPlayer int         `json:"currentPlayer"`
}

// Replay is a whole recorded game, ready for the viewer.
type Replay struct {
	Game   stats.GameRow `json:"game"`
	Frames []ReplayFrame `json:"frames"`
}

// orEmptyInts turns a nil slice into an empty one. JSON writes nil as `null`, and the dashboard
// iterates these fields: `null.length` is a crash, while an empty array is a board with nothing on
// it -- which is a real and common position (an empty table, a pile not won yet, a move that only
// placed a card).
func orEmptyInts(values []int) []int {
	if values == nil {
		return []int{}
	}
	return values
}

var (
	errReplayMissing = errors.New("this game has no record to rebuild")
	errReplayDrift   = errors.New("this game cannot be replayed: it was recorded with different rules, so a move is no longer legal in the position it rebuilds")
)

// buildReplay rebuilds a game from its seed and its recorded moves.
//
// WHY REPLAY THE ENGINE INSTEAD OF STORING SNAPSHOTS: the deal comes from the seed and every move
// is recorded, so seed + moves reproduces the game exactly, and the database keeps a few small
// rows per game instead of a snapshot of the world per ply. The cost is that a replay is only as
// good as the rules that recorded it: if the engine changes so that a stored move is no longer
// legal, this refuses to show the game rather than showing a plausible wrong one.
func buildReplay(store *stats.Store, gameID string, row stats.GameRow) (Replay, error) {
	meta, found, err := store.GameMeta(gameID)
	if err != nil {
		return Replay{}, err
	}
	if !found {
		return Replay{}, errReplayMissing
	}

	plies, err := store.Plies(gameID)
	if err != nil {
		return Replay{}, err
	}

	env := game.NewEnv(meta.Seed)
	env.Reset()

	replay := Replay{Game: row, Frames: []ReplayFrame{frameOf(env, 0, nil, nil, nil)}}
	for _, ply := range plies {
		action, err := actionFromRecorded(ply)
		if err != nil {
			return Replay{}, err
		}
		if !game.ActionIsLegal(action, env.LegalActions()) {
			return Replay{}, errReplayDrift
		}
		if _, _, _, err := env.Step(action); err != nil {
			return Replay{}, errReplayDrift
		}

		seat, kind := ply.Seat, ply.Kind
		move := &ReplayMove{
			Card:     ply.Card,
			Captures: orEmptyInts(ply.Captures),
			Points:   ply.Points,
			Caida:    ply.Caida,
			Limpia:   ply.Limpia,
		}
		replay.Frames = append(replay.Frames, frameOf(env, ply.Ply, &seat, &kind, move))
	}

	return replay, nil
}

// actionFromRecorded turns a stored move back into an engine action.
func actionFromRecorded(ply stats.RecordedPly) (game.Action, error) {
	cards, err := cardsFromIndices(append([]int{ply.Card}, ply.Captures...))
	if err != nil {
		return game.Action{}, err
	}
	action := game.Action{Card: cards[0]}
	if len(cards) > 1 {
		action.Capture = cards[1:]
	}
	return action, nil
}

// frameOf reads the current position out of the engine.
func frameOf(env *game.Env, ply int, seat *int, kind *string, move *ReplayMove) ReplayFrame {
	state := env.State()

	hands := make([][]int, 4)
	for index := 0; index < 4; index++ {
		hands[index] = orEmptyInts(cardIndices(state.Hands[index]))
	}
	captured := make([][]int, 2)
	for team := 0; team < 2; team++ {
		captured[team] = orEmptyInts(cardIndices(state.CardsCaptured[team]))
	}

	frame := ReplayFrame{
		Ply:           ply,
		Seat:          seat,
		Kind:          kind,
		Move:          move,
		Scores:        []int{state.Scores[0], state.Scores[1]},
		Table:         orEmptyInts(cardIndices(state.Table)),
		Hands:         hands,
		Captured:      captured,
		DeckRemaining: len(state.Deck),
		CurrentPlayer: state.CurrentPlayer,
	}
	if state.LastPlaced != nil {
		index := state.LastPlaced.Index()
		frame.LastPlaced = &index
	}
	return frame
}
