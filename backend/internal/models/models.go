package models

import (
	"time"
)

type User struct {
	Id       string `json:"id"`
	Username string `json:"username"`
	// Email is the login field accounts used BEFORE usernames existed. It is kept only so
	// databases written back then still decode -- the store is gob, which matches fields by
	// name and silently drops fields it cannot find -- and `MemDatabase.migrateLegacyLogins`
	// promotes it to Username on load, because that address is what those people signed in
	// with. New accounts never set it, and it is cleared as it is migrated.
	Email    string `json:"email,omitempty"`
	Password string `json:"-"`
}

type HumanPlayer struct {
	UserId   string `json:"userId"`
	Index    int    `json:"index"`
	Team     int    `json:"team"`
	JoinedAt string `json:"joinedAt"`
}

type Move struct {
	CardIndex      int   `json:"cardIndex"`
	CaptureIndices []int `json:"captureIndices,omitempty"`
	PlayerIndex    int   `json:"playerIndex"`
}

type CaptureOption struct {
	CaptureIndices []int `json:"captureIndices"`
}

type WsTicket struct {
	Code         string
	ExpiresAtUTC time.Time
	UserId       string
}

// RoundEnd mirrors game.RoundEnd on the wire: what a deal-ending carton looked like, so
// the client can show the counting before the next deal is dealt.
type RoundEnd struct {
	Captured   [][]int `json:"captured"`
	Counted    []int   `json:"counted"`
	Carton     []int   `json:"carton"`
	TableCards []int   `json:"tableCards,omitempty"`
}

type Game struct {
	Id            string        `json:"id"`
	HumanPlayers  []HumanPlayer `json:"humanPlayers"`
	TimeSteps     int           `json:"timeSteps"`
	Cards         [][]int       `json:"cards"`
	Table         []int         `json:"table,omitempty"`
	Scores        []int         `json:"scores,omitempty"`
	CardsCaptured [][]int       `json:"cardsCaptured,omitempty"`
	CurrentPlayer int           `json:"currentPlayer"`
	LastPlaced    *int          `json:"lastPlaced,omitempty"`
	Winner        *int          `json:"winner,omitempty"`
	Done          bool          `json:"done,omitempty"`
	DeckRemaining int           `json:"deckRemaining,omitempty"`
	NewRoundStart bool          `json:"newRoundStart,omitempty"`
	// DealId counts the deals of this match (1 = the opening deal). The client animates a
	// deal whenever it changes.
	DealId int `json:"dealId,omitempty"`
	// RoundEnd is present on the ONE snapshot published by the move that ended a deal and
	// paid the carton: the piles that were counted, the cards left on the table, and the
	// points paid. Nil on every other snapshot, so a client can trigger its animation once.
	RoundEnd        *RoundEnd       `json:"roundEnd,omitempty"`
	CreatedBy       string          `json:"createdBy"`
	CreatedAt       string          `json:"createdAt"`
	DurationSeconds int             `json:"durationSeconds"`
	StartedAt       string          `json:"startedAt,omitempty"`
	Status          string          `json:"status"`
	Move            *Move           `json:"move,omitempty"`
	CaptureOptions  []CaptureOption `json:"captureOptions,omitempty"`
	AwaitingCapture bool            `json:"awaitingCapture,omitempty"`
}
