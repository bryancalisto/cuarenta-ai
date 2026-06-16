package models

import (
	"time"
)

type User struct {
	Id       string `json:"id"`
	Email    string `json:"email"`
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

type Game struct {
	Id              string          `json:"id"`
	HumanPlayers    []HumanPlayer   `json:"humanPlayers"`
	TimeSteps       int             `json:"timeSteps"`
	Cards           [][]int         `json:"cards"`
	Table           []int           `json:"table,omitempty"`
	Scores          []int           `json:"scores,omitempty"`
	CurrentPlayer   int             `json:"currentPlayer"`
	LastPlaced      *int            `json:"lastPlaced,omitempty"`
	Winner          *int            `json:"winner,omitempty"`
	Done            bool            `json:"done,omitempty"`
	DeckRemaining   int             `json:"deckRemaining,omitempty"`
	CreatedBy       string          `json:"createdBy"`
	CreatedAt       string          `json:"createdAt"`
	DurationSeconds int             `json:"durationSeconds"`
	StartedAt       string          `json:"startedAt,omitempty"`
	Status          string          `json:"status"`
	PendingMoves    []Move          `json:"pendingMoves,omitempty"`
	CaptureOptions  []CaptureOption `json:"captureOptions,omitempty"`
	AwaitingCapture bool            `json:"awaitingCapture,omitempty"`
}
