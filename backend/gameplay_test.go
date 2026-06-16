package main

import (
	"cuarenta/internal/game"
	"cuarenta/internal/models"
	"testing"
	"time"

	"github.com/benbjohnson/clock"
)

func TestGameDurationSecondsOnGameEnd(t *testing.T) {
	// arrange
	clockMock := clock.NewMock()
	DB = NewDatabase(clockMock)

	// Set initial time
	initialTime := time.Date(2026, 6, 16, 10, 0, 0, 0, time.UTC)
	clockMock.Set(initialTime)

	// Create a game
	creatorID := "creator-1"
	creator := models.HumanPlayer{
		UserId: creatorID,
		Index:  1, // Seat 1, PlayerIndex 0
		Team:   1,
	}

	gameObj, err := DB.CreateGame(creatorID, creator)
	if err != nil {
		t.Fatalf("Failed to create game: %v", err)
	}

	// Start the game
	gameObj, err = DB.StartGame(gameObj.Id, creatorID)
	if err != nil {
		t.Fatalf("Failed to start game: %v", err)
	}

	// Prepare session using the game snapshot
	session, _, err := prepareSession(gameObj)
	if err != nil {
		t.Fatalf("Failed to prepare session: %v", err)
	}

	// Manipulate the state of the session environment to be near winning
	state := session.env.State()
	if state == nil {
		t.Fatalf("Env state is nil")
	}

	// Cinco of Brillos is index 4.
	// Cinco of CorazonesRojos is index 14.
	cincoBrillos := game.Card{Rank: game.Cinco, Suit: game.Brillos}
	cincoCorazones := game.Card{Rank: game.Cinco, Suit: game.CorazonesRojos}

	state.Table = []game.Card{cincoBrillos}
	state.Hands[0] = []game.Card{cincoCorazones}
	state.LastPlaced = &cincoBrillos
	state.Scores[0] = 38 // 38 points. Caida gives +2, which will reach 40.
	state.CurrentPlayer = 0

	// Advance mock clock by 45 seconds to simulate game duration
	clockMock.Add(45 * time.Second)

	// Play the card (Cinco of CorazonesRojos) and capture Cinco of Brillos
	dto := PlayGameDTO{
		CardIndex:      14,
		CaptureIndices: []int{4},
	}

	// Act
	_, err = playHumanTurn(gameObj.Id, creatorID, dto)
	if err != nil {
		t.Fatalf("playHumanTurn failed: %v", err)
	}

	// Assert
	// Get the updated game from the database
	updatedGame, exists := DB.GetGame(gameObj.Id)
	if !exists {
		t.Fatalf("Game not found in DB after play")
	}

	if !updatedGame.Done {
		t.Errorf("Expected game to be Done, but it was not")
	}

	if updatedGame.Status != "finished" {
		t.Errorf("Expected game status to be 'finished', got '%s'", updatedGame.Status)
	}

	expectedDuration := 45
	if updatedGame.DurationSeconds != expectedDuration {
		t.Errorf("Expected DurationSeconds to be %d, got %d", expectedDuration, updatedGame.DurationSeconds)
	}
}
