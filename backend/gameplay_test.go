package cua

import (
	"context"
	"cua/internal/game"
	infer "cua/internal/inference"
	"cua/internal/models"
	"testing"
	"time"

	"github.com/benbjohnson/clock"
)

func TestGameDurationSecondsOnGameEnd(t *testing.T) {
	// arrange
	clockMock := clock.NewMock()
	DB := NewMemDatabase(context.Background(), clockMock)

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
	inferService := infer.NewInferenceService()
	gameplayHandler := NewMainGameplay(DB, clockMock, inferService)
	session, _, err := gameplayHandler.prepareSession(gameObj)
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
	_, err = gameplayHandler.playHumanTurnAndAdvanceAI(gameObj.Id, creatorID, dto)
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

func TestCancelGame(t *testing.T) {
	t.Run("creator can cancel game and it is deleted", func(t *testing.T) {
		// arrange
		clockMock := clock.NewMock()
		DB := NewMemDatabase(context.Background(), clockMock)
		creatorID := "creator-1"
		creator := models.HumanPlayer{
			UserId: creatorID,
			Index:  1,
			Team:   1,
		}
		gameObj, _ := DB.CreateGame(creatorID, creator)
		inferService := infer.NewInferenceService()
		gameplayHandler := NewMainGameplay(DB, clockMock, inferService)

		// Create a session so we can test that it gets cleaned up too
		session := gameplayHandler.newGameSession(gameObj)
		Sessions.Mu.Lock()
		Sessions.M[normalizeGameId(gameObj.Id)] = session
		Sessions.Mu.Unlock()

		// act
		canceledGame, err := DB.CancelGame(gameObj.Id, creatorID)
		if err != nil {
			t.Fatalf("CancelGame failed: %v", err)
		}

		// Clean up session (normally done in controller, mock here to assert)
		gameplayHandler.deleteSession(gameObj.Id)

		// assert
		if canceledGame.Status != "canceled" {
			t.Errorf("Expected status to be 'canceled', got %s", canceledGame.Status)
		}

		_, exists := DB.GetGame(gameObj.Id)
		if exists {
			t.Error("Expected game to be deleted from DB, but it still exists")
		}

		_, sessionExists := getSession(gameObj.Id)
		if sessionExists {
			t.Error("Expected game session to be deleted, but it still exists")
		}
	})

	t.Run("non-creator cannot cancel game", func(t *testing.T) {
		// arrange
		clockMock := clock.NewMock()
		DB := NewMemDatabase(context.Background(), clockMock)
		creatorID := "creator-1"
		creator := models.HumanPlayer{
			UserId: creatorID,
			Index:  1,
			Team:   1,
		}
		gameObj, _ := DB.CreateGame(creatorID, creator)

		// act
		_, err := DB.CancelGame(gameObj.Id, "non-creator")

		// assert
		if err != ErrForbiddenAction {
			t.Errorf("Expected ErrForbiddenAction, got %v", err)
		}

		_, exists := DB.GetGame(gameObj.Id)
		if !exists {
			t.Error("Expected game to still exist in DB")
		}
	})
}
