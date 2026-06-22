package cua

import (
	"context"
	"cua/internal/models"
	"reflect"
	"testing"
	"time"

	"github.com/benbjohnson/clock"
)

func TestValidateWsTicket(t *testing.T) {
	t.Run("accepts a valid ticket", func(t *testing.T) {
		// arrange
		clockMock := clock.NewMock()
		db := NewMemDatabase(context.Background(), clockMock)
		userId := "mockId"
		ticket := db.CreateWsTicket(userId)
		// db.tickets[ticket.Code] = ticket
		clockMock.Add(2 * time.Second) // still within the 3 seconds window. Ticket should be valid

		// act
		receivedTicket, valid := db.ValidateWsTicket(ticket.Code)

		// assert
		if valid != true {
			t.Error("Expected ticket to be valid")
		}

		if receivedTicket.ExpiresAtUTC != ticket.ExpiresAtUTC {
			t.Error("Expected received ticket to be what was put in the database")
		}
	})

	t.Run("rejects non existent ticket", func(t *testing.T) {
		// arrange
		clockMock := clock.NewMock()
		db := NewMemDatabase(context.Background(), clockMock)
		nonExistentTicketId := getUid()

		// act
		receivedTicket, valid := db.ValidateWsTicket(nonExistentTicketId)

		// assert
		if valid != false {
			t.Error("Expected non existent ticket to be rejected")
		}

		if receivedTicket != nil {
			t.Error("Expected to receive nil, not valid ticket instance")
		}
	})

	t.Run("rejects expired ticket", func(t *testing.T) {
		// arrange
		clockMock := clock.NewMock()
		db := NewMemDatabase(context.Background(), clockMock)
		userId := "mockId"
		ticket := db.CreateWsTicket(userId)
		clockMock.Add(3001 * time.Millisecond) // past the 3 seconds window. Ticket should be rejected

		// act
		receivedTicket, valid := db.ValidateWsTicket(ticket.Code)

		// assert
		if valid != false {
			t.Error("Expected expired ticket to be rejected")
		}

		if receivedTicket.ExpiresAtUTC != ticket.ExpiresAtUTC {
			t.Error("Expected received ticket to be what was put in the database")
		}
	})
}

func TestUpdateGame(t *testing.T) {
	t.Run("Update the game in the games map", func(t *testing.T) {
		db := NewMemDatabase(context.Background(), clock.NewMock())
		// create a mock game
		creatorId := getUid()
		game, err := db.CreateGame(creatorId, models.HumanPlayer{UserId: creatorId})

		if err != nil {
			t.Error("CreateGame failed")
		}

		game.DurationSeconds = 123456

		if err := db.UpdateGame(game.Id, game); err != nil {
			t.Errorf("Update Game failed: %v", err)
		}

		updatedGame, _ := db.FindGame(game.Id)

		if reflect.DeepEqual(game, updatedGame) {
			t.Error("game and updatedGame look the same")
		}
	})
}
