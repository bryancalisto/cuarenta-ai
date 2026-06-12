package main

import (
	"testing"
	"time"

	"github.com/benbjohnson/clock"
)

func TestValidateWsTicket(t *testing.T) {
	t.Run("accepts a valid ticket", func(t *testing.T) {
		// arrange
		clockMock := clock.NewMock()
		db := NewDatabase(clockMock)
		userId := "mockId"
		ticket := db.CreateWsTicket(userId)
		db.tickets[ticket.Code] = ticket
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
		db := NewDatabase(clockMock)
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
		db := NewDatabase(clockMock)
		userId := "mockId"
		ticket := db.CreateWsTicket(userId)
		db.tickets[ticket.Code] = ticket
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
