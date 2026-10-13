package cua

import (
	"context"
	"cua/internal/models"
	"encoding/gob"
	"errors"
	"os"
	"path/filepath"
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
		game, err := db.CreateGame(creatorId, models.HumanPlayer{UserId: creatorId, Index: 1, Team: 1})

		if err != nil {
			t.Errorf("CreateGame failed: %v", err)
		}
		gameCopy := game

		gameCopy.DurationSeconds = 123456

		if err := db.UpdateGame(game.Id, gameCopy); err != nil {
			t.Errorf("Update Game failed: %v", err)
		}

		updatedGame, _ := db.GetGame(game.Id)

		if reflect.DeepEqual(game, updatedGame) {
			t.Error("game and updatedGame look the same")
		}
	})
}

// ---------------------------------------------------------------------------------------------
// Usernames replaced emails as the login identity. The store is gob, so the accounts created
// before that switch carry an `Email` field and no `Username`: without a migration they would
// end up with an empty username, which locks them out AND makes every one of them collide with
// the next empty username. These tests pin the upgrade path and the username rules.
// ---------------------------------------------------------------------------------------------

// legacyUserFile / legacyDatabaseFile mirror the on-disk shape from when email was the login
// field, so a database written by the old code can be reproduced here.
type legacyUserFile struct {
	Id       string
	Email    string
	Password string
}

type legacyDatabaseFile struct {
	Users map[string]legacyUserFile
}

func TestLegacyEmailAccountsKeepWorking(t *testing.T) {
	t.Run("the stored email becomes the username", func(t *testing.T) {
		// arrange: a database file in the pre-username shape
		path := filepath.Join(t.TempDir(), "db.gob")
		file, err := os.Create(path)
		if err != nil {
			t.Fatalf("could not create the legacy db file: %v", err)
		}
		legacy := legacyDatabaseFile{Users: map[string]legacyUserFile{
			"u1": {Id: "u1", Email: "someone@example.com", Password: "hash"},
		}}
		if err := gob.NewEncoder(file).Encode(legacy); err != nil {
			t.Fatalf("could not encode the legacy db: %v", err)
		}
		file.Close()

		// act
		db := NewMemDatabase(context.Background(), clock.NewMock())
		if err := db.LoadFromDisk(path); err != nil {
			t.Fatalf("LoadFromDisk failed: %v", err)
		}

		// assert: the account is reachable by the address it always signed in with
		user, found := db.FindUserByUsername("someone@example.com")
		if !found {
			t.Fatal("Expected the migrated account to be found by its old email address")
		}
		if user.Username != "someone@example.com" {
			t.Errorf("Expected the email to become the username, got %q", user.Username)
		}
		// and the legacy field is cleared, so it cannot be re-migrated or leak into a response
		if user.Email != "" {
			t.Errorf("Expected the legacy email field to be cleared, got %q", user.Email)
		}
		if user.Password != "hash" {
			t.Errorf("Expected the password hash to survive, got %q", user.Password)
		}
	})

	t.Run("an account that already has a username is left alone", func(t *testing.T) {
		// arrange: the same legacy file, but the account was already renamed
		path := filepath.Join(t.TempDir(), "db.gob")
		file, err := os.Create(path)
		if err != nil {
			t.Fatalf("could not create the db file: %v", err)
		}
		current := serializableDatabase{Users: map[string]models.User{
			"u1": {Id: "u1", Username: "already.named", Password: "hash"},
		}}
		if err := gob.NewEncoder(file).Encode(current); err != nil {
			t.Fatalf("could not encode the db: %v", err)
		}
		file.Close()

		// act
		db := NewMemDatabase(context.Background(), clock.NewMock())
		if err := db.LoadFromDisk(path); err != nil {
			t.Fatalf("LoadFromDisk failed: %v", err)
		}

		// assert
		user, found := db.FindUserByUsername("already.named")
		if !found || user.Username != "already.named" {
			t.Errorf("Expected the username to survive a load, got %+v (found=%v)", user, found)
		}
	})
}

func TestUsernameProblem(t *testing.T) {
	valid := []string{"ana", "Ana", "player_1", "a.b-c", "cuarenta40", "0123456789012345678"}
	for _, name := range valid {
		if problem := UsernameProblem(name); problem != "" {
			t.Errorf("Expected %q to be a valid username, got %q", name, problem)
		}
	}

	invalid := []string{"", "   ", "ab", "012345678901234567890", "_ana", ".ana", "-ana", "ana maria", "ana@example.com", "ana!"}
	for _, name := range invalid {
		if problem := UsernameProblem(name); problem == "" {
			t.Errorf("Expected %q to be rejected as a username", name)
		}
	}
}

func TestAddUserTreatsUsernamesCaseInsensitively(t *testing.T) {
	t.Run("a second account with different casing is a duplicate", func(t *testing.T) {
		// arrange
		db := NewMemDatabase(context.Background(), clock.NewMock())
		if err := db.AddUser(models.User{Id: "u1", Username: "Ana"}); err != nil {
			t.Fatalf("first AddUser failed: %v", err)
		}

		// act
		err := db.AddUser(models.User{Id: "u2", Username: "  ana  "})

		// assert
		if !errors.Is(err, ErrUserExists) {
			t.Errorf("Expected ErrUserExists for a case/space variant, got %v", err)
		}
		// the stored spelling is the one that was typed, which is what players are shown
		user, found := db.FindUserByUsername("ANA")
		if !found || user.Username != "Ana" {
			t.Errorf("Expected to find the account by any casing with its original spelling, got %+v", user)
		}
	})
}
