package main

import (
	"crypto/rand"
	"cuarenta/internal/models"
	"errors"
	"fmt"
	"math/big"
	"strings"
	"sync"
	"time"
)

type Database struct {
	mu    sync.RWMutex
	users map[string]models.User
	games map[string]models.Game
}

var (
	ErrUserExists       = errors.New("user already exists")
	ErrGameNotFound     = errors.New("game not found")
	ErrSeatUnavailable  = errors.New("seat unavailable")
	ErrGameNotReady     = errors.New("game not ready")
	ErrGameNotRunning   = errors.New("game not running")
	ErrForbiddenAction  = errors.New("forbidden action")
	ErrNotYourTurn      = errors.New("not your turn")
	ErrNotAITurn        = errors.New("not AI turn")
	ErrIllegalAction    = errors.New("illegal action")
	ErrInvalidSeat      = errors.New("invalid seat")
	ErrInvalidTeam      = errors.New("invalid team")
	ErrSeatTeamMismatch = errors.New("seat does not belong to team")
	ErrAlreadyInGame    = errors.New("player already in game")
)

func NewDatabase() *Database {
	return &Database{
		users: make(map[string]models.User),
		games: make(map[string]models.Game),
	}
}

func normalizeEmail(email string) string {
	return strings.ToLower(strings.TrimSpace(email))
}

func normalizeGameId(gameId string) string {
	return strings.ToUpper(strings.TrimSpace(gameId))
}

func teamAllowsSeat(team int, seat int) bool {
	switch team {
	case 1:
		return seat == 1 || seat == 3
	case 2:
		return seat == 2 || seat == 4
	default:
		return false
	}
}

func validatePlayerPlacement(player models.HumanPlayer) error {
	if player.Index < 1 || player.Index > 4 {
		return ErrInvalidSeat
	}
	if player.Team != 1 && player.Team != 2 {
		return ErrInvalidTeam
	}
	if !teamAllowsSeat(player.Team, player.Index) {
		return ErrSeatTeamMismatch
	}

	return nil
}

func (db *Database) AddUser(user models.User) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	for _, existing := range db.users {
		if normalizeEmail(existing.Email) == normalizeEmail(user.Email) {
			return ErrUserExists
		}
	}

	db.users[user.Id] = user
	return nil
}

func (db *Database) FindUserByEmail(email string) (models.User, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	for _, user := range db.users {
		if normalizeEmail(user.Email) == normalizeEmail(email) {
			return user, true
		}
	}

	return models.User{}, false
}

func (db *Database) GetUser(userId string) (models.User, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	user, ok := db.users[userId]
	return user, ok
}

func (db *Database) GetGame(gameId string) (models.Game, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	game, ok := db.games[normalizeGameId(gameId)]
	return game, ok
}

func (db *Database) uniqueGameId() (string, error) {
	const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

	for attempts := 0; attempts < 64; attempts++ {
		var builder strings.Builder
		builder.Grow(4)

		for i := 0; i < 4; i++ {
			n, err := rand.Int(rand.Reader, big.NewInt(int64(len(alphabet))))
			if err != nil {
				return "", err
			}
			builder.WriteByte(alphabet[n.Int64()])
		}

		id := builder.String()
		if _, exists := db.games[id]; !exists {
			return id, nil
		}
	}

	return "", fmt.Errorf("could not generate unique game id")
}

func (db *Database) CreateGame(creatorId string, player models.HumanPlayer) (models.Game, error) {
	if err := validatePlayerPlacement(player); err != nil {
		return models.Game{}, err
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	id, err := db.uniqueGameId()
	if err != nil {
		return models.Game{}, err
	}

	player.JoinedAt = time.Now().UTC().Format(time.RFC3339)
	game := models.Game{
		Id:           id,
		HumanPlayers: []models.HumanPlayer{player},
		TimeSteps:    0,
		Cards:        [][]int{},
		CreatedBy:    creatorId,
		CreatedAt:    time.Now().UTC().Format(time.RFC3339),
		Status:       "lobby",
	}

	db.games[id] = game
	return game, nil
}

func (db *Database) JoinGame(gameId string, player models.HumanPlayer) (models.Game, error) {
	if err := validatePlayerPlacement(player); err != nil {
		return models.Game{}, err
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	gameId = normalizeGameId(gameId)
	game, ok := db.games[gameId]

	if !ok {
		return models.Game{}, ErrGameNotFound
	}

	if game.Status == "running" {
		return models.Game{}, ErrGameNotReady
	}

	for _, existing := range game.HumanPlayers {
		if existing.UserId == player.UserId {
			return models.Game{}, ErrAlreadyInGame
		}
		if existing.Index == player.Index {
			return models.Game{}, ErrSeatUnavailable
		}
	}

	player.JoinedAt = time.Now().UTC().Format(time.RFC3339)
	game.HumanPlayers = append(game.HumanPlayers, player)
	db.games[gameId] = game
	return game, nil
}

func (db *Database) StartGame(gameId string, requesterId string) (models.Game, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	gameId = normalizeGameId(gameId)
	game, ok := db.games[gameId]
	if !ok {
		return models.Game{}, ErrGameNotFound
	}
	if game.CreatedBy != requesterId {
		return models.Game{}, ErrForbiddenAction
	}

	if game.Status == "running" {
		return models.Game{}, ErrGameNotReady
	}

	if len(game.HumanPlayers) == 0 || len(game.HumanPlayers) > 4 {
		return models.Game{}, ErrGameNotReady
	}

	game.Status = "running"
	game.StartedAt = time.Now().UTC().Format(time.RFC3339)
	db.games[gameId] = game
	return game, nil
}
