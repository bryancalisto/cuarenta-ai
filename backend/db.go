package main

import (
	"crypto/rand"
	"cuarenta/internal/config"
	"cuarenta/internal/models"
	"encoding/gob"
	"errors"
	"fmt"
	"log"
	"math/big"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/benbjohnson/clock"
)

// to be able to write it to disk
type serializableDatabase struct {
	Users     map[string]models.User
	Games     map[string]models.Game
	Tickets   map[string]models.WsTicket
	CreatedAt time.Time
}

type Database struct {
	clock     clock.Clock
	mu        sync.RWMutex
	users     map[string]models.User
	games     map[string]models.Game
	tickets   map[string]models.WsTicket
	createdAt time.Time
}

func (d *Database) WriteToDisk() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	dbPath := config.GetEnv().DbPath

	file, err := os.Create(dbPath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(serializableDatabase{Users: d.users, Games: d.games, Tickets: d.tickets})
	if err != nil {
		return err
	}
	fmt.Println("DB successfully saved to ", dbPath)
	return nil
}

func (d *Database) LoadFromDisk() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	dbPath := config.GetEnv().DbPath
	file, err := os.Open(dbPath)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	serDb := serializableDatabase{}
	err = decoder.Decode(&serDb)
	if err != nil {
		return err
	}

	d.createdAt = serDb.CreatedAt
	d.games = serDb.Games
	d.tickets = serDb.Tickets
	d.users = serDb.Users

	return nil
}

var (
	ErrUserExists              = errors.New("user already exists")
	ErrInvalidConnectionTicket = errors.New("invalid connection ticket")
	ErrGameNotFound            = errors.New("game not found")
	ErrSeatUnavailable         = errors.New("seat unavailable")
	ErrGameNotReady            = errors.New("game not ready")
	ErrGameNotRunning          = errors.New("game not running")
	ErrForbiddenAction         = errors.New("forbidden action")
	ErrNotYourTurn             = errors.New("not your turn")
	ErrNotAITurn               = errors.New("not AI turn")
	ErrIllegalAction           = errors.New("illegal action")
	ErrInvalidSeat             = errors.New("invalid seat")
	ErrInvalidTeam             = errors.New("invalid team")
	ErrSeatTeamMismatch        = errors.New("seat does not belong to team")
	ErrAlreadyInGame           = errors.New("player already in game")
	ErrLoadDbFromDisk          = errors.New("could not load DB from disk")
	ErrWriteDbToDisk           = errors.New("could not write DB into disk")
)

func NewDatabase(clock clock.Clock) *Database {
	return &Database{
		clock:     clock,
		createdAt: clock.Now().UTC(),
		users:     make(map[string]models.User),
		games:     make(map[string]models.Game),
		tickets:   make(map[string]models.WsTicket),
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

	player.JoinedAt = db.clock.Now().UTC().Format(time.RFC3339)
	game := models.Game{
		Id:           id,
		HumanPlayers: []models.HumanPlayer{player},
		TimeSteps:    0,
		Cards:        [][]int{},
		CreatedBy:    creatorId,
		CreatedAt:    db.clock.Now().UTC().Format(time.RFC3339),
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

	player.JoinedAt = db.clock.Now().UTC().Format(time.RFC3339)
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
	game.StartedAt = db.clock.Now().UTC().Format(time.RFC3339)
	db.games[gameId] = game
	return game, nil
}

func (db *Database) CancelGame(gameId string, requesterId string) (models.Game, error) {
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

	delete(db.games, gameId)

	game.Status = "canceled"
	return game, nil
}

func (db *Database) CreateWsTicket(userId string) models.WsTicket {
	const secondsToLive = 3
	code := getUid()
	expiresAt := db.clock.Now().UTC().Add(time.Second * secondsToLive)
	ticket := models.WsTicket{Code: code, ExpiresAtUTC: expiresAt, UserId: userId}

	db.mu.Lock()
	db.tickets[code] = ticket
	db.mu.Unlock()

	return ticket
}

func (db *Database) DeleteExpiredWsTickets() int {
	count := 0
	currentTime := db.clock.Now().UTC()
	for _, ticket := range db.tickets {
		if ticket.ExpiresAtUTC.Before(currentTime) {
			db.mu.Lock()
			delete(db.tickets, ticket.Code)
			db.mu.Unlock()
			count++
		}
	}

	return count
}

func (db *Database) ValidateWsTicket(ticketCode string) (*models.WsTicket, bool) {
	db.mu.RLock()
	ticket, ok := db.tickets[ticketCode]
	db.mu.RUnlock()

	if !ok {
		log.Println("ticket not found")
		return nil, false
	}

	currentTime := db.clock.Now().UTC()
	valid := currentTime.Before(ticket.ExpiresAtUTC)
	return &ticket, valid
}

// Cancels in-progress games. Used on shutdown because currently the session and games are corrupted on server shutdown / restart.
func (db *Database) CancelInProgressGames() int {
	db.mu.Lock()
	defer db.mu.Unlock()

	count := 0
	for gameId, game := range db.games {
		if game.Status == "running" {
			game.Status = "canceled"
			db.games[gameId] = game
			count++
		}
	}

	return count
}

func (db *Database) GetGamesByCreator(creatorId string) []models.Game {
	db.mu.RLock()
	defer db.mu.RUnlock()

	var games []models.Game
	for _, game := range db.games {
		if game.CreatedBy == creatorId {
			games = append(games, game)
		}
	}
	return games
}

func (db *Database) GetGamesByUser(userId string) []models.Game {
	db.mu.RLock()
	defer db.mu.RUnlock()

	seen := make(map[string]struct{})
	var games []models.Game

	for _, game := range db.games {
		if _, exists := seen[game.Id]; exists {
			continue
		}

		// User created the game OR user is a player in the game
		isCreator := game.CreatedBy == userId
		isPlayer := false
		for _, player := range game.HumanPlayers {
			if player.UserId == userId {
				isPlayer = true
				break
			}
		}

		if isCreator || isPlayer {
			seen[game.Id] = struct{}{}
			games = append(games, game)
		}
	}
	return games
}
