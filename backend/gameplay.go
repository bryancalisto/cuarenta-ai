package cua

import (
	"errors"
	"fmt"
	"log"
	"math/rand"
	"sort"
	"sync"
	"time"

	"cua/internal/game"
	"cua/internal/models"

	"github.com/benbjohnson/clock"
)

type Gameplay interface {
	processGameAction(clien *wsClient, payload map[string]interface{}) error
	startRunningGame(snapshot models.Game) (models.Game, error)
	playHumanTurn(gameId string, userID string, dto PlayGameDTO) (models.Game, error)
	playAITurns(gameId string) error
	cancelRunningGames() int
}

type MainGameplay struct {
	clock        clock.Clock
	db           Database
	inferService InferenceService
}

func NewMainGameplay(db Database, clock clock.Clock, inferService InferenceService) Gameplay {
	return &MainGameplay{db: db, clock: clock, inferService: inferService}
}

type PlayGameDTO struct {
	CardIndex      int   `json:"cardIndex"`
	CaptureIndices []int `json:"captureIndices"`
}

type gameSession struct {
	Mu         sync.Mutex
	AiTurnChan chan int
	env        *game.Env
	stepCount  int
	humanSeats map[int]string
}

var Sessions = struct {
	Mu sync.RWMutex
	M  map[string]*gameSession
}{
	M: make(map[string]*gameSession),
}

var (
	ErrMissingGameId   = errors.New("missing game ID")
	ErrMissingCardIdx  = errors.New("missing cardIndex")
	ErrHumanTurnFailed = errors.New("error on human turn")
)

func (g MainGameplay) processGameAction(client *wsClient, payload map[string]interface{}) error {
	// log.Println("game.action:", payload)
	gameId, ok := payload["gameId"].(string)
	if !ok {
		return ErrMissingGameId
	}
	cardIndex, ok := payload["cardIndex"].(float64)
	if !ok {
		return ErrMissingCardIdx
	}

	captureIndices := []int{}
	if captures, ok := payload["captureIndices"].([]interface{}); ok {
		for _, c := range captures {
			if idx, ok := c.(float64); ok {
				captureIndices = append(captureIndices, int(idx))
			}
		}
	}

	dto := PlayGameDTO{
		CardIndex:      int(cardIndex),
		CaptureIndices: captureIndices,
	}

	if _, err := g.playHumanTurn(gameId, client.userId, dto); err != nil {
		return ErrHumanTurnFailed
	}

	return nil
}

func normalizeCaptureIndices(indices []int) []int {
	if len(indices) == 0 {
		return nil
	}

	seen := make(map[int]struct{}, len(indices))
	result := make([]int, 0, len(indices))
	for _, idx := range indices {
		if _, ok := seen[idx]; ok {
			continue
		}
		seen[idx] = struct{}{}
		result = append(result, idx)
	}
	sort.Ints(result)
	return result
}

func cardsFromIndices(indices []int) ([]game.Card, error) {
	deck := game.BuildDeck()
	cards := make([]game.Card, 0, len(indices))
	for _, idx := range indices {
		if idx < 0 || idx >= len(deck) {
			return nil, fmt.Errorf("card index %d is out of range", idx)
		}
		cards = append(cards, deck[idx])
	}
	return cards, nil
}

func cardIndices(cards []game.Card) []int {
	if len(cards) == 0 {
		return nil
	}
	out := make([]int, len(cards))
	for i, card := range cards {
		out[i] = card.Index()
	}
	sort.Ints(out)
	return out
}

func humanSeats(snapshot models.Game) map[int]string {
	result := make(map[int]string, len(snapshot.HumanPlayers))
	for _, player := range snapshot.HumanPlayers {
		result[player.Index-1] = player.UserId
	}
	return result
}

func getSession(gameId string) (*gameSession, bool) {
	Sessions.Mu.RLock()
	defer Sessions.Mu.RUnlock()

	session, ok := Sessions.M[normalizeGameId(gameId)]
	return session, ok
}

func setSession(gameId string, session *gameSession) {
	Sessions.Mu.Lock()
	defer Sessions.Mu.Unlock()
	Sessions.M[normalizeGameId(gameId)] = session
}

func deleteSession(gameId string) {
	Sessions.Mu.Lock()
	defer Sessions.Mu.Unlock()
	delete(Sessions.M, normalizeGameId(gameId))
}

func sessionSnapshot(base models.Game, session *gameSession, move *models.Move) models.Game {
	state := session.env.State()
	snapshot := base
	snapshot.TimeSteps = session.stepCount
	snapshot.Cards = make([][]int, 4)
	for seat := 0; seat < 4; seat++ {
		snapshot.Cards[seat] = cardIndices(state.Hands[seat])
	}
	snapshot.Table = cardIndices(state.Table)
	snapshot.Scores = []int{state.Scores[0], state.Scores[1]}
	snapshot.CardsCaptured = make([][]int, 2)
	snapshot.CardsCaptured[0] = cardIndices(state.CardsCaptured[0])
	snapshot.CardsCaptured[1] = cardIndices(state.CardsCaptured[1])
	snapshot.CurrentPlayer = state.CurrentPlayer
	snapshot.DeckRemaining = len(state.Deck)
	snapshot.NewRoundStart = state.DecksConsumed > 0 && len(state.CardsCaptured[0]) == 0 && len(state.CardsCaptured[1]) == 0
	snapshot.Done = state.Done
	if state.LastPlaced != nil {
		last := state.LastPlaced.Index()
		snapshot.LastPlaced = &last
	} else {
		snapshot.LastPlaced = nil
	}
	if state.Winner != nil {
		winner := *state.Winner
		snapshot.Winner = &winner
	} else {
		snapshot.Winner = nil
	}
	if state.Done {
		snapshot.Status = "finished"
	} else {
		snapshot.Status = "running"
	}

	// Reset capture selection state
	snapshot.AwaitingCapture = false
	snapshot.CaptureOptions = nil

	if move != nil {
		snapshot.Move = move
	} else {
		snapshot.Move = nil
	}

	return snapshot
}

func (g *MainGameplay) publishGameSnapshot(snapshot models.Game) {
	g.db.UpdateGame(snapshot.Id, snapshot)
	hub.broadcast(snapshot)
}

func newGameSession(snapshot models.Game) *gameSession {
	aiTurnChan := make(chan int, 1)
	return &gameSession{
		env:        game.NewEnv(time.Now().UnixNano(), aiTurnChan),
		humanSeats: humanSeats(snapshot),
		AiTurnChan: aiTurnChan,
	}
}

func prepareSession(snapshot models.Game) (*gameSession, models.Game, error) {
	session, ok := getSession(snapshot.Id)
	if ok {
		return session, sessionSnapshot(snapshot, session, nil), nil
	}

	session = newGameSession(snapshot)
	session.env.Reset()
	setSession(snapshot.Id, session)
	return session, sessionSnapshot(snapshot, session, nil), nil
}

func (g *MainGameplay) startRunningGame(snapshot models.Game) (models.Game, error) {
	_, updated, err := prepareSession(snapshot)
	if err != nil {
		return models.Game{}, err
	}

	g.publishGameSnapshot(updated)

	session, ok := getSession(snapshot.Id)

	if !ok {
		log.Fatal("Failed on getting session on game start")
	}

	updated = sessionSnapshot(snapshot, session, nil)

	go g.aiTurnsRunner(updated.Id, session.AiTurnChan)

	session.AiTurnChan <- updated.CurrentPlayer
	return updated, nil
}

func actionFromDTO(dto PlayGameDTO) (game.Action, error) {
	cards, err := cardsFromIndices(append([]int{dto.CardIndex}, normalizeCaptureIndices(dto.CaptureIndices)...))
	if err != nil {
		return game.Action{}, err
	}

	action := game.Action{Card: cards[0]}
	if len(cards) > 1 {
		action.Capture = cards[1:]
	}
	return action, nil
}

func matchLegalAction(action game.Action, legal []game.Action) (game.Action, bool) {
	for _, candidate := range legal {
		if candidate.Card != action.Card {
			continue
		}

		if len(candidate.Capture) != len(action.Capture) {
			continue
		}

		if len(candidate.Capture) == 0 {
			return candidate, true
		}

		candidateSet := make(map[game.Card]struct{}, len(candidate.Capture))
		for _, card := range candidate.Capture {
			candidateSet[card] = struct{}{}
		}

		ok := true
		for _, card := range action.Capture {
			if _, exists := candidateSet[card]; !exists {
				ok = false
				break
			}
		}
		if ok {
			return candidate, true
		}
	}

	return game.Action{}, false
}

func (g *MainGameplay) aiTurnsRunner(gameId string, aiTurnChan chan int) {
	for range aiTurnChan {
		g.playAITurns(gameId)
	}
}

func legalCapturesForCard(card game.Card, legal []game.Action) []game.Action {
	matches := make([]game.Action, 0)
	for _, action := range legal {
		if action.Card == card {
			matches = append(matches, action)
		}
	}
	return matches
}

func (g *MainGameplay) getGameDurationSeconds(game models.Game) int {
	duration, err := time.Parse(time.RFC3339, game.StartedAt)
	if err != nil {
		log.Fatal("Failed to parsed date")
	}
	return int(g.clock.Now().Sub(duration).Seconds())
}

func (g *MainGameplay) playAITurn(gameId string) error {
	base, exists := g.db.GetGame(gameId)

	if !exists {
		return ErrGameNotFound
	}

	if base.Status != "running" {
		return ErrGameNotRunning
	}
	session, ok := getSession(gameId)
	if !ok {
		return ErrGameNotRunning
	}

	session.Mu.Lock()
	defer session.Mu.Unlock()

	state := session.env.State()
	if state == nil {
		return fmt.Errorf("game session is not initialized")
	}

	if state.Done {
		snapshot := sessionSnapshot(base, session, nil)
		g.publishGameSnapshot(snapshot)
		return nil
	}

	currentSeat := state.CurrentPlayer

	legal := session.env.LegalActions()
	if len(legal) == 0 {
		return fmt.Errorf("ai turn has no legal actions")
	}
	obs := game.FlattenObservation(session.env.Observe(currentSeat))
	action, _, err := g.inferService.PredictAction(obs, legal)
	if err != nil {
		log.Printf("AI inference failed for game %s: %v; falling back to first legal action", base.Id, err)
		action = legal[0]
	}

	_, done, _, err := session.env.Step(action)

	if done {
		base.DurationSeconds = g.getGameDurationSeconds(base)
		log.Println("Game has ended in AI turn")
	}

	session.stepCount++

	move := models.Move{
		CardIndex:      action.Card.Index(),
		CaptureIndices: cardIndices(action.Capture),
		PlayerIndex:    currentSeat,
	}

	snapshot := sessionSnapshot(base, session, &move)
	if err != nil {
		g.publishGameSnapshot(snapshot)
		return err
	}

	g.publishGameSnapshot(snapshot)
	return nil
}

func (g *MainGameplay) cancelRunningGames() int {
	count := g.db.CancelInProgressGames()
	return count
}

// Return random milliseconds for slee[1, 3]
func randomSleep() {
	millis := []int{1500, 2000, 2500, 3000, 3500}
	time.Sleep(time.Duration(millis[rand.Intn(len(millis))]) * time.Millisecond)
}

func (g *MainGameplay) playAITurns(gameId string) error {
	base, exists := g.db.GetGame(gameId)

	if !exists {
		return ErrGameNotFound
	}

	if base.Status != "running" {
		return ErrGameNotRunning
	}
	session, ok := getSession(gameId)
	if !ok {
		return ErrGameNotRunning
	}

	state := session.env.State()
	if state == nil {
		return fmt.Errorf("game session is not initialized")
	}

	// play the AI agents turns
	for currentSeat := state.CurrentPlayer; currentSeat < 4; currentSeat++ {
		_, isHumanSeat := session.humanSeats[currentSeat]

		if isHumanSeat {
			break
		}

		if state.Done {
			log.Println("Skipping AI turns as game has ended")
			break
		}

		randomSleep()

		if err := g.playAITurn(gameId); err != nil {
			log.Fatalf("ERROR while playing AI turn: %v", err)
		}
	}

	return nil
}

func (g *MainGameplay) playHumanTurn(gameId string, userID string, dto PlayGameDTO) (models.Game, error) {
	base, exists := g.db.GetGame(gameId)
	if !exists {
		return models.Game{}, ErrGameNotFound
	}
	if base.Status != "running" {
		return models.Game{}, ErrGameNotRunning
	}

	session, ok := getSession(gameId)
	if !ok {
		return models.Game{}, ErrGameNotRunning
	}

	session.Mu.Lock()
	defer session.Mu.Unlock()

	state := session.env.State()
	if state == nil {
		return models.Game{}, fmt.Errorf("game session is not initialized")
	}
	if state.Done {
		snapshot := sessionSnapshot(base, session, nil)
		g.publishGameSnapshot(snapshot)
		return snapshot, nil
	}

	currentSeat := state.CurrentPlayer
	humanUserID, isHumanSeat := session.humanSeats[currentSeat]
	if !isHumanSeat {
		return models.Game{}, ErrNotYourTurn
	}
	if humanUserID != userID {
		return models.Game{}, ErrForbiddenAction
	}

	action, err := actionFromDTO(dto)
	if err != nil {
		return models.Game{}, err
	}

	legal := session.env.LegalActionsForCard(action.Card)

	// Check if this card has legal captures
	matchingActions := legalCapturesForCard(action.Card, legal)

	if len(matchingActions) > 1 && len(dto.CaptureIndices) == 0 {
		// Multiple captures possible, ask user to choose
		snapshot := sessionSnapshot(base, session, nil)
		snapshot.AwaitingCapture = true
		snapshot.CaptureOptions = make([]models.CaptureOption, len(matchingActions))
		for i, act := range matchingActions {
			snapshot.CaptureOptions[i] = models.CaptureOption{
				CaptureIndices: cardIndices(act.Capture),
			}
		}
		g.publishGameSnapshot(snapshot)
		return snapshot, nil
	}

	// If exactly one matching action (with capture), use it automatically
	if len(matchingActions) == 1 && len(dto.CaptureIndices) == 0 {
		action = matchingActions[0]
	} else if matched, ok := matchLegalAction(action, legal); ok {
		action = matched
	} else {
		return models.Game{}, ErrIllegalAction
	}

	_, done, _, err := session.env.Step(action)

	if done {
		base.DurationSeconds = g.getGameDurationSeconds(base)
		log.Println("Game has ended in human turn")
	}

	session.stepCount++

	move := models.Move{
		CardIndex:      action.Card.Index(),
		CaptureIndices: cardIndices(action.Capture),
		PlayerIndex:    currentSeat,
	}

	snapshot := sessionSnapshot(base, session, &move)
	if err != nil {
		g.publishGameSnapshot(snapshot)
		return snapshot, err
	}

	g.publishGameSnapshot(snapshot)

	session.AiTurnChan <- snapshot.CurrentPlayer
	return models.Game{}, nil
}
