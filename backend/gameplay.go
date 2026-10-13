package cua

import (
	"errors"
	"fmt"
	"log"
	"math/rand"
	"sort"
	"sync"
	"time"

	"cua/internal/config"
	"cua/internal/game"
	"cua/internal/models"
	"cua/internal/search"

	"github.com/benbjohnson/clock"
)

type Gameplay interface {
	newGameSession(snapshot models.Game) *gameSession
	prepareSession(snapshot models.Game) (*gameSession, models.Game, error)
	processGameAction(clien *wsClient, payload map[string]interface{}) error
	startRunningGame(snapshot models.Game) (models.Game, error)
	playHumanTurnAndAdvanceAI(gameId string, userID string, dto PlayGameDTO) (models.Game, error)
	playAITurns(gameId string) error
	cancelRunningGames() int
	deleteSession(gameId string)
}

type MainGameplay struct {
	clock        clock.Clock
	db           Database
	inferService InferenceService
}

func NewMainGameplay(db Database, clock clock.Clock, inferService InferenceService) *MainGameplay {
	return &MainGameplay{db: db, clock: clock, inferService: inferService}
}

type PlayGameDTO struct {
	CardIndex      int   `json:"cardIndex"`
	CaptureIndices []int `json:"captureIndices"`
}

type gameSession struct {
	Mu         sync.Mutex
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

	if _, err := g.playHumanTurnAndAdvanceAI(gameId, client.userId, dto); err != nil {
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

func (g *MainGameplay) newGameSession(snapshot models.Game) *gameSession {
	return &gameSession{
		env:        game.NewEnv(time.Now().UnixNano()),
		humanSeats: humanSeats(snapshot),
	}
}

func (g *MainGameplay) prepareSession(snapshot models.Game) (*gameSession, models.Game, error) {
	session, ok := getSession(snapshot.Id)
	if ok {
		return session, sessionSnapshot(snapshot, session, nil), nil
	}

	session = g.newGameSession(snapshot)
	session.env.Reset()
	// A brand new game gets a brand new recurrent state: the LSTM memory of a previous
	// game (same game id, e.g. after a restart) must not leak into this one.
	g.inferService.ClearGameState(snapshot.Id)
	setSession(snapshot.Id, session)
	return session, sessionSnapshot(snapshot, session, nil), nil
}

func (g *MainGameplay) startRunningGame(snapshot models.Game) (models.Game, error) {
	_, updated, err := g.prepareSession(snapshot)
	if err != nil {
		return models.Game{}, err
	}

	g.publishGameSnapshot(updated)

	session, ok := getSession(snapshot.Id)

	if !ok {
		log.Fatal("Failed on getting session on game start")
	}

	updated = sessionSnapshot(snapshot, session, nil)

	if err := g.playAITurns(snapshot.Id); err != nil {
		return updated, err
	}

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

// searchWarnOnce makes the "search unavailable" message appear once per process instead of
// on every AI turn of every game.
var searchWarnOnce sync.Once

// searchTurn runs one ISMCTS decision for `seat`, or reports that the search is not
// available so the caller can fall back to the bare policy.
//
// WHY THE SEARCH IS THE POINT: the served network is a POLICY, and on its own it is worth
// about a coin flip against the rule-based heuristic (measured over 200 games: 49.0%), while
// the same weights WITH the search win 77%. `AI_SEARCH_ITERATIONS=0` turns this off.
//
// The caller holds the session's lock, so a game's AI turns are serialized; the ONNX session
// is shared process-wide and the search runs one decision at a time.
func (g *MainGameplay) searchTurn(gameId string, seat int, state *game.State,
	legal []game.Action) (game.Action, bool) {
	iterations := config.GetEnv().AiSearchIterations
	if iterations <= 0 {
		return game.Action{}, false
	}

	evaluator, err := g.inferService.SearchEvaluator()
	if err != nil {
		searchWarnOnce.Do(func() {
			log.Printf("AI search unavailable: %v (falling back to the bare policy)", err)
		})
		return game.Action{}, false
	}

	cfg := evaluator.Config(iterations, config.GetEnv().AiSearchCPuct)
	searcher, err := search.New(state, seat, evaluator, cfg,
		rand.New(rand.NewSource(time.Now().UnixNano())))
	if err != nil {
		log.Printf("AI search could not start for game %s: %v", gameId, err)
		return game.Action{}, false
	}

	action, stats, err := searcher.Decide(state, g.inferService.History(gameId, seat))
	if err != nil {
		log.Printf("AI search failed for game %s: %v; falling back to the bare policy",
			gameId, err)
		return game.Action{}, false
	}
	if !game.ActionIsLegal(action, legal) {
		log.Printf("AI search returned %s, which is not legal for game %s; falling back",
			action, gameId)
		return game.Action{}, false
	}

	log.Printf("AI search: game %s seat %d -> %s (%d sims, %d evals, root value %.2f pts)",
		gameId, seat, action, stats.Simulations, stats.Evaluations, stats.RootValuePts)
	return action, true
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

	state := session.env.State()
	if state == nil {
		session.Mu.Unlock()
		return fmt.Errorf("game session is not initialized")
	}

	if state.Done {
		snapshot := sessionSnapshot(base, session, nil)
		g.publishGameSnapshot(snapshot)
		session.Mu.Unlock()
		return nil
	}

	currentSeat := state.CurrentPlayer

	legal := session.env.LegalActions()
	if len(legal) == 0 {
		session.Mu.Unlock()
		return fmt.Errorf("ai turn has no legal actions")
	}

	// The policy scores candidate ACTIONS by features, so the observation, the
	// [64][84] action-feature block and the [64] mask are built in exactly the order of
	// `legal`: the slot the model answers with indexes this same list.
	obs := game.FlattenObservation(session.env.Observe(currentSeat))
	feats, mask, slots := game.BuildActionFeatures(legal, state)

	action := legal[0]
	switch {
	case slots < len(legal):
		log.Printf("AI turn for game %s has %d legal actions but only %d policy slots; falling back to first legal action", base.Id, len(legal), slots)
	default:
		if searchAction, ok := g.searchTurn(gameId, currentSeat, state, legal); ok {
			// The SEARCH played this turn: the same network, but with a tree over
			// determinizations instead of a single forward pass (see searchTurn).
			action = searchAction
			break
		}
		slot, _, err := g.inferService.PredictActionSlot(gameId, obs, feats.Flatten(), mask.Flatten(), slots)
		if err != nil {
			log.Printf("AI inference failed for game %s: %v; falling back to first legal action", base.Id, err)
		} else if picked, ok := game.ActionForSlot(slot, legal); ok {
			action = picked
		} else {
			log.Printf("AI inference returned slot %d for game %s with %d legal actions; falling back to first legal action", slot, base.Id, len(legal))
		}
	}

	// Remember this decision for the search's next root evaluation: the network reads the
	// acting seat's OWN history window, so the observation it saw and the features of the
	// action it is about to play are appended here (see InferenceService.RecordHistory).
	g.inferService.RecordHistory(gameId, currentSeat, obs, game.ActionFeatures(action, state))

	_, done, info, err := session.env.Step(action)

	if done {
		base.DurationSeconds = g.getGameDurationSeconds(base)
		log.Println("Game has ended in AI turn")
		g.inferService.ClearGameState(gameId)
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
		session.Mu.Unlock()
		return err
	}

	g.publishGameSnapshot(snapshot)

	// If newRound happened during step, recursively continue AI turns from new state
	//
	// The recurrent state is deliberately NOT cleared here: the Python reference keeps
	// one (h, c) per match and resets it only when the game ends (src/ppo/rollout.py:
	// replay_state_embeddings, src/ppo/eval_harness.py), so a mid-game reset would feed
	// the network a state it never saw while training. It IS cleared when a new game
	// session starts (prepareSession) and when a game ends, above.
	if newRound, ok := info["newRound"].(bool); ok && newRound {
		session.Mu.Unlock()
		if err := g.playAITurns(gameId); err != nil {
			return err
		}
		return nil
	}

	session.Mu.Unlock()
	return nil
}

// advanceRecurrentState feeds one turn of a match to the policy without using its
// answer, so the per-game LSTM memory is advanced on the HUMAN turns too.
//
// The network was trained with its recurrent state advanced on EVERY turn of a match,
// each turn fed the ACTING player's observation and legal-action features
// (src/ppo/trainer.py). Advancing it only on the AI's own decisions would hand the
// network, at its own decisions, a hidden state it never saw during training — hence
// every human turn goes through here: after the action has been validated (a rejected
// play must not pollute the memory with a turn that never happened) and before it is
// applied.
//
// `seat` is the acting player, whose point of view the observation is built from; while
// the turn is being played it is the current player. A net that cannot run is logged and
// ignored: a human's move must never fail because inference was unavailable.
func (g *MainGameplay) advanceRecurrentState(gameId string, session *gameSession, seat int) {
	state := session.env.State()
	if state == nil {
		return
	}

	legal := session.env.LegalActions()
	if len(legal) == 0 {
		return
	}

	obs := game.FlattenObservation(session.env.Observe(seat))
	feats, mask, slots := game.BuildActionFeatures(legal, state)
	if slots < len(legal) {
		log.Printf("Not advancing the recurrent state of game %s: %d legal actions do not fit the %d policy slots", gameId, len(legal), slots)
		return
	}

	if err := g.inferService.AdvanceRecurrentState(gameId, obs, feats.Flatten(), mask.Flatten(), slots); err != nil {
		log.Printf("AI recurrent state was not advanced for game %s (seat %d): %v", gameId, seat, err)
	}
}

// deleteSession drops a game's session together with the policy's recurrent state for
// it, so a cancelled or removed game cannot leave memory behind (lstmStates is keyed by
// game id and would otherwise grow without bound).
func (g *MainGameplay) deleteSession(gameId string) {
	Sessions.Mu.Lock()
	delete(Sessions.M, normalizeGameId(gameId))
	Sessions.Mu.Unlock()

	g.inferService.ClearGameState(gameId)
}

func (g *MainGameplay) cancelRunningGames() int {
	count := g.db.CancelInProgressGames()
	return count
}

// Return random milliseconds for slee[1, 3]
func randomSleep() {
	millis := []int{1500, 1800, 2000, 2400, 3000}
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

	// play the AI agents turns until hitting a human seat or game end
	for {
		state := session.env.State()
		if state == nil {
			return fmt.Errorf("game session is not initialized")
		}

		if state.Done {
			log.Println("Skipping AI turns as game has ended")
			break
		}

		currentSeat := state.CurrentPlayer
		if currentSeat >= 4 {
			break
		}

		_, isHumanSeat := session.humanSeats[currentSeat]
		if isHumanSeat {
			break
		}

		randomSleep()

		if err := g.playAITurn(gameId); err != nil {
			log.Fatalf("ERROR while playing AI turn: %v", err)
		}
	}

	return nil
}

func (g *MainGameplay) playHumanTurnAndAdvanceAI(gameId string, userID string, dto PlayGameDTO) (models.Game, error) {
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

	state := session.env.State()
	if state == nil {
		session.Mu.Unlock()
		return models.Game{}, fmt.Errorf("game session is not initialized")
	}
	if state.Done {
		snapshot := sessionSnapshot(base, session, nil)
		g.publishGameSnapshot(snapshot)
		session.Mu.Unlock()
		return snapshot, nil
	}

	currentSeat := state.CurrentPlayer
	humanUserID, isHumanSeat := session.humanSeats[currentSeat]
	if !isHumanSeat {
		session.Mu.Unlock()
		return models.Game{}, ErrNotYourTurn
	}
	if humanUserID != userID {
		session.Mu.Unlock()
		return models.Game{}, ErrForbiddenAction
	}

	action, err := actionFromDTO(dto)
	if err != nil {
		session.Mu.Unlock()
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
		session.Mu.Unlock()
		return snapshot, nil
	}

	// If exactly one matching action (with capture), use it automatically
	if len(matchingActions) == 1 && len(dto.CaptureIndices) == 0 {
		action = matchingActions[0]
	} else if matched, ok := matchLegalAction(action, legal); ok {
		action = matched
	} else {
		session.Mu.Unlock()
		return models.Game{}, ErrIllegalAction
	}

	// `legal` above is scoped to the played card and does not know whose hand it is in,
	// so re-check the resolved action against the acting player's own legal actions
	// before anything is committed: an illegal or rejected play must leave both the game
	// AND the policy's memory untouched.
	if !game.ActionIsLegal(action, session.env.LegalActions()) {
		session.Mu.Unlock()
		return models.Game{}, ErrIllegalAction
	}

	// The turn is real from here on, so it counts for the policy's memory: advance the
	// recurrent state with the acting (human) player's own decision point, exactly as
	// the AI turns do, and only then apply the action.
	g.advanceRecurrentState(gameId, session, currentSeat)

	_, done, _, err := session.env.Step(action)

	if done {
		base.DurationSeconds = g.getGameDurationSeconds(base)
		log.Println("Game has ended in human turn")
		// The game is over: its memory is of no further use (playAITurn does the same).
		g.inferService.ClearGameState(gameId)
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
		session.Mu.Unlock()
		return snapshot, err
	}

	g.publishGameSnapshot(snapshot)
	session.Mu.Unlock()

	if err := g.playAITurns(gameId); err != nil {
		if err == ErrGameNotRunning {
			return snapshot, nil
		}

		return snapshot, err
	}

	return snapshot, nil
}
