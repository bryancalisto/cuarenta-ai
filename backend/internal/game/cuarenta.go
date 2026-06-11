package game

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
)

const (
	NumPlayers = 4
	HandSize   = 5
	WinScore   = 40

	// Matches the Python training code.
	MaxAchievablePointsByNonCaidaActions = 38
)

var teamByPlayer = map[int]int{
	0: 0,
	1: 1,
	2: 0,
	3: 1,
}

type Suit int
type Rank int

const (
	Brillos         Suit = 0
	CorazonesRojos  Suit = 1
	CorazonesNegros Suit = 2
	Treboles        Suit = 3
)

const (
	As     Rank = 1
	Dos    Rank = 2
	Tres   Rank = 3
	Cuatro Rank = 4
	Cinco  Rank = 5
	Seis   Rank = 6
	Siete  Rank = 7
	Jack   Rank = 11
	Queen  Rank = 12
	King   Rank = 13
)

var allRanks = []Rank{As, Dos, Tres, Cuatro, Cinco, Seis, Siete, Jack, Queen, King}

type Card struct {
	Rank Rank
	Suit Suit
}

func (c Card) Value() int {
	return int(c.Rank)
}

func (c Card) Index() int {
	rankPos := 0
	for i, rank := range allRanks {
		if rank == c.Rank {
			rankPos = i
			break
		}
	}
	return int(c.Suit)*10 + rankPos
}

func (c Card) String() string {
	return fmt.Sprintf("%s%s", rankDisplay[c.Rank], suitDisplay[c.Suit])
}

func (c Card) Less(other Card) bool {
	return c.Index() < other.Index()
}

var rankDisplay = map[Rank]string{
	As:     "A",
	Dos:    "2",
	Tres:   "3",
	Cuatro: "4",
	Cinco:  "5",
	Seis:   "6",
	Siete:  "7",
	Jack:   "J",
	Queen:  "Q",
	King:   "K",
}

var suitDisplay = map[Suit]string{
	Brillos:         "B",
	CorazonesRojos:  "CR",
	CorazonesNegros: "CN",
	Treboles:        "T",
}

func BuildDeck() []Card {
	deck := make([]Card, 0, 40)
	for suit := Brillos; suit <= Treboles; suit++ {
		for _, rank := range allRanks {
			deck = append(deck, Card{Rank: rank, Suit: suit})
		}
	}
	return deck
}

func ShuffledDeck(rng *rand.Rand) []Card {
	deck := BuildDeck()
	if rng != nil {
		rng.Shuffle(len(deck), func(i, j int) {
			deck[i], deck[j] = deck[j], deck[i]
		})
		return deck
	}
	rand.Shuffle(len(deck), func(i, j int) {
		deck[i], deck[j] = deck[j], deck[i]
	})
	return deck
}

type Action struct {
	Card    Card
	Capture []Card
}

func (a Action) String() string {
	if len(a.Capture) > 0 {
		parts := make([]string, len(a.Capture))
		for i, c := range a.Capture {
			parts[i] = c.String()
		}
		sort.Strings(parts)
		return fmt.Sprintf("play %s -> capture [%s]", a.Card, strings.Join(parts, ","))
	}
	return fmt.Sprintf("play %s -> place", a.Card)
}

func canonicalCards(cards []Card) []Card {
	if len(cards) == 0 {
		return nil
	}
	seen := make(map[Card]struct{}, len(cards))
	unique := make([]Card, 0, len(cards))
	for _, card := range cards {
		if _, ok := seen[card]; ok {
			continue
		}
		seen[card] = struct{}{}
		unique = append(unique, card)
	}
	sort.Slice(unique, func(i, j int) bool {
		if unique[i].Value() == unique[j].Value() {
			return unique[i].Index() < unique[j].Index()
		}
		return unique[i].Value() < unique[j].Value()
	})
	return unique
}

func actionKey(action Action) string {
	parts := make([]string, 0, len(action.Capture)+1)
	parts = append(parts, fmt.Sprintf("%d", action.Card.Index()))
	capture := canonicalCards(action.Capture)
	for _, card := range capture {
		parts = append(parts, fmt.Sprintf("%d", card.Index()))
	}
	return strings.Join(parts, ":")
}

func sameAction(a, b Action) bool {
	return actionKey(a) == actionKey(b)
}

func sortedByRank(cards []Card) []Card {
	out := make([]Card, len(cards))
	copy(out, cards)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Value() == out[j].Value() {
			return out[i].Index() < out[j].Index()
		}
		return out[i].Value() < out[j].Value()
	})
	return out
}

func isFigure(card Card) bool {
	switch card.Rank {
	case Jack, Queen, King, As:
		return true
	default:
		return false
	}
}

func sameRankCaptures(played Card, table []Card) [][]Card {
	captures := make([][]Card, 0)
	for _, card := range table {
		if card.Rank == played.Rank {
			captures = append(captures, []Card{card})
		}
	}
	return captures
}

func sumCaptures(played Card, table []Card) [][]Card {
	if len(table) < 2 || isFigure(played) {
		return nil
	}

	sortedTable := sortedByRank(table)
	playedValue := played.Value()
	complementMap := map[int][]Card{}
	captures := make([][]Card, 0)

	for i := 0; i < len(sortedTable); i++ {
		card := sortedTable[i]
		complement := playedValue - card.Value()
		if complementCards, ok := complementMap[complement]; ok {
			for _, complementCard := range complementCards {
				captures = append(captures, []Card{card, complementCard})
			}
		}
		complementMap[card.Value()] = append(complementMap[card.Value()], card)
	}

	return captures
}

func stairsCaptures(played Card, table []Card) [][]Card {
	if len(table) == 0 {
		return nil
	}

	sortedTable := sortedByRank(table)
	captures := make([][]Card, 0)

	for i := 0; i < len(sortedTable); i++ {
		if sortedTable[i].Rank != played.Rank {
			continue
		}

		matched := sortedTable[i]
		current := []Card{matched}

		for j := i + 1; j < len(sortedTable); j++ {
			if sortedTable[j].Rank == matched.Rank {
				continue
			}
			if sortedTable[j].Rank == matched.Rank+1 || (matched.Rank == Siete && sortedTable[j].Rank == Jack) {
				current = append(current, sortedTable[j])
				matched = sortedTable[j]
				continue
			}
			break
		}

		if len(current) > 1 {
			capture := make([]Card, len(current))
			copy(capture, current)
			captures = append(captures, capture)
		}
	}

	return captures
}

func ValidCaptures(played Card, table []Card) [][]Card {
	if len(table) == 0 {
		return nil
	}

	captures := make([][]Card, 0)
	sameRank := sameRankCaptures(played, table)
	captures = append(captures, sameRank...)

	sum := sumCaptures(played, table)
	captures = append(captures, sum...)

	stairs := stairsCaptures(played, table)
	captures = append(captures, stairs...)

	if len(sum) > 0 && len(stairs) > 0 {
		for _, sumCap := range sum {
			sumSet := make(map[Card]struct{}, len(sumCap))
			for _, card := range sumCap {
				sumSet[card] = struct{}{}
			}
			for _, stairCap := range stairs {
				merged := make([]Card, 0, len(sumCap)+len(stairCap))
				seen := make(map[Card]struct{}, len(sumCap)+len(stairCap))
				for _, card := range sumCap {
					if _, ok := seen[card]; ok {
						continue
					}
					seen[card] = struct{}{}
					merged = append(merged, card)
				}
				for _, card := range stairCap {
					if _, ok := seen[card]; ok {
						continue
					}
					seen[card] = struct{}{}
					merged = append(merged, card)
				}
				_ = sumSet
				captures = append(captures, canonicalCards(merged))
			}
		}
	}

	seen := map[string]struct{}{}
	unique := make([][]Card, 0, len(captures))
	for _, capSet := range captures {
		canon := canonicalCards(capSet)
		keyParts := make([]string, 0, len(canon))
		for _, card := range canon {
			keyParts = append(keyParts, fmt.Sprintf("%d", card.Index()))
		}
		key := strings.Join(keyParts, ":")
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		unique = append(unique, canon)
	}

	return unique
}

func detectCaida(played Card, captureSet []Card, lastPlaced *Card) bool {
	if lastPlaced == nil {
		return false
	}
	if lastPlaced.Value() != played.Value() {
		return false
	}
	for _, card := range captureSet {
		if card == *lastPlaced {
			return true
		}
	}
	return false
}

func detectLimpia(captureSet []Card, table []Card) bool {
	if len(captureSet) != len(table) {
		return false
	}
	tableSet := make(map[Card]struct{}, len(table))
	for _, card := range table {
		tableSet[card] = struct{}{}
	}
	for _, card := range captureSet {
		if _, ok := tableSet[card]; !ok {
			return false
		}
	}
	return true
}

func scorePlay(played Card, captureSet []Card, table []Card, lastPlaced *Card, nonCaidaAllowed bool) (int, []string) {
	points := 0
	reasons := make([]string, 0, 2)

	if len(captureSet) > 0 {
		if detectCaida(played, captureSet, lastPlaced) {
			points += 2
			reasons = append(reasons, "caída (+2)")
		} else if detectLimpia(captureSet, table) && nonCaidaAllowed {
			points += 2
			reasons = append(reasons, "limpia (+2)")
		}
	}

	return points, reasons
}

type State struct {
	Hands         map[int][]Card
	Table         []Card
	Deck          []Card
	DecksConsumed int
	Scores        [2]int
	CardsCaptured [2][]Card
	CurrentPlayer int
	LastPlaced    *Card
	Done          bool
	Winner        *int
}

type Env struct {
	rng                  *rand.Rand
	state                *State
	caidaExposurePenalty float64
}

func NewEnv(seed int64, caidaExposurePenalty float64) *Env {
	var rng *rand.Rand
	if seed != 0 {
		rng = rand.New(rand.NewSource(seed))
	} else {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}
	return &Env{
		rng:                  rng,
		caidaExposurePenalty: caidaExposurePenalty,
	}
}

func (e *Env) State() *State {
	return e.state
}

func (e *Env) Reset() Observation {
	deck := ShuffledDeck(e.rng)
	hands, deck := deal(deck)
	e.state = &State{
		Hands:         hands,
		Table:         nil,
		Deck:          deck,
		Scores:        [2]int{0, 0},
		CardsCaptured: [2][]Card{},
		CurrentPlayer: 0,
		DecksConsumed: 0,
		LastPlaced:    nil,
		Done:          false,
		Winner:        nil,
	}
	return e.Observe(0)
}

func deal(deck []Card) (map[int][]Card, []Card) {
	hands := make(map[int][]Card, NumPlayers)
	for p := 0; p < NumPlayers; p++ {
		hand := make([]Card, HandSize)
		copy(hand, deck[:HandSize])
		hands[p] = hand
		deck = deck[HandSize:]
	}
	return hands, deck
}

func (e *Env) redeal() {
	s := e.state
	for p := 0; p < NumPlayers; p++ {
		hand := make([]Card, len(s.Deck[:HandSize]))
		copy(hand, s.Deck[:HandSize])
		s.Hands[p] = hand
		s.Deck = s.Deck[HandSize:]
	}
}

func (e *Env) newRound() {
	s := e.state
	newDeck := ShuffledDeck(e.rng)
	s.Hands, s.Deck = deal(newDeck)
	s.CardsCaptured = [2][]Card{}
	s.Table = nil
	s.LastPlaced = nil
	s.CurrentPlayer = (s.CurrentPlayer + 1) % NumPlayers
}

func cardsToBitVec(cards []Card) []float32 {
	vec := make([]float32, 40)
	for _, card := range cards {
		vec[card.Index()] = 1
	}
	return vec
}

func (e *Env) Observe(player int) Observation {
	s := e.state
	allCards := BuildDeck()
	allSet := make(map[Card]struct{}, len(allCards))
	for _, card := range allCards {
		allSet[card] = struct{}{}
	}

	known := make(map[Card]struct{}, len(allCards))
	for _, card := range s.Hands[player] {
		known[card] = struct{}{}
	}
	for _, card := range s.Table {
		known[card] = struct{}{}
	}
	for _, card := range s.Deck {
		known[card] = struct{}{}
	}
	for p := 0; p < NumPlayers; p++ {
		if p == player {
			continue
		}
		for _, card := range s.Hands[p] {
			known[card] = struct{}{}
		}
	}

	playedThisDeal := make([]Card, 0)
	for _, card := range allCards {
		if _, ok := known[card]; !ok {
			playedThisDeal = append(playedThisDeal, card)
		}
	}
	_ = allSet

	obs := Observation{
		"my_hand":           cardsToBitVec(s.Hands[player]),
		"table":             cardsToBitVec(s.Table),
		"played_this_deal":  cardsToBitVec(playedThisDeal),
		"my_team_score":     []float32{float32(s.Scores[teamByPlayer[player]])},
		"opp_team_score":    []float32{float32(s.Scores[1-teamByPlayer[player]])},
		"my_team_captures":  cardsToBitVec(s.CardsCaptured[teamByPlayer[player]]),
		"opp_team_captures": cardsToBitVec(s.CardsCaptured[1-teamByPlayer[player]]),
		"current_player":    []float32{float32(s.CurrentPlayer)},
		"last_placed": cardsToBitVec(func() []Card {
			if s.LastPlaced != nil {
				return []Card{*s.LastPlaced}
			}
			return nil
		}()),
		"deck_remaining": []float32{float32(len(s.Deck))},
	}
	return obs
}

type Observation map[string][]float32

func FlattenObservation(obs Observation) []float32 {
	keys := make([]string, 0, len(obs))
	for k := range obs {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	flat := make([]float32, 0, 244)
	for _, key := range keys {
		flat = append(flat, obs[key]...)
	}
	return flat
}

func (e *Env) legalActions() []Action {
	s := e.state
	player := s.CurrentPlayer
	hand := s.Hands[player]
	actions := make([]Action, 0, len(hand))

	for _, card := range hand {
		captures := ValidCaptures(card, s.Table)
		if len(captures) > 0 {
			for _, capSet := range captures {
				action := Action{Card: card, Capture: canonicalCards(capSet)}
				actions = append(actions, action)
			}
			continue
		}
		actions = append(actions, Action{Card: card})
	}

	return actions
}

func (e *Env) LegalActions() []Action {
	return e.legalActions()
}

func (e *Env) LegalActionsForCard(played Card) []Action {
	s := e.state
	player := s.CurrentPlayer
	actions := make([]Action, 0, len(s.Hands[player]))

	// the card alone the player just throwed is a valid action
	actions = append(actions, Action{Card: played})

	// according to the thrown card, there's a set of actions that can be taken
	captures := ValidCaptures(played, s.Table)
	for _, capSet := range captures {
		action := Action{Card: played, Capture: canonicalCards(capSet)}
		actions = append(actions, action)
	}

	return actions
}

func actionExists(action Action, legal []Action) bool {
	for _, candidate := range legal {
		if sameAction(candidate, action) {
			return true
		}
	}
	return false
}

func removeCard(cards []Card, target Card) []Card {
	for i, card := range cards {
		if card == target {
			out := make([]Card, 0, len(cards)-1)
			out = append(out, cards[:i]...)
			out = append(out, cards[i+1:]...)
			return out
		}
	}
	return cards
}

func (e *Env) step(action Action) (Observation, float64, bool, map[string]any, error) {
	s := e.state
	if s.Done {
		return nil, 0, true, nil, fmt.Errorf("game is already over")
	}

	player := s.CurrentPlayer
	legal := e.legalActions()
	if !actionExists(action, legal) {
		return nil, 0, false, nil, fmt.Errorf("illegal action: %s", action.String())
	}

	hand := s.Hands[player]
	found := false
	for _, card := range hand {
		if card == action.Card {
			found = true
			break
		}
	}
	if !found {
		return nil, 0, false, nil, fmt.Errorf("player %d does not hold %s", player, action.Card)
	}

	s.Hands[player] = removeCard(s.Hands[player], action.Card)

	team := teamByPlayer[player]
	points, reasons := scorePlay(action.Card, action.Capture, s.Table, s.LastPlaced, s.Scores[team] < MaxAchievablePointsByNonCaidaActions)
	info := map[string]any{
		"player":                 player,
		"team":                   team,
		"action":                 action.String(),
		"points":                 points,
		"reasons":                reasons,
		"caida_exposure_penalty": 0.0,
	}

	caidaExposurePenalty := 0.0
	if len(action.Capture) > 0 {
		for _, captured := range canonicalCards(action.Capture) {
			s.Table = removeCard(s.Table, captured)
			s.CardsCaptured[team] = append(s.CardsCaptured[team], captured)
		}
		s.LastPlaced = nil
	} else {
		s.Table = append(s.Table, action.Card)
		s.LastPlaced = &action.Card
		caidaExposurePenalty = e.caidaExposurePenaltyForNextPlayer(action.Card)
		info["caida_exposure_penalty"] = caidaExposurePenalty
	}

	s.Scores[team] += points
	reward := float64(points) - caidaExposurePenalty
	if len(action.Capture) > 0 {
		reward += float64(len(action.Capture)) * 0.25
	}

	if s.Scores[team] >= WinScore {
		s.Done = true
		winner := team
		s.Winner = &winner
		reward += 100
		info["winner"] = team
		return e.Observe(player), reward, true, info, nil
	}

	s.CurrentPlayer = (player + 1) % NumPlayers

	allHandsEmpty := true
	for p := 0; p < NumPlayers; p++ {
		if len(s.Hands[p]) != 0 {
			allHandsEmpty = false
			break
		}
	}

	if allHandsEmpty {
		if len(s.Deck) >= NumPlayers*HandSize {
			e.redeal()
		} else {
			for teamIdx := 0; teamIdx < 2; teamIdx++ {
				cartonPoints := len(s.CardsCaptured[teamIdx]) - 19
				if cartonPoints <= 0 {
					continue
				}
				cartonPoints += 5
				if cartonPoints%2 != 0 {
					cartonPoints++
				}
				if cartonPoints < MaxAchievablePointsByNonCaidaActions {
					s.Scores[teamIdx] += MaxAchievablePointsByNonCaidaActions
				} else {
					s.Scores[teamIdx] += cartonPoints
				}
				if teamIdx == team {
					reward += float64(cartonPoints)
				}
			}

			if s.Scores[0] >= WinScore || s.Scores[1] >= WinScore {
				s.Done = true
				winner := 0
				if s.Scores[1] > s.Scores[0] {
					winner = 1
				}
				s.Winner = &winner
				info["winner"] = winner
				return e.Observe(s.CurrentPlayer), reward + 100, true, info, nil
			}

			e.newRound()
		}
	}

	return e.Observe(s.CurrentPlayer), reward, false, info, nil
}

func (e *Env) caidaExposurePenaltyForNextPlayer(played Card) float64 {
	if e.caidaExposurePenalty <= 0 {
		return 0
	}

	s := e.state
	nextPlayer := (s.CurrentPlayer + 1) % NumPlayers
	for _, responseCard := range s.Hands[nextPlayer] {
		for _, captureSet := range ValidCaptures(responseCard, s.Table) {
			if containsCard(captureSet, played) && detectCaida(responseCard, captureSet, &played) {
				return e.caidaExposurePenalty
			}
		}
	}

	return 0
}

func containsCard(cards []Card, target Card) bool {
	for _, card := range cards {
		if card == target {
			return true
		}
	}
	return false
}

func (e *Env) Render() string {
	s := e.state
	lines := []string{
		strings.Repeat("=", 50),
		fmt.Sprintf("  CUARENTA  |  Team 0 (P0,P2): %d  vs  Team 1 (P1,P3): %d", s.Scores[0], s.Scores[1]),
		fmt.Sprintf("  Current player: %d  |  Deck remaining: %d", s.CurrentPlayer, len(s.Deck)),
		fmt.Sprintf("  Table: %s", func() string {
			if len(s.Table) == 0 {
				return "(empty)"
			}
			parts := make([]string, len(s.Table))
			for i, c := range s.Table {
				parts[i] = c.String()
			}
			return strings.Join(parts, " ")
		}()),
		fmt.Sprintf("  Last placed: %v", s.LastPlaced),
	}
	for p := 0; p < NumPlayers; p++ {
		hand := make([]Card, len(s.Hands[p]))
		copy(hand, s.Hands[p])
		sort.Slice(hand, func(i, j int) bool { return hand[i].Less(hand[j]) })
		parts := make([]string, len(hand))
		for i, c := range hand {
			parts[i] = c.String()
		}
		marker := ""
		if p == s.CurrentPlayer {
			marker = " <-"
		}
		lines = append(lines, fmt.Sprintf("  P%d (team %d): %s%s", p, teamByPlayer[p], strings.Join(parts, " "), marker))
	}
	lines = append(lines, strings.Repeat("=", 50))
	return strings.Join(lines, "\n")
}

func (e *Env) Step(action Action) (Observation, float64, bool, map[string]any, error) {
	return e.step(action)
}

func EncodeAction(action Action, legal []Action) int {
	cardIdx := action.Card.Index()
	captureIdx := 0
	if len(action.Capture) > 0 {
		sameCard := make([]Action, 0)
		for _, candidate := range legal {
			if candidate.Card == action.Card {
				sameCard = append(sameCard, candidate)
			}
		}
		for i, candidate := range sameCard {
			if sameAction(candidate, action) {
				captureIdx = i + 1
				break
			}
		}
	}
	return cardIdx*41 + captureIdx
}

func DecodeAction(actionIdx int, legal []Action) (Action, bool) {
	if len(legal) == 0 {
		return Action{}, false
	}
	if actionIdx < 0 {
		return legal[0], false
	}

	cardIdx := actionIdx / 41
	captureIdx := actionIdx % 41
	deck := BuildDeck()
	if cardIdx < 0 || cardIdx >= len(deck) {
		return legal[actionIdx%len(legal)], false
	}
	playedCard := deck[cardIdx]
	matching := make([]Action, 0)
	for _, candidate := range legal {
		if candidate.Card == playedCard {
			matching = append(matching, candidate)
		}
	}
	if len(matching) == 0 {
		return legal[actionIdx%len(legal)], true
	}
	if captureIdx == 0 {
		for _, candidate := range matching {
			if len(candidate.Capture) == 0 {
				return candidate, true
			}
		}
		return legal[actionIdx%len(legal)], true
	}

	captureActions := make([]Action, 0)
	for _, candidate := range matching {
		if len(candidate.Capture) > 0 {
			captureActions = append(captureActions, candidate)
		}
	}
	captureOffset := captureIdx - 1
	if captureOffset < len(captureActions) {
		return captureActions[captureOffset], true
	}

	return legal[actionIdx%len(legal)], true
}
