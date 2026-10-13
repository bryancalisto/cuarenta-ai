package game

import (
	"fmt"
	"log"
	"math/rand"
	"slices"
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

// ---------------------------------------------------------------------------
// Policy input contract (the ONNX model's tensors)
// ---------------------------------------------------------------------------
//
// The neural players are the PPO actor-critic exported by scripts/pth2onnx.py: one
// call per decision, taking the observation of the player about to move, a FEATURE
// vector per candidate action and a mask of which slots are legal. The mirror of
// these constants in Python is src/ppo/env.py (MAX_ACTIONS, ACTION_FEAT_DIM),
// src/ppo/utils.py (OBS_KEYS_ISMCTS, OBS_SIZE_ISMCTS) and backend/models/*.json.
//
// Slot i of the model's logits is the i-th entry of the []Action list the caller
// passes to BuildActionFeatures — there is no other ordering contract.
const (
	// MaxActionSlots is MAX_ACTIONS: the policy scores at most this many candidate
	// actions per decision, the legal ones packed into slots 0..n-1.
	MaxActionSlots = 64

	// ActionFeatureDim is ACTION_FEAT_DIM: the width of one candidate action's
	// description, see ActionFeatures.
	ActionFeatureDim = 84

	// CardVecSize is the width of a card bit vector: one float per Card.Index().
	CardVecSize = 40

	// Flat observation layout — OBS_KEYS_ISMCTS in src/ppo/utils.py, in this exact
	// order, 208 floats total. The set, the order and the widths are pinned by every
	// exported model: do not reorder, do not add keys.
	ObsOffsetMyHand          = 0   // [0, 40)    cards in the moving player's hand
	ObsOffsetTable           = 40  // [40, 80)   cards face-up on the table
	ObsOffsetMyTeamCaptures  = 80  // [80, 120)  capture pile of the mover's team
	ObsOffsetOppTeamCaptures = 120 // [120, 160) capture pile of the other team
	ObsOffsetLastPlaced      = 160 // [160, 200) the caída reference card, or all zeros
	ObsOffsetMyTeamScore     = 200 // mover's team score
	ObsOffsetOppTeamScore    = 201 // other team's score
	ObsOffsetCurrentPlayer   = 202 // seat index of the mover (0..3)
	ObsOffsetDeckRemaining   = 203 // cards left to deal
	ObsOffsetPlaysThisDeal   = 204 // [204, 208) turns taken this deal by seats 0,1,2,3
	ObsSizeISMCTS            = 208

	// Action feature layout — ACTION_FEAT_DIM in src/ppo/env.py.
	ActionFeatureCapturesStart = 40 // [40, 80) bitvec of the cards the action captures
	ActionFeatureIsCapture     = 80 // 1.0 iff the action captures
	ActionFeatureIsCaida       = 81 // 1.0 iff the action is a caída (+2)
	ActionFeatureIsLimpia      = 82 // 1.0 iff the action is a limpia (+2)
	// Index 83 is reserved by the exporter and always left at 0.
)

// obsLayoutISMCTS is OBS_KEYS_ISMCTS made explicit: the key, its offset and its width
// in the flattened observation, in the order the network reads them. Adding a row here
// without re-exporting the model silently shifts every later block, which is why
// TestFlattenObservationLayout pins each offset.
var obsLayoutISMCTS = []struct {
	key    string
	offset int
	width  int
}{
	{"my_hand", ObsOffsetMyHand, CardVecSize},
	{"table", ObsOffsetTable, CardVecSize},
	{"my_team_captures", ObsOffsetMyTeamCaptures, CardVecSize},
	{"opp_team_captures", ObsOffsetOppTeamCaptures, CardVecSize},
	{"last_placed", ObsOffsetLastPlaced, CardVecSize},
	{"my_team_score", ObsOffsetMyTeamScore, 1},
	{"opp_team_score", ObsOffsetOppTeamScore, 1},
	{"current_player", ObsOffsetCurrentPlayer, 1},
	{"deck_remaining", ObsOffsetDeckRemaining, 1},
	{"plays_this_deal", ObsOffsetPlaysThisDeal, NumPlayers},
}

// Reasons reported by scorePlay. They are awarded as a list (one play can be both a
// caída and a limpia in principle), so always test membership with hasReason.
const (
	ReasonCaida  = "caída (+2)"
	ReasonLimpia = "limpia (+2)"
)

// hasReason reports whether `reason` is one of the reasons scorePlay returned.
//
// The Python pipelines once compared the LIST of reasons against a single string with
// `==` — always false — which silently zeroed the caída/limpia rewards and the
// is_caida/is_limpia features. Keep the membership test (see src/ppo/utils.py).
func hasReason(reasons []string, reason string) bool {
	return slices.Contains(reasons, reason)
}

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
		// A caída wins over a limpia: a play that is both scores once, as a caída.
		if detectCaida(played, captureSet, lastPlaced) {
			points += 2
			reasons = append(reasons, ReasonCaida)
		} else if detectLimpia(captureSet, table) && nonCaidaAllowed {
			points += 2
			reasons = append(reasons, ReasonLimpia)
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

	// PlaysThisDeal counts the turns each seat has taken in the current deal; it is
	// the last four floats of the flattened observation (ObsOffsetPlaysThisDeal). It
	// advances on every step and resets when a new deal starts (redeal / newRound),
	// mirroring `state.plays_this_deal` in src/ppo/env.py.
	PlaysThisDeal [NumPlayers]int

	// DealsDealt counts the deals handed out in this match (the opening deal is 1). The
	// web client animates a deal whenever it changes, so it must survive a restart of the
	// deal's own bookkeeping — it is a match counter, not a per-deal one.
	DealsDealt int
}

// RoundEnd describes a deal that ended and paid the carton, so a client can SHOW what
// happened before the state moves on: the piles that were counted, the cards left on the
// table when the deal ended, and the points each team was paid. Everything here is public
// information by the time it is sent (the deal is over and the scores are updated).
type RoundEnd struct {
	// Captured holds the two piles that were counted, card by card, so a client can show
	// the counting rather than just its result.
	Captured [2][]int
	// Counted is len(Captured[t]) — the number actually counted.
	Counted [2]int
	// Carton is the points the carton paid each team (0 for a team under 20 cards).
	Carton [2]int
	// TableCards is what was still on the table when the deal ended, i.e. the cards the
	// new round clears. Sent so a client can show them being swept without having to
	// remember the previous snapshot.
	TableCards []int
}

func (r RoundEnd) tableString() string {
	parts := make([]string, 0, len(r.TableCards))
	for _, card := range r.TableCards {
		parts = append(parts, fmt.Sprintf("%d", card))
	}
	return strings.Join(parts, ",")
}

type Env struct {
	rng   *rand.Rand
	state *State
}

func NewEnv(seed int64) *Env {
	var rng *rand.Rand
	if seed != 0 {
		rng = rand.New(rand.NewSource(seed))
	} else {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}
	return &Env{
		rng: rng,
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
		PlaysThisDeal: [NumPlayers]int{},
		DealsDealt:    1,
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
	// The observation reports the turns of the CURRENT deal.
	s.PlaysThisDeal = [NumPlayers]int{}
	s.DealsDealt++
}

func (e *Env) newRound() {
	s := e.state
	newDeck := ShuffledDeck(e.rng)
	s.Hands, s.Deck = deal(newDeck)
	s.CardsCaptured = [2][]Card{}
	s.Table = nil
	s.LastPlaced = nil
	s.CurrentPlayer = (s.CurrentPlayer + 1) % NumPlayers
	s.PlaysThisDeal = [NumPlayers]int{}
	s.DealsDealt++
}

// CardIndices maps cards to their 0-39 index, in order (the wire format for hands, the
// table and the captured piles).
func CardIndices(cards []Card) []int {
	out := make([]int, len(cards))
	for i, card := range cards {
		out[i] = card.Index()
	}
	return out
}

func cardsToBitVec(cards []Card) []float32 {
	vec := make([]float32, 40)
	for _, card := range cards {
		vec[card.Index()] = 1
	}
	return vec
}

func (e *Env) Observe(player int) Observation {
	return ObserveState(e.state, player)
}

// ObserveState is `Env.Observe` for a bare state.
//
// The search needs it: it explores SAMPLED worlds that are not the env's own state (the
// hidden hands are dealt from the unseen pool), and it must build the same observation the
// policy was trained on for such a world. Keeping one implementation is what stops the
// search's inputs from drifting away from the engine's.
func ObserveState(s *State, player int) Observation {
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
		// Turns taken this deal by seats 0..3 — required by the ISMCTS layout the ONNX
		// model uses. ("played_this_deal" above is only kept for the legacy layout,
		// which is the union of both capture piles and not part of the model input.)
		"plays_this_deal": playsToFloats(s.PlaysThisDeal),
	}
	return obs
}

func playsToFloats(plays [NumPlayers]int) []float32 {
	out := make([]float32, NumPlayers)
	for seat, count := range plays {
		out[seat] = float32(count)
	}
	return out
}

type Observation map[string][]float32

// FlattenObservation concatenates an Observation into the flat 208-float vector the
// ONNX policy takes, in the fixed OBS_KEYS_ISMCTS order (src/ppo/utils.py). The
// obsLayoutISMCTS table above documents which slice of the result is which block, and
// ObsSizeISMCTS is the total the model declares as its observation input.
//
// A key that is missing (or shorter than its block) leaves its slice zeroed, so an
// incomplete observation degrades to "nothing known" instead of shifting every block
// after it, which is the failure mode that would silently feed the network nonsense.
func FlattenObservation(obs Observation) []float32 {
	flat := make([]float32, ObsSizeISMCTS)
	for _, block := range obsLayoutISMCTS {
		copy(flat[block.offset:block.offset+block.width], obs[block.key])
	}
	return flat
}

// ---------------------------------------------------------------------------
// Candidate-action features (the policy head's second input)
// ---------------------------------------------------------------------------

// ActionFeatureMatrix is the [MaxActionSlots][ActionFeatureDim] feature block of one
// decision: row i describes the i-th legal action the caller offered.
type ActionFeatureMatrix [MaxActionSlots][ActionFeatureDim]float32

// ActionSlotMask is the [MaxActionSlots] mask sent alongside it: 1.0 on the slots that
// hold a legal action, 0.0 everywhere else.
type ActionSlotMask [MaxActionSlots]float32

// Flatten returns the row-major block the ONNX `action_features` tensor ([1,64,84])
// expects.
func (m *ActionFeatureMatrix) Flatten() []float32 {
	flat := make([]float32, 0, MaxActionSlots*ActionFeatureDim)
	for slot := 0; slot < MaxActionSlots; slot++ {
		flat = append(flat, m[slot][:]...)
	}
	return flat
}

// Flatten returns the mask as the ONNX `action_mask` tensor ([1,64]).
func (m *ActionSlotMask) Flatten() []float32 {
	return m[:]
}

// ActionFeatures describes ONE candidate action in ActionFeatureDim floats, exactly as
// `action_features` in src/ppo/env.py does:
//
//	[0:40]   the card being played (one bit at Card.Index())
//	[40:80]  the cards this action captures (empty when it only places a card)
//	[80]     is_capture
//	[81]     is_caida  — the capture includes LastPlaced and the played card has its rank
//	[82]     is_limpia — the capture takes every card on the table, and the acting team
//	         is below MaxAchievablePointsByNonCaidaActions (38), because above that a
//	         limpia no longer pays
//	[83]     reserved, always 0
//
// The two bonuses are read from scorePlay rather than re-derived, so the features and
// the rules cannot drift apart; scorePlay also settles caída-over-limpia precedence.
func ActionFeatures(action Action, s *State) []float32 {
	feats := make([]float32, ActionFeatureDim)

	feats[action.Card.Index()] = 1
	for _, card := range action.Capture {
		feats[ActionFeatureCapturesStart+card.Index()] = 1
	}
	if len(action.Capture) > 0 {
		feats[ActionFeatureIsCapture] = 1
	}

	team := teamByPlayer[s.CurrentPlayer]
	_, reasons := scorePlay(action.Card, action.Capture, s.Table, s.LastPlaced,
		s.Scores[team] < MaxAchievablePointsByNonCaidaActions)

	if hasReason(reasons, ReasonCaida) {
		feats[ActionFeatureIsCaida] = 1
	}
	if hasReason(reasons, ReasonLimpia) {
		feats[ActionFeatureIsLimpia] = 1
	}

	return feats
}

// BuildActionFeatures packs `legal` into the feature block and mask of one decision:
// slot i gets the features of legal[i] and mask[i] is set, so the model's answer (a
// slot) can be used directly as `legal[slot]`.
//
// It returns the number of slots filled — min(len(legal), MaxActionSlots). A caller
// with more legal actions than slots must not query the model at all: the policy can
// only see, and therefore only choose between, the first MaxActionSlots of them.
func BuildActionFeatures(legal []Action, s *State) (ActionFeatureMatrix, ActionSlotMask, int) {
	var feats ActionFeatureMatrix
	var mask ActionSlotMask

	slots := min(len(legal), MaxActionSlots)
	for i := 0; i < slots; i++ {
		copy(feats[i][:], ActionFeatures(legal[i], s))
		mask[i] = 1
	}

	return feats, mask, slots
}

// ActionForSlot maps a slot predicted by the model back to the action to play. Slots
// are indices into the very same list BuildActionFeatures was given, so an in-range
// slot is legal by construction; out of range means the model/caller disagree and the
// caller must fall back.
func ActionForSlot(slot int, legal []Action) (Action, bool) {
	if slot < 0 || slot >= len(legal) {
		return Action{}, false
	}
	return legal[slot], true
}

// ActionIsLegal reports whether `action` is one of `legal`, comparing actions the way the
// rules do (the played card plus the captured cards as a set).
//
// It is the hand-aware check: LegalActionsForCard answers "what could this card do",
// ignoring whose hand it is in, so a caller that must not commit an illegal turn — for
// instance before feeding that turn to the policy's memory — has to test the resolved
// action against the acting player's own LegalActions().
func ActionIsLegal(action Action, legal []Action) bool {
	return actionExists(action, legal)
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
	actions := make([]Action, 0)

	// according to the thrown card, there's a set of actions that can be taken
	captures := ValidCaptures(played, s.Table)
	if len(captures) > 0 {
		for _, capSet := range captures {
			action := Action{Card: played, Capture: canonicalCards(capSet)}
			actions = append(actions, action)
		}
	} else {
		// only if no captures are available, the card alone is a valid action
		actions = append(actions, Action{Card: played})
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

// Moves one step on the game environment.
// Returns (observation, done, info, error)
func (e *Env) step(action Action) (Observation, bool, map[string]any, error) {
	s := e.state
	if s.Done {
		return nil, true, nil, fmt.Errorf("game is already over")
	}

	player := s.CurrentPlayer
	legal := e.legalActions()
	if !actionExists(action, legal) {
		return nil, false, nil, fmt.Errorf("illegal action: %s", action.String())
	}

	hand := s.Hands[player]
	found := slices.Contains(hand, action.Card)
	if !found {
		return nil, false, nil, fmt.Errorf("player %d does not hold %s", player, action.Card)
	}

	s.Hands[player] = removeCard(s.Hands[player], action.Card)
	// Count the turn for this deal: `plays_this_deal` is part of the observation.
	s.PlaysThisDeal[player]++

	team := teamByPlayer[player]
	points, reasons := scorePlay(action.Card, action.Capture, s.Table, s.LastPlaced, s.Scores[team] < MaxAchievablePointsByNonCaidaActions)
	info := map[string]any{
		"player":  player,
		"team":    team,
		"action":  action.String(),
		"points":  points,
		"reasons": reasons,
	}

	if len(action.Capture) > 0 {
		for _, captured := range canonicalCards(action.Capture) {
			s.Table = removeCard(s.Table, captured)
			s.CardsCaptured[team] = append(s.CardsCaptured[team], captured)
		}
		s.CardsCaptured[team] = append(s.CardsCaptured[team], action.Card)
		s.LastPlaced = nil
	} else {
		s.Table = append(s.Table, action.Card)
		s.LastPlaced = &action.Card
	}

	s.Scores[team] += points

	if s.Scores[team] >= WinScore {
		s.Done = true
		winner := team
		s.Winner = &winner
		info["winner"] = team
		log.Printf("Team %d won\n", team+1)
		return e.Observe(player), true, info, nil
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
			// Count the carton, and publish what happened so a client can show it: the
			// piles are about to be cleared for the next round, and this is the last
			// moment they exist.
			roundEnd := RoundEnd{TableCards: CardIndices(s.Table)}
			for teamIdx := 0; teamIdx < 2; teamIdx++ {
				roundEnd.Captured[teamIdx] = CardIndices(s.CardsCaptured[teamIdx])
				roundEnd.Counted[teamIdx] = len(s.CardsCaptured[teamIdx])

				cartonPoints := len(s.CardsCaptured[teamIdx]) - 19
				if cartonPoints <= 0 {
					continue
				}
				cartonPoints += 5
				if cartonPoints%2 != 0 {
					cartonPoints++
				}
				roundEnd.Carton[teamIdx] = cartonPoints

				s.Scores[teamIdx] = min(s.Scores[teamIdx]+cartonPoints, MaxAchievablePointsByNonCaidaActions)
			}
			info["roundEnd"] = roundEnd

			if s.Scores[0] >= WinScore || s.Scores[1] >= WinScore {
				s.Done = true
				winner := 0
				if s.Scores[1] > s.Scores[0] {
					winner = 1
				}
				s.Winner = &winner
				info["winner"] = winner
				log.Printf("Team %d won\n", team+1)
				return e.Observe(s.CurrentPlayer), true, info, nil
			}

			e.newRound()
			info["newRound"] = true
		}
	}

	return e.Observe(s.CurrentPlayer), false, info, nil
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

func (e *Env) Step(action Action) (Observation, bool, map[string]any, error) {
	return e.step(action)
}

// EncodeAction and DecodeAction translate the LEGACY flat action id used by the old
// DAgger/ONNX model (`card_index * 41 + capture_option_index`, a 1640-wide policy head).
// The current policy does not use them: it scores each candidate action by its features
// (ActionFeatures) and answers with a slot into the caller's own legal-action list, so
// nothing has to agree on a global action numbering any more. Kept for the legacy
// tooling that still speaks the flat encoding.
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

// DecodeAction is the inverse of EncodeAction, for the legacy flat action id. See the
// note above: the feature-based policy does not need it.
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
