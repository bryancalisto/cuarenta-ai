package game

import (
	"math/rand"
	"slices"
	"testing"
)

// These tests pin the Go side of the ONNX policy contract: the 208-dim observation
// layout, the [64][84] candidate-action feature block with its mask, and the slot ->
// action mapping the AI path uses. A layout regression here is invisible at runtime
// (the model happily consumes a reordered vector and plays worse), so every index is
// asserted explicitly.

// craftedPolicyState is a small hand-made position used by several tests:
//
//	seat 2 (team 0) is to move, holding the Cinco of Corazones Rojos,
//	the table holds the Siete of Brillos, and there is one card left in the deck.
func craftedPolicyState() (*Env, *State) {
	handCard := Card{Rank: Cinco, Suit: CorazonesRojos}   // index 14
	tableCard := Card{Rank: Siete, Suit: Brillos}         // index 6
	myCapture := Card{Rank: As, Suit: Treboles}           // index 30
	oppCapture := Card{Rank: King, Suit: CorazonesNegros} // index 29
	lastPlaced := Card{Rank: Tres, Suit: Brillos}         // index 2
	deckCard := Card{Rank: Dos, Suit: Brillos}            // index 1
	state := &State{
		Hands: map[int][]Card{
			0: {},
			1: {},
			2: {handCard},
			3: {},
		},
		Table:         []Card{tableCard},
		Deck:          []Card{deckCard},
		Scores:        [2]int{7, 11},
		CardsCaptured: [2][]Card{{myCapture}, {oppCapture}},
		CurrentPlayer: 2,
		LastPlaced:    &lastPlaced,
		PlaysThisDeal: [NumPlayers]int{1, 2, 3, 4},
	}
	return &Env{state: state}, state
}

func TestObsSizeMatchesTheExportedModel(t *testing.T) {
	if ObsSizeISMCTS != 208 {
		t.Errorf("ObsSizeISMCTS = %d, the exported models declare 208", ObsSizeISMCTS)
	}
	if MaxActionSlots != 64 || ActionFeatureDim != 84 {
		t.Errorf("action head geometry is %d slots x %d features, the models declare 64 x 84", MaxActionSlots, ActionFeatureDim)
	}
	if ActionFeatureCapturesStart != CardVecSize {
		t.Errorf("capture features must start right after the played-card bitvec: %d != %d", ActionFeatureCapturesStart, CardVecSize)
	}
}

// TestObservationLayoutIsContiguous checks obsLayoutISMCTS against OBS_KEYS_ISMCTS:
// the blocks must be in order and add up to exactly the model's input width. Adding or
// reordering a key without re-exporting the model fails here.
func TestObservationLayoutIsContiguous(t *testing.T) {
	wantKeys := []string{
		"my_hand", "table", "my_team_captures", "opp_team_captures", "last_placed",
		"my_team_score", "opp_team_score", "current_player", "deck_remaining",
		"plays_this_deal",
	}

	if len(obsLayoutISMCTS) != len(wantKeys) {
		t.Fatalf("layout has %d blocks, OBS_KEYS_ISMCTS has %d", len(obsLayoutISMCTS), len(wantKeys))
	}

	next := 0
	for i, block := range obsLayoutISMCTS {
		if block.key != wantKeys[i] {
			t.Errorf("block %d is %q, expected %q (OBS_KEYS_ISMCTS order)", i, block.key, wantKeys[i])
		}
		if block.offset != next {
			t.Errorf("block %q starts at %d, expected %d (blocks must be contiguous)", block.key, block.offset, next)
		}
		next += block.width
	}

	if next != ObsSizeISMCTS {
		t.Errorf("layout covers %d floats, expected %d", next, ObsSizeISMCTS)
	}
}

func TestFlattenObservationLayout(t *testing.T) {
	env, state := craftedPolicyState()

	obs := FlattenObservation(env.Observe(state.CurrentPlayer))
	if len(obs) != ObsSizeISMCTS {
		t.Fatalf("flattened observation has %d floats, expected %d", len(obs), ObsSizeISMCTS)
	}

	// The expected vector, written one index at a time so a reader can verify every
	// offset against the layout comment in cuarenta.go.
	want := make([]float32, ObsSizeISMCTS)
	want[ObsOffsetMyHand+14] = 1          // Cinco de Corazones Rojos, the mover's only card
	want[ObsOffsetTable+6] = 1            // Siete de Brillos on the table
	want[ObsOffsetMyTeamCaptures+30] = 1  // As de Tréboles captured by team 0 (seat 2's team)
	want[ObsOffsetOppTeamCaptures+29] = 1 // Rey de Corazones Negros captured by team 1
	want[ObsOffsetLastPlaced+2] = 1       // Tres de Brillos was the last card placed
	want[ObsOffsetMyTeamScore] = 7        // mover is seat 2 -> team 0
	want[ObsOffsetOppTeamScore] = 11      // the other team
	want[ObsOffsetCurrentPlayer] = 2      // seat of the mover
	want[ObsOffsetDeckRemaining] = 1      // one card left in the deck
	want[ObsOffsetPlaysThisDeal+0] = 1    // turns taken this deal: seats 0..3
	want[ObsOffsetPlaysThisDeal+1] = 2
	want[ObsOffsetPlaysThisDeal+2] = 3
	want[ObsOffsetPlaysThisDeal+3] = 4

	for i, got := range obs {
		if got != want[i] {
			t.Fatalf("observation[%d] = %v, expected %v\n(offset %d is the first difference; "+
				"check the block order against OBS_KEYS_ISMCTS)", i, got, want[i], i)
		}
	}

	// Derived from the vector: the mover's hand block holds exactly the hand size.
	handOnes := 0
	for _, v := range obs[ObsOffsetMyHand : ObsOffsetMyHand+CardVecSize] {
		if v == 1 {
			handOnes++
		}
	}
	if handOnes != len(state.Hands[state.CurrentPlayer]) {
		t.Errorf("my_hand block has %d ones, the mover holds %d cards", handOnes, len(state.Hands[state.CurrentPlayer]))
	}
}

func TestObservationKeepsLegacyKeyAndToleratesMissingBlocks(t *testing.T) {
	env, state := craftedPolicyState()
	obs := env.Observe(state.CurrentPlayer)

	// The ISMCTS layout needs plays_this_deal; played_this_deal is still published for
	// the legacy layout, but it is NOT part of the model input.
	if _, ok := obs["plays_this_deal"]; !ok {
		t.Error("observation is missing plays_this_deal")
	}
	if len(obs["plays_this_deal"]) != NumPlayers {
		t.Errorf("plays_this_deal has %d entries, expected %d", len(obs["plays_this_deal"]), NumPlayers)
	}
	if _, ok := obs["played_this_deal"]; !ok {
		t.Error("observation is missing played_this_deal (legacy layout key)")
	}

	// A key that is absent leaves its block zeroed instead of shifting the rest.
	partial := Observation{"my_hand": obs["my_hand"]}
	flat := FlattenObservation(partial)
	if len(flat) != ObsSizeISMCTS {
		t.Fatalf("flattened partial observation has %d floats, expected %d", len(flat), ObsSizeISMCTS)
	}
	if flat[ObsOffsetMyHand+14] != 1 {
		t.Error("my_hand block was not copied for a partial observation")
	}
	for _, offset := range []int{ObsOffsetTable, ObsOffsetMyTeamScore, ObsOffsetPlaysThisDeal} {
		if flat[offset] != 0 {
			t.Errorf("missing key left %v at offset %d, expected 0", flat[offset], offset)
		}
	}
}

func TestPlaysThisDealTracksAndResetsEachDeal(t *testing.T) {
	// Four cards of different ranks, none of which can capture the others, so each
	// turn is a plain placement.
	hand := []Card{
		{Rank: Cinco, Suit: Brillos}, // index 4
		{Rank: Siete, Suit: Brillos}, // index 6
		{Rank: Jack, Suit: Brillos},  // index 7
		{Rank: King, Suit: Brillos},  // index 9
	}

	deck := make([]Card, 0, 36)
	for _, card := range BuildDeck() {
		if !slices.Contains(hand, card) {
			deck = append(deck, card)
		}
	}

	env := &Env{
		rng: rand.New(rand.NewSource(7)),
		state: &State{
			Hands: map[int][]Card{
				0: {hand[0]},
				1: {hand[1]},
				2: {hand[2]},
				3: {hand[3]},
			},
			Deck:          deck,
			CardsCaptured: [2][]Card{},
			CurrentPlayer: 0,
		},
	}

	// One step counts for the acting seat only, and shows up in the observation.
	if _, _, _, err := env.Step(Action{Card: hand[0]}); err != nil {
		t.Fatalf("step failed: %v", err)
	}
	if got := env.State().PlaysThisDeal; got != [NumPlayers]int{1, 0, 0, 0} {
		t.Fatalf("PlaysThisDeal after one turn = %v, expected [1 0 0 0]", got)
	}
	flat := FlattenObservation(env.Observe(env.State().CurrentPlayer))
	if flat[ObsOffsetPlaysThisDeal] != 1 || flat[ObsOffsetPlaysThisDeal+1] != 0 {
		t.Errorf("observation plays_this_deal = %v, expected [1 0 0 0]", flat[ObsOffsetPlaysThisDeal:ObsOffsetPlaysThisDeal+NumPlayers])
	}

	// Play the deal out: with every hand empty the deck (36 cards) still holds another
	// deal, so the env redeals and the per-deal counters start over.
	for seat := 1; seat < NumPlayers; seat++ {
		if _, _, _, err := env.Step(Action{Card: hand[seat]}); err != nil {
			t.Fatalf("step for seat %d failed: %v", seat, err)
		}
	}

	state := env.State()
	if got := state.PlaysThisDeal; got != [NumPlayers]int{} {
		t.Errorf("PlaysThisDeal after the redeal = %v, expected all zeros", got)
	}
	for seat := 0; seat < NumPlayers; seat++ {
		if len(state.Hands[seat]) != HandSize {
			t.Errorf("seat %d holds %d cards after the redeal, expected %d", seat, len(state.Hands[seat]), HandSize)
		}
	}
	if len(state.Deck) != 36-NumPlayers*HandSize {
		t.Errorf("deck has %d cards after the redeal, expected %d", len(state.Deck), 36-NumPlayers*HandSize)
	}
}

// TestBuildActionFeaturesShapeMaskAndSlotOrder is the core contract test: the feature
// block must have one row per legal action, in the SAME order as the []Action list the
// model's answer will index, and the mask must mark exactly those rows.
func TestBuildActionFeaturesShapeMaskAndSlotOrder(t *testing.T) {
	tableCard := Card{Rank: Cinco, Suit: Brillos}        // index 4
	capturing := Card{Rank: Cinco, Suit: CorazonesRojos} // index 14, captures the table card
	placing := Card{Rank: Siete, Suit: CorazonesNegros}  // index 26, captures nothing

	env := &Env{state: &State{
		Hands:         map[int][]Card{0: {capturing, placing}},
		Table:         []Card{tableCard},
		CardsCaptured: [2][]Card{},
		CurrentPlayer: 0,
	}}

	legal := env.LegalActions()
	if len(legal) != 2 {
		t.Fatalf("expected 2 legal actions (one capture + one placement), got %d", len(legal))
	}

	feats, mask, slots := BuildActionFeatures(legal, env.State())
	if slots != len(legal) {
		t.Fatalf("packed %d slots for %d legal actions", slots, len(legal))
	}

	// Shape: [MaxActionSlots][ActionFeatureDim], asserted both at compile time ...
	var shape [64][84]float32 = feats
	_ = shape
	// ... and at run time against the constants the model was exported with.
	if len(feats) != 64 || len(feats[0]) != 84 {
		t.Fatalf("feature block is [%d][%d], expected [64][84]", len(feats), len(feats[0]))
	}
	if len(mask) != 64 {
		t.Fatalf("mask has %d entries, expected 64", len(mask))
	}

	// The mask marks exactly the packed slots, nothing else.
	ones := 0
	for slot := 0; slot < MaxActionSlots; slot++ {
		want := float32(0)
		if slot < slots {
			want = 1
		}
		if mask[slot] != want {
			t.Errorf("mask[%d] = %v, expected %v", slot, mask[slot], want)
		}
		if mask[slot] == 1 {
			ones++
		}
	}
	if ones != len(legal) {
		t.Errorf("mask marks %d slots, expected %d", ones, len(legal))
	}

	for i, action := range legal {
		// Slot i describes legal[i]: this is the slot -> action mapping.
		if feats[i][action.Card.Index()] != 1 {
			t.Errorf("slot %d does not set the played-card bit %d (%s)", i, action.Card.Index(), action)
		}
		cardBits := 0
		for card := 0; card < CardVecSize; card++ {
			if feats[i][card] == 1 {
				cardBits++
			}
		}
		if cardBits != 1 {
			t.Errorf("slot %d sets %d played-card bits, expected exactly 1", i, cardBits)
		}

		isCapture := len(action.Capture) > 0
		if got := feats[i][ActionFeatureIsCapture]; (got == 1) != isCapture {
			t.Errorf("slot %d [80] = %v, the action captures: %v", i, got, isCapture)
		}

		// [40:80] holds exactly the captured cards (empty for a placement).
		captureBits := 0
		for card := 0; card < CardVecSize; card++ {
			if feats[i][ActionFeatureCapturesStart+card] != 0 {
				captureBits++
			}
		}
		if captureBits != len(action.Capture) {
			t.Errorf("slot %d marks %d captured cards, expected %d", i, captureBits, len(action.Capture))
		}
		for _, card := range action.Capture {
			if feats[i][ActionFeatureCapturesStart+card.Index()] != 1 {
				t.Errorf("slot %d does not mark %s as captured", i, card)
			}
		}

		if feats[i][ActionFeatureDim-1] != 0 {
			t.Errorf("slot %d sets the reserved feature [83]", i)
		}
	}

	// Slots beyond the packed ones stay empty: they are dead weight the model masks out.
	for slot := slots; slot < MaxActionSlots; slot++ {
		for at, v := range feats[slot] {
			if v != 0 {
				t.Fatalf("unused slot %d has %v at feature %d, expected 0", slot, v, at)
			}
		}
	}

	// Flatten must be row-major: slot-major rows of ActionFeatureDim, which is the
	// [1, 64, 84] tensor the model declares. A transposed block would still have the
	// right length, so compare it against the rows it must be made of.
	flat := feats.Flatten()
	if len(flat) != MaxActionSlots*ActionFeatureDim {
		t.Fatalf("flattened features have %d floats, expected %d", len(flat), MaxActionSlots*ActionFeatureDim)
	}
	wantFlat := make([]float32, 0, MaxActionSlots*ActionFeatureDim)
	for slot := 0; slot < MaxActionSlots; slot++ {
		wantFlat = append(wantFlat, feats[slot][:]...)
	}
	if !slices.Equal(flat, wantFlat) {
		t.Error("flattened features are not the rows of the matrix in slot order (row-major)")
	}
	for i := range legal {
		if flat[i*ActionFeatureDim+legal[i].Card.Index()] != 1 {
			t.Errorf("flattened features[%d] (slot %d, played card %d) is not set", i*ActionFeatureDim+legal[i].Card.Index(), i, legal[i].Card.Index())
		}
	}
	if flat[slots*ActionFeatureDim] != 0 {
		t.Errorf("flattened features start a non-empty unused slot at %d", slots*ActionFeatureDim)
	}

	flatMask := mask.Flatten()
	if len(flatMask) != MaxActionSlots {
		t.Fatalf("flattened mask has %d floats, expected %d", len(flatMask), MaxActionSlots)
	}
	if flatMask[0] != 1 || flatMask[slots] != 0 {
		t.Errorf("flattened mask = %v at the packed boundary", flatMask[:slots+1])
	}
}

func TestActionFeaturesCaptureAndCaidaFlags(t *testing.T) {
	lastPlaced := Card{Rank: Cinco, Suit: Brillos}    // index 4
	played := Card{Rank: Cinco, Suit: CorazonesRojos} // index 14, same rank as the last placed card

	env := &Env{state: &State{
		Hands:         map[int][]Card{0: {played}},
		Table:         []Card{lastPlaced},
		Scores:        [2]int{10, 4},
		CardsCaptured: [2][]Card{},
		CurrentPlayer: 0,
		LastPlaced:    &lastPlaced,
	}}

	action := Action{Card: played, Capture: []Card{lastPlaced}}
	feats := ActionFeatures(action, env.State())

	if len(feats) != ActionFeatureDim {
		t.Fatalf("feature vector has %d floats, expected %d", len(feats), ActionFeatureDim)
	}
	if feats[played.Index()] != 1 {
		t.Errorf("played-card bit %d is not set", played.Index())
	}
	if feats[ActionFeatureCapturesStart+lastPlaced.Index()] != 1 {
		t.Errorf("captured-card bit %d is not set", lastPlaced.Index())
	}
	if feats[ActionFeatureIsCapture] != 1 {
		t.Error("[80] is_capture should be 1")
	}
	if feats[ActionFeatureIsCaida] != 1 {
		t.Error("[81] is_caida should be 1: the capture takes the last-placed card and the ranks match")
	}
	// Caída wins over limpia: this capture also clears the table, but it pays a caída.
	if feats[ActionFeatureIsLimpia] != 0 {
		t.Error("[82] is_limpia should be 0 when the action is a caída")
	}
	if feats[83] != 0 {
		t.Error("[83] is reserved and must stay 0")
	}

	// The rules agree with the features (scorePlay is the single source of truth here).
	if points, reasons := scorePlay(action.Card, action.Capture, env.State().Table, env.State().LastPlaced, true); points != 2 || !hasReason(reasons, ReasonCaida) {
		t.Errorf("scorePlay says points=%d reasons=%v, expected a caída", points, reasons)
	}

	// A placement sets no capture feature at all.
	place := ActionFeatures(Action{Card: played}, env.State())
	if place[ActionFeatureIsCapture] != 0 || place[ActionFeatureIsCaida] != 0 || place[ActionFeatureIsLimpia] != 0 {
		t.Error("a placement must not set the capture/caída/limpia features")
	}
	for card := 0; card < CardVecSize; card++ {
		if place[ActionFeatureCapturesStart+card] != 0 {
			t.Errorf("a placement marks card %d as captured", card)
		}
	}
}

func TestActionFeaturesLimpiaNeedsScoreBelow38(t *testing.T) {
	tableCard := Card{Rank: Cinco, Suit: Brillos} // index 4
	played := Card{Rank: Cinco, Suit: CorazonesRojos}
	action := Action{Card: played, Capture: []Card{tableCard}}

	newState := func(team0Score int, lastPlaced *Card) *State {
		return &State{
			Hands:         map[int][]Card{0: {played}},
			Table:         []Card{tableCard},
			Scores:        [2]int{team0Score, 0},
			CardsCaptured: [2][]Card{},
			CurrentPlayer: 0,
			LastPlaced:    lastPlaced,
		}
	}

	// Clearing the whole table below 38 points is a limpia.
	feats := ActionFeatures(action, newState(0, nil))
	if feats[ActionFeatureIsLimpia] != 1 {
		t.Error("[82] is_limpia should be 1: the capture takes every card on the table")
	}
	if feats[ActionFeatureIsCaida] != 0 {
		t.Error("[81] is_caida should be 0 without a last-placed card")
	}

	// At 38 points the acting team can no longer score a limpia (the cap), so the
	// feature is off — the model must not be told a bonus that the rules will not pay.
	feats = ActionFeatures(action, newState(MaxAchievablePointsByNonCaidaActions, nil))
	if feats[ActionFeatureIsLimpia] != 0 {
		t.Error("[82] is_limpia should be 0 while the acting team is at 38 points")
	}

	// A caída still pays at 38 (it is what wins the game).
	feats = ActionFeatures(action, newState(MaxAchievablePointsByNonCaidaActions, &tableCard))
	if feats[ActionFeatureIsCaida] != 1 {
		t.Error("[81] is_caida should stay 1 at 38 points")
	}
}

// TestActionForSlotKeepsTheAIPathLegal mirrors what playAITurn does with the model's
// answer: the slot indexes the legal list it was built from, so it must always resolve
// to one of those legal actions.
func TestActionForSlotKeepsTheAIPathLegal(t *testing.T) {
	env := &Env{state: &State{
		Hands: map[int][]Card{0: {
			{Rank: Cinco, Suit: CorazonesRojos},  // captures the five on the table
			{Rank: Siete, Suit: CorazonesNegros}, // places
		}},
		Table:         []Card{{Rank: Cinco, Suit: Brillos}},
		CardsCaptured: [2][]Card{},
		CurrentPlayer: 0,
	}}
	legal := env.LegalActions()
	if len(legal) < 2 {
		t.Fatalf("expected at least 2 legal actions, got %d", len(legal))
	}

	for slot := range legal {
		picked, ok := ActionForSlot(slot, legal)
		if !ok {
			t.Fatalf("slot %d was rejected although it is in range", slot)
		}
		if !sameAction(picked, legal[slot]) {
			t.Errorf("slot %d resolved to %s, expected %s", slot, picked, legal[slot])
		}
		if !actionExists(picked, legal) {
			t.Errorf("slot %d resolved to an action the game does not consider legal: %s", slot, picked)
		}
	}

	for _, slot := range []int{-1, len(legal), MaxActionSlots} {
		if picked, ok := ActionForSlot(slot, legal); ok {
			t.Errorf("out-of-range slot %d was accepted as %s", slot, picked)
		}
	}
	if _, ok := ActionForSlot(0, nil); ok {
		t.Error("no legal action can be resolved from an empty list")
	}
}
