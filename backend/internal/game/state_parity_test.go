package game

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"testing"
)

// ---------------------------------------------------------------------------
// Cross-language parity with the Python reference (src/ppo/env.py)
// ---------------------------------------------------------------------------
//
// The other tests in this package pin the policy's input layout against this package's
// own constants, so they catch a Go regression but not a DISAGREEMENT WITH PYTHON. That
// is the dangerous case: if the card index convention, the hand order, the last_placed
// normalisation or the score scaling differed from `src/ppo/env.py`, every Go test would
// still pass while the model received a well-formed but WRONG vector — and would quietly
// play a worse game, which no assertion would notice.
//
// `scripts/state_parity_fixture.py` therefore dumps real positions taken from complete
// games together with exactly what Python computes for them: the 208-dim observation,
// the legal actions IN ORDER, the [64, 84] candidate-action features and the [64] mask.
// This test rebuilds each position in Go from the fixture's own description of it and
// compares all four.
//
// Python is the reference implementation — it is what trained the network — so a
// mismatch found here is a Go bug to fix, never a fixture to adjust.

const stateFixtureTolerance = 1e-6

// parityFixture is backend/internal/game/testdata/state_fixture.json.
type parityFixture struct {
	Generator     string       `json:"generator"`
	ObsKeys       []string     `json:"obs_keys"`
	ObsDim        int          `json:"obs_dim"`
	ActionFeatDim int          `json:"action_feat_dim"`
	NSlots        int          `json:"n_slots"`
	Cases         []parityCase `json:"cases"`
}

type parityCase struct {
	Name                   string        `json:"name"`
	Game                   int           `json:"game"`
	Step                   int           `json:"step"`
	Player                 int           `json:"player"`
	State                  parityState   `json:"state"`
	ExpectedObs            []float32     `json:"expected_obs"`
	ExpectedLegal          []parityLegal `json:"expected_legal"`
	ExpectedActionFeatures []float32     `json:"expected_action_features"`
	ExpectedActionMask     []float32     `json:"expected_action_mask"`
}

type parityState struct {
	Hands         [4][]parityCard `json:"hands"`
	Table         []parityCard    `json:"table"`
	Deck          []parityCard    `json:"deck"`
	Scores        [2]int          `json:"scores"`
	CardsCaptured [2][]parityCard `json:"cards_captured"`
	CurrentPlayer int             `json:"current_player"`
	LastPlaced    *parityCard     `json:"last_placed"`
	PlaysThisDeal [4]int          `json:"plays_this_deal"`
	DecksConsumed int             `json:"decks_consumed"`
	Done          bool            `json:"done"`
}

type parityLegal struct {
	Card    parityCard   `json:"card"`
	Capture []parityCard `json:"capture"` // null when the action only places a card
}

// parityCard is a card in the fixture: [suit name, rank value].
type parityCard struct {
	suitName string
	rank     int
}

func (c *parityCard) UnmarshalJSON(data []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if len(raw) != 2 {
		return fmt.Errorf("a card is [suit name, rank], got %s", data)
	}
	if err := json.Unmarshal(raw[0], &c.suitName); err != nil {
		return fmt.Errorf("card suit: %w", err)
	}
	return json.Unmarshal(raw[1], &c.rank)
}

// String renders a fixture card for diagnostics, in the fixture's own vocabulary.
func (c parityCard) String() string {
	return fmt.Sprintf("%s#%d", c.suitName, c.rank)
}

// The fixture names its suits and ranks so that neither side depends on the other's enum
// order; these tables are what turn those names into this package's constants. They are
// deliberately explicit (rather than, say, parsing the name into an integer) so the test
// exercises the name -> enum mapping, and an unknown name is a failure rather than a
// silent zero suit.
var suitByName = map[string]Suit{
	"BRILLOS":          Brillos,
	"CORAZONES_ROJOS":  CorazonesRojos,
	"CORAZONES_NEGROS": CorazonesNegros,
	"TREBOLES":         Treboles,
}

var rankByValue = map[int]Rank{
	1:  As,
	2:  Dos,
	3:  Tres,
	4:  Cuatro,
	5:  Cinco,
	6:  Seis,
	7:  Siete,
	11: Jack,
	12: Queen,
	13: King,
}

func TestStateFixtureMatchesPythonPolicyInputs(t *testing.T) {
	fixture := loadStateFixture(t)
	checkFixtureShape(t, fixture)
	checkFixtureCoverage(t, fixture)

	maxObsDiff := float32(0)
	maxFeatureDiff := float32(0)
	legalMatched := 0

	for _, testCase := range fixture.Cases {
		t.Run(testCase.Name, func(t *testing.T) {
			state := stateFromFixture(t, testCase)
			env := &Env{rng: rand.New(rand.NewSource(1)), state: state}

			// The fixture captures the observation of the player about to move.
			if testCase.Player != testCase.State.CurrentPlayer {
				t.Fatalf("the fixture says player %d is to move but current_player is %d",
					testCase.Player, testCase.State.CurrentPlayer)
			}

			// 1) The observation.
			obs := FlattenObservation(env.Observe(testCase.Player))
			maxObsDiff = max(maxObsDiff, comparePolicyVector(t, "observation", obs,
				testCase.ExpectedObs, obsBlockName))

			// 2) The legal actions, in order, with their capture sets.
			legal := env.LegalActions()
			if compareLegalActions(t, legal, testCase.ExpectedLegal) {
				legalMatched++
			}

			// 3) The candidate-action features and the mask they are packed with.
			feats, mask, slots := BuildActionFeatures(legal, state)
			if slots != len(legal) {
				t.Fatalf("packed %d slots for %d legal actions", slots, len(legal))
			}
			maxFeatureDiff = max(maxFeatureDiff, comparePolicyVector(t, "action_features",
				feats.Flatten(), testCase.ExpectedActionFeatures, actionFeatureName))
			comparePolicyVector(t, "action_mask", mask.Flatten(), testCase.ExpectedActionMask, nil)
		})
	}

	// Reported after the comparisons so a wrong header cannot hide their result: the
	// header is metadata the generator writes, the vectors above are what Python computed.
	checkFixtureObsDim(t, fixture)

	t.Logf("parity with Python over %d positions: max observation difference %g, max action-feature difference %g, "+
		"%d/%d legal-action lists matched in order and content",
		len(fixture.Cases), maxObsDiff, maxFeatureDiff, legalMatched, len(fixture.Cases))
}

// checkFixtureShape compares the fixture's declared contract with this package's. The
// per-case vectors are the authoritative part of the fixture (they are what Python
// actually computed), so they are checked in the cases themselves; this function checks
// the header the generator writes beside them.
func checkFixtureShape(t *testing.T, fixture parityFixture) {
	t.Helper()

	if len(fixture.Cases) == 0 {
		t.Fatal("the fixture contains no cases")
	}

	// The key list is the layout's real specification: same names, same order.
	keys := make([]string, 0, len(obsLayoutISMCTS))
	for _, block := range obsLayoutISMCTS {
		keys = append(keys, block.key)
	}
	if !slices.Equal(fixture.ObsKeys, keys) {
		t.Fatalf("the fixture flattens the observation in the order %v, this package uses %v",
			fixture.ObsKeys, keys)
	}
	if fixture.ActionFeatDim != ActionFeatureDim || fixture.NSlots != MaxActionSlots {
		t.Fatalf("the fixture describes %d features and %d slots; this package builds %d and %d",
			fixture.ActionFeatDim, fixture.NSlots, ActionFeatureDim, MaxActionSlots)
	}

	// obs_dim is NOT checked against ObsSizeISMCTS here on purpose: a mismatch is
	// reported by TestStateFixtureMatchesPythonPolicyInputs after every case has been
	// compared, so a wrong header cannot hide the parity result of the vectors.
}

// checkFixtureObsDim reports a fixture header that disagrees with the vectors it carries.
func checkFixtureObsDim(t *testing.T, fixture parityFixture) {
	t.Helper()

	if fixture.ObsDim == ObsSizeISMCTS {
		return
	}

	// Both numbers come from Python, so name the more likely culprit precisely: the
	// generator writes `len(OBS_KEYS_ISMCTS)` (the number of KEYS) where it means
	// `OBS_SIZE_ISMCTS` (the flattened width).
	width := -1
	for _, testCase := range fixture.Cases {
		if width == -1 {
			width = len(testCase.ExpectedObs)
		}
		if len(testCase.ExpectedObs) != width {
			t.Errorf("the fixture's observation vectors have inconsistent widths (%d and %d)",
				width, len(testCase.ExpectedObs))
			return
		}
	}

	t.Errorf("the fixture header declares obs_dim=%d while its own observation vectors hold "+
		"%d values (and this package builds %d); scripts/state_parity_fixture.py writes "+
		"`len(OBS_KEYS_ISMCTS)`, the number of observation KEYS, where it means "+
		"`OBS_SIZE_ISMCTS`, the flattened width. The vectors themselves are the ones it "+
		"computed and they are compared above; fix the header (or drop it) and regenerate "+
		"with `python3 scripts/state_parity_fixture.py`",
		fixture.ObsDim, width, ObsSizeISMCTS)
}

// checkFixtureCoverage guards the guard: a stale or degenerate fixture must not be able to
// pass. The generator refuses to write such a file, and this fails if one is committed
// anyway.
func checkFixtureCoverage(t *testing.T, fixture parityFixture) {
	t.Helper()

	withCapture := 0
	withTable := 0
	withFreeSlots := 0
	withCaida := 0
	withLimpia := 0

	for _, testCase := range fixture.Cases {
		if len(testCase.ExpectedObs) != ObsSizeISMCTS {
			t.Fatalf("%s: the fixture observation has %d values, expected %d",
				testCase.Name, len(testCase.ExpectedObs), ObsSizeISMCTS)
		}
		if len(testCase.ExpectedActionFeatures) != MaxActionSlots*ActionFeatureDim {
			t.Fatalf("%s: the fixture feature block has %d values, expected %d",
				testCase.Name, len(testCase.ExpectedActionFeatures), MaxActionSlots*ActionFeatureDim)
		}
		if len(testCase.ExpectedActionMask) != MaxActionSlots {
			t.Fatalf("%s: the fixture mask has %d values, expected %d",
				testCase.Name, len(testCase.ExpectedActionMask), MaxActionSlots)
		}
		if len(testCase.ExpectedLegal) > MaxActionSlots {
			t.Fatalf("%s: %d legal actions do not fit %d slots",
				testCase.Name, len(testCase.ExpectedLegal), MaxActionSlots)
		}

		if slices.ContainsFunc(testCase.ExpectedLegal, func(action parityLegal) bool {
			return len(action.Capture) > 0
		}) {
			withCapture++
		}
		if len(testCase.State.Table) > 0 {
			withTable++
		}
		if len(testCase.ExpectedLegal) < MaxActionSlots {
			withFreeSlots++
		}
		for slot := range testCase.ExpectedLegal {
			if testCase.ExpectedActionFeatures[slot*ActionFeatureDim+ActionFeatureIsCaida] == 1 {
				withCaida++
			}
			if testCase.ExpectedActionFeatures[slot*ActionFeatureDim+ActionFeatureIsLimpia] == 1 {
				withLimpia++
			}
		}
	}

	switch {
	case withCapture == 0:
		t.Error("no fixture case offers a capture: regenerate it with scripts/state_parity_fixture.py")
	case withTable == 0:
		t.Error("no fixture case has cards on the table: regenerate it with scripts/state_parity_fixture.py")
	case withFreeSlots == 0:
		t.Error("every fixture case fills all 64 slots: regenerate it with scripts/state_parity_fixture.py")
	}

	// Coverage, reported rather than required: the generator guarantees captures, a
	// non-empty table and free slots, but not that it sampled a caída or a limpia.
	t.Logf("fixture coverage: %d cases, %d with a capture, %d with cards on the table, "+
		"%d with free slots, %d caída flags, %d limpia flags",
		len(fixture.Cases), withCapture, withTable, withFreeSlots, withCaida, withLimpia)
	if withCaida == 0 {
		t.Log("note: no caída among the sampled positions, so the is_caida feature is not compared against Python")
	}
	if withLimpia == 0 {
		t.Log("note: no limpia among the sampled positions, so the is_limpia feature is not compared against Python")
	}
}

func loadStateFixture(t *testing.T) parityFixture {
	t.Helper()

	path := stateFixturePath(t)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("cannot read %s (%v); regenerate it with "+
			"`python3 scripts/state_parity_fixture.py`", path, err)
	}

	var fixture parityFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("cannot parse %s: %v", path, err)
	}
	return fixture
}

// stateFromFixture rebuilds the game state the fixture describes, field by field. The
// hands keep the fixture's card order: the legal-action order (and therefore the meaning
// of every policy slot) follows the order the cards are held in.
func stateFromFixture(t *testing.T, testCase parityCase) *State {
	t.Helper()

	state := &State{
		Hands:         make(map[int][]Card, NumPlayers),
		Table:         cardsFromFixture(t, testCase.Name, testCase.State.Table),
		Deck:          cardsFromFixture(t, testCase.Name, testCase.State.Deck),
		Scores:        [2]int{testCase.State.Scores[0], testCase.State.Scores[1]},
		CardsCaptured: [2][]Card{},
		CurrentPlayer: testCase.State.CurrentPlayer,
		DecksConsumed: testCase.State.DecksConsumed,
		Done:          testCase.State.Done,
		PlaysThisDeal: [NumPlayers]int{
			testCase.State.PlaysThisDeal[0],
			testCase.State.PlaysThisDeal[1],
			testCase.State.PlaysThisDeal[2],
			testCase.State.PlaysThisDeal[3],
		},
	}

	for seat := 0; seat < NumPlayers; seat++ {
		state.Hands[seat] = cardsFromFixture(t, testCase.Name, testCase.State.Hands[seat])
	}
	for team := 0; team < 2; team++ {
		state.CardsCaptured[team] = cardsFromFixture(t, testCase.Name, testCase.State.CardsCaptured[team])
	}
	if testCase.State.LastPlaced != nil {
		lastPlaced := cardFromFixture(t, testCase.Name, *testCase.State.LastPlaced)
		state.LastPlaced = &lastPlaced
	}

	return state
}

func cardsFromFixture(t *testing.T, caseName string, cards []parityCard) []Card {
	t.Helper()

	if len(cards) == 0 {
		return nil
	}
	out := make([]Card, len(cards))
	for i, card := range cards {
		out[i] = cardFromFixture(t, caseName, card)
	}
	return out
}

func cardFromFixture(t *testing.T, caseName string, card parityCard) Card {
	t.Helper()

	suit, ok := suitByName[card.suitName]
	if !ok {
		t.Fatalf("%s: unknown suit %q in the fixture", caseName, card.suitName)
	}
	rank, ok := rankByValue[card.rank]
	if !ok {
		t.Fatalf("%s: unknown rank %d in the fixture", caseName, card.rank)
	}
	return Card{Rank: rank, Suit: suit}
}

// compareLegalActions checks the legal-action list the AI indexes the model's answer
// with: the same number of actions, in the same order, each playing the same card and
// capturing the same set of cards. A capture set has no order of its own, so both sides
// are compared as sets (rendered by captureKey). It reports whether the whole list
// matched, so the test can state how many positions agreed on the action order.
func compareLegalActions(t *testing.T, got []Action, want []parityLegal) bool {
	t.Helper()

	if len(got) != len(want) {
		t.Errorf("the game offers %d legal actions, Python had %d", len(got), len(want))
		for i, action := range got {
			t.Logf("  go   legal[%d] = %s", i, action)
		}
		for i, action := range want {
			t.Logf("  py   legal[%d] = %s -> capture %v", i, action.Card, action.Capture)
		}
		return false
	}

	matched := true
	for i := range want {
		wantCard := cardFromFixture(t, fmt.Sprintf("legal[%d]", i), want[i].Card)
		if got[i].Card != wantCard {
			t.Errorf("legal[%d] plays %s, Python plays %s", i, got[i].Card, wantCard)
			matched = false
		}

		wantCapture := captureKey(cardsFromFixture(t, fmt.Sprintf("legal[%d]", i), want[i].Capture))
		gotCapture := captureKey(got[i].Capture)
		if !slices.Equal(gotCapture, wantCapture) {
			t.Errorf("legal[%d] (%s) captures %v, Python captures %v", i, got[i].Card, gotCapture, wantCapture)
			matched = false
		}
	}

	return matched
}

// comparePolicyVector compares one flat policy input against the fixture's and reports
// the first mismatching index (named, for the observation and the feature block), both
// values and the largest difference over the whole vector — a bare "not equal" on 5376
// values would be useless.
func comparePolicyVector(t *testing.T, what string, got, want []float32, describe func(int) string) float32 {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s has %d values, the fixture has %d", what, len(got), len(want))
	}

	maxDiff := float32(0)
	firstMismatch := -1
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		if firstMismatch == -1 && diff > stateFixtureTolerance {
			firstMismatch = i
		}
	}

	if firstMismatch >= 0 {
		at := fmt.Sprintf("[%d]", firstMismatch)
		if describe != nil {
			at = fmt.Sprintf("[%d] (%s)", firstMismatch, describe(firstMismatch))
		}
		t.Errorf("%s%s = %v, Python's is %v; largest difference over the %d values is %v",
			what, at, got[firstMismatch], want[firstMismatch], len(got), maxDiff)
	}

	return maxDiff
}

// obsBlockName names the observation block a flat index falls in, so a mismatch reads as
// "my_hand[3]" instead of "index 83".
func obsBlockName(index int) string {
	for _, block := range obsLayoutISMCTS {
		if index < block.offset || index >= block.offset+block.width {
			continue
		}
		if block.width == 1 {
			return block.key
		}
		return fmt.Sprintf("%s[%d]", block.key, index-block.offset)
	}
	return "outside the layout"
}

// actionFeatureName names the part of an 84-wide feature vector an index falls in.
func actionFeatureName(index int) string {
	slot, at := index/ActionFeatureDim, index%ActionFeatureDim

	var field string
	switch {
	case at < CardVecSize:
		field = fmt.Sprintf("played card %d", at)
	case at < ActionFeatureCapturesStart*2:
		field = fmt.Sprintf("captured card %d", at-ActionFeatureCapturesStart)
	case at == ActionFeatureIsCapture:
		field = "is_capture"
	case at == ActionFeatureIsCaida:
		field = "is_caida"
	case at == ActionFeatureIsLimpia:
		field = "is_limpia"
	default:
		field = "reserved"
	}
	return fmt.Sprintf("slot %d %s", slot, field)
}

// captureKey renders a set of cards for comparison and diagnostics: each card as
// "rank+suit(index)" — the same form on both sides, since the fixture's cards go through
// cardFromFixture first — sorted so that the comparison does not depend on the order the
// capture set happens to be stored in.
func captureKey(cards []Card) []string {
	out := make([]string, len(cards))
	for i, card := range cards {
		out[i] = fmt.Sprintf("%s(%d)", card, card.Index())
	}
	sort.Strings(out)
	return out
}

// stateFixturePath resolves the fixture relative to this test file, so the test does not
// depend on the working directory.
func stateFixturePath(t *testing.T) string {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate this test file")
	}
	return filepath.Join(filepath.Dir(file), "testdata", "state_fixture.json")
}
