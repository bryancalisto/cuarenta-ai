package infer

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"cua/internal/game"

	"github.com/joho/godotenv"
	ort "github.com/yalue/onnxruntime_go"
)

// The tests that need the engine call startEngine(modelPath, sharedLibPath) directly
// instead of StartEngine(): StartEngine reads config.GetEnv(), which loads .env from the
// process's working directory (the package directory under `go test`) and exits the test
// binary when the file is not there.

func TestSelectSlot(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		nLegal int
		want   int
	}{
		{"argmax over the legal prefix", []float32{0.1, 2.0, 5.0}, 2, 1},
		{"illegal slots cannot win", []float32{-1e9, -1e9, 3.0}, 2, 0},
		{"last legal slot wins", []float32{1, 2, 3}, 3, 2},
		{"ties keep the first slot", []float32{1, 1, 1}, 3, 0},
		{"negative logits", []float32{-5, -2, -9}, 3, 1},
		{"a single legal action leaves no choice", []float32{-3, 9}, 1, 0},
		{"nLegal past the tensor is clamped", []float32{1, 4}, 10, 1},
		{"no legal action falls back to slot 0", []float32{1, 2}, 0, 0},
		{"empty logits fall back to slot 0", nil, 4, 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := SelectSlot(test.logits, test.nLegal); got != test.want {
				t.Errorf("SelectSlot(%v, %d) = %d, expected %d", test.logits, test.nLegal, got, test.want)
			}
		})
	}
}

// TestPredictActionSlotLoadsTheExportedModel runs the real ONNX model exported by
// scripts/pth2onnx.py end to end: it feeds tensors built by the game package and checks
// that the 5-in / 4-out contract holds and that the answer is a slot into the same
// []Action list. It skips when the model or the onnxruntime shared library is not on the
// machine — the layout itself is covered by internal/game's tests.
func TestPredictActionSlotLoadsTheExportedModel(t *testing.T) {
	service := requireEngine(t)

	// Geometry the game package builds against.
	if obsSize != game.ObsSizeISMCTS || nSlots != game.MaxActionSlots || featDim != game.ActionFeatureDim {
		t.Fatalf("model geometry is observation %d, %d slots x %d features; the game builds %d, %d x %d",
			obsSize, nSlots, featDim, game.ObsSizeISMCTS, game.MaxActionSlots, game.ActionFeatureDim)
	}
	if lstmStateSize != 128 {
		t.Errorf("recurrent state is %d floats, expected 128 for lstm_h/lstm_c [1,1,128]", lstmStateSize)
	}

	// The allocated tensors must have exactly the shapes scripts/pth2onnx.py declares
	// (dynamic batch axes pinned to 1).
	wantInputShapes := map[string]ort.Shape{
		inputObservation:    {1, 1, 208},
		inputActionFeatures: {1, 64, 84},
		inputActionMask:     {1, 64},
		inputLSTMH:          {1, 1, 128},
		inputLSTMC:          {1, 1, 128},
	}
	for name, want := range wantInputShapes {
		if got := inputTensors[name].GetShape(); !got.Equals(want) {
			t.Errorf("input %q has shape %s, expected %s", name, got, want)
		}
	}
	wantOutputShapes := map[string]ort.Shape{
		outputActionLogits: {1, 64},
		outputValue:        {1},
		outputNewLSTMH:     {1, 1, 128},
		outputNewLSTMC:     {1, 1, 128},
	}
	for name, want := range wantOutputShapes {
		if got := outputTensors[name].GetShape(); !got.Equals(want) {
			t.Errorf("output %q has shape %s, expected %s", name, got, want)
		}
	}

	// A real position, described by the game package exactly as playAITurn does it.
	env := game.NewEnv(12345)
	env.Reset()
	state := env.State()
	legal := env.LegalActions()
	obs := game.FlattenObservation(env.Observe(state.CurrentPlayer))
	feats, mask, nLegal := game.BuildActionFeatures(legal, state)
	if nLegal != len(legal) {
		t.Fatalf("packed %d slots for %d legal actions", nLegal, len(legal))
	}

	const gameID = "test-game"
	slot, value, err := service.PredictActionSlot(gameID, obs, feats.Flatten(), mask.Flatten(), nLegal)
	if err != nil {
		t.Fatalf("PredictActionSlot failed: %v", err)
	}
	if slot < 0 || slot >= nLegal {
		t.Fatalf("slot %d is outside the %d legal actions", slot, nLegal)
	}
	action, ok := game.ActionForSlot(slot, legal)
	if !ok {
		t.Fatalf("slot %d does not resolve to a legal action", slot)
	}
	t.Logf("decision 1: %d legal actions, slot %d -> %s, value %v", nLegal, slot, action, value)
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		t.Errorf("value is not finite: %v", value)
	}

	// The recurrent state was written back from new_lstm_h / new_lstm_c: this is the
	// part of the contract that silently degrades into "no memory" if the output
	// tensors are read in the wrong order.
	h, c := lstmState(gameID)
	if len(h) != lstmStateSize || len(c) != lstmStateSize {
		t.Fatalf("stored recurrent state is %d/%d floats, expected %d", len(h), len(c), lstmStateSize)
	}
	if isZeroFloats(h) || isZeroFloats(c) {
		t.Errorf("the model returned an all-zero recurrent state; new_lstm_h/new_lstm_c do not look connected (h=%v...)", h[:4])
	}

	// A second decision of the same game, one turn later, must work with the fed-back
	// state and still answer with a legal slot.
	if _, _, _, err := env.Step(action); err != nil {
		t.Fatalf("stepping the game failed: %v", err)
	}
	state = env.State()
	legal = env.LegalActions()
	obs = game.FlattenObservation(env.Observe(state.CurrentPlayer))
	feats, mask, nLegal = game.BuildActionFeatures(legal, state)
	slot, value, err = service.PredictActionSlot(gameID, obs, feats.Flatten(), mask.Flatten(), nLegal)
	if err != nil {
		t.Fatalf("second PredictActionSlot failed: %v", err)
	}
	if slot < 0 || slot >= nLegal {
		t.Fatalf("second slot %d is outside the %d legal actions", slot, nLegal)
	}
	t.Logf("decision 2: %d legal actions, slot %d, value %v", nLegal, slot, value)

	// A new game must start from a zeroed state, not from the state of another game.
	otherH, otherC := lstmState("another-game")
	if !isZeroFloats(otherH) || !isZeroFloats(otherC) {
		t.Error("a game that has not moved yet must start with a zeroed recurrent state")
	}

	// Misuse must be reported, never guessed at.
	if _, _, err := service.PredictActionSlot(gameID, obs[:len(obs)-1], feats.Flatten(), mask.Flatten(), nLegal); err == nil {
		t.Error("a wrong-length observation must be rejected")
	}
	if _, _, err := service.PredictActionSlot(gameID, obs, feats.Flatten()[:10], mask.Flatten(), nLegal); err == nil {
		t.Error("a wrong-length action feature block must be rejected")
	}
	if _, _, err := service.PredictActionSlot(gameID, obs, feats.Flatten(), mask.Flatten(), nSlots+1); err == nil {
		t.Error("more legal actions than slots must be rejected")
	}
	if _, _, err := service.PredictActionSlot(gameID, obs, feats.Flatten(), mask.Flatten(), 0); err == nil {
		t.Error("an empty legal-action list must be rejected")
	}

	// ClearGameState forgets the game: nothing is left behind and the next decision of
	// the same id starts fresh.
	service.ClearGameState(gameID)
	lstmStateMu.Lock()
	_, stillThere := lstmStates[gameID]
	lstmStateMu.Unlock()
	if stillThere {
		t.Error("ClearGameState left the recurrent state behind")
	}
	if _, _, err := service.PredictActionSlot(gameID, obs, feats.Flatten(), mask.Flatten(), nLegal); err != nil {
		t.Fatalf("PredictActionSlot after ClearGameState failed: %v", err)
	}
}

func testModelPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(packageDir(t), "..", "..", "models", "ppo_bootstrap.onnx")
}

// requireEngine loads the exported model once per test binary (the ONNX runtime and the
// session are process-wide) and skips the test when the model or the shared library is
// missing, so the suite still runs on a machine without onnxruntime.
func requireEngine(t *testing.T) *InferenceService {
	t.Helper()
	return requireEngineAt(t, testModelPath(t))
}

// requireEngineAt is requireEngine for a specific model file, which is how the ISMCTS
// parity test loads a second model type in the same process (the service tears the previous
// session down when the path changes).
func requireEngineAt(t *testing.T, modelPath string) *InferenceService {
	t.Helper()

	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("exported model not available at %s: %v", modelPath, err)
	}

	libPath := onnxRuntimeLibPath(t)
	if libPath == "" {
		t.Skip("ONNXRUNTIME_SHARED_LIBRARY_PATH is not set (check backend/.env)")
	}
	if _, err := os.Stat(libPath); err != nil {
		t.Skipf("onnxruntime shared library not available at %s: %v", libPath, err)
	}

	service := NewInferenceService()
	if err := service.startEngine(modelPath, libPath); err != nil {
		// Only a missing/broken runtime is a skip; a model that does not match the
		// contract is a real failure and must be visible.
		if strings.Contains(err.Error(), "initialize onnx runtime") {
			t.Skipf("onnxruntime could not be loaded (%v)", err)
		}
		t.Fatalf("loading %s failed: %v", modelPath, err)
	}

	return service
}

// TestPredictActionSlotDrivesTheGameAcrossDeals runs the policy for a couple of hundred
// decisions of one game, so the shapes, the recurrent state hand-off and the deal
// boundaries (redeal / new round inside Env.step) are exercised together.
func TestPredictActionSlotDrivesTheGameAcrossDeals(t *testing.T) {
	service := requireEngine(t)

	const maxDecisions = 200
	const gameID = "driven-game"

	env := game.NewEnv(4242)
	env.Reset()

	dealBoundaries := 0
	previousPlays := 0
	decisions := 0

	for decisions < maxDecisions && !env.State().Done {
		state := env.State()
		legal := env.LegalActions()
		if len(legal) == 0 {
			t.Fatalf("decision %d has no legal actions", decisions)
		}

		obs := game.FlattenObservation(env.Observe(state.CurrentPlayer))
		feats, mask, nLegal := game.BuildActionFeatures(legal, state)

		slot, value, err := service.PredictActionSlot(gameID, obs, feats.Flatten(), mask.Flatten(), nLegal)
		if err != nil {
			t.Fatalf("decision %d failed: %v", decisions, err)
		}
		action, ok := game.ActionForSlot(slot, legal)
		if !ok {
			t.Fatalf("decision %d returned slot %d outside the %d legal actions", decisions, slot, nLegal)
		}
		if math.IsNaN(float64(value)) {
			t.Fatalf("decision %d returned NaN value", decisions)
		}

		if _, _, _, err := env.Step(action); err != nil {
			t.Fatalf("decision %d played an illegal action %s: %v", decisions, action, err)
		}

		next := env.State()
		plays := 0
		for _, count := range next.PlaysThisDeal {
			plays += count
		}
		if plays < previousPlays {
			dealBoundaries++ // the per-deal counters restarted: a new deal was dealt
		}
		previousPlays = plays
		decisions++
	}

	t.Logf("drove %d decisions through %d deal boundaries (scores %v, done=%v)",
		decisions, dealBoundaries, env.State().Scores, env.State().Done)
	if dealBoundaries == 0 {
		t.Error("no deal boundary was crossed: the observation's plays_this_deal is not tracking turns")
	}
}

// onnxRuntimeLibPath looks for the shared library the server uses, first in the
// environment and then in backend/.env.
func onnxRuntimeLibPath(t *testing.T) string {
	t.Helper()

	if path := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); path != "" {
		return path
	}
	values, err := godotenv.Read(filepath.Join(packageDir(t), "..", "..", ".env"))
	if err != nil {
		return ""
	}
	return values["ONNXRUNTIME_SHARED_LIBRARY_PATH"]
}

func packageDir(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate this test file")
	}
	return filepath.Dir(file)
}

func isZeroFloats(values []float32) bool {
	for _, v := range values {
		if v != 0 {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Per-turn recurrent advance (human turns included)
// ---------------------------------------------------------------------------

// recordedTurn is the decision point of one turn of a match, captured so it can be
// replayed into the policy later.
type recordedTurn struct {
	seat   int
	obs    []float32
	feats  []float32
	mask   []float32
	nLegal int
	action game.Action
}

// readState copies the recurrent state stored for a game; a game that has not been fed
// yet reads as the zero state it would be initialized with.
func readState(gameID string) [2][]float32 {
	lstmStateMu.Lock()
	defer lstmStateMu.Unlock()

	stored, ok := lstmStates[gameID]
	if !ok {
		return [2][]float32{make([]float32, lstmStateSize), make([]float32, lstmStateSize)}
	}
	return [2][]float32{
		append([]float32(nil), stored[0]...),
		append([]float32(nil), stored[1]...),
	}
}

// stateMaxDiff is the largest absolute difference between two (h, c) pairs.
func stateMaxDiff(a, b [2][]float32) float32 {
	var worst float32
	for part := range a {
		if len(a[part]) != len(b[part]) {
			return float32(math.Inf(1))
		}
		for i := range a[part] {
			if d := float32(math.Abs(float64(a[part][i] - b[part][i]))); d > worst {
				worst = d
			}
		}
	}
	return worst
}

// TestAdvanceRecurrentStateKeepsTheNewState covers the plain contract of the new method:
// it runs a forward pass for a turn the network does not decide and stores the state the
// model returned, feeding it back on the next call.
func TestAdvanceRecurrentStateKeepsTheNewState(t *testing.T) {
	service := requireEngine(t)

	env := game.NewEnv(7)
	env.Reset()
	state := env.State()
	legal := env.LegalActions()
	obs := game.FlattenObservation(env.Observe(state.CurrentPlayer))
	feats, mask, nLegal := game.BuildActionFeatures(legal, state)

	const gameID = "advance-state"
	before := readState(gameID)
	if stateMaxDiff(before, [2][]float32{make([]float32, lstmStateSize), make([]float32, lstmStateSize)}) != 0 {
		t.Fatal("a game that has not been fed must start from a zeroed state")
	}

	if err := service.AdvanceRecurrentState(gameID, obs, feats.Flatten(), mask.Flatten(), nLegal); err != nil {
		t.Fatalf("AdvanceRecurrentState failed: %v", err)
	}
	after := readState(gameID)
	if len(after[0]) != lstmStateSize || len(after[1]) != lstmStateSize {
		t.Fatalf("stored state is %d/%d floats, expected %d", len(after[0]), len(after[1]), lstmStateSize)
	}
	if stateMaxDiff(before, after) == 0 {
		t.Error("the human turn did not change the stored recurrent state")
	}

	// A second advance moves it again: the state the model returned is fed back in.
	if err := service.AdvanceRecurrentState(gameID, obs, feats.Flatten(), mask.Flatten(), nLegal); err != nil {
		t.Fatalf("second AdvanceRecurrentState failed: %v", err)
	}
	third := readState(gameID)
	if stateMaxDiff(after, third) == 0 {
		t.Error("the second advance produced the same state: the new state is not being fed back")
	}

	// Another game is unaffected by it.
	other := readState("advance-state-other")
	if stateMaxDiff(other, before) != 0 {
		t.Error("a different game's recurrent state was modified")
	}

	service.ClearGameState(gameID)
	if diff := stateMaxDiff(readState(gameID), before); diff != 0 {
		t.Error("ClearGameState did not drop the recurrent state")
	}
}

// TestEveryTurnChainMatchesTheInterleavedChain is the fidelity test for the per-turn
// advance: driving a match where the other seats' turns go through
// AdvanceRecurrentState while the AI's turns go through PredictActionSlot must leave the
// AI's hidden state identical to a chain where EVERY turn is fed in order — and different
// from the chain that skipped the other seats' turns (which is what the backend used to
// do, and would hand the network a state it never saw in training).
func TestEveryTurnChainMatchesTheInterleavedChain(t *testing.T) {
	service := requireEngine(t)

	const aiSeat = 0
	const maxTurns = 40
	const interleavedID = "interleaved-match"
	const everyTurnID = "every-turn-match"
	const aiOnlyID = "ai-only-match"

	// Pass 1: play the match. The AI seat is decided by the network, every other seat
	// plays the first legal action; each turn's decision point is recorded, together with
	// the state the AI held when it decided.
	turns := make([]recordedTurn, 0, maxTurns)
	interleaved := make(map[int][2][]float32, maxTurns)

	env := game.NewEnv(20260101)
	env.Reset()
	for len(turns) < maxTurns && !env.State().Done {
		state := env.State()
		seat := state.CurrentPlayer
		legal := env.LegalActions()
		if len(legal) == 0 {
			t.Fatalf("no legal actions at turn %d", len(turns))
		}
		obs := game.FlattenObservation(env.Observe(seat))
		feats, mask, nLegal := game.BuildActionFeatures(legal, state)

		interleaved[len(turns)] = readState(interleavedID)

		var next game.Action
		if seat == aiSeat {
			slot, _, err := service.PredictActionSlot(interleavedID, obs, feats.Flatten(), mask.Flatten(), nLegal)
			if err != nil {
				t.Fatalf("turn %d: PredictActionSlot failed: %v", len(turns), err)
			}
			picked, ok := game.ActionForSlot(slot, legal)
			if !ok {
				t.Fatalf("turn %d: slot %d is outside the %d legal actions", len(turns), slot, nLegal)
			}
			next = picked
		} else {
			if err := service.AdvanceRecurrentState(interleavedID, obs, feats.Flatten(), mask.Flatten(), nLegal); err != nil {
				t.Fatalf("turn %d: AdvanceRecurrentState failed: %v", len(turns), err)
			}
			next = legal[0]
		}

		turns = append(turns, recordedTurn{
			seat:   seat,
			obs:    obs,
			feats:  feats.Flatten(),
			mask:   mask.Flatten(),
			nLegal: nLegal,
			action: next,
		})

		if _, _, _, err := env.Step(next); err != nil {
			t.Fatalf("turn %d: stepping %s failed: %v", len(turns)-1, next, err)
		}
	}

	// Pass 2: replay the same decision points, feeding EVERY turn.
	everyTurn := make(map[int][2][]float32, len(turns))
	for i, turn := range turns {
		everyTurn[i] = readState(everyTurnID)
		if err := service.AdvanceRecurrentState(everyTurnID, turn.obs, turn.feats, turn.mask, turn.nLegal); err != nil {
			t.Fatalf("reference turn %d failed: %v", i, err)
		}
	}

	// Pass 3: same replay, feeding ONLY the AI seat's turns (the old behaviour).
	aiOnly := make(map[int][2][]float32, len(turns))
	for i, turn := range turns {
		aiOnly[i] = readState(aiOnlyID)
		if turn.seat != aiSeat {
			continue
		}
		if err := service.AdvanceRecurrentState(aiOnlyID, turn.obs, turn.feats, turn.mask, turn.nLegal); err != nil {
			t.Fatalf("ai-only turn %d failed: %v", i, err)
		}
	}

	aiDecisions := 0
	differences := 0
	for i, turn := range turns {
		if turn.seat != aiSeat {
			continue
		}
		aiDecisions++

		// What the AI had when it decided (interleaved) must be the every-turn chain.
		if diff := stateMaxDiff(interleaved[i], everyTurn[i]); diff > 1e-4 {
			t.Errorf("turn %d: the AI's hidden state differs from the every-turn chain by %v", i, diff)
		}
		// And skipping the other seats' turns really does change it, so this test can
		// fail: the assertion above is not vacuous.
		if diff := stateMaxDiff(aiOnly[i], everyTurn[i]); diff > 1e-3 {
			differences++
		}
	}

	t.Logf("compared %d AI decisions over %d turns; %d of them would differ if the other seats' turns were skipped",
		aiDecisions, len(turns), differences)
	if aiDecisions < 2 {
		t.Fatalf("needed at least 2 AI decisions to compare, got %d", aiDecisions)
	}
	if differences == 0 {
		t.Error("skipping the other seats' turns changed nothing: this test would not catch a skipped turn")
	}
}

// TestDecisionValidationIsSharedByBothEntryPoints: the two entry points take the same
// arguments and must reject the same misuse — and a rejected call must leave the stored
// recurrent state untouched, or a turn that never happened would enter the memory.
func TestDecisionValidationIsSharedByBothEntryPoints(t *testing.T) {
	service := requireEngine(t)

	env := game.NewEnv(11)
	env.Reset()
	state := env.State()
	legal := env.LegalActions()
	obs := game.FlattenObservation(env.Observe(state.CurrentPlayer))
	feats, mask, nLegal := game.BuildActionFeatures(legal, state)
	flatFeats, flatMask := feats.Flatten(), mask.Flatten()

	const gameID = "validation-game"
	if err := service.AdvanceRecurrentState(gameID, obs, flatFeats, flatMask, nLegal); err != nil {
		t.Fatalf("seeding the recurrent state failed: %v", err)
	}
	seeded := readState(gameID)

	cases := []struct {
		name   string
		obs    []float32
		feats  []float32
		mask   []float32
		nLegal int
	}{
		{"more legal actions than slots", obs, flatFeats, flatMask, game.MaxActionSlots + 1},
		{"no legal actions", obs, flatFeats, flatMask, 0},
		{"negative legal actions", obs, flatFeats, flatMask, -1},
		{"short observation", obs[:len(obs)-1], flatFeats, flatMask, nLegal},
		{"empty observation", nil, flatFeats, flatMask, nLegal},
		{"short feature block", obs, flatFeats[:len(flatFeats)-1], flatMask, nLegal},
		{"short mask", obs, flatFeats, flatMask[:1], nLegal},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if _, _, err := service.PredictActionSlot(gameID, test.obs, test.feats, test.mask, test.nLegal); err == nil {
				t.Error("PredictActionSlot accepted invalid inputs")
			}
			if err := service.AdvanceRecurrentState(gameID, test.obs, test.feats, test.mask, test.nLegal); err == nil {
				t.Error("AdvanceRecurrentState accepted invalid inputs")
			}
			if diff := stateMaxDiff(readState(gameID), seeded); diff != 0 {
				t.Errorf("a rejected call changed the recurrent state (largest difference %v)", diff)
			}
		})
	}
}
