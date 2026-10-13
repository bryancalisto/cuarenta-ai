package infer

import (
	"fmt"
	"log"
	"sync"

	"cua/internal/config"
	"cua/internal/game"
	"cua/internal/search"

	ort "github.com/yalue/onnxruntime_go"
)

// The neural players run one of TWO exported model types. Both take ONE call per
// decision; they differ in where the history lives.
//
// (a) STATEFUL — the PPO actor-critic, exported by scripts/pth2onnx.py. The recurrent
// state is carried by the caller and advanced on every turn.
//
//	observation      float32 [1, 1, 208]  ISMCTS-layout observation of the player about
//	                                      to move (game.FlattenObservation)
//	action_features  float32 [1, 64, 84]  candidate actions, legal ones packed first
//	                                      (game.BuildActionFeatures)
//	action_mask      float32 [1, 64]      1.0 on the legal slots, 0.0 otherwise
//	lstm_h           float32 [1, 1, 128]  recurrent state in  (layers, batch, hidden)
//	lstm_c           float32 [1, 1, 128]  recurrent state in
//
//	action_logits    float32 [1, 64]      -1e9 on the illegal slots
//	value            float32 [1]          win estimate in [-1,1] for the mover's team
//	new_lstm_h       float32 [1, 1, 128]  recurrent state out, fed back next decision
//	new_lstm_c       float32 [1, 1, 128]  recurrent state out
//
// (b) STATELESS — the ISMCTS `CuarentaNet`, exported by scripts/pth2onnx_ismcts.py.
// Same three inputs and the first two outputs, NO lstm_h/lstm_c: that network is not
// recurrent, it reads the action history from an input sequence, and the search feeds it
// a single "pending" step (the regime it also uses for its interior nodes). For such a
// model AdvanceRecurrentState is a no-op, because nothing about a human turn changes the
// next evaluation.
//
// The caller plays `argmax(action_logits[:n_legal])`, where slot i is the i-th entry of
// ITS OWN legal-action list: the feature block must be built in the same order as the
// []game.Action the resulting slot is used to index. There is no other ordering
// contract between Go and the network.
//
// WHICH MODEL AM I SERVING? The graph says so (presence of lstm_h/lstm_c), and the
// sidecar `model.json` next to it records the value target: for ISMCTS the `value` output
// is the mover's shaped return divided by RETURN_VALUE_SCALE (game points = value*scale,
// NOT bounded by 1), not the PPO win estimate in [-1,1]. The action is unaffected — it is
// an argmax over logits either way — but anything that displays or thresholds `value`
// must read the sidecar first.
const (
	inputObservation    = "observation"
	inputActionFeatures = "action_features"
	inputActionMask     = "action_mask"
	inputLSTMH          = "lstm_h"
	inputLSTMC          = "lstm_c"

	outputActionLogits = "action_logits"
	outputValue        = "value"
	outputNewLSTMH     = "new_lstm_h"
	outputNewLSTMC     = "new_lstm_c"
)

// The two signatures, in the order the tensors are handed to the session. They are also
// the order of the contract checks, so a model that renamed or dropped a tensor is
// rejected at startup with a message naming the missing input.
var (
	statefulInputs = []string{
		inputObservation,
		inputActionFeatures,
		inputActionMask,
		inputLSTMH,
		inputLSTMC,
	}
	statefulOutputs = []string{
		outputActionLogits,
		outputValue,
		outputNewLSTMH,
		outputNewLSTMC,
	}
	statelessInputs = []string{
		inputObservation,
		inputActionFeatures,
		inputActionMask,
	}
	statelessOutputs = []string{
		outputActionLogits,
		outputValue,
	}
)

// modelKind is which of the two signatures the loaded model uses. It is decided from the
// graph itself (see detectKind), never from configuration, so a model file cannot be
// served under the wrong contract.
type modelKind int

const (
	kindStateful  modelKind = iota // PPO actor-critic: recurrent state in and out
	kindStateless                  // ISMCTS CuarentaNet: no recurrent state
)

// inputOrder / outputOrder / kind are set by initialize for the detected model.
var (
	inputOrder  []string
	outputOrder []string
	kind        modelKind
)

var (
	once      sync.Once
	initErr   error
	sessionMu sync.Mutex
	// engineMu serializes (re)initialization, which the test-facing startEngine can do more
	// than once per process; loadedModelPath is what it compares against.
	engineMu        sync.Mutex
	loadedModelPath string

	onnxSession *ort.AdvancedSession

	// Tensors are allocated once and reused, keyed by the model's declared name.
	inputTensors  map[string]*ort.Tensor[float32]
	outputTensors map[string]*ort.Tensor[float32]

	// Geometry read from the model graph and checked against the game package, so a
	// stale or incompatible .onnx cannot be loaded silently.
	obsSize       int // flattened observation length
	nSlots        int // action slots per decision
	featDim       int // width of one action's features
	lstmStateSize int // flattened (layers, 1, hidden) recurrent state

	// Per-game recurrent state, so the LSTM sees the game's history and not just the
	// current position. Dropped by ClearGameState when a game ends.
	lstmStateMu sync.Mutex
	lstmStates  map[string][2][]float32 // game_id -> [h, c]

	// Per-game, per-seat history of that seat's OWN completed decisions, for the search's
	// root evaluation. The ISMCTS network reads history from an input SEQUENCE (unlike PPO,
	// which carries it in the recurrent state), and `src/ismcts/history.py: HistoryTracker`
	// stores exactly this per player: what it saw and the action it took, oldest first.
	// Dropped by ClearGameState with the rest of the game's state.
	historyMu sync.Mutex
	histories map[string]map[int][]search.HistoryStep
)

// MaxHistorySteps is the window the search keeps per seat: the training-time T_MAX
// (`src/ismcts/train.py`), which is also the longest window the network ever saw.
const MaxHistorySteps = 32

type InferenceService struct{}

func NewInferenceService() *InferenceService {
	return &InferenceService{}
}

// StartEngine loads the ONNX policy from the configured model path. It is safe to call
// before every prediction: the environment, the session and the tensors are created once
// per process, and the configuration is read at most that once too.
func (i *InferenceService) StartEngine() error {
	once.Do(func() {
		env := config.GetEnv()
		initErr = initialize(env.OnnxModelPath, env.OnnxRuntimeSharedLibPath)
		if initErr != nil {
			log.Println("failed to initialize onnx inference:", initErr)
		}
	})
	return initErr
}

// ensureEngine makes sure SOME model is loaded, without insisting that it came from the
// environment: a session already loaded through startEngine (tests load explicit models)
// counts as initialized. Without this, the first prediction would re-enter StartEngine and
// try to read .env from the CWD.
func (i *InferenceService) ensureEngine() error {
	engineMu.Lock()
	loaded := onnxSession != nil
	engineMu.Unlock()
	if loaded {
		return nil
	}
	return i.StartEngine()
}

// startEngine is StartEngine with explicit paths and with model SWITCHING, so tests can
// load more than one model in a process. The ONNX environment and the session are
// process-wide, so a different model means tearing the previous one down first.
//
// Production goes through StartEngine, whose `once` keeps the environment built once.
func (i *InferenceService) startEngine(modelPath, sharedLibPath string) error {
	engineMu.Lock()
	defer engineMu.Unlock()

	if onnxSession != nil && loadedModelPath == modelPath {
		return nil
	}
	if onnxSession != nil {
		if err := i.ShutdownEngine(); err != nil {
			return err
		}
	}

	initErr = initialize(modelPath, sharedLibPath)
	if initErr != nil {
		log.Println("failed to initialize onnx inference:", initErr)
		return initErr
	}
	loadedModelPath = modelPath
	return nil
}

func initialize(modelPath, sharedLibPath string) error {
	if modelPath == "" {
		return fmt.Errorf("ONNX_MODEL_PATH is not set")
	}
	if sharedLibPath == "" {
		return fmt.Errorf("ONNXRUNTIME_SHARED_LIBRARY_PATH is not set")
	}

	ort.SetSharedLibraryPath(sharedLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize onnx runtime: %w", err)
	}

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return fmt.Errorf("inspect onnx model: %w", err)
	}

	inputOrder, outputOrder, kind, err = detectKind(inputs, outputs)
	if err != nil {
		return err
	}

	inByName, err := checkTensors("input", inputs, inputOrder)
	if err != nil {
		return err
	}
	outByName, err := checkTensors("output", outputs, outputOrder)
	if err != nil {
		return err
	}

	// Allocate every tensor with the shape the graph declares. Dynamic axes (the batch
	// dimension, and the batch dimension inside lstm_h/lstm_c) come back as a
	// non-positive dim and are pinned to 1, which is what this backend always uses.
	inputTensors = make(map[string]*ort.Tensor[float32], len(inputOrder))
	inputVals := make([]ort.Value, 0, len(inputOrder))
	for _, name := range inputOrder {
		shape := normalizeShape(inByName[name].Dimensions)
		tensor, err := ort.NewTensor(shape, make([]float32, shape.FlattenedSize()))
		if err != nil {
			return fmt.Errorf("create input tensor %q (shape %s): %w", name, shape, err)
		}
		inputTensors[name] = tensor
		inputVals = append(inputVals, tensor)
	}

	outputTensors = make(map[string]*ort.Tensor[float32], len(outputOrder))
	outputVals := make([]ort.Value, 0, len(outputOrder))
	for _, name := range outputOrder {
		shape := normalizeShape(outByName[name].Dimensions)
		tensor, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			return fmt.Errorf("create output tensor %q (shape %s): %w", name, shape, err)
		}
		outputTensors[name] = tensor
		outputVals = append(outputVals, tensor)
	}

	if err := readGeometry(); err != nil {
		return err
	}

	onnxSession, err = ort.NewAdvancedSession(
		modelPath,
		inputOrder,
		outputOrder,
		inputVals,
		outputVals,
		nil,
	)
	if err != nil {
		return fmt.Errorf("create onnx session: %w", err)
	}

	lstmStateMu.Lock()
	lstmStates = make(map[string][2][]float32)
	lstmStateMu.Unlock()

	historyMu.Lock()
	histories = make(map[string]map[int][]search.HistoryStep)
	historyMu.Unlock()

	if kind == kindStateful {
		log.Printf("onnx policy loaded (PPO, recurrent): observation %d, %d action slots x "+
			"%d features, lstm state %d", obsSize, nSlots, featDim, lstmStateSize)
	} else {
		log.Printf("onnx policy loaded (ISMCTS, stateless): observation %d, %d action slots "+
			"x %d features, no recurrent state", obsSize, nSlots, featDim)
	}

	return nil
}

// detectKind reads the model's signature off its tensors and returns the matching input and
// output orders. A model with lstm_h/lstm_c is stateful; a model with exactly the three
// stateless inputs and nothing else is stateless; anything else is rejected with the
// difference spelled out, because a partially-matching signature would otherwise be
// discovered as a silently wrong prediction at runtime.
func detectKind(inputs, outputs []ort.InputOutputInfo) ([]string, []string, modelKind, error) {
	has := func(infos []ort.InputOutputInfo, name string) bool {
		for _, info := range infos {
			if info.Name == name {
				return true
			}
		}
		return false
	}

	if has(inputs, inputLSTMH) || has(inputs, inputLSTMC) {
		return statefulInputs, statefulOutputs, kindStateful, nil
	}

	if len(inputs) == len(statelessInputs) && len(outputs) == len(statelessOutputs) {
		for _, name := range statelessInputs {
			if !has(inputs, name) {
				return nil, nil, 0, fmt.Errorf(
					"model has no input %q (it has %v); re-export with "+
						"scripts/pth2onnx_ismcts.py", name, tensorNames(inputs))
			}
		}
		for _, name := range statelessOutputs {
			if !has(outputs, name) {
				return nil, nil, 0, fmt.Errorf(
					"model has no output %q (it has %v); re-export with "+
						"scripts/pth2onnx_ismcts.py", name, tensorNames(outputs))
			}
		}
		return statelessInputs, statelessOutputs, kindStateless, nil
	}

	return nil, nil, 0, fmt.Errorf(
		"unrecognised model signature: %d inputs %v, %d outputs %v. Expected either the "+
			"PPO contract (obs, action_features, action_mask, lstm_h, lstm_c -> "+
			"action_logits, value, new_lstm_h, new_lstm_c) or the ISMCTS one (obs, "+
			"action_features, action_mask -> action_logits, value)",
		len(inputs), tensorNames(inputs), len(outputs), tensorNames(outputs))
}

func tensorNames(infos []ort.InputOutputInfo) []string {
	names := make([]string, 0, len(infos))
	for _, info := range infos {
		names = append(names, info.Name)
	}
	return names
}

// checkTensors indexes the model's tensors by name and validates that every name of the
// PPO contract is present as a float32 tensor.
func checkTensors(kind string, infos []ort.InputOutputInfo, expected []string) (map[string]ort.InputOutputInfo, error) {
	byName := make(map[string]ort.InputOutputInfo, len(infos))
	for _, info := range infos {
		byName[info.Name] = info
	}

	if len(infos) != len(expected) {
		return nil, fmt.Errorf("model has %d %ss, the PPO action-feature contract needs %d (%v); re-export with scripts/pth2onnx.py",
			len(infos), kind, len(expected), expected)
	}

	for _, name := range expected {
		info, ok := byName[name]
		if !ok {
			return nil, fmt.Errorf("model has no %s named %q; re-export with scripts/pth2onnx.py", kind, name)
		}
		if info.OrtValueType != ort.ONNXTypeTensor || info.DataType != ort.TensorElementDataTypeFloat {
			return nil, fmt.Errorf("%s %q: expected a float32 tensor, got %s", kind, name, info.String())
		}
	}

	return byName, nil
}

// readGeometry derives the tensor geometry from the allocated tensors and checks it
// against the sizes the game package builds, which is the cross-language part of the
// contract (src/ppo/utils.py, src/ppo/env.py, backend/models/*.json).
func readGeometry() error {
	obsSize = len(inputTensors[inputObservation].GetData())
	nSlots = len(inputTensors[inputActionMask].GetData())
	// Stateless models have no recurrent state at all, so there is nothing to size.
	lstmStateSize = 0
	if kind == kindStateful {
		lstmStateSize = len(inputTensors[inputLSTMH].GetData())
	}

	if nSlots <= 0 {
		return fmt.Errorf("action_mask tensor is empty")
	}
	featDim = len(inputTensors[inputActionFeatures].GetData()) / nSlots

	switch {
	case obsSize != game.ObsSizeISMCTS && obsSize%game.ObsSizeISMCTS == 0:
		return fmt.Errorf("model takes a %d-step observation window (%d floats); this backend "+
			"feeds ONE step (%d). Re-export with "+
			"`python3 -m scripts.pth2onnx_ismcts <checkpoint> --window 1`",
			obsSize/game.ObsSizeISMCTS, obsSize, game.ObsSizeISMCTS)
	case obsSize != game.ObsSizeISMCTS:
		return fmt.Errorf("observation size mismatch: model takes %d, game builds %d (OBS_SIZE_ISMCTS)", obsSize, game.ObsSizeISMCTS)
	case nSlots != game.MaxActionSlots:
		return fmt.Errorf("action slots mismatch: model scores %d, game packs %d (MAX_ACTIONS)", nSlots, game.MaxActionSlots)
	case featDim != game.ActionFeatureDim:
		return fmt.Errorf("action feature size mismatch: model takes %d, game builds %d (ACTION_FEAT_DIM)", featDim, game.ActionFeatureDim)
	case len(inputTensors[inputActionFeatures].GetData()) != nSlots*featDim:
		return fmt.Errorf("action_features tensor is not n_slots x feat_dim")
	case len(outputTensors[outputActionLogits].GetData()) != nSlots:
		return fmt.Errorf("action_logits size mismatch: %d logits for %d slots", len(outputTensors[outputActionLogits].GetData()), nSlots)
	case len(outputTensors[outputValue].GetData()) != 1:
		return fmt.Errorf("value output must be a single float, got %d", len(outputTensors[outputValue].GetData()))
	}

	if kind == kindStateful {
		switch {
		case len(inputTensors[inputLSTMC].GetData()) != lstmStateSize:
			return fmt.Errorf("lstm_h and lstm_c have different sizes")
		case len(outputTensors[outputNewLSTMH].GetData()) != lstmStateSize:
			return fmt.Errorf("new_lstm_h size mismatch: %d, expected %d", len(outputTensors[outputNewLSTMH].GetData()), lstmStateSize)
		case len(outputTensors[outputNewLSTMC].GetData()) != lstmStateSize:
			return fmt.Errorf("new_lstm_c size mismatch: %d, expected %d", len(outputTensors[outputNewLSTMC].GetData()), lstmStateSize)
		}
	}

	return nil
}

func (i *InferenceService) ShutdownEngine() error {
	var firstErr error

	if onnxSession != nil {
		if err := onnxSession.Destroy(); err != nil {
			firstErr = err
			log.Println("failed to destroy onnx session:", err)
		}
		onnxSession = nil
	}

	for name, tensor := range inputTensors {
		if err := tensor.Destroy(); err != nil && firstErr == nil {
			firstErr = err
			log.Println("failed to destroy input tensor", name, ":", err)
		}
	}
	inputTensors = nil

	for name, tensor := range outputTensors {
		if err := tensor.Destroy(); err != nil && firstErr == nil {
			firstErr = err
			log.Println("failed to destroy output tensor", name, ":", err)
		}
	}
	outputTensors = nil

	lstmStateMu.Lock()
	lstmStates = make(map[string][2][]float32)
	lstmStateMu.Unlock()

	if err := ort.DestroyEnvironment(); err != nil && firstErr == nil {
		firstErr = err
		log.Println("failed to destroy onnx env:", err)
	}

	return firstErr
}

func normalizeShape(shape ort.Shape) ort.Shape {
	if len(shape) == 0 {
		return ort.Shape{1}
	}

	normalized := make(ort.Shape, len(shape))
	for i, dim := range shape {
		// Dynamic dimensions (the batch axes) are reported as non-positive; this
		// backend always runs one decision at a time.
		if dim <= 0 {
			normalized[i] = 1
			continue
		}
		normalized[i] = dim
	}
	return normalized
}

// PredictActionSlot runs one decision for `gameID` and returns the slot of the action
// to play plus the model's value estimate.
//
// WHAT THE VALUE MEANS depends on the model type, and neither value is used to choose the
// action (that is the argmax over the logits): for the PPO net it is a win estimate in
// [-1,1] for the moving player's team, and for an ISMCTS net it is the mover's shaped
// return divided by RETURN_VALUE_SCALE (game points = value * scale, and NOT bounded by 1).
// Read the sidecar `model.json`'s `value_target`/`value_semantics` before displaying or
// thresholding it.
//
//	obs    game.FlattenObservation output, game.ObsSizeISMCTS floats
//	feats  game.BuildActionFeatures output flattened, MaxActionSlots x ActionFeatureDim
//	mask   its mask flattened, MaxActionSlots floats
//	nLegal how many of the slots hold legal actions (they are packed first)
//
// The returned slot indexes the caller's own legal-action list: use game.ActionForSlot.
// Any failure (engine unavailable, wrong lengths, a model that does not match the
// contract, a failed Run) is returned as an error and no action is chosen, so the
// caller can fall back to a legal action it already holds.
func (i *InferenceService) PredictActionSlot(gameID string, obs []float32, feats []float32, mask []float32, nLegal int) (int, float32, error) {
	return i.runDecision(gameID, obs, feats, mask, nLegal)
}

// AdvanceRecurrentState runs the same forward pass as PredictActionSlot for a turn whose
// action is NOT chosen by the network — a human turn — and keeps the new recurrent
// state, discarding the slot.
//
// WHY EVERY TURN MUST BE FED: the training contract advances the learner's (h, c) on
// every turn of the match, not only on the learner's own turns, and each turn is fed the
// ACTING player's observation and legal-action features (src/ppo/trainer.py: "With an
// LSTM the state is advanced on EVERY turn, not just the learner's own, so the memory
// covers the whole match"). A backend that skipped the human turns would hand the
// network, at its own decisions, a hidden state it never saw during training.
//
// It takes and validates exactly the same arguments as PredictActionSlot, so an
// unrepresentable position or a stale model fails the same way (and, importantly, leaves
// the stored state untouched rather than advancing it with a turn that never happened).
//
// For a STATELESS model (ISMCTS) this is a no-op: it has no recurrent state, and the single
// step it is fed means a human turn carries nothing into the next evaluation.
func (i *InferenceService) AdvanceRecurrentState(gameID string, obs []float32, feats []float32, mask []float32, nLegal int) error {
	if err := i.ensureEngine(); err != nil {
		return err
	}
	if kind == kindStateless {
		// An ISMCTS CuarentaNet has no recurrent state: it reads whatever history it is
		// given as a SEQUENCE, and this backend feeds it a single step, so a human turn
		// has nothing to contribute to the next evaluation. Skipping the forward pass is
		// not an optimisation here, it is the whole semantics.
		return nil
	}
	_, _, err := i.runDecision(gameID, obs, feats, mask, nLegal)
	return err
}

// runDecision is the single entry point for one forward pass of the policy: it owns the
// tensor copies, the recurrent-state in/out plumbing and the update of lstmStates.
// Both PredictActionSlot (which uses the slot it returns) and AdvanceRecurrentState
// (which drops it) are thin wrappers, so a turn costs the same forward pass either way
// and the two can never feed the network differently shaped inputs.
func (i *InferenceService) runDecision(gameID string, obs []float32, feats []float32, mask []float32, nLegal int) (int, float32, error) {
	if err := i.ensureEngine(); err != nil {
		return 0, 0, err
	}
	if nLegal <= 0 {
		return 0, 0, fmt.Errorf("no legal actions to score")
	}

	sessionMu.Lock()
	defer sessionMu.Unlock()

	if onnxSession == nil {
		return 0, 0, fmt.Errorf("onnx engine is not initialized")
	}
	if nLegal > nSlots {
		return 0, 0, fmt.Errorf("model scores at most %d actions per decision, got %d legal actions", nSlots, nLegal)
	}
	if len(obs) != obsSize {
		return 0, 0, fmt.Errorf("expected an observation of %d floats, got %d", obsSize, len(obs))
	}
	if len(feats) != nSlots*featDim {
		return 0, 0, fmt.Errorf("expected %d action feature floats, got %d", nSlots*featDim, len(feats))
	}
	if len(mask) != nSlots {
		return 0, 0, fmt.Errorf("expected a mask of %d floats, got %d", nSlots, len(mask))
	}

	copy(inputTensors[inputObservation].GetData(), obs)
	copy(inputTensors[inputActionFeatures].GetData(), feats)
	copy(inputTensors[inputActionMask].GetData(), mask)

	if kind == kindStateful {
		// The recurrent state of this game, zero on its first turn.
		h, c := lstmState(gameID)
		copy(inputTensors[inputLSTMH].GetData(), h)
		copy(inputTensors[inputLSTMC].GetData(), c)
	}

	if err := onnxSession.Run(); err != nil {
		return 0, 0, fmt.Errorf("run onnx session: %w", err)
	}

	// Slots past nLegal describe other positions (the model writes -1e9 on the illegal
	// ones), so the choice is an argmax over the caller's own legal slots only.
	slot := SelectSlot(outputTensors[outputActionLogits].GetData(), nLegal)

	if kind == kindStateful {
		// Feed the new state back on the next turn of this game, whatever the turn was for.
		freshH := cloneFloats(outputTensors[outputNewLSTMH].GetData())
		freshC := cloneFloats(outputTensors[outputNewLSTMC].GetData())
		lstmStateMu.Lock()
		lstmStates[gameID] = [2][]float32{freshH, freshC}
		lstmStateMu.Unlock()
	}

	value := outputTensors[outputValue].GetData()[0]
	return slot, value, nil
}

// SelectSlot is the policy's action selection: the argmax over the first `nLegal`
// logits, i.e. over the legal slots the caller packed. Out-of-range slots can never
// win, and an empty request falls back to slot 0.
func SelectSlot(logits []float32, nLegal int) int {
	if len(logits) == 0 || nLegal <= 0 {
		return 0
	}
	if nLegal > len(logits) {
		nLegal = len(logits)
	}

	best, bestScore := 0, logits[0]
	for slot := 1; slot < nLegal; slot++ {
		if logits[slot] > bestScore {
			best, bestScore = slot, logits[slot]
		}
	}
	return best
}

// lstmState returns the recurrent state to send for `gameID`, zero-initialized the
// first time a game is seen.
func lstmState(gameID string) ([]float32, []float32) {
	lstmStateMu.Lock()
	defer lstmStateMu.Unlock()

	state, ok := lstmStates[gameID]
	if !ok {
		state = [2][]float32{
			make([]float32, lstmStateSize),
			make([]float32, lstmStateSize),
		}
		lstmStates[gameID] = state
	}
	return state[0], state[1]
}

func cloneFloats(src []float32) []float32 {
	out := make([]float32, len(src))
	copy(out, src)
	return out
}

// ClearGameState forgets the recurrent state of `gameID`. Call it when a game ends and
// when a new game starts, so one game's memory cannot leak into another.
func (i *InferenceService) ClearGameState(gameID string) {
	lstmStateMu.Lock()
	delete(lstmStates, gameID)
	lstmStateMu.Unlock()

	historyMu.Lock()
	delete(histories, gameID)
	historyMu.Unlock()
}

// RecordHistory appends one completed decision of `seat`: the observation it saw and the
// features of the action it took. The search feeds this to the network as its input window
// at the root — exactly what `HistoryTracker.record` does in the training code
// ("saw this, then did that").
func (i *InferenceService) RecordHistory(gameID string, seat int, obs, actionFeatures []float32) {
	if len(obs) != game.ObsSizeISMCTS || len(actionFeatures) != game.ActionFeatureDim {
		log.Printf("search: refusing to record history for seat %d in game %s: got %d obs / "+
			"%d action features, expected %d / %d", seat, gameID, len(obs),
			len(actionFeatures), game.ObsSizeISMCTS, game.ActionFeatureDim)
		return
	}

	historyMu.Lock()
	defer historyMu.Unlock()

	if histories == nil {
		histories = make(map[string]map[int][]search.HistoryStep)
	}
	bySeat, ok := histories[gameID]
	if !ok {
		bySeat = make(map[int][]search.HistoryStep, game.NumPlayers)
		histories[gameID] = bySeat
	}
	steps := append(bySeat[seat], search.HistoryStep{
		Obs:   cloneFloats(obs),
		Feats: cloneFloats(actionFeatures),
	})
	if len(steps) > MaxHistorySteps {
		steps = steps[len(steps)-MaxHistorySteps:]
	}
	bySeat[seat] = steps
}

// History returns the recorded window for `seat`, oldest step first.
func (i *InferenceService) History(gameID string, seat int) []search.HistoryStep {
	historyMu.Lock()
	defer historyMu.Unlock()
	bySeat, ok := histories[gameID]
	if !ok {
		return nil
	}
	steps := bySeat[seat]
	out := make([]search.HistoryStep, len(steps))
	copy(out, steps)
	return out
}
