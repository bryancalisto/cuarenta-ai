package infer

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"cua/internal/config"
	"cua/internal/game"
	"cua/internal/search"
)

// The search needs one model call per simulation, with inputs it builds itself (the
// observation of a SAMPLED world, the action features of that world's legal actions).
// `SearchEvaluator` is the adapter between `internal/search` and the ONNX session.
//
// It also carries the two facts the search cannot guess and must not get wrong:
//
//	value_target  "win" or "return" — in return mode the search accumulates the dense reward
//	              and reads a LINEAR head; in win mode it backs up ±1 results and ignores
//	              rewards. Serving a return-trained head through a win-mode search (or the
//	              reverse) produces legal, plausible, and much weaker play: measured on the
//	              project's own checkpoint, 64% instead of 76% against the heuristic.
//	value_scale   RETURN_VALUE_SCALE, the divisor between game points and the head's units.
//
// Both come from the `model.json` sidecar the exporter writes next to the .onnx.
// SearchEvaluator must satisfy the search's evaluator interface.
var _ search.ModelEvaluator = (*SearchEvaluator)(nil)

type SearchEvaluator struct {
	steps       int
	valueTarget string
	valueScale  float64
}

// modelSidecar is the subset of `scripts/pth2onnx_ismcts.py`'s metadata this package reads.
type modelSidecar struct {
	Observation struct {
		Size  int `json:"size"`
		Steps int `json:"steps"`
	} `json:"observation"`
	ValueTarget string  `json:"value_target"`
	ValueScale  float64 `json:"value_scale"`
}

// NewSearchEvaluator builds the search's evaluator for the loaded model, reading its
// sidecar for the value semantics. A missing sidecar is not fatal — the search falls back to
// win mode, which is only correct for a win-trained head, so it is logged loudly.
func NewSearchEvaluator(modelPath, sharedLibPath string) (*SearchEvaluator, error) {
	service := NewInferenceService()
	if err := service.startEngine(modelPath, sharedLibPath); err != nil {
		return nil, err
	}
	if kind != kindStateless {
		return nil, fmt.Errorf("the search needs a stateless ISMCTS model, but %s loaded as "+
			"the PPO contract (it carries a recurrent state that belongs to the live game, "+
			"so sampled positions cannot be evaluated). Export the ISMCTS checkpoint with "+
			"scripts/pth2onnx_ismcts.py, or set AI_SEARCH_ITERATIONS=0", modelPath)
	}

	eval := &SearchEvaluator{steps: 1, valueTarget: search.ValueTargetWin, valueScale: 1}

	sidecarPath := strings.TrimSuffix(modelPath, ".onnx") + ".json"
	raw, err := os.ReadFile(sidecarPath)
	if err != nil {
		log.Printf("search: no model sidecar at %s (%v); assuming value_target=%q. If the "+
			"model was trained with --value-target return, the search would optimise the "+
			"wrong objective — re-export with scripts/pth2onnx_ismcts.py, which writes the "+
			"sidecar", sidecarPath, err, search.ValueTargetWin)
		return eval, nil
	}
	var sidecar modelSidecar
	if err := json.Unmarshal(raw, &sidecar); err != nil {
		return nil, fmt.Errorf("parse %s: %w", sidecarPath, err)
	}

	if sidecar.Observation.Steps > 0 {
		eval.steps = sidecar.Observation.Steps
	}
	switch sidecar.ValueTarget {
	case search.ValueTargetWin, search.ValueTargetReturn:
		eval.valueTarget = sidecar.ValueTarget
	case "":
		log.Printf("search: %s does not record a value_target; assuming %q",
			sidecarPath, search.ValueTargetWin)
	default:
		return nil, fmt.Errorf("%s has an unknown value_target %q", sidecarPath,
			sidecar.ValueTarget)
	}
	if sidecar.ValueScale > 0 {
		eval.valueScale = sidecar.ValueScale
	}
	if eval.valueTarget == search.ValueTargetReturn && eval.valueScale <= 0 {
		return nil, fmt.Errorf("%s: return mode needs a value_scale", sidecarPath)
	}

	if declared := observationSteps(); declared > 0 && declared != eval.steps {
		return nil, fmt.Errorf("the model graph takes %d observation steps but %s says %d; "+
			"re-export with scripts/pth2onnx_ismcts.py", declared, sidecarPath, eval.steps)
	}

	log.Printf("search: evaluator ready (value_target=%s, scale=%g, observation steps=%d)",
		eval.valueTarget, eval.valueScale, eval.steps)
	return eval, nil
}

func (e *SearchEvaluator) Steps() int { return e.steps }

// Forward implements search.Evaluator by filling the session's tensors and running it.
//
// The session holds ONE set of tensors sized for one decision at a time and is guarded by
// `sessionMu`; the search is single-threaded per decision, and a whole game's AI turns are
// serialized by the game's own lock, so this is safe — but it does mean two games' searches
// cannot run concurrently in one process (they queue on the mutex).
func (e *SearchEvaluator) Forward(obsSeq, actSeq, actMask, feats, mask []float32,
	steps int) ([]float32, float32, error) {
	return forwardRaw(obsSeq, actSeq, actMask, feats, mask, steps)
}

// forwardRaw runs one session call with caller-supplied tensors.
//
// It exists because `runDecision` is tied to the LIVE game: it reads and writes the
// per-game recurrent state and assumes the position it is given is the real one. The
// search evaluates SAMPLED positions (a determinized world, or an interior node of its
// tree), and the ISMCTS model is stateless, so there is no state to thread.
//
// The returned logits are a copy: `outputTensors` is reused by the next call.
func forwardRaw(obsSeq, actSeq, actMask, feats, mask []float32, steps int) ([]float32, float32, error) {
	if err := serviceStartEngine(); err != nil {
		return nil, 0, err
	}
	if kind != kindStateless {
		return nil, 0, fmt.Errorf("the search cannot use a stateful (PPO) model")
	}
	if steps < 1 {
		return nil, 0, fmt.Errorf("steps must be >= 1, got %d", steps)
	}
	if len(obsSeq) != steps*obsSize {
		return nil, 0, fmt.Errorf("observation window: %d floats for %d step(s) of %d",
			len(obsSeq), steps, obsSize)
	}
	if len(actSeq) != steps*featDim {
		return nil, 0, fmt.Errorf("action history: %d floats for %d step(s) of %d",
			len(actSeq), steps, featDim)
	}
	if len(actMask) != steps {
		return nil, 0, fmt.Errorf("action history mask: %d values for %d step(s)",
			len(actMask), steps)
	}
	if len(feats) != nSlots*featDim {
		return nil, 0, fmt.Errorf("action features: %d floats, expected %d", len(feats),
			nSlots*featDim)
	}
	if len(mask) != nSlots {
		return nil, 0, fmt.Errorf("action mask: %d floats, expected %d", len(mask), nSlots)
	}

	sessionMu.Lock()
	defer sessionMu.Unlock()

	if onnxSession == nil {
		return nil, 0, fmt.Errorf("onnx engine is not initialized")
	}

	copy(inputTensors[inputObservation].GetData(), obsSeq)
	copy(inputTensors[inputActionFeatures].GetData(), feats)
	copy(inputTensors[inputActionMask].GetData(), mask)

	if err := onnxSession.Run(); err != nil {
		return nil, 0, fmt.Errorf("run onnx session: %w", err)
	}

	logits := cloneFloats(outputTensors[outputActionLogits].GetData())
	value := outputTensors[outputValue].GetData()[0]
	return logits, value, nil
}

// observationSteps is how many observation steps the loaded graph declares (1 for a
// single-step export, 32 for a windowed one).
func observationSteps() int {
	if obsSize == 0 {
		return 0
	}
	return obsSize / game.ObsSizeISMCTS
}

// serviceStartEngine makes sure a model is loaded, using the environment's paths.
func serviceStartEngine() error {
	return (&InferenceService{}).ensureEngine()
}

// SearchEvaluator returns the evaluator for the LOADED model, building it once per process
// from the environment's paths. It is the entry point the server uses.
func (i *InferenceService) SearchEvaluator() (search.ModelEvaluator, error) {
	searchEvalOnce.Do(func() {
		env := config.GetEnv()
		searchEval, searchEvalErr = NewSearchEvaluator(env.OnnxModelPath,
			env.OnnxRuntimeSharedLibPath)
	})
	if searchEval == nil {
		return nil, searchEvalErr
	}
	return searchEval, searchEvalErr
}

var (
	searchEvalOnce sync.Once
	searchEval     *SearchEvaluator
	searchEvalErr  error
)

// Config builds a search configuration carrying this model's value semantics, so the
// search optimises the objective the weights were trained for. `*SearchEvaluator` satisfies
// `search.ModelEvaluator`.
func (e *SearchEvaluator) Config(iterations int, cpuct float64) search.Config {
	cfg := search.DefaultConfig()
	cfg.Iterations = iterations
	cfg.CPuct = cpuct
	cfg.ValueTarget = e.valueTarget
	cfg.ValueScale = e.valueScale
	return cfg
}
