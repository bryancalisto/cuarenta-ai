package infer

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"cua/internal/game"
	"cua/internal/search"
)

// TestSearchThroughTheRealModel runs one full search decision through the actual ONNX
// session, so the search and the model are known to fit together (tensor sizes, the
// value/prior split, the sidecar's value target).
func TestSearchThroughTheRealModel(t *testing.T) {
	// The ISMCTS export, not the PPO one: the search needs a stateless model (it evaluates
	// sampled positions, which a recurrent state tied to the live game cannot describe).
	modelPath := os.Getenv("ISMCTS_TEST_MODEL")
	if modelPath == "" {
		modelPath = filepath.Join(packageDir(t), "..", "..", "models", "ismcts_return.onnx")
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("ISMCTS model not available at %s (%v); export one with "+
			"`make export_ismcts_onnx`", modelPath, err)
	}
	libPath := onnxRuntimeLibPath(t)
	if libPath == "" {
		t.Skip("ONNXRUNTIME_SHARED_LIBRARY_PATH is not set")
	}
	eval, err := NewSearchEvaluator(modelPath, libPath)
	if err != nil {
		t.Fatalf("NewSearchEvaluator: %v", err)
	}

	env := game.NewEnv(31)
	env.Reset()
	state := env.State()
	cfg := eval.Config(32, 1.5)
	s, err := search.New(state, state.CurrentPlayer, eval, cfg, rand.New(rand.NewSource(5)))
	if err != nil {
		t.Fatalf("search.New: %v", err)
	}
	action, stats, err := s.Decide(state, nil)
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	legal := env.LegalActions()
	if !game.ActionIsLegal(action, legal) {
		t.Fatalf("the search returned an illegal action: %s", action)
	}
	if stats.Simulations != 32 || stats.Evaluations == 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	total := 0
	for _, v := range stats.Visits {
		total += v
	}
	if total != 32 {
		t.Fatalf("visits sum to %d, expected 32", total)
	}
	t.Logf("chose %s | visits %v | root value %.3f (%s, scale %g) | %d evaluations",
		action, stats.Visits, stats.RootValue, eval.valueTarget, eval.valueScale,
		stats.Evaluations)
}
