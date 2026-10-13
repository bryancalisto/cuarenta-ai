package infer

import (
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"cua/internal/game"
	"cua/internal/search"
)

// TestSearchBeatsTheBarePolicy is the end-to-end check that the server-side search is worth
// its cost: team 0 plays WITH the search, team 1 plays the bare policy (argmax of the same
// network's logits, i.e. exactly what the backend did before the search existed).
//
// This is the Go equivalent of the Python `teacher_edge` measurement, and the number to
// compare against is Python's: the search beat the same bare policy 56% of the time at 64
// simulations and 67-85% at 400 (100 games each). If the port were subtly wrong — a
// different value target, a broken backup, a determinizer that leaks information — the
// search would not clear 50% here, and nothing else in the suite would notice.
//
// It is OPT-IN because it is slow (each search decision is `iterations` model calls), so it
// does not run in the normal suite:
//
//	ISMCTS_MATCH_GAMES=20 ISMCTS_MATCH_ITER=128 \
//	  go test ./internal/inference/ -run TestSearchBeatsTheBarePolicy -v
func TestSearchBeatsTheBarePolicy(t *testing.T) {
	gamesRaw := os.Getenv("ISMCTS_MATCH_GAMES")
	if gamesRaw == "" {
		t.Skip("set ISMCTS_MATCH_GAMES to run this match (it is slow: iterations x decisions)")
	}
	games, err := strconv.Atoi(gamesRaw)
	if err != nil || games <= 0 {
		t.Fatalf("ISMCTS_MATCH_GAMES=%q is not a positive integer", gamesRaw)
	}
	iterations := 128
	if raw := os.Getenv("ISMCTS_MATCH_ITER"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			iterations = parsed
		}
	}

	modelPath := os.Getenv("ISMCTS_TEST_MODEL")
	if modelPath == "" {
		modelPath = filepath.Join(packageDir(t), "..", "..", "models", "ismcts_return.onnx")
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not available at %s: %v", modelPath, err)
	}
	libPath := onnxRuntimeLibPath(t)
	if libPath == "" {
		t.Skip("ONNXRUNTIME_SHARED_LIBRARY_PATH is not set")
	}
	eval, err := NewSearchEvaluator(modelPath, libPath)
	if err != nil {
		t.Fatalf("NewSearchEvaluator: %v", err)
	}
	cfgTemplate := eval.Config(iterations, 1.5)

	wins, losses := 0, 0
	searchedDecisions, policyDecisions := 0, 0
	for g := 0; g < games; g++ {
		env := game.NewEnv(int64(g) + 1)
		env.Reset()
		plies := 0
		for !env.State().Done {
			state := env.State()
			seat := state.CurrentPlayer
			var action game.Action
			if seat%2 == 0 {
				cfg := cfgTemplate
				searcher, err := search.New(state, seat, eval, cfg,
					rand.New(rand.NewSource(int64(g)*1_000_000+int64(plies))))
				if err != nil {
					t.Fatalf("game %d ply %d: search.New: %v", g, plies, err)
				}
				chosen, _, err := searcher.Decide(state, nil)
				if err != nil {
					t.Fatalf("game %d ply %d: Decide: %v", g, plies, err)
				}
				action = chosen
				searchedDecisions++
			} else {
				legal := env.LegalActions()
				feats, mask, slots := game.BuildActionFeatures(legal, state)
				obs := game.FlattenObservation(game.ObserveState(state, seat))
				logits, _, err := eval.Forward(obs, make([]float32, game.ActionFeatureDim),
					[]float32{0}, feats.Flatten(), mask.Flatten(), 1)
				if err != nil {
					t.Fatalf("game %d ply %d: policy forward: %v", g, plies, err)
				}
				best := 0
				for i := 1; i < slots; i++ {
					if logits[i] > logits[best] {
						best = i
					}
				}
				action = legal[best]
				policyDecisions++
			}
			if _, _, _, err := env.Step(action); err != nil {
				t.Fatalf("game %d ply %d: the engine rejected %s: %v", g, plies, action, err)
			}
			plies++
			if plies > 600 {
				t.Fatalf("game %d did not terminate", g)
			}
		}
		winner := 0
		if env.State().Winner != nil {
			winner = *env.State().Winner
		}
		if winner == 0 {
			wins++
		} else {
			losses++
		}
		t.Logf("game %d: team %d won (%d plies)", g, winner, plies)
	}

	rate := float64(wins) / float64(wins+losses)
	t.Logf("SEARCH vs BARE POLICY: %d-%d (%.1f%%) over %d games at %d sims/decision "+
		"(%d searched decisions, %d policy decisions, model %s)", wins, losses, rate*100,
		games, iterations, searchedDecisions, policyDecisions, filepath.Base(modelPath))
	if rate <= 0.5 {
		t.Fatalf("the search did not beat the bare policy (%.1f%%), which is the one thing "+
			"it must do: check the value target in the model sidecar, the backup, and whether "+
			"the determinizer leaks information", rate*100)
	}
}
