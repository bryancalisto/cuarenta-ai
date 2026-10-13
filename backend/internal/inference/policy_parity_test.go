package infer

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// policyFixture is the JSON written by `scripts/onnx_parity_fixture.py`: PyTorch's own
// logits/value for a handful of real decisions, captured from the checkpoint the ONNX
// file was exported from.
type policyFixture struct {
	Checkpoint       string `json:"checkpoint"`
	CheckpointSHA256 string `json:"checkpoint_sha256"`
	ONNX             string `json:"onnx"`
	Seed             int    `json:"seed"`
	ObsDim           int    `json:"obs_dim"`
	ActionFeatDim    int    `json:"action_feat_dim"`
	NSlots           int    `json:"n_slots"`
	LSTMHidden       int    `json:"lstm_hidden"`
	Decisions        []struct {
		Name           string    `json:"name"`
		Step           int       `json:"step"`
		Player         int       `json:"player"`
		NLegal         int       `json:"n_legal"`
		Obs            []float32 `json:"obs"`
		ActionFeatures []float32 `json:"action_features"`
		ActionMask     []float32 `json:"action_mask"`
		ExpectedSlot   int       `json:"expected_slot"`
		ExpectedValue  float32   `json:"expected_value"`
		ExpectedLogits []float32 `json:"expected_logits"`
	} `json:"decisions"`
}

// TestONNXMatchesPyTorchPolicy is the guard that the export in backend/models still
// means what the trained network meant. Without it, a transposed action-feature matrix,
// a shuffled observation layout or a mislabelled output would keep "working" -- the
// service would return a legal slot every turn, just a worse one -- and no other test
// here would notice.
//
// Regenerate both sides after re-exporting a model:
//
//	python3 scripts/onnx_parity_fixture.py
func TestONNXMatchesPyTorchPolicy(t *testing.T) {
	fixturePath := filepath.Join(packageDir(t), "testdata", "policy_fixture.json")
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("cannot read %s (%v); regenerate it with "+
			"`python3 scripts/onnx_parity_fixture.py`", fixturePath, err)
	}
	var fixture policyFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("cannot parse %s: %v", fixturePath, err)
	}
	if len(fixture.Decisions) == 0 {
		t.Fatalf("%s contains no decisions", fixturePath)
	}

	// Staleness guard: the fixture only validates the model it was generated from.
	modelPath := testModelPath(t)
	sidecarRaw, err := os.ReadFile(replaceExt(modelPath, ".json"))
	if err != nil {
		t.Fatalf("cannot read the ONNX metadata beside %s: %v", modelPath, err)
	}
	var sidecar struct {
		Observation struct {
			Size int `json:"size"`
		} `json:"observation"`
		ActionFeatures struct {
			Size int `json:"size"`
		} `json:"action_features"`
		ActionSlots            int    `json:"action_slots"`
		SourceCheckpoint       string `json:"source_checkpoint"`
		SourceCheckpointSHA256 string `json:"source_checkpoint_sha256"`
	}
	if err := json.Unmarshal(sidecarRaw, &sidecar); err != nil {
		t.Fatalf("cannot parse the ONNX metadata: %v", err)
	}
	if sidecar.SourceCheckpointSHA256 == "" {
		t.Fatalf("the ONNX metadata has no source_checkpoint_sha256; re-export with "+
			"`python3 -m scripts.pth2onnx %s`", sidecar.SourceCheckpoint)
	}
	if sidecar.SourceCheckpointSHA256 != fixture.CheckpointSHA256 {
		t.Fatalf("the ONNX in backend/models came from checkpoint %s (%s…), but the parity "+
			"fixture was generated from %s (%s…). Regenerate it with "+
			"`python3 scripts/onnx_parity_fixture.py --checkpoint <that checkpoint>`",
			sidecar.SourceCheckpoint, short(sidecar.SourceCheckpointSHA256),
			fixture.Checkpoint, short(fixture.CheckpointSHA256))
	}

	// The fixture must describe the same contract the service enforces.
	if sidecar.Observation.Size != fixture.ObsDim ||
		sidecar.ActionFeatures.Size != fixture.ActionFeatDim ||
		sidecar.ActionSlots != fixture.NSlots {
		t.Fatalf("fixture contract %d/%d/%d does not match the ONNX sidecar %d/%d/%d",
			fixture.ObsDim, fixture.ActionFeatDim, fixture.NSlots,
			sidecar.Observation.Size, sidecar.ActionFeatures.Size, sidecar.ActionSlots)
	}

	service := requireEngine(t)

	nonZeroSlots := 0
	for _, decision := range fixture.Decisions {
		if len(decision.Obs) != fixture.ObsDim {
			t.Fatalf("%s: fixture observation has %d floats, expected %d",
				decision.Name, len(decision.Obs), fixture.ObsDim)
		}
		if len(decision.ActionFeatures) != fixture.NSlots*fixture.ActionFeatDim {
			t.Fatalf("%s: fixture action features have %d floats, expected %d",
				decision.Name, len(decision.ActionFeatures),
				fixture.NSlots*fixture.ActionFeatDim)
		}
		if len(decision.ActionMask) != fixture.NSlots {
			t.Fatalf("%s: fixture action mask has %d floats, expected %d",
				decision.Name, len(decision.ActionMask), fixture.NSlots)
		}

		// The fixture was captured with PyTorch called with hidden=None, i.e. from a
		// zeroed recurrent state. Clear it explicitly rather than relying on the game id
		// being new: the state is keyed by game id and persists between calls (including
		// between the iterations of `go test -count=N` in one process), so a stale state
		// would quietly change the value this test compares.
		gameID := "parity-" + decision.Name
		service.ClearGameState(gameID)
		slot, value, err := service.PredictActionSlot(
			gameID, decision.Obs, decision.ActionFeatures, decision.ActionMask, decision.NLegal)
		if err != nil {
			t.Fatalf("%s: PredictActionSlot failed: %v", decision.Name, err)
		}

		if slot != decision.ExpectedSlot {
			t.Errorf("%s: ONNX chose slot %d, PyTorch chose %d",
				decision.Name, slot, decision.ExpectedSlot)
		}
		if diff := math.Abs(float64(value - decision.ExpectedValue)); diff > 1e-3 {
			t.Errorf("%s: ONNX value %.6f, PyTorch value %.6f (diff %.6f > 1e-3)",
				decision.Name, value, decision.ExpectedValue, diff)
		}
		if decision.ExpectedSlot != 0 {
			nonZeroSlots++
		}
	}

	// Guards the guard: a fixture whose answers are all slot 0 would pass even if the
	// service always returned 0.
	if nonZeroSlots == 0 {
		t.Error("every fixture decision expects slot 0, so this test cannot detect a " +
			"service that always answers 0; regenerate the fixture")
	}
}

func replaceExt(path, ext string) string {
	return path[:len(path)-len(filepath.Ext(path))] + ext
}

func short(hash string) string {
	if len(hash) <= 12 {
		return hash
	}
	return hash[:12]
}
