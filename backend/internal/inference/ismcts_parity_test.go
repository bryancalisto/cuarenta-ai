package infer

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// ismctsFixture is the JSON written by `scripts/onnx_parity_fixture_ismcts.py`: PyTorch's
// own logits/value for a handful of real decisions, taken from the checkpoint the ISMCTS
// ONNX was exported from.
type ismctsFixture struct {
	Pipeline         string `json:"pipeline"`
	Checkpoint       string `json:"checkpoint"`
	CheckpointSHA256 string `json:"checkpoint_sha256"`
	ONNX             string `json:"onnx"`
	ValueTarget      string `json:"value_target"`
	Seed             int    `json:"seed"`
	ObsDim           int    `json:"obs_dim"`
	ActionFeatDim    int    `json:"action_feat_dim"`
	NSlots           int    `json:"n_slots"`
	RecurrentState   bool   `json:"recurrent_state"`
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

// TestONNXMatchesPyTorchISMCTS is the ISMCTS twin of TestONNXMatchesPyTorchPolicy: it runs
// the STATELESS exported `CuarentaNet` through the real inference service and compares the
// logits, the value and the chosen slot with PyTorch's.
//
// Three things it protects that nothing else would:
//
//  1. The stateless signature loads at all. Before this model type was supported,
//     initialize() demanded lstm_h/lstm_c and rejected every ISMCTS export outright.
//  2. The VALUE is not squashed. In return mode the head is linear and its output is a
//     shaped return in game points (value * RETURN_VALUE_SCALE), NOT a win estimate in
//     [-1,1]; exporting it under the wrong value target wraps it in tanh and the comparison
//     here fails loudly instead of the backend quietly showing nonsense.
//  3. Human turns cost nothing. AdvanceRecurrentState is a no-op for this model type
//     because it has no recurrent state — asserted below by checking that it neither errors
//     nor changes what the next prediction returns.
//
// Regenerate both sides after re-exporting:
//
//	python3 -m scripts.pth2onnx_ismcts <ckpt> --value-target return \
//	    --out backend/models/ismcts_return.onnx
//	python3 scripts/onnx_parity_fixture_ismcts.py --checkpoint <ckpt> \
//	    --onnx backend/models/ismcts_return.onnx \
//	    --out backend/internal/inference/testdata/ismcts_fixture.json
func TestONNXMatchesPyTorchISMCTS(t *testing.T) {
	fixturePath := filepath.Join(packageDir(t), "testdata", "ismcts_fixture.json")
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Skipf("ISMCTS fixture not available (%v); generate it with "+
			"`python3 scripts/onnx_parity_fixture_ismcts.py`", err)
	}
	var fixture ismctsFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("cannot parse %s: %v", fixturePath, err)
	}
	if len(fixture.Decisions) == 0 {
		t.Fatalf("%s contains no decisions", fixturePath)
	}
	if fixture.RecurrentState {
		t.Fatalf("fixture says the model has a recurrent state; this test is for the "+
			"stateless ISMCTS contract (fixture %s)", fixturePath)
	}

	modelPath := filepath.Join(packageDir(t), "..", "..", "models", "ismcts_return.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("ISMCTS model not available at %s: %v", modelPath, err)
	}

	// Staleness guard, same as the PPO parity test: the fixture only validates the model
	// exported from the checkpoint it was generated from.
	sidecarRaw, err := os.ReadFile(replaceExt(modelPath, ".json"))
	if err != nil {
		t.Fatalf("cannot read the ONNX metadata beside %s: %v", modelPath, err)
	}
	var sidecar struct {
		ValueTarget            string `json:"value_target"`
		ValueTargetSource      string `json:"value_target_source"`
		RecurrentState         any    `json:"recurrent_state"`
		SourceCheckpoint       string `json:"source_checkpoint"`
		SourceCheckpointSHA256 string `json:"source_checkpoint_sha256"`
	}
	if err := json.Unmarshal(sidecarRaw, &sidecar); err != nil {
		t.Fatalf("cannot parse the ONNX metadata: %v", err)
	}
	if sidecar.SourceCheckpointSHA256 != fixture.CheckpointSHA256 {
		t.Fatalf("the ISMCTS ONNX came from checkpoint %s (%s…), the fixture from %s (%s…). "+
			"A fixture only validates the weights it was generated from — regenerate both:\n"+
			"  python3 -m scripts.pth2onnx_ismcts %s --value-target %s --out %s\n"+
			"  python3 scripts/onnx_parity_fixture_ismcts.py --checkpoint %s --value-target %s "+
			"--onnx %s --out %s",
			sidecar.SourceCheckpoint, short(sidecar.SourceCheckpointSHA256),
			fixture.Checkpoint, short(fixture.CheckpointSHA256),
			fixture.Checkpoint, fixture.ValueTarget, modelPath,
			fixture.Checkpoint, fixture.ValueTarget, modelPath, fixturePath)
	}
	if sidecar.ValueTarget != fixture.ValueTarget {
		t.Fatalf("the ONNX sidecar says value_target=%q, the fixture says %q; the exported "+
			"head's units do not match the numbers this test compares against",
			sidecar.ValueTarget, fixture.ValueTarget)
	}

	service := requireEngineAt(t, modelPath)

	if kind != kindStateless {
		t.Fatalf("the service loaded %s as %v, expected stateless", modelPath, kind)
	}
	if obsSize != fixture.ObsDim || nSlots != fixture.NSlots || featDim != fixture.ActionFeatDim {
		t.Fatalf("geometry %d/%d/%d does not match the fixture %d/%d/%d",
			obsSize, nSlots, featDim, fixture.ObsDim, fixture.NSlots, fixture.ActionFeatDim)
	}

	const tol = 1e-4
	// The value target decides what the number means; for return mode it is NOT bounded by
	// 1, so this is also the assertion that the head is linear.
	sawLargeValue := false

	for _, decision := range fixture.Decisions {
		slot, value, err := service.PredictActionSlot("ismcts-fixture", decision.Obs,
			decision.ActionFeatures, decision.ActionMask, decision.NLegal)
		if err != nil {
			t.Fatalf("%s: PredictActionSlot: %v", decision.Name, err)
		}
		if slot != decision.ExpectedSlot {
			t.Fatalf("%s: Go chose slot %d, PyTorch chose %d (n_legal=%d)",
				decision.Name, slot, decision.ExpectedSlot, decision.NLegal)
		}
		if math.Abs(float64(value-decision.ExpectedValue)) > tol {
			t.Fatalf("%s: value %f, PyTorch says %f (tolerance %g)", decision.Name,
				value, decision.ExpectedValue, tol)
		}
		if math.Abs(float64(value)) > 1 {
			sawLargeValue = true
		}

		// The logits themselves, not just the argmax: a scaled or mis-wired policy head can
		// keep the argmax on this fixture and still be a different (worse) policy.
		got := outputTensors[outputActionLogits].GetData()
		for i := 0; i < fixture.NSlots; i++ {
			if math.Abs(float64(got[i]-decision.ExpectedLogits[i])) > 1e-3 {
				t.Fatalf("%s: logit[%d] %f, PyTorch says %f", decision.Name, i,
					got[i], decision.ExpectedLogits[i])
			}
		}
	}

	if fixture.ValueTarget == "return" && !sawLargeValue {
		t.Logf("note: every fixture value happened to be within [-1,1]; the linear-head " +
			"assertion in this test is then not exercised by these positions")
	}

	// A stateless model must ignore human turns entirely: no error, and no effect on the
	// next prediction.
	before, _, err := service.PredictActionSlot("ismcts-fixture", fixture.Decisions[0].Obs,
		fixture.Decisions[0].ActionFeatures, fixture.Decisions[0].ActionMask,
		fixture.Decisions[0].NLegal)
	if err != nil {
		t.Fatalf("PredictActionSlot before the human turn: %v", err)
	}
	for i := 0; i < 3; i++ {
		if err := service.AdvanceRecurrentState("ismcts-fixture", fixture.Decisions[0].Obs,
			fixture.Decisions[0].ActionFeatures, fixture.Decisions[0].ActionMask,
			fixture.Decisions[0].NLegal); err != nil {
			t.Fatalf("AdvanceRecurrentState on a stateless model: %v", err)
		}
	}
	after, _, err := service.PredictActionSlot("ismcts-fixture", fixture.Decisions[0].Obs,
		fixture.Decisions[0].ActionFeatures, fixture.Decisions[0].ActionMask,
		fixture.Decisions[0].NLegal)
	if err != nil {
		t.Fatalf("PredictActionSlot after the human turn: %v", err)
	}
	if before != after {
		t.Fatalf("a stateless model changed its answer (%d -> %d) after AdvanceRecurrentState",
			before, after)
	}
}
