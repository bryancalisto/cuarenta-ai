package infer

import (
	"cua/internal/config"
	"cua/internal/game"
	"fmt"
	"log"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	once           sync.Once
	initErr        error
	sessionMu      sync.Mutex
	onnxSession    *ort.AdvancedSession
	InputTensor    *ort.Tensor[float32]
	OutputTensor   *ort.Tensor[float32]
	inputNames     []string
	outputNames    []string
	inputTensors   []*ort.Tensor[float32]
	outputTensors  []*ort.Tensor[float32]
	isLSTM         bool
	lstmHiddenSize int
	lstmStateMu    sync.Mutex
	lstmStates     map[string][2][]float32 // game_id -> [h, c]
)

type InferenceService struct{}

// detects and handles both stateless (1-1) and stateful (LSTM, 3-3) models automatically
func NewInferenceService() *InferenceService {
	return &InferenceService{}
}

func (i *InferenceService) StartEngine() error {
	once.Do(func() {
		sharedLibPath := config.GetEnv().OnnxRuntimeSharedLibPath
		ort.SetSharedLibraryPath(sharedLibPath)

		if err := ort.InitializeEnvironment(); err != nil {
			initErr = fmt.Errorf("initialize onnx runtime: %w", err)
			log.Println("failed to init onnx env:", err)
			return
		}

		modelPath := config.GetEnv().OnnxModelPath
		if modelPath == "" {
			initErr = fmt.Errorf("ONNX_MODEL_PATH is not set")
			return
		}

		inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
		if err != nil {
			initErr = fmt.Errorf("inspect onnx model: %w", err)
			return
		}

		// detect LSTM model: 3 inputs (obs, lstm_h, lstm_c) and 3 outputs (logits, new_h, new_c)
		if len(inputs) == 3 && len(outputs) == 3 {
			isLSTM = true
			log.Println("Detected LSTM model")
		} else if len(inputs) != 1 || len(outputs) != 1 {
			initErr = fmt.Errorf("expected either (1 input, 1 output) or (3 inputs, 3 outputs), got %d inputs and %d outputs", len(inputs), len(outputs))
			return
		}

		// validate all inputs/outputs are float32 tensors
		for _, inp := range inputs {
			if inp.OrtValueType != ort.ONNXTypeTensor || inp.DataType != ort.TensorElementDataTypeFloat {
				initErr = fmt.Errorf("expected float32 tensor inputs")
				return
			}
		}
		for _, out := range outputs {
			if out.OrtValueType != ort.ONNXTypeTensor || out.DataType != ort.TensorElementDataTypeFloat {
				initErr = fmt.Errorf("expected float32 tensor outputs")
				return
			}
		}

		// build input/output names in order
		inputNames = make([]string, len(inputs))
		for i, inp := range inputs {
			inputNames[i] = inp.Name
		}
		outputNames = make([]string, len(outputs))
		for i, out := range outputs {
			outputNames[i] = out.Name
		}

		// for LSTM, extract hidden size from lstm_h shape (should be [1, lstm_hidden])
		if isLSTM {
			for _, inp := range inputs {
				if inp.Name == "lstm_h" {
					dims := normalizeShape(inp.Dimensions)
					if len(dims) >= 2 {
						lstmHiddenSize = int(dims[1])
					}
					break
				}
			}
			lstmStates = make(map[string][2][]float32)
		}

		// create tensors for all inputs
		inputVals := make([]ort.Value, len(inputs))
		inputTensors = make([]*ort.Tensor[float32], len(inputs))
		for i, inp := range inputs {
			shape := normalizeShape(inp.Dimensions)
			data := make([]float32, shape.FlattenedSize())
			tensor, err := ort.NewTensor(shape, data)
			if err != nil {
				initErr = fmt.Errorf("create input tensor %d: %w", i, err)
				return
			}
			inputTensors[i] = tensor
			inputVals[i] = tensor
		}

		// Create output tensors
		outputVals := make([]ort.Value, len(outputs))
		outputTensors = make([]*ort.Tensor[float32], len(outputs))
		for i, out := range outputs {
			shape := normalizeShape(out.Dimensions)
			tensor, err := ort.NewEmptyTensor[float32](shape)
			if err != nil {
				initErr = fmt.Errorf("create output tensor %d: %w", i, err)
				return
			}
			outputTensors[i] = tensor
			outputVals[i] = tensor
		}

		// For backward compatibility (with non-lstm models), point InputTensor/OutputTensor to first tensors
		if len(inputTensors) > 0 {
			InputTensor = inputTensors[0]
		}
		if len(outputTensors) > 0 {
			OutputTensor = outputTensors[0]
		}

		onnxSession, err = ort.NewAdvancedSession(
			modelPath,
			inputNames,
			outputNames,
			inputVals,
			outputVals,
			nil,
		)
		if err != nil {
			initErr = fmt.Errorf("create onnx session: %w", err)
			return
		}
	})

	return initErr
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

	if InputTensor != nil {
		if err := InputTensor.Destroy(); err != nil && firstErr == nil {
			firstErr = err
			log.Println("failed to destroy input tensor:", err)
		}
		InputTensor = nil
	}

	if OutputTensor != nil {
		if err := OutputTensor.Destroy(); err != nil && firstErr == nil {
			firstErr = err
			log.Println("failed to destroy output tensor:", err)
		}
		OutputTensor = nil
	}

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
		if dim <= 0 {
			normalized[i] = 1
			continue
		}
		normalized[i] = dim
	}
	return normalized
}

func (i *InferenceService) Infer(input []float32) ([]float32, error) {
	if err := i.StartEngine(); err != nil {
		return nil, err
	}

	sessionMu.Lock()
	defer sessionMu.Unlock()

	if InputTensor == nil || OutputTensor == nil || onnxSession == nil {
		return nil, fmt.Errorf("onnx engine is not initialized")
	}

	if len(input) != len(InputTensor.GetData()) {
		return nil, fmt.Errorf("expected input length %d, got %d", len(InputTensor.GetData()), len(input))
	}

	copy(InputTensor.GetData(), input)
	if err := onnxSession.Run(); err != nil {
		return nil, err
	}

	out := make([]float32, len(OutputTensor.GetData()))
	copy(out, OutputTensor.GetData())
	return out, nil
}

func (i *InferenceService) PredictAction(gameID string, input []float32, legal []game.Action) (game.Action, int, error) {
	var logits []float32
	var err error

	if isLSTM {
		logits, err = i.inferWithLSTM(gameID, input)
	} else {
		logits, err = i.Infer(input)
	}

	if err != nil {
		return game.Action{}, 0, err
	}
	if len(legal) == 0 {
		return game.Action{}, 0, fmt.Errorf("no legal actions available")
	}

	bestIdx := -1
	bestScore := float32(-1 << 30)
	for i, action := range legal {
		actionIdx := game.EncodeAction(action, legal)
		if actionIdx < 0 || actionIdx >= len(logits) {
			continue
		}
		score := logits[actionIdx]
		if bestIdx == -1 || score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	if bestIdx == -1 {
		return legal[0], 0, nil
	}

	return legal[bestIdx], bestIdx, nil
}

func (i *InferenceService) inferWithLSTM(gameID string, input []float32) ([]float32, error) {
	if err := i.StartEngine(); err != nil {
		return nil, err
	}

	sessionMu.Lock()
	defer sessionMu.Unlock()

	if onnxSession == nil || len(inputTensors) < 3 || len(outputTensors) < 3 {
		return nil, fmt.Errorf("onnx engine not initialized or not LSTM model")
	}

	if len(input) != len(inputTensors[0].GetData()) {
		return nil, fmt.Errorf("expected input length %d, got %d", len(inputTensors[0].GetData()), len(input))
	}

	// Get or initialize LSTM state
	lstmStateMu.Lock()
	state, exists := lstmStates[gameID]
	if !exists {
		state = [2][]float32{
			make([]float32, lstmHiddenSize),
			make([]float32, lstmHiddenSize),
		}
		lstmStates[gameID] = state
	}
	lstmStateMu.Unlock()

	// copy inputs: observation, lstm_h, lstm_c
	copy(inputTensors[0].GetData(), input)
	copy(inputTensors[1].GetData(), state[0])
	copy(inputTensors[2].GetData(), state[1])

	// Run inference
	if err := onnxSession.Run(); err != nil {
		return nil, err
	}

	// extract logits
	logits := make([]float32, len(outputTensors[0].GetData()))
	copy(logits, outputTensors[0].GetData())

	// extract and store new state
	newH := make([]float32, len(outputTensors[1].GetData()))
	newC := make([]float32, len(outputTensors[2].GetData()))
	copy(newH, outputTensors[1].GetData())
	copy(newC, outputTensors[2].GetData())

	lstmStateMu.Lock()
	lstmStates[gameID] = [2][]float32{newH, newC}
	lstmStateMu.Unlock()

	return logits, nil
}

// ClearGameState removes LSTM state for a completed game
func (i *InferenceService) ClearGameState(gameID string) {
	lstmStateMu.Lock()
	delete(lstmStates, gameID)
	lstmStateMu.Unlock()
}
