package infer

import (
	"cuarenta/internal/config"
	"cuarenta/internal/game"
	"fmt"
	"log"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	once         sync.Once
	initErr      error
	sessionMu    sync.Mutex
	onnxSession  *ort.AdvancedSession
	InputTensor  *ort.Tensor[float32]
	OutputTensor *ort.Tensor[float32]
	inputName    string
	outputName   string
)

func StartEngine() error {
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
		if len(inputs) != 1 || len(outputs) != 1 {
			initErr = fmt.Errorf("expected exactly 1 input and 1 output, got %d inputs and %d outputs", len(inputs), len(outputs))
			return
		}
		if inputs[0].OrtValueType != ort.ONNXTypeTensor || outputs[0].OrtValueType != ort.ONNXTypeTensor {
			initErr = fmt.Errorf("expected tensor input/output")
			return
		}
		if inputs[0].DataType != ort.TensorElementDataTypeFloat || outputs[0].DataType != ort.TensorElementDataTypeFloat {
			initErr = fmt.Errorf("expected float32 input/output tensors")
			return
		}

		inputName = inputs[0].Name
		outputName = outputs[0].Name

		inputShape := normalizeShape(inputs[0].Dimensions)
		outputShape := normalizeShape(outputs[0].Dimensions)

		inputData := make([]float32, inputShape.FlattenedSize())
		outputData := make([]float32, outputShape.FlattenedSize())

		InputTensor, err = ort.NewTensor[float32](inputShape, inputData)
		if err != nil {
			initErr = fmt.Errorf("create input tensor: %w", err)
			return
		}

		OutputTensor, err = ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			initErr = fmt.Errorf("create output tensor: %w", err)
			return
		}

		// Ensure the output buffer is zeroed and has the expected size.
		copy(OutputTensor.GetData(), outputData)

		onnxSession, err = ort.NewAdvancedSession(
			modelPath,
			[]string{inputName},
			[]string{outputName},
			[]ort.Value{InputTensor},
			[]ort.Value{OutputTensor},
			nil,
		)
		if err != nil {
			initErr = fmt.Errorf("create onnx session: %w", err)
			return
		}
	})

	return initErr
}

func ShutdownEngine() error {
	var firstErr error

	if onnxSession != nil {
		if err := onnxSession.Destroy(); err != nil && firstErr == nil {
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

func Infer(input []float32) ([]float32, error) {
	if err := StartEngine(); err != nil {
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

func PredictAction(input []float32, legal []game.Action) (game.Action, int, error) {
	logits, err := Infer(input)
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
