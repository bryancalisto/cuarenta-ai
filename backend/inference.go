package cua

import "cua/internal/game"

type InferenceService interface {
	StartEngine() error
	ShutdownEngine() error
	PredictAction(input []float32, legal []game.Action) (game.Action, int, error)
	Infer(input []float32) ([]float32, error)
}
