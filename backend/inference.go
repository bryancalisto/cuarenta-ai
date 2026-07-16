package cua

import "cua/internal/game"

type InferenceService interface {
	StartEngine() error
	ShutdownEngine() error
	PredictAction(gameID string, input []float32, legal []game.Action) (game.Action, int, error)
	ClearGameState(gameID string)
	Infer(input []float32) ([]float32, error)
}
