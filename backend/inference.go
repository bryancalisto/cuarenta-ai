package cua

import (
	"cua/internal/search"
)

type InferenceService interface {
	StartEngine() error
	ShutdownEngine() error
	// PredictActionSlot runs one decision of the neural policy and returns the slot of
	// the chosen action inside the caller's own legal-action list, plus the model's
	// value estimate for the mover's team ([-1,1]). The slot indexes the same []Action
	// list the feature block was built from, so `legal[slot]` is the action to play.
	PredictActionSlot(gameID string, obs []float32, feats []float32, mask []float32, nLegal int) (int, float32, error)
	// AdvanceRecurrentState runs the same forward pass for a turn the network does NOT
	// decide (a human turn) and keeps the new recurrent state, so the policy's memory
	// covers every turn of the match — the contract it was trained under (see
	// src/ppo/trainer.py: the state is advanced on EVERY turn, with the acting player's
	// observation). The action the caller then plays is the caller's business, and the
	// slot the model returned is discarded.
	AdvanceRecurrentState(gameID string, obs []float32, feats []float32, mask []float32, nLegal int) error
	// ClearGameState drops everything kept for a game: the recurrent (LSTM) state and the
	// search's per-seat history.
	ClearGameState(gameID string)

	// SearchEvaluator returns the ISMCTS evaluator for the loaded model: the adapter the
	// server-side search uses to run one model call per simulation, carrying the model's
	// value target (and scale) so the search optimises the objective the policy learned.
	// It fails for a stateful (PPO) model, and for a model whose sidecar is missing or
	// inconsistent — the caller then falls back to the bare policy.
	SearchEvaluator() (search.ModelEvaluator, error)

	// RecordHistory appends one completed decision of `seat` (the observation it saw and
	// the features of the action it took), which is the window the search's root evaluation
	// feeds to the network. `src/ismcts/history.py: HistoryTracker` keeps exactly this.
	RecordHistory(gameID string, seat int, obs, actionFeatures []float32)

	// History returns that window, oldest step first.
	History(gameID string, seat int) []search.HistoryStep
}
