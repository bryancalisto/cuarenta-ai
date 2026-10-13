package config

import (
	"log"
	"sync"

	envpkg "github.com/caarlos0/env/v11"
	"github.com/joho/godotenv"
)

type Environ struct {
	JwtKey                   string `env:"JWT_KEY"`
	OnnxModelPath            string `env:"ONNX_MODEL_PATH"`
	OnnxRuntimeSharedLibPath string `env:"ONNXRUNTIME_SHARED_LIBRARY_PATH"`
	DbPath                   string `env:"DB_PATH"`

	// AiSearchIterations is how many ISMCTS simulations each AI turn runs. 0 (or unset)
	// keeps the old behaviour: one forward pass, argmax over the policy.
	//
	// This is the strength/latency knob, and it is worth being explicit about what it buys.
	// The served network is a POLICY: on its own it is worth about a coin flip against the
	// rule-based heuristic (measured over 200 games: 49.0%), while the same weights WITH the
	// search win 77%. Measured budget curve per AI turn, against the heuristic:
	//
	//	0 sims   49.0%   (policy only, no search)
	//	32 sims  57.0%
	//	64 sims  59.0%
	//	128 sims 72.0%
	//	400 sims 77.0%
	//
	// Each simulation costs one model forward (~1-3 ms at batch 1), so 128 is roughly
	// 0.15-0.4 s of "thinking" per turn.
	AiSearchIterations int     `env:"AI_SEARCH_ITERATIONS" envDefault:"128"`
	AiSearchCPuct      float64 `env:"AI_SEARCH_CPUCT" envDefault:"1.5"`

	// AiDealGraceMs is how long the FIRST AI move of a deal is held back, so the client's deal
	// animation (five cards to one seat, then the next) has finished before a card flies across
	// the board. Keep it in step with the web UI: it is DEAL_TOTAL_MS + DEAL_HOLD_MS in
	// web-ui/src/lib/GameBoard.svelte, which the deal timings there define. 0 disables the wait.
	AiDealGraceMs int `env:"AI_DEAL_GRACE_MS" envDefault:"6500"`
}

var env Environ
var once sync.Once

func GetEnv() Environ {
	once.Do(func() {
		var err error
		err = godotenv.Load()

		if err != nil {
			log.Fatalf("Couldn't not load environment variables: %v", err)
		}

		env, err = envpkg.ParseAs[Environ]()

		if err != nil {
			log.Fatalf("Couldn't parse environment variables: %v", err)
		}

		if err != nil {
			log.Fatalf("Couldn't parse environment variables 2: %v", err)
		}
	})

	return env
}
