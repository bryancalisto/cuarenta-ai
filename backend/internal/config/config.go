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
