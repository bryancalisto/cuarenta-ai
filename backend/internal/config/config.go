package config

import (
	"log"
	"sync"

	envpkg "github.com/caarlos0/env/v11"
)

type Environ struct {
	JwtKey string `env:"JWT_KEY"`
}

var env Environ
var once sync.Once

func GetEnv() Environ {
	once.Do(func() {
		var err error
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
