package main

import (
	"context"
	"cua"
	"cua/internal/auth"
	infer "cua/internal/inference"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/benbjohnson/clock"
	"github.com/gin-gonic/gin"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	clock := clock.New()

	// DB
	db := cua.NewMemDatabase(ctx, clock)
	db.Setup()

	// HTTP server
	r := gin.Default()
	authService := auth.NewAuthService()
	jobsService := cua.NewMainJobsService(db)
	inferService := infer.NewInferenceService()
	gameplayHandler := cua.NewMainGameplay(db, clock, inferService)
	server := cua.NewServer(r, clock, db, authService, jobsService, inferService, gameplayHandler)
	server.Setup()

	// jobs
	go server.JobsService.RunRecurrentJobs(ctx)

	// AI inference
	if err := inferService.StartEngine(); err != nil {
		log.Fatalf("Couldn't start inference engine: %v", err)
	}

	go server.Start()
	defer server.Shutdown()

	<-quit
}
