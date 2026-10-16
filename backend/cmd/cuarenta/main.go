package main

import (
	"context"
	"cua"
	"cua/internal/auth"
	"cua/internal/config"
	infer "cua/internal/inference"
	"cua/internal/stats"
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

	// Statistics for the admin dashboard, in their own SQLite file. A failure here must not stop
	// the game server: without the store the dashboard has nothing to show, and everything else
	// keeps working.
	statsStore, err := stats.Open(config.GetEnv().StatsDbPath)
	if err != nil {
		log.Printf("statistics are off: %v", err)
		statsStore = nil
	} else {
		defer statsStore.Close()
		log.Printf("Recording game statistics to %s", config.GetEnv().StatsDbPath)
		if _, configured := auth.AdminUsername(); !configured {
			log.Printf("ADMIN_USERNAME/ADMIN_PASSWORD are not set: the dashboard stays closed")
		}
	}

	// HTTP server
	r := gin.Default()
	authService := auth.NewAuthService()
	jobsService := cua.NewMainJobsService(db)
	inferService := infer.NewInferenceService()
	gameplayHandler := cua.NewMainGameplay(db, clock, inferService, stats.NewRecorder(statsStore))
	server := cua.NewServer(r, clock, db, authService, jobsService, inferService, gameplayHandler, statsStore)
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
