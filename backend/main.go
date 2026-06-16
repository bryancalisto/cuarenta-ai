package main

import (
	"context"
	"cuarenta/internal/auth"
	infer "cuarenta/internal/inference"
	"cuarenta/internal/models"
	"log"
	"net/http"
	"os"
	"os/signal"
	"reflect"
	"syscall"
	"time"

	"github.com/benbjohnson/clock"
	"github.com/gin-gonic/gin"
	"github.com/lithammer/shortuuid/v4"
)

func getUid() string {
	return shortuuid.New()
}

var DB *Database

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.GetHeader("Origin")
		if origin != "" {
			c.Header("Access-Control-Allow-Origin", origin)
			c.Header("Vary", "Origin")
		} else {
			c.Header("Access-Control-Allow-Origin", "*")
		}
		c.Header("Access-Control-Allow-Credentials", "true")
		c.Header("Access-Control-Allow-Headers", "Authorization, Content-Type")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

func shutdown() {
	if err := infer.ShutdownEngine(); err != nil {
		log.Println("failed to shutdown inference engine:", err)
	}
}

type RecurrentJob struct {
	ticker  *time.Ticker
	handler func()
}

func runRecurrentJobs(ctx context.Context, recurrentJobs []RecurrentJob) {
	defer func() {
		for _, j := range recurrentJobs {
			j.ticker.Stop()
		}
	}()

	var switchCases []reflect.SelectCase
	for _, j := range recurrentJobs {
		switchCases = append(switchCases, reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(j.ticker.C)})
	}
	switchCases = append(switchCases, reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(ctx.Done())})

	for {
		chosenIndex, _, ok := reflect.Select(switchCases)
		if !ok || chosenIndex == len(switchCases)-1 {
			return
		}
		if ok && chosenIndex < len(recurrentJobs) {
			recurrentJobs[chosenIndex].handler()
		}
	}
}

func deleteExpiredWsTicketsHandler() {
	nDeleted := DB.DeleteExpiredWsTickets()
	log.Printf("Deleted %d tickets\n", nDeleted)
}

func main() {
	r := gin.Default()
	DB = NewDatabase(clock.New())

	// Start recurrent jobs runner
	recurrentJobsCtx, cancelRecurrentJobs := context.WithCancel(context.Background())
	defer cancelRecurrentJobs()
	recurrentJobs := []RecurrentJob{{ticker: time.NewTicker(1 * time.Minute), handler: deleteExpiredWsTicketsHandler}}
	go runRecurrentJobs(recurrentJobsCtx, recurrentJobs)

	// add default user for convenient testing
	userId := getUid()
	passwHash, _ := auth.HashPassword("1234")
	DB.users[userId] = models.User{Id: userId, Email: "1@1.com", Password: passwHash}
	r.Use(corsMiddleware())

	if err := infer.StartEngine(); err != nil {
		log.Fatalf("Couldn't start inference engine: %v", err)
	}
	defer shutdown()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "pong",
		})
	})

	// r.GET("/debug/pprof/cmdline", gin.WrapH(http.HandlerFunc(pprof.Cmdline)))
	// r.GET("/debug/pprof/profile", gin.WrapH(http.HandlerFunc(pprof.Profile)))
	// r.GET("/debug/pprof/symbol", gin.WrapH(http.HandlerFunc(pprof.Symbol)))
	// r.GET("/debug/pprof/trace", gin.WrapH(http.HandlerFunc(pprof.Trace)))
	// // For the handlers that return a Handler interface:
	// r.GET("/debug/pprof/goroutine", gin.WrapH(pprof.Handler("goroutine")))
	// r.GET("/debug/pprof/heap", gin.WrapH(pprof.Handler("heap")))
	// r.GET("/debug/pprof/threadcreate", gin.WrapH(pprof.Handler("threadcreate")))
	// r.GET("/debug/pprof/block", gin.WrapH(pprof.Handler("block")))
	// r.GET("/debug/pprof/mutex", gin.WrapH(pprof.Handler("mutex")))

	authGroup := r.Group("/auth")
	authGroup.POST("/sign-up", SignUpCtl)
	authGroup.POST("/sign-in", SignInCtl)
	authGroup.Use(auth.AuthMiddleware())
	authGroup.POST("/ws-ticket", CreateWsTicketCtl)

	api := r.Group("")
	api.Use(auth.AuthMiddleware())
	api.GET("/games/:code", GameSnapshotCtl)
	api.POST("/games", CreateGameCtl)
	api.POST("/games/:code/join", JoinGameCtl)
	api.POST("/games/:code/start", StartGameCtl)
	api.POST("/games/:code/play", PlayGameCtl)

	r.GET("/exit", ExitCtl(quit))
	r.GET("/ws", WsCtl)

	srv := &http.Server{
		Addr:    ":8080",
		Handler: r,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Couldn't start server: %v", err)
		}
	}()

	<-quit

	log.Println("[+] Received exit signal. Cleaning things up...")

	// Shutdowns and cleaning

	// Shutdown the server
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("[!] Forced shutdown: %v", err)
	} else {
		log.Println("[+] Server exited gracefully")
	}

	// Shutdown the recurrent jobs
	cancelRecurrentJobs()
	log.Println("[+] Finishing recurrent Jobs")
}
