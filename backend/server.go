package cua

import (
	"context"
	"cua/internal/stats"
	"log"
	"net/http"
	"net/http/pprof"
	"os"
	"time"

	"github.com/benbjohnson/clock"
	"github.com/gin-gonic/gin"
)

type Server struct {
	r     *gin.Engine
	srv   *http.Server
	clock clock.Clock
	// signal channel to start the server shutdown
	quitChan chan os.Signal

	// i know it's dirty, but some things need a centralized way to get the DB
	DB Database

	// Stats is the SQLite database behind the admin dashboard. Nil (or not enabled) means the
	// dashboard answers "statistics are not available" instead of pretending there is no data.
	Stats *stats.Store

	// services
	AuthService  AuthService
	JobsService  JobsService
	gameplay     Gameplay
	inferService InferenceService
}

// NewServer wires the HTTP layer. The statistics store is optional (variadic) so every existing
// call site keeps working: a server without it simply has no dashboard data.
func NewServer(r *gin.Engine, clock clock.Clock, db Database, authService AuthService, jobsService JobsService, inferService InferenceService, gameplay Gameplay, statsStores ...*stats.Store) *Server {
	server := &Server{r: r, clock: clock, DB: db, AuthService: authService, JobsService: jobsService, inferService: inferService, gameplay: gameplay}
	if len(statsStores) > 0 {
		server.Stats = statsStores[0]
	}
	return server
}

func (s *Server) Setup() {
	// set routing
	s.r.Use(corsMiddleware())
	api := s.r.Group("/api")
	api.GET("/ping", s.PingCtl)
	api.GET("/ws", s.WsCtl)

	// s.r.GET("/debug/pprof/cmdline", gin.WrapH(http.HandlerFunc(pprof.Cmdline)))
	// s.r.GET("/debug/pprof/profile", gin.WrapH(http.HandlerFunc(pprof.Profile)))
	// s.r.GET("/debug/pprof/symbol", gin.WrapH(http.HandlerFunc(pprof.Symbol)))
	// s.r.GET("/debug/pprof/trace", gin.WrapH(http.HandlerFunc(pprof.Trace)))
	// // For the handlers that return a Handler interface:
	s.r.GET("/debug/pprof/goroutine", gin.WrapH(pprof.Handler("goroutine")))
	// s.r.GET("/debug/pprof/heap", gin.WrapH(pprof.Handler("heap")))
	// s.r.GET("/debug/pprof/threadcreate", gin.WrapH(pprof.Handler("threadcreate")))
	// s.r.GET("/debug/pprof/block", gin.WrapH(pprof.Handler("block")))
	// s.r.GET("/debug/pprof/mutex", gin.WrapH(pprof.Handler("mutex")))

	authGroup := api.Group("/auth")
	authGroup.POST("/sign-up", s.SignUpCtl)
	authGroup.POST("/sign-in", s.SignInCtl)
	authGroup.Use(s.AuthService.AuthMiddleware())
	authGroup.POST("/refresh-token", s.RefreshJwtCtl)
	authGroup.POST("/ws-ticket", s.CreateWsTicketCtl)

	// The admin dashboard. Sign-in is public (it IS the credential check) and everything else
	// hangs off it behind AdminMiddleware, which accepts only a token carrying `role: "admin"`.
	// It is registered before the player middleware below so an admin request is never judged by
	// the player rule, and a player request never reaches these handlers.
	adminGroup := api.Group("/admin")
	adminGroup.POST("/sign-in", s.AdminSignInCtl)
	adminGroup.Use(s.AuthService.AdminMiddleware())
	adminGroup.GET("/stats/overview", s.AdminOverviewCtl)
	adminGroup.GET("/stats/games", s.AdminGamesCtl)
	adminGroup.GET("/stats/games/:id/replay", s.AdminReplayCtl)
	adminGroup.GET("/stats/timeseries", s.AdminTimeseriesCtl)

	api.Use(s.AuthService.AuthMiddleware())
	api.GET("/games", s.MyGamesCtl)
	api.GET("/games/:code", s.GameSnapshotCtl)
	api.POST("/games", s.CreateGameCtl)
	api.POST("/games/:code/join", s.JoinGameCtl)
	api.POST("/games/:code/start", s.StartGameCtl)
	api.POST("/games/:code/play", s.PlayGameCtl)
	api.POST("/games/:code/cancel", s.CancelGameCtl)

	api.GET("/exit", ExitCtl(s.quitChan))
}

func (s *Server) Start() {
	s.srv = &http.Server{
		Addr:    ":8080",
		Handler: s.r,
	}

	if err := s.srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Couldn't start server: %v", err)
	}
}

func (s *Server) Shutdown() {

	log.Println("[+] Received exit signal. Cleaning things up...")

	// Shutdowns and cleaning

	// Shutdown the server
	ctxShut, cancelShut := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancelShut()

	log.Println("[+] Stopping server")
	if err := s.srv.Shutdown(ctxShut); err != nil {
		log.Fatalf("[!] Forced shutdown: %v", err)
	} else {
		log.Println("[+] Server exited gracefully")
	}

	log.Println("[+] Stopping running games")
	canceledCount := s.gameplay.cancelRunningGames()
	if canceledCount > 0 {
		log.Printf("Canceled %d games before shutdown\n", canceledCount)
	}

	log.Println("[+] Shutting down inference engine")
	if err := s.inferService.ShutdownEngine(); err != nil {
		log.Println("[!] Failed to shutdown inference engine:", err)
	}

	log.Println("[+] Shutting down DB")
	if err := s.DB.WriteToDisk(); err != nil {
		log.Println("[!] Failed to save DB to disk")
	}
}

// func cancelRunningGames() {
// 	nCanceled := j.db.CancelInProgressGames()

// 	cua.Sessions.Mu.Lock()
// 	for _, session := range cua.Sessions.M {
// 		close(session.AiTurnChan)
// 	}
// 	cua.Sessions.Mu.Unlock()

// 	if nCanceled > 0 {
// 		log.Printf("Canceled %d games before shutdown\n", nCanceled)
// 	}
// }

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
