package main

import (
	"context"
	"cuarenta/internal/auth"
	"cuarenta/internal/models"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

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

func main() {
	r := gin.Default()
	DB = NewDatabase()
	// add default user for convenient testing
	userId := getUid()
	passwHash, _ := auth.HashPassword("1234")
	DB.users[userId] = models.User{Id: userId, Email: "1@1.com", Password: passwHash}
	r.Use(corsMiddleware())

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "pong",
		})
	})

	authGroup := r.Group("/auth")
	authGroup.POST("/sign-up", SignUpCtl)
	authGroup.POST("/sign-in", SignInCtl)

	api := r.Group("")
	api.Use(auth.AuthMiddleware())
	api.GET("/games/:code", GameSnapshotCtl)
	api.POST("/games", CreateGameCtl)
	api.POST("/games/:code/join", JoinGameCtl)
	api.POST("/games/:code/start", StartGameCtl)

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

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("[!] Forced shutdown: %v", err)
	} else {
		log.Println("[+] Server exited gracefully")
	}
}
