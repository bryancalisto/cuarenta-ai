package cua

import "github.com/gin-gonic/gin"

type AuthService interface {
	GenerateJWT(userId string) (string, error)
	ValidateJWT(tokenString string) (bool, string, error)
	RefreshJWT(userId string) (string, error)
	AuthMiddleware() gin.HandlerFunc
	HashPassword(password string) (string, error)
	CheckPasswordHash(password, hash string) bool

	// The dashboard's own audience: a separate token (role: admin) and a separate middleware, so
	// that a player's token cannot open the statistics and an admin token cannot play games.
	GenerateAdminJWT(username string) (string, error)
	AdminMiddleware() gin.HandlerFunc
}
