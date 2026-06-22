package cua

import "github.com/gin-gonic/gin"

type AuthService interface {
	GenerateJWT(userId string) (string, error)
	ValidateJWT(tokenString string) (bool, string, error)
	RefreshJWT(userId string) (string, error)
	AuthMiddleware() gin.HandlerFunc
	HashPassword(password string) (string, error)
	CheckPasswordHash(password, hash string) bool
}
