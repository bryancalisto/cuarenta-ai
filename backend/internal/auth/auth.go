package auth

import (
	"cua/internal/config"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

type AuthService struct{}

func NewAuthService() *AuthService {
	return &AuthService{}
}

func (a *AuthService) GenerateJWT(userId string) (string, error) {
	t := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"userId": userId,
	})
	s, err := t.SignedString([]byte(config.GetEnv().JwtKey))

	if err != nil {
		return "", err
	}

	return s, nil
}

func (a *AuthService) ValidateJWT(tokenString string) (bool, string, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, jwt.ErrSignatureInvalid
		}
		return []byte(config.GetEnv().JwtKey), nil
	})

	if err != nil {
		return false, "", err
	}
	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return false, "", jwt.ErrTokenMalformed
	}

	userId, ok := claims["userId"].(string)
	if !ok {
		return false, "", jwt.ErrTokenMalformed
	}

	return token.Valid, userId, nil
}

func (a *AuthService) RefreshJWT(userId string) (string, error) {
	return a.GenerateJWT(userId)
}

func (a *AuthService) AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Unauthorized"})
			return
		}

		tokenString := parts[1]

		valid, userId, err := a.ValidateJWT(tokenString)

		if err != nil || !valid {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Unauthorized"})
			return
		}

		c.Set("userId", userId)

		c.Next()
	}
}

func (a *AuthService) HashPassword(password string) (string, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	return string(bytes), err
}

func (a *AuthService) CheckPasswordHash(password, hash string) bool {
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}
