package auth

import (
	"crypto/subtle"
	"cua/internal/config"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

type AuthService struct{}

// ---------------------------------------------------------------------------------------------
// Admin tokens
//
// The dashboard is a separate audience from the game: its credential comes from
// ADMIN_USERNAME/ADMIN_PASSWORD in the environment (not from the user table), and its token
// carries `role: "admin"` so that a stolen *player* token cannot open the statistics. The two
// kinds of token are signed with the same key, and each validator refuses the other's kind:
// `ValidateJWT` requires `userId`, `ValidateAdminJWT` requires the role claim.
// ---------------------------------------------------------------------------------------------

// AdminRole is the claim value that marks a token as the dashboard's.
const AdminRole = "admin"

// AdminUsername reports the configured admin account, and whether the dashboard is enabled at all.
// No ADMIN_PASSWORD means no admin: the endpoints stay closed instead of falling back to a default.
func AdminUsername() (string, bool) {
	env := config.GetEnv()
	username := strings.TrimSpace(env.AdminUsername)
	if username == "" || env.AdminPassword == "" {
		return "", false
	}
	return username, true
}

// CheckAdminCredentials compares the credentials the dashboard sent with the configured ones.
// The comparison is constant-time: a byte-by-byte `==` leaks how much of the password matched.
func CheckAdminCredentials(username, password string) bool {
	wantUser, ok := AdminUsername()
	if !ok {
		return false
	}
	wantPass := config.GetEnv().AdminPassword
	userMatches := subtle.ConstantTimeCompare([]byte(strings.TrimSpace(username)), []byte(wantUser)) == 1
	passMatches := subtle.ConstantTimeCompare([]byte(password), []byte(wantPass)) == 1
	// Both comparisons are always performed, so the timing does not say which one failed.
	return userMatches && passMatches
}

// GenerateAdminJWT signs a token for the dashboard account.
func (a *AuthService) GenerateAdminJWT(username string) (string, error) {
	t := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"role":     AdminRole,
		"username": username,
		// The dashboard is an operations tool, not a session: it expires on its own so a token
		// left in a browser does not stay valid forever.
		"exp": time.Now().Add(AdminTokenTTL).Unix(),
	})
	return t.SignedString([]byte(config.GetEnv().JwtKey))
}

// ValidateAdminJWT accepts only a live admin token, returning its username.
func (a *AuthService) ValidateAdminJWT(tokenString string) (string, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, jwt.ErrSignatureInvalid
		}
		return []byte(config.GetEnv().JwtKey), nil
	}, jwt.WithExpirationRequired())

	if err != nil {
		return "", err
	}
	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok || !token.Valid {
		return "", jwt.ErrTokenMalformed
	}
	// A player token has no `role`, so it is rejected here even though it is otherwise valid.
	if role, _ := claims["role"].(string); role != AdminRole {
		return "", ErrNotAdmin
	}
	username, _ := claims["username"].(string)
	return username, nil
}

// AdminMiddleware guards the dashboard endpoints. It is registered on its own group, so the
// player middleware never runs for an admin request and vice versa.
func (a *AuthService) AdminMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		parts := strings.Split(c.GetHeader("Authorization"), " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Admin authentication required"})
			return
		}

		username, err := a.ValidateAdminJWT(parts[1])
		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Admin authentication required"})
			return
		}

		c.Set("adminUsername", username)
		c.Next()
	}
}

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

// AdminTokenTTL is how long a dashboard token stays valid.
const AdminTokenTTL = 12 * time.Hour

// ErrNotAdmin is returned when a token is valid but belongs to a player instead of the dashboard.
var ErrNotAdmin = errors.New("token is not an admin token")
