package cua

import (
	"bytes"
	"context"
	"cua/internal/auth"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/benbjohnson/clock"
	"github.com/gin-gonic/gin"
)

func MockJsonPost(c *gin.Context, content any) {
	c.Request = &http.Request{
		Header: make(http.Header),
	}
	c.Request.Method = "POST"
	c.Request.Header.Set("Content-Type", "application/json")

	jsonbytes, err := json.Marshal(content)
	if err != nil {
		panic(err)
	}

	c.Request.Body = io.NopCloser(bytes.NewBuffer(jsonbytes))
}

type M map[string]any

func TestBindJSONDTO(t *testing.T) {
	t.Run("Sends bad request error and returns error on bad data", func(t *testing.T) {
		r := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(r)
		MockJsonPost(c, M{"secre": "whatever"})
		var expectedDTO ExitDTO

		err := bindJSONDTO(c, &expectedDTO)

		if err == nil {
			t.Errorf("Expected error")
		}

		if r.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", r.Code)
		}
	})

	t.Run("Does not send bad request error and returns nil error on good data", func(t *testing.T) {
		r := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(r)
		MockJsonPost(c, M{"secret": "whatever"})
		var expectedDTO ExitDTO

		err := bindJSONDTO(c, &expectedDTO)

		if err != nil {
			t.Errorf("Expected error to be nil")
		}

		if r.Code != 200 {
			t.Errorf("expected default status (200), got %d", r.Code)
		}
	})
}

// newAuthTestServer wires just enough of the server for the auth handlers: they only touch the
// database and the auth service, so jobs/inference/gameplay stay nil.
func newAuthTestServer(t *testing.T) *Server {
	t.Helper()
	mockClock := clock.NewMock()
	return NewServer(
		gin.New(),
		mockClock,
		NewMemDatabase(context.Background(), mockClock),
		auth.NewAuthService(),
		nil,
		nil,
		nil,
	)
}

// callAuth runs one of the auth handlers against a JSON body and returns the status and body.
func callAuth(t *testing.T, s *Server, handler gin.HandlerFunc, body M) (int, map[string]any) {
	t.Helper()
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	MockJsonPost(c, body)

	handler(c)

	decoded := map[string]any{}
	_ = json.Unmarshal(recorder.Body.Bytes(), &decoded)
	return recorder.Code, decoded
}

// Sign-in identity is a USERNAME, not an email. These tests pin the contract the web UI speaks:
// the field name, the shape rules, case-insensitive uniqueness, and that the response never
// carries a login address.
func TestAuthWithUsernames(t *testing.T) {
	t.Run("signs up with a username", func(t *testing.T) {
		s := newAuthTestServer(t)

		status, body := callAuth(t, s, s.SignUpCtl, M{"username": "Ana", "password": "secret"})

		if status != http.StatusOK {
			t.Fatalf("expected 200, got %d (%v)", status, body)
		}
		if body["accessToken"] == nil || body["accessToken"] == "" {
			t.Errorf("expected an access token, got %v", body["accessToken"])
		}
		user, ok := body["user"].(map[string]any)
		if !ok {
			t.Fatalf("expected a user object, got %v", body["user"])
		}
		if user["username"] != "Ana" {
			t.Errorf("expected the username to round-trip as typed, got %v", user["username"])
		}
		if _, present := user["email"]; present {
			t.Error("expected the response to have no email field")
		}
	})

	t.Run("rejects an email as a username", func(t *testing.T) {
		s := newAuthTestServer(t)

		status, body := callAuth(t, s, s.SignUpCtl, M{"username": "ana@example.com", "password": "secret"})

		if status != http.StatusBadRequest {
			t.Errorf("expected 400 for an email-shaped username, got %d (%v)", status, body)
		}
	})

	t.Run("rejects the old email payload", func(t *testing.T) {
		s := newAuthTestServer(t)

		status, _ := callAuth(t, s, s.SignUpCtl, M{"email": "ana@example.com", "password": "secret"})

		if status != http.StatusBadRequest {
			t.Errorf("expected 400 when the client still sends `email`, got %d", status)
		}
	})

	t.Run("rejects a username already taken in another casing", func(t *testing.T) {
		s := newAuthTestServer(t)
		callAuth(t, s, s.SignUpCtl, M{"username": "Ana", "password": "secret"})

		status, body := callAuth(t, s, s.SignUpCtl, M{"username": "ana", "password": "other"})

		if status != http.StatusConflict {
			t.Errorf("expected 409 for a duplicate username, got %d (%v)", status, body)
		}
	})

	t.Run("signs in with any casing and the right password", func(t *testing.T) {
		s := newAuthTestServer(t)
		callAuth(t, s, s.SignUpCtl, M{"username": "Ana", "password": "secret"})

		status, body := callAuth(t, s, s.SignInCtl, M{"username": "ANA", "password": "secret"})

		if status != http.StatusOK {
			t.Fatalf("expected 200, got %d (%v)", status, body)
		}
		user, _ := body["user"].(map[string]any)
		if user["username"] != "Ana" {
			t.Errorf("expected the stored spelling in the response, got %v", user["username"])
		}
	})

	t.Run("rejects a wrong password and an unknown user alike", func(t *testing.T) {
		s := newAuthTestServer(t)
		callAuth(t, s, s.SignUpCtl, M{"username": "Ana", "password": "secret"})

		for _, attempt := range []M{
			{"username": "Ana", "password": "nope"},
			{"username": "nobody", "password": "secret"},
		} {
			status, body := callAuth(t, s, s.SignInCtl, attempt)
			if status != http.StatusUnauthorized {
				t.Errorf("expected 401 for %v, got %d (%v)", attempt, status, body)
			}
			// The message must not reveal which of the two was wrong.
			if message, _ := body["error"].(string); message != "Invalid username or password" {
				t.Errorf("expected a neutral error message, got %q", message)
			}
		}
	})
}
