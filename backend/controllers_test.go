package cua

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

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

func TestSignUpCtl(t *testing.T) {
	t.Run("StatusConflict if user already exists", func(t *testing.T) {})
	t.Run("StatusConflict if user already exists", func(t *testing.T) {})
}
