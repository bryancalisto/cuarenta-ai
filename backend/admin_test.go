package cua

import (
	"bytes"
	"context"
	"cua/internal/auth"
	"cua/internal/config"
	"cua/internal/stats"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/benbjohnson/clock"
	"github.com/gin-gonic/gin"
)

// The dashboard is the only part of this service that shows internal data, so its access rules get
// tested from the outside: through the real router, with real tokens, checking what each kind of
// caller can reach. A permission that is only enforced in the UI is not enforced at all.

// adminTestServer builds the full router with a statistics database and the real auth service.
func adminTestServer(t *testing.T, store *stats.Store) *Server {
	t.Helper()
	gin.SetMode(gin.TestMode)

	mockClock := clock.NewMock()
	server := NewServer(
		gin.New(),
		mockClock,
		NewMemDatabase(context.Background(), mockClock),
		auth.NewAuthService(),
		nil,
		nil,
		nil,
		store,
	)
	server.Setup()
	return server
}

func doRequest(t *testing.T, server *Server, method, path, token string, body any) *httptest.ResponseRecorder {
	t.Helper()

	var payload *bytes.Buffer
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("could not encode the request body: %v", err)
		}
		payload = bytes.NewBuffer(encoded)
	} else {
		payload = bytes.NewBuffer(nil)
	}

	request := httptest.NewRequest(method, path, payload)
	request.Header.Set("Content-Type", "application/json")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}

	recorder := httptest.NewRecorder()
	server.r.ServeHTTP(recorder, request)
	return recorder
}

// configuredAdmin returns the credentials the server is actually configured with, so the test
// asserts the real policy instead of a copy of it.
func configuredAdmin(t *testing.T) (string, string) {
	t.Helper()
	username, ok := auth.AdminUsername()
	if !ok {
		t.Skip("ADMIN_USERNAME/ADMIN_PASSWORD are not set in backend/.env: nothing to sign in with")
	}
	return username, config.GetEnv().AdminPassword
}

func adminToken(t *testing.T, server *Server) string {
	t.Helper()
	username, password := configuredAdmin(t)

	recorder := doRequest(t, server, http.MethodPost, "/api/admin/sign-in", "",
		AdminSignInDTO{Username: username, Password: password})
	if recorder.Code != http.StatusOK {
		t.Fatalf("admin sign-in failed with %d: %s", recorder.Code, recorder.Body.String())
	}

	var response struct {
		AccessToken string `json:"accessToken"`
		Admin       struct {
			Username string `json:"username"`
		} `json:"admin"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("could not read the sign-in response: %v", err)
	}
	if response.AccessToken == "" {
		t.Fatal("sign-in must return a token")
	}
	if response.Admin.Username != username {
		t.Errorf("expected the response to name the admin %q, got %q", username, response.Admin.Username)
	}
	return response.AccessToken
}

func TestAdminSignInRejectsWrongCredentials(t *testing.T) {
	store, _ := statsTestStore(t)
	server := adminTestServer(t, store)
	username, _ := configuredAdmin(t)

	for _, attempt := range []AdminSignInDTO{
		{Username: username, Password: "definitely-not-the-password"},
		{Username: "not-the-admin", Password: "whatever"},
		{Username: ""},
	} {
		recorder := doRequest(t, server, http.MethodPost, "/api/admin/sign-in", "", attempt)
		if recorder.Code != http.StatusUnauthorized && recorder.Code != http.StatusBadRequest {
			t.Errorf("attempt %+v should be refused, got %d", attempt, recorder.Code)
		}
	}
}

// TestAdminRoutesRequireAnAdminToken is the access matrix: no token, a player's token, and the
// admin's token, against every dashboard route.
func TestAdminRoutesRequireAnAdminToken(t *testing.T) {
	store, recorder := statsTestStore(t)
	server := adminTestServer(t, store)
	token := adminToken(t, server)

	// A perfectly valid PLAYER token: the dashboard must refuse it, or every player could read the
	// statistics.
	playerToken, err := server.AuthService.GenerateJWT("some-user")
	if err != nil {
		t.Fatalf("could not mint a player token: %v", err)
	}

	// Record one game so the endpoints have something to answer with.
	handler, gameObj, session := statsCleanFixture(t, recorder)
	playRecordedGame(t, handler, gameObj, session, 3)

	for _, path := range []string{
		"/api/admin/stats/overview",
		"/api/admin/stats/games",
		"/api/admin/stats/timeseries",
	} {
		t.Run("no token: "+path, func(t *testing.T) {
			if code := doRequest(t, server, http.MethodGet, path, "", nil).Code; code != http.StatusUnauthorized {
				t.Errorf("expected 401 without a token, got %d", code)
			}
		})
		t.Run("player token: "+path, func(t *testing.T) {
			if code := doRequest(t, server, http.MethodGet, path, playerToken, nil).Code; code != http.StatusUnauthorized {
				t.Errorf("expected 401 for a player token, got %d", code)
			}
		})
		t.Run("admin token: "+path, func(t *testing.T) {
			if code := doRequest(t, server, http.MethodGet, path, token, nil).Code; code != http.StatusOK {
				t.Errorf("expected 200 for the admin token, got %d", code)
			}
		})
	}
}

// TestAdminTokenCannotPlayGames is the other direction: the dashboard's token is not a session for
// the game, and using it to play would be a privilege escalation in reverse.
func TestAdminTokenCannotPlayGames(t *testing.T) {
	store, _ := statsTestStore(t)
	server := adminTestServer(t, store)
	token := adminToken(t, server)

	for _, path := range []string{"/api/games"} {
		if code := doRequest(t, server, http.MethodGet, path, token, nil).Code; code != http.StatusUnauthorized {
			t.Errorf("expected 401 when playing with an admin token on %s, got %d", path, code)
		}
	}
}

// TestAdminOverviewReadsTheRecordedGames: the dashboard's numbers must come from the recorded
// games, end to end through the HTTP layer.
func TestAdminOverviewReadsTheRecordedGames(t *testing.T) {
	store, recorder := statsTestStore(t)
	server := adminTestServer(t, store)
	token := adminToken(t, server)

	handler, gameObj, session := statsCleanFixture(t, recorder)
	played := playRecordedGame(t, handler, gameObj, session, 12)

	recorderOut := doRequest(t, server, http.MethodGet, "/api/admin/stats/overview", token, nil)
	if recorderOut.Code != http.StatusOK {
		t.Fatalf("overview failed with %d: %s", recorderOut.Code, recorderOut.Body.String())
	}

	var overview stats.Overview
	if err := json.Unmarshal(recorderOut.Body.Bytes(), &overview); err != nil {
		t.Fatalf("could not read the overview: %v", err)
	}
	if overview.Totals.Games != 1 {
		t.Errorf("expected the recorded game in the totals, got %d", overview.Totals.Games)
	}
	if overview.Totals.Plies != played {
		t.Errorf("the dashboard should count the %d recorded plies, it counted %d", played, overview.Totals.Plies)
	}
	if len(overview.BySeatKind) != 2 {
		t.Errorf("expected a win rate per seat kind, got %d", len(overview.BySeatKind))
	}
	if len(overview.RecentGames) != 1 || len(overview.RecentGames[0].Seats) != 4 {
		t.Errorf("the recent list must carry the game and its seats: %+v", overview.RecentGames)
	}
}

// TestAdminReplayEndpointServesFrames: one request per seat per ply is what the viewer needs, so
// the endpoint is checked for its shape, not just its status.
func TestAdminReplayEndpointServesFrames(t *testing.T) {
	store, recorder := statsTestStore(t)
	server := adminTestServer(t, store)
	token := adminToken(t, server)

	handler, gameObj, session := statsCleanFixture(t, recorder)
	played := playRecordedGame(t, handler, gameObj, session, 10)

	response := doRequest(t, server, http.MethodGet, "/api/admin/stats/games/"+gameObj.Id+"/replay", token, nil)
	if response.Code != http.StatusOK {
		t.Fatalf("replay failed with %d: %s", response.Code, response.Body.String())
	}

	var replay Replay
	if err := json.Unmarshal(response.Body.Bytes(), &replay); err != nil {
		t.Fatalf("could not read the replay: %v", err)
	}
	if len(replay.Frames) != played+1 {
		t.Errorf("expected %d frames (the opening deal included), got %d", played+1, len(replay.Frames))
	}
	if replay.Game.ID != gameObj.Id {
		t.Errorf("the replay should carry its game row, got %q", replay.Game.ID)
	}
	opening := replay.Frames[0]
	if opening.Move != nil || len(opening.Hands) != 4 {
		t.Errorf("the first frame is the opening position with four hands: %+v", opening)
	}
	second := replay.Frames[1]
	if second.Move == nil || second.Seat == nil || second.Kind == nil {
		t.Errorf("every frame after the opening one describes a move: %+v", second)
	}
	if second.Scores[0]+second.Scores[1] < 0 {
		t.Error("scores must be present")
	}

	// An unknown game is a 404 the dashboard can show as "no replay", not a 500.
	if code := doRequest(t, server, http.MethodGet, "/api/admin/stats/games/ZZZZ/replay", token, nil).Code; code != http.StatusNotFound {
		t.Errorf("expected 404 for an unknown game, got %d", code)
	}
}

// TestAdminRoutesWithoutAStatisticsDatabase: a server with no stats must say so instead of
// answering with an empty dashboard, which would look exactly like "no games played yet".
func TestAdminRoutesWithoutAStatisticsDatabase(t *testing.T) {
	server := adminTestServer(t, nil)
	token := adminToken(t, server)

	code := doRequest(t, server, http.MethodGet, "/api/admin/stats/overview", token, nil).Code
	if code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 when statistics are not configured, got %d", code)
	}
}

// TestAdminJSONSendsArraysNotEmpty is a regression guard for a class of bug that no Go test would
// otherwise catch and no compiler can see: a nil slice marshals to `null`, and the dashboard
// iterates these fields. `null.length` is a blank page, so the wire contract is "array, always".
func TestAdminJSONSendsArraysNotEmpty(t *testing.T) {
	store, recorder := statsTestStore(t)
	server := adminTestServer(t, store)
	token := adminToken(t, server)

	handler, gameObj, session := statsCleanFixture(t, recorder)
	playRecordedGame(t, handler, gameObj, session, 6)

	assertNoNulls := func(t *testing.T, body []byte, where string) {
		t.Helper()
		var decoded any
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("%s: could not decode: %v", where, err)
		}
		var walk func(value any, path string)
		walk = func(value any, path string) {
			switch typed := value.(type) {
			case map[string]any:
				for key, child := range typed {
					if child == nil {
						// Null is fine for a field that MEANS "nothing": an absent winner, a day
						// with no win rate, a last-placed card that does not exist.
						switch key {
						case "winner", "loser", "seat", "kind", "move", "lastPlaced",
							"startedAt", "endedAt", "aiTeamWinRate", "humanTeamWinRate":
							continue
						}
					}
					walk(child, path+"."+key)
				}
			case []any:
				for index, child := range typed {
					walk(child, path+"[]")
					_ = index
				}
			case nil:
				t.Errorf("%s: %s is null, and the dashboard expects an array or a value there", where, path)
			}
		}
		walk(decoded, "")
	}

	for _, path := range []string{
		"/api/admin/stats/overview",
		"/api/admin/stats/games",
		"/api/admin/stats/timeseries?days=3",
		"/api/admin/stats/games/" + gameObj.Id + "/replay",
	} {
		response := doRequest(t, server, http.MethodGet, path, token, nil)
		if response.Code != http.StatusOK {
			t.Fatalf("%s: expected 200, got %d (%s)", path, response.Code, response.Body.String())
		}
		assertNoNulls(t, response.Body.Bytes(), path)
	}
}

// TestAdminReplayOnAnEmptyDatabase: the empty state has to be a well-formed payload too, because it
// is the first thing anyone sees on a fresh install.
func TestAdminEmptyDatabaseJSON(t *testing.T) {
	store, _ := statsTestStore(t)
	server := adminTestServer(t, store)
	token := adminToken(t, server)

	response := doRequest(t, server, http.MethodGet, "/api/admin/stats/overview", token, nil)
	if response.Code != http.StatusOK {
		t.Fatalf("overview on an empty database: %d", response.Code)
	}

	var overview stats.Overview
	if err := json.Unmarshal(response.Body.Bytes(), &overview); err != nil {
		t.Fatalf("could not read the overview: %v", err)
	}
	if overview.BySearchBudget == nil {
		t.Error("bySearchBudget must be [] and not null even with no games")
	}
	if overview.RecentGames == nil {
		t.Error("recentGames must be [] and not null even with no games")
	}
	if overview.ByComposition == nil {
		t.Error("byComposition must be [] and not null even with no games")
	}
}
