package main

import (
	"cuarenta/internal/auth"
	"cuarenta/internal/models"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

type SignUpDTO struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

type SignInDTO struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

type GameCreateDTO struct {
	SpotIndex int `json:"spotIndex"`
	Team      int `json:"team"`
}

type GameJoinDTO struct {
	SpotIndex int `json:"spotIndex"`
	Team      int `json:"team"`
}

type wsClientMessage struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

type wsSubscribePayload struct {
	GameId string `json:"gameId"`
}

type wsServerMessage struct {
	Type  string          `json:"type"`
	Game  *models.Game    `json:"game,omitempty"`
	Error *wsErrorPayload `json:"error,omitempty"`
}

type wsErrorPayload struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type wsClient struct {
	conn    *websocket.Conn
	userId  string
	gameId  string
	writeMu sync.Mutex
}

type wsHub struct {
	mu      sync.RWMutex
	clients map[string]map[*wsClient]struct{}
}

var hub = &wsHub{
	clients: make(map[string]map[*wsClient]struct{}),
}

func (h *wsHub) attach(client *wsClient, gameId string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	gameId = normalizeGameId(gameId)
	if client.gameId != "" && client.gameId != gameId {
		if peers, ok := h.clients[client.gameId]; ok {
			delete(peers, client)
			if len(peers) == 0 {
				delete(h.clients, client.gameId)
			}
		}
	}

	if gameId == "" {
		client.gameId = ""
		return
	}

	if _, ok := h.clients[gameId]; !ok {
		h.clients[gameId] = make(map[*wsClient]struct{})
	}
	h.clients[gameId][client] = struct{}{}
	client.gameId = gameId
}

func (h *wsHub) detach(client *wsClient) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if client.gameId == "" {
		return
	}

	if peers, ok := h.clients[client.gameId]; ok {
		delete(peers, client)
		if len(peers) == 0 {
			delete(h.clients, client.gameId)
		}
	}

	client.gameId = ""
}

func (h *wsHub) broadcast(game models.Game) {
	h.mu.RLock()
	clients := make([]*wsClient, 0, len(h.clients[game.Id]))
	for client := range h.clients[game.Id] {
		clients = append(clients, client)
	}
	h.mu.RUnlock()

	message := wsServerMessage{
		Type: "game.snapshot",
		Game: &game,
	}

	for _, client := range clients {
		_ = client.send(message)
	}
}

func (client *wsClient) send(message wsServerMessage) error {
	client.writeMu.Lock()
	defer client.writeMu.Unlock()

	return client.conn.WriteJSON(message)
}

func sendWsError(client *wsClient, code int, message string) {
	_ = client.send(wsServerMessage{
		Type: "error",
		Error: &wsErrorPayload{
			Code:    code,
			Message: message,
		},
	})
}

func mapErrorToStatus(err error) (int, string) {
	switch {
	case errors.Is(err, ErrUserExists):
		return http.StatusConflict, "A user with that email already exists"
	case errors.Is(err, ErrInvalidConnectionTicket):
		return http.StatusUnauthorized, "invalid connection ticket"
	case errors.Is(err, ErrGameNotFound):
		return http.StatusNotFound, "Game not found"
	case errors.Is(err, ErrSeatUnavailable):
		return http.StatusForbidden, "That spot index is already taken"
	case errors.Is(err, ErrForbiddenAction):
		return http.StatusForbidden, "You are not allowed to perform that action"
	case errors.Is(err, ErrGameNotReady):
		return http.StatusConflict, "The game is not ready to start yet"
	case errors.Is(err, ErrGameNotRunning):
		return http.StatusConflict, "The game is not running"
	case errors.Is(err, ErrInvalidSeat):
		return http.StatusBadRequest, "Spot index must be between 0 and 3"
	case errors.Is(err, ErrInvalidTeam):
		return http.StatusBadRequest, "Team must be 1 or 2"
	case errors.Is(err, ErrSeatTeamMismatch):
		return http.StatusBadRequest, "Team 1 can only use seats 1 and 3; team 2 can only use seats 2 and 4"
	case errors.Is(err, ErrAlreadyInGame):
		return http.StatusConflict, "You are already registered in this game"
	case errors.Is(err, ErrNotYourTurn):
		return http.StatusForbidden, "It is not your turn"
	case errors.Is(err, ErrIllegalAction):
		return http.StatusBadRequest, "That action is not legal"
	default:
		return http.StatusBadRequest, err.Error()
	}
}

func publicUser(user models.User) gin.H {
	return gin.H{
		"id":    user.Id,
		"email": user.Email,
	}
}

func GameSnapshotCtl(c *gin.Context) {
	gameId := normalizeGameId(c.Param("code"))
	game, exists := DB.GetGame(gameId)
	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Game not found"})
		return
	}

	if err := playAITurns(gameId); err != nil {
		log.Printf("GameSnapshotCtl error: %v", err)
	}

	c.JSON(http.StatusOK, gin.H{"game": game})
}

func SignUpCtl(c *gin.Context) {
	var dto SignUpDTO
	if err := c.ShouldBindJSON(&dto); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	email := normalizeEmail(dto.Email)
	if email == "" || strings.TrimSpace(dto.Password) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Email and password are required"})
		return
	}

	if _, exists := DB.FindUserByEmail(email); exists {
		c.JSON(http.StatusConflict, gin.H{"error": "A user with that email already exists"})
		return
	}

	hash, err := auth.HashPassword(dto.Password)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not hash password"})
		return
	}

	user := models.User{
		Id:       getUid(),
		Email:    email,
		Password: hash,
	}

	if err := DB.AddUser(user); err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	token, err := auth.GenerateJWT(user.Id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not generate access token"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"accessToken": token,
		"user":        publicUser(user),
	})
}

func SignInCtl(c *gin.Context) {
	var dto SignInDTO
	if err := c.ShouldBindJSON(&dto); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	email := normalizeEmail(dto.Email)
	user, exists := DB.FindUserByEmail(email)
	if !exists || !auth.CheckPasswordHash(dto.Password, user.Password) {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid email or password"})
		return
	}

	token, err := auth.GenerateJWT(user.Id)
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not generate access token"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"accessToken": token,
		"user":        publicUser(user),
	})
}

func RefreshJwtCtl(c *gin.Context) {
	userId := c.GetString("userId")

	token, err := auth.GenerateJWT(userId)
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not generate access token"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"accessToken": token,
	})
}

func CreateWsTicketCtl(c *gin.Context) {
	userId := c.GetString("userId")
	ticket := DB.CreateWsTicket(userId)
	c.JSON(http.StatusOK, gin.H{"ticket": ticket.Code})
}

func CreateGameCtl(c *gin.Context) {
	var dto GameCreateDTO
	if err := c.ShouldBindJSON(&dto); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	userId := c.GetString("userId")
	game, err := DB.CreateGame(userId, models.HumanPlayer{
		UserId: userId,
		Index:  dto.SpotIndex,
		Team:   dto.Team,
	})
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	hub.broadcast(game)
	c.JSON(http.StatusOK, gin.H{"game": game})
}

func JoinGameCtl(c *gin.Context) {
	var dto GameJoinDTO
	if err := c.ShouldBindJSON(&dto); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	userId := c.GetString("userId")
	game, err := DB.JoinGame(c.Param("code"), models.HumanPlayer{
		UserId: userId,
		Index:  dto.SpotIndex,
		Team:   dto.Team,
	})
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	hub.broadcast(game)
	c.JSON(http.StatusOK, gin.H{"game": game})
}

func StartGameCtl(c *gin.Context) {
	userId := c.GetString("userId")
	game, err := DB.StartGame(c.Param("code"), userId)
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	updated, err := startRunningGame(game)
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	c.JSON(http.StatusOK, gin.H{"game": updated})
}

func CancelGameCtl(c *gin.Context) {
	userId := c.GetString("userId")
	gameId := c.Param("code")

	game, err := DB.CancelGame(gameId, userId)
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	hub.broadcast(game)
	deleteSession(gameId)

	c.JSON(http.StatusOK, gin.H{"message": "Game canceled and deleted successfully"})
}

func PlayGameCtl(c *gin.Context) {
	var dto PlayGameDTO
	if err := c.ShouldBindJSON(&dto); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	userId := c.GetString("userId")
	log.Println("Calling playHumanTurn in PlayGameCtl")
	updated, err := playHumanTurn(c.Param("code"), userId, dto)
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	c.JSON(http.StatusOK, gin.H{"game": updated})
}

func ExitCtl(quit chan os.Signal) func(c *gin.Context) {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "Shutting down server...",
		})
		go func() {
			time.Sleep(2 * time.Second)
			quit <- os.Interrupt
		}()
	}
}

func subscribeToGame(client *wsClient, gameId string) {
	gameId = normalizeGameId(gameId)
	if gameId == "" {
		return
	}

	if game, exists := DB.GetGame(gameId); exists {
		hub.attach(client, gameId)
		_ = client.send(wsServerMessage{
			Type: "game.snapshot",
			Game: &game,
		})
		return
	}

	sendWsError(client, http.StatusNotFound, "Game not found")
}

func WsCtl(c *gin.Context) {
	ticketCode := strings.TrimSpace(c.Query("ticket"))
	ticket, valid := DB.ValidateWsTicket(ticketCode)

	userId := ticket.UserId

	if !valid {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Unauthorized"})
		return
	}

	upgrader := websocket.Upgrader{
		ReadBufferSize:  2048,
		WriteBufferSize: 2048,
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Println("websocket upgrade error:", err)
		return
	}

	client := &wsClient{
		conn:   conn,
		userId: userId,
	}
	defer func() {
		hub.detach(client)
		_ = conn.Close()
	}()

	_ = client.send(wsServerMessage{Type: "connection.ready"})

	if initialGameId := c.Query("gameId"); initialGameId != "" {
		subscribeToGame(client, initialGameId)
	}

	for {
		var message wsClientMessage
		if err := conn.ReadJSON(&message); err != nil {
			log.Println("websocket read:", err)
			break
		}

		switch message.Type {
		case "game.subscribe":
			var payload wsSubscribePayload
			if err := json.Unmarshal(message.Payload, &payload); err != nil {
				sendWsError(client, http.StatusBadRequest, "Invalid subscribe payload")
				continue
			}
			subscribeToGame(client, payload.GameId)
		case "game.action":
			var payload map[string]interface{}
			if err := json.Unmarshal(message.Payload, &payload); err != nil {
				sendWsError(client, http.StatusBadRequest, "Invalid action payload")
				continue
			}
			log.Println("game.action:", payload)
			gameId, ok := payload["gameId"].(string)
			if !ok {
				sendWsError(client, http.StatusBadRequest, "Missing gameId")
				continue
			}
			cardIndex, ok := payload["cardIndex"].(float64)
			if !ok {
				sendWsError(client, http.StatusBadRequest, "Missing cardIndex")
				continue
			}

			captureIndices := []int{}
			if captures, ok := payload["captureIndices"].([]interface{}); ok {
				for _, c := range captures {
					if idx, ok := c.(float64); ok {
						captureIndices = append(captureIndices, int(idx))
					}
				}
			}

			dto := PlayGameDTO{
				CardIndex:      int(cardIndex),
				CaptureIndices: captureIndices,
			}
			_, err := playHumanTurn(gameId, client.userId, dto)
			if err != nil {
				sendWsError(client, http.StatusBadRequest, err.Error())
				continue
			}
		default:
			sendWsError(client, http.StatusBadRequest, "Unknown websocket event")
		}
	}
}
