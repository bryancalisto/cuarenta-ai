package cua

import (
	"cua/internal/models"
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
	Email    string `json:"email" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type SignInDTO struct {
	Email    string `json:"email" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type GameCreateDTO struct {
	SpotIndex int `json:"spotIndex" binding:"required"`
	Team      int `json:"team" binding:"required"`
}

type GameJoinDTO struct {
	SpotIndex int `json:"spotIndex" binding:"required"`
	Team      int `json:"team" binding:"required"`
}

type ExitDTO struct {
	Secret string `json:"secret" binding:"required"`
}

type wsClientMessage struct {
	Type    string          `json:"type" binding:"required"`
	Payload json.RawMessage `json:"payload" binding:"required"`
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

func bindJSONDTO[T any](c *gin.Context, dto *T) error {

	if err := c.ShouldBindJSON(&dto); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return err
	}

	return nil
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

func (s *Server) PingCtl(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message": "pong",
	})
}

func (s *Server) GameSnapshotCtl(c *gin.Context) {
	gameId := normalizeGameId(c.Param("code"))
	game, exists := s.DB.GetGame(gameId)
	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Game not found"})
		return
	}

	if err := s.gameplay.playAITurns(gameId); err != nil {
		log.Printf("GameSnapshotCtl error: %v", err)
	}

	c.JSON(http.StatusOK, gin.H{"game": game})
}

func (s *Server) SignUpCtl(c *gin.Context) {
	var dto SignUpDTO
	if err := bindJSONDTO(c, &dto); err != nil {
		return
	}

	email := normalizeEmail(dto.Email)
	if email == "" || strings.TrimSpace(dto.Password) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Email and password are required"})
		return
	}

	if _, exists := s.DB.FindUserByEmail(email); exists {
		c.JSON(http.StatusConflict, gin.H{"error": "A user with that email already exists"})
		return
	}

	hash, err := s.AuthService.HashPassword(dto.Password)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not hash password"})
		return
	}

	user := models.User{
		Id:       getUid(),
		Email:    email,
		Password: hash,
	}

	if err := s.DB.AddUser(user); err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	token, err := s.AuthService.GenerateJWT(user.Id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not generate access token"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"accessToken": token,
		"user":        publicUser(user),
	})
}

func (s *Server) SignInCtl(c *gin.Context) {
	var dto SignInDTO
	if err := bindJSONDTO(c, &dto); err != nil {
		return
	}

	email := normalizeEmail(dto.Email)
	user, exists := s.DB.FindUserByEmail(email)
	if !exists || !s.AuthService.CheckPasswordHash(dto.Password, user.Password) {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid email or password"})
		return
	}

	token, err := s.AuthService.GenerateJWT(user.Id)
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

func (s *Server) RefreshJwtCtl(c *gin.Context) {
	userId := c.GetString("userId")

	token, err := s.AuthService.GenerateJWT(userId)
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not generate access token"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"accessToken": token,
	})
}

func (s *Server) MyGamesCtl(c *gin.Context) {
	userId := c.GetString("userId")
	games := s.DB.GetGamesByUser(userId)
	c.JSON(http.StatusOK, gin.H{"games": games})
}

func (s *Server) CreateWsTicketCtl(c *gin.Context) {
	userId := c.GetString("userId")
	ticket := s.DB.CreateWsTicket(userId)
	c.JSON(http.StatusOK, gin.H{"ticket": ticket.Code})
}

func (s *Server) CreateGameCtl(c *gin.Context) {
	var dto GameCreateDTO
	if err := bindJSONDTO(c, &dto); err != nil {
		return
	}

	userId := c.GetString("userId")
	game, err := s.DB.CreateGame(userId, models.HumanPlayer{
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

func (s *Server) JoinGameCtl(c *gin.Context) {
	var dto GameJoinDTO

	if err := bindJSONDTO(c, &dto); err != nil {
		return
	}

	userId := c.GetString("userId")
	game, err := s.DB.JoinGame(c.Param("code"), models.HumanPlayer{
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

func (s *Server) StartGameCtl(c *gin.Context) {
	userId := c.GetString("userId")
	game, err := s.DB.StartGame(c.Param("code"), userId)
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	updated, err := s.gameplay.startRunningGame(game)
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	c.JSON(http.StatusOK, gin.H{"game": updated})
}

func (s *Server) CancelGameCtl(c *gin.Context) {
	userId := c.GetString("userId")
	gameId := c.Param("code")

	game, err := s.DB.CancelGame(gameId, userId)
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	hub.broadcast(game)

	if session, ok := getSession(gameId); ok {
		close(session.AiTurnChan)
	}
	deleteSession(gameId)

	c.JSON(http.StatusOK, gin.H{"message": "Game canceled and deleted successfully"})
}

func (s *Server) PlayGameCtl(c *gin.Context) {
	var dto PlayGameDTO

	if err := bindJSONDTO(c, &dto); err != nil {
		return
	}

	userId := c.GetString("userId")
	log.Println("Calling playHumanTurn in PlayGameCtl")
	updated, err := s.gameplay.playHumanTurn(c.Param("code"), userId, dto)
	if err != nil {
		status, message := mapErrorToStatus(err)
		c.JSON(status, gin.H{"error": message})
		return
	}

	c.JSON(http.StatusOK, gin.H{"game": updated})
}

func ExitCtl(quit chan os.Signal) func(c *gin.Context) {
	return func(c *gin.Context) {
		var dto ExitDTO

		if err := bindJSONDTO(c, &dto); err != nil {
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"message": "Shutting down server...",
		})
		go func() {
			time.Sleep(2 * time.Second)
			quit <- os.Interrupt
		}()
	}
}

func (s *Server) subscribeToGame(client *wsClient, gameId string) {
	gameId = normalizeGameId(gameId)
	if gameId == "" {
		return
	}

	if game, exists := s.DB.GetGame(gameId); exists {
		hub.attach(client, gameId)
		_ = client.send(wsServerMessage{
			Type: "game.snapshot",
			Game: &game,
		})
		return
	}

	sendWsError(client, http.StatusNotFound, "Game not found")
}

func (s *Server) WsCtl(c *gin.Context) {
	ticketCode := strings.TrimSpace(c.Query("ticket"))
	ticket, valid := s.DB.ValidateWsTicket(ticketCode)

	userId := ticket.UserId

	if !valid {
		log.Println("Invalid ticket")
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Unauthorized"})
		return
	}

	upgrader := websocket.Upgrader{
		ReadBufferSize:  2048,
		WriteBufferSize: 2048,
		CheckOrigin: func(r *http.Request) bool {
			log.Println("Host in ws upgrader: ", r.Host)
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
		log.Printf("[WS] Client %s disconnected from game %s", userId, client.gameId)
		hub.detach(client)
		_ = conn.Close()
	}()

	log.Printf("[WS] Client %s connected", userId)
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		// log.Printf("[WS] Pong received from %s, resetting deadline", userId)
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	done := make(chan struct{})
	defer close(done)

	go func() {
		for {
			select {
			case <-ticker.C:
				// log.Printf("[WS] Sending ping to %s", userId)
				client.writeMu.Lock()
				if err := conn.WriteMessage(websocket.PingMessage, []byte{}); err != nil {
					log.Printf("[WS] Ping write failed for %s: %v", userId, err)
					client.writeMu.Unlock()
					return
				}
				client.writeMu.Unlock()
			case <-done:
				log.Printf("[WS] Ping ticker stopped for %s", userId)
				return
			}
		}
	}()

	_ = client.send(wsServerMessage{Type: "connection.ready"})
	log.Printf("[WS] Sent connection.ready to %s", userId)

	if initialGameId := c.Query("gameId"); initialGameId != "" {
		log.Printf("[WS] Client %s subscribing to game %s", userId, initialGameId)
		s.subscribeToGame(client, initialGameId)
	}

	for {
		var message wsClientMessage
		if err := conn.ReadJSON(&message); err != nil {
			log.Printf("[WS] Read error from %s (game %s): %v", userId, client.gameId, err)
			break
		}

		// log.Printf("[WS] Message from %s: type=%s", userId, message.Type)
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))

		switch message.Type {
		case "game.subscribe":
			var payload wsSubscribePayload
			if err := json.Unmarshal(message.Payload, &payload); err != nil {
				sendWsError(client, http.StatusBadRequest, "Invalid subscribe payload")
				continue
			}
			s.subscribeToGame(client, payload.GameId)
		case "game.action":
			var payload map[string]interface{}
			if err := json.Unmarshal(message.Payload, &payload); err != nil {
				sendWsError(client, http.StatusBadRequest, "Invalid action payload")
				continue
			}
			// log.Println("game.action:", payload)
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
			_, err := s.gameplay.playHumanTurn(gameId, client.userId, dto)
			if err != nil {
				sendWsError(client, http.StatusBadRequest, err.Error())
				continue
			}
		default:
			sendWsError(client, http.StatusBadRequest, "Unknown websocket event")
		}
	}
}
