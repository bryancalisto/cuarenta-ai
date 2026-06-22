package game

import "cua/internal/models"

func NewGame(id string, humanPlayers []models.HumanPlayer, createdBy string, createdAt string) models.Game {
	return models.Game{Id: id, HumanPlayers: humanPlayers, CreatedBy: createdBy, CreatedAt: createdAt}
}
