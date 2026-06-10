package models

type User struct {
	Id       string `json:"id"`
	Email    string `json:"email"`
	Password string `json:"password"`
}

type HumanPlayer struct {
	UserId    string `json:"userId"`
	Index     int    `json:"index"`
	Team      int    `json:"team"`
	JoinedAt  string `json:"joinedAt"`
}

type Game struct {
	Id           string        `json:"id"`
	HumanPlayers []HumanPlayer `json:"humanPlayers"`
	TimeSteps    int           `json:"timeSteps"`
	Cards        [][]int       `json:"cards"`
	CreatedBy    string        `json:"createdBy"`
	CreatedAt    string        `json:"createdAt"`
	StartedAt    string        `json:"startedAt,omitempty"`
	Status       string        `json:"status"`
}
