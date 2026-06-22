package cua

import (
	"context"
	"log"
	"reflect"
	"time"
)

type JobsService interface {
	RunRecurrentJobs(ctx context.Context)
}

type RecurrentJob struct {
	ticker  *time.Ticker
	handler func()
}

type MainJobsService struct {
	db Database
}

func NewMainJobsService(db Database) JobsService {
	return &MainJobsService{db: db}
}

func (j *MainJobsService) RunRecurrentJobs(ctx context.Context) {
	recurrentJobs := []RecurrentJob{
		{ticker: time.NewTicker(12 * time.Hour), handler: j.deleteExpiredWsTicketsHandler},
		{ticker: time.NewTicker(1 * time.Hour), handler: j.dbToDiskHandler},
	}

	defer func() {
		for _, j := range recurrentJobs {
			j.ticker.Stop()
		}
	}()

	var switchCases []reflect.SelectCase
	for _, j := range recurrentJobs {
		switchCases = append(switchCases, reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(j.ticker.C)})
	}
	switchCases = append(switchCases, reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(ctx.Done())})
	doneCaseIndex := len(switchCases) - 1

	for {
		chosenIndex, _, ok := reflect.Select(switchCases)
		if !ok || chosenIndex == doneCaseIndex {
			log.Println("[+] Finishing recurrent Jobs")
			return
		}
		if ok && chosenIndex < len(recurrentJobs) {
			recurrentJobs[chosenIndex].handler()
		}
	}

}

func (j *MainJobsService) deleteExpiredWsTicketsHandler() {
	nDeleted := j.db.DeleteExpiredWsTickets()
	log.Printf("Deleted %d tickets\n", nDeleted)
}

func (j *MainJobsService) dbToDiskHandler() {
	if err := j.db.WriteToDisk(); err != nil {
		log.Printf("Failed to write DB file to disk: %v", err)
	}
}
