package main

import (
	"context"
	"cua"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/benbjohnson/clock"
)

func printDBEntityRecords[T any](title string, records []T) {
	fmt.Printf("\n=== %s ===\n\n", title)

	if len(records) == 0 {
		fmt.Println("[]")
		return
	}

	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error marshaling %s: %v\n", title, err)
		return
	}
	fmt.Println(string(data))

	recordSuffix := "record"

	if len(records) != 1 {
		recordSuffix = "records"
	}

	fmt.Printf("\n%d %s\n", len(records), recordSuffix)
}

func dumpDB(db *cua.MemDatabase, entities map[string]bool) {
	if entities["users"] {
		printDBEntityRecords("users", db.Users())
	}
	if entities["games"] {
		printDBEntityRecords("games", db.Games())
	}
	if entities["wstickets"] {
		printDBEntityRecords("tickets", db.WsTickets())
	}
}

func parseEntities(raw string) map[string]bool {
	valid := map[string]bool{
		"users":     false,
		"games":     false,
		"wstickets": false,
	}

	parts := strings.Split(raw, ",")
	seen := make(map[string]bool)

	for _, p := range parts {
		p = strings.ToLower(strings.TrimSpace(p))

		if _, ok := valid[p]; !ok {
			fmt.Fprintf(os.Stderr, "warning: ignoring invalid entity \"%s\" (valid: users, games, wstickets)\n", p)
			continue
		}

		if seen[p] {
			continue
		}

		seen[p] = true
		valid[p] = true
	}

	return valid
}

func main() {
	dbPath := flag.String("path", "cuarenta.db", "path to cuarenta.db")
	fDump := flag.Bool("dump", true, "print all DB content to stdout")
	fEntities := flag.String("entities", "", "comma-separated list of entities to dump (users, games, wstickets)")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	clk := clock.New()
	db := cua.NewMemDatabase(ctx, clk).(*cua.MemDatabase)

	if err := db.LoadFromDisk(*dbPath); err != nil {
		fmt.Fprintf(os.Stderr, "error loading database from %s: %v\n", *dbPath, err)
		os.Exit(1)
	}

	if *fDump {
		entities := parseEntities(*fEntities)
		dumpDB(db, entities)
	}
}
