// Command oracle drives the official Battlesnake rules engine
// (github.com/BattlesnakeOfficial/rules) so the slinky test-suite can compare
// its JAX implementation against it. Test-only tool; see README.md.
package main

import (
	"fmt"
	"os"
)

const usage = `usage: oracle <command> [flags]

commands:
  step    read JSON Lines step requests on stdin, answer one JSON line each
  games   play full games (map + ruleset, built-in seeded policy), emit JSON Lines
          run "oracle games -h" for flags
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "step":
		err = runStep(os.Args[2:])
	case "games":
		err = runGames(os.Args[2:])
	case "-h", "--help", "help":
		fmt.Print(usage)
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "oracle %s: %v\n", os.Args[1], err)
		os.Exit(1)
	}
}
