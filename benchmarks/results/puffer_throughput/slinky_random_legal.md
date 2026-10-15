# slinky throughput benchmark

- jax 0.11.2, jaxlib 0.11.2, numpy 2.5.3, python 3.13.16
- backend: cpu, 1 device(s): cpu
- host: x86_64, 4 logical CPUs
- prng: threefry2x32
- load average at exit: 2.5, 2.4, 3.0
- policy: random_legal, turns per call: 100, best of 3
- configs: duel: 11x11 standard 2 snakes; standard-4p: 11x11 standard 4 snakes; constrictor-2p: 11x11 constrictor 2 snakes

## Throughput

| config         | variant     | batch | steps/s | agent-steps/s |
| -------------- | ----------- | ----: | ------: | ------------: |
| duel           | sim         |    64 | 130,651 |       261,302 |
| duel           | sim         | 1,024 | 330,584 |       661,168 |
| duel           | sim         | 8,192 | 510,498 |     1,020,997 |
| duel           | sim+ego-obs |    64 |  60,860 |       121,721 |
| duel           | sim+ego-obs | 1,024 | 110,749 |       221,499 |
| duel           | sim+ego-obs | 8,192 |  90,058 |       180,115 |
| standard-4p    | sim         |    64 |  99,679 |       398,718 |
| standard-4p    | sim         | 1,024 | 237,055 |       948,219 |
| standard-4p    | sim         | 8,192 | 342,117 |     1,368,468 |
| standard-4p    | sim+ego-obs |    64 |  39,272 |       157,088 |
| standard-4p    | sim+ego-obs | 1,024 |  48,701 |       194,804 |
| standard-4p    | sim+ego-obs | 8,192 |  37,726 |       150,906 |
| constrictor-2p | sim         |    64 | 131,331 |       262,661 |
| constrictor-2p | sim         | 1,024 | 332,759 |       665,518 |
| constrictor-2p | sim         | 8,192 | 509,879 |     1,019,758 |
| constrictor-2p | sim+ego-obs |    64 |  60,346 |       120,692 |
| constrictor-2p | sim+ego-obs | 1,024 | 106,243 |       212,486 |
| constrictor-2p | sim+ego-obs | 8,192 |  85,569 |       171,137 |

`steps/s` = games advanced one turn per second; `agent-steps/s` = steps/s x snakes.
Rows for `reset-only` count resets instead of steps.
`sim-no-autoreset` games stay done once over; they still pay for a full step.

## Game length (random_legal policy, 4096 games, 1000 turns)

| config         | mean | median | p90 | max |  finished |
| -------------- | ---: | -----: | --: | --: | --------: |
| duel           | 70.8 |     66 | 130 | 285 | 4096/4096 |
| standard-4p    | 79.6 |     76 | 133 | 256 | 4096/4096 |
| constrictor-2p | 19.2 |     18 |  30 |  50 | 4096/4096 |
