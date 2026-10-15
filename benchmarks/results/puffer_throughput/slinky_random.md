# slinky throughput benchmark, `--policy random` (`benchmarks/throughput.py --configs duel,standard-4p --batch-sizes 1024,8192 --variants sim,sim+ego-obs --policy random`; same machine and versions as slinky_random_legal.md)

## Throughput

| config      | variant     | batch | steps/s | agent-steps/s |
| ----------- | ----------- | ----: | ------: | ------------: |
| duel        | sim         | 1,024 | 394,581 |       789,163 |
| duel        | sim         | 8,192 | 558,613 |     1,117,226 |
| duel        | sim+ego-obs | 1,024 | 114,730 |       229,459 |
| duel        | sim+ego-obs | 8,192 |  89,613 |       179,226 |
| standard-4p | sim         | 1,024 | 281,939 |     1,127,758 |
| standard-4p | sim         | 8,192 | 365,576 |     1,462,303 |
| standard-4p | sim+ego-obs | 1,024 |  40,224 |       160,895 |
| standard-4p | sim+ego-obs | 8,192 |  38,125 |       152,502 |

`steps/s` = games advanced one turn per second; `agent-steps/s` = steps/s x snakes.
Rows for `reset-only` count resets instead of steps.
`sim-no-autoreset` games stay done once over; they still pay for a full step.

## Game length (random policy, 4096 games, 1000 turns)
