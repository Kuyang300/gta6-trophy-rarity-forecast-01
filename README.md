# GTA VI trophy rarity forecast

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
on make.fun ($250, 20 places).

`main.ts` (Node 22, standard library only, no network) reads one case per
`*.in` file from `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`/output/<id>.out`: for every Grand Theft Auto VI PlayStation 5 trophy, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's
target (2026-11-26T00:00:00Z for the final case).

## Model

All fitting is per trophy in logit space, rates clamped to [0.01, 99.99]:

- **Direct trend** — weighted least squares of logit(rate) on
  ln(1 + hours since release), recency-weighted with a 24 h half-life.
- **Counts decomposition** — the earn rate is earned / tracked, so
  ln(earned) and ln(players tracked) are fitted separately on the same clock
  and their ratio is taken at the target. This captures dilution: when
  Exophase refreshes and newly tracked players arrive holding few trophies,
  rates fall even while earned counts keep rising.
- **Tier shrinkage** — each trophy's slope is shrunk toward its tier's
  median slope (platinum / gold / silver / bronze), steadying secret and
  late-game trophies with thin histories.
- **Median** — a blend of the two fits and persistence (the last read).
- **Spread** — a random-walk term growing with the horizon, plus the fit's
  residual scale, the slope's standard error projected to the target, and a
  refresh-jump term from the largest single-read logit drop in the history.

The program is deterministic: two runs on the same case write identical bytes.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```
