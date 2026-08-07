# cricket-dml-runs

Double / debiased machine learning (**DML**) for **T20 batter run effects**, with synthetic ball-by-ball data and a **RAE vs DML** benchmark.

## What this estimates

Partially linear model on each delivery:

\[
Y = D\theta + g(X) + \varepsilon
\]

- **\(Y\)**: runs off the bat
- **\(D\)**: batter indicators (vs a reference batter)
- **\(X\)**: series/league, venue, season, innings, phase, bowler, home, over, batting position, day/night
- **\(\theta\)**: debiased batter effects (runs per ball), after cross-fitting out \(g(X)\) and \(E[D\mid X]\)

Cluster-robust SEs use **match_id**. Multiple testing across batters uses BH–FDR.

## Requirements

| Package | Role |
|---------|------|
| **R** ≥ 4.1 | Runtime |
| **data.table** | Data wrangling / I/O |
| **Matrix** | Sparse design matrices |
| **glmnet** | Cross-fitted nuisance models |
| **ggplot2** | Forest plots |

```r
install.packages(c("data.table", "Matrix", "glmnet", "ggplot2"))
```

## Quick start

From the repo root:

```bash
Rscript R/run_all.R
```

Useful flags:

```bash
Rscript R/run_all.R --matches=80 --min-balls=80 --folds=5 --seed=42 --out=outputs/run_t20
```

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Simulate T20 BBB | `R/generate_synthetic_bbb.R` | `data/raw/t20_ball_by_ball.csv` |
| 2. Analysis frame | `R/prepare_analysis_frame.R` | Batter \(D\), controls \(X\), outcome \(Y\) |
| 3. DML | `R/estimate_player_dml.R` | `player_effects_dml.csv`, forest plot |
| 4. RAE vs DML | `R/compare_rae_dml.R` | `player_effects_rae.csv`, `rae_vs_dml.csv`, comparison plots |

**RAE** here is mean ball-level residual from a cross-fitted \(E[Y\mid X]\) model (same controls as DML, no batter), contrasted vs the reference batter. It is a simple benchmark, not ground truth.

## Synthetic data fields

Ball-level columns include: `match_id`, `series_league`, `season`, `venue`, `day_night`, `innings`, `over`, `ball_in_over`, `phase`, `batting_team`, `bowling_team`, `batter_id`, `non_striker_id`, `batting_position`, `bowler_id`, `batter_is_home`, `runs_off_bat`, `extras`, `is_wicket`, `target` (2nd innings), and cumulative score/wickets.

## Project layout

```
R/           # generators + DML
data/raw/    # generated BBB CSV
data/processed/
outputs/     # estimates, plots, manifests
```

## Notes

- Format is **T20 only** (20 overs, powerplay / middle / death).
- Effects are **per ball**, relative to the reference batter (most balls faced).
- This is a simulation sandbox for methods; swap in real BBB with the same schema when ready.
