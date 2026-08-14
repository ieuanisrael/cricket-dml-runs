#' Generate synthetic T20 ball-by-ball cricket data with known player effects.
#'
#' Ground-truth batter effects are stored alongside the deliveries so DML
#' estimates can be validated. Format is T20 only.
#'
#' @param n_matches Number of matches to simulate.
#' @param n_batters Size of the global batter pool.
#' @param n_bowlers Size of the global bowler pool.
#' @param n_venues Number of venues.
#' @param seed RNG seed.
#' @return Named list: `deliveries` (data.table), `true_effects` (list), `meta` (list).
generate_synthetic_t20_bbb <- function(
    n_matches = 80L,
    n_batters = 48L,
    n_bowlers = 36L,
    n_venues = 8L,
    seed = 42L
) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table: install.packages(\"data.table\")", call. = FALSE)
  }
  set.seed(as.integer(seed))

  leagues <- c("Premier Smash", "Coastal T20", "Capital League")
  venues <- sprintf("Venue_%02d", seq_len(n_venues))
  batters <- sprintf("Batter_%03d", seq_len(n_batters))
  bowlers <- sprintf("Bowler_%03d", seq_len(n_bowlers))

  # True additive effects on expected runs per ball
  true_batter <- stats::setNames(stats::rnorm(n_batters, 0, 0.28), batters)
  # A few star / weak batters for clearer signal
  star_ix <- sample.int(n_batters, 4L)
  weak_ix <- sample.int(n_batters, 4L)
  true_batter[star_ix] <- true_batter[star_ix] + 0.45
  true_batter[weak_ix] <- true_batter[weak_ix] - 0.40

  true_bowler <- stats::setNames(stats::rnorm(n_bowlers, 0, 0.20), bowlers)
  league_effect <- c(`Premier Smash` = 0.06, `Coastal T20` = 0, `Capital League` = -0.05)
  innings_effect <- c(`1` = 0.04, `2` = -0.03)
  phase_effect <- c(powerplay = 0.18, middle = 0, death = 0.28)
  home_effect <- 0.05
  venue_effect <- stats::setNames(stats::rnorm(n_venues, 0, 0.10), venues)

  teams <- sprintf("Team_%02d", 1:12)
  team_batters <- split(sample(batters), rep_len(seq_along(teams), n_batters))
  names(team_batters) <- teams
  team_bowlers <- split(sample(bowlers), rep_len(seq_along(teams), n_bowlers))
  names(team_bowlers) <- teams

  rows <- list()
  match_meta <- list()

  for (m in seq_len(n_matches)) {
    match_id <- sprintf("M%04d", m)
    league <- sample(leagues, 1L, prob = c(0.45, 0.35, 0.20))
    venue <- sample(venues, 1L)
    day_night <- sample(c(FALSE, TRUE), 1L, prob = c(0.35, 0.65))
    season <- sample(c("2023/24", "2024/25", "2025/26"), 1L, prob = c(0.3, 0.4, 0.3))
    home_team <- sample(teams, 1L)
    away_team <- sample(setdiff(teams, home_team), 1L)
    if (stats::runif(1) < 0.5) {
      team1 <- home_team
      team2 <- away_team
    } else {
      team1 <- away_team
      team2 <- home_team
    }

    match_meta[[m]] <- data.table::data.table(
      match_id = match_id,
      series = league,
      venue = venue,
      season = season,
      day_night = day_night,
      home_team = home_team,
      team1 = team1,
      team2 = team2
    )

    target <- NA_real_
    for (inn in 1:2) {
      batting_team <- if (inn == 1L) team1 else team2
      bowling_team <- if (inn == 1L) team2 else team1
      bat_xi <- sample(team_batters[[batting_team]], size = min(8L, length(team_batters[[batting_team]])))
      bowl_xi <- sample(team_bowlers[[bowling_team]], size = min(6L, length(team_bowlers[[bowling_team]])))

      order_ix <- seq_along(bat_xi)
      striker_pos <- 1L
      non_pos <- 2L
      wickets <- 0L
      innings_balls <- 0L
      innings_runs <- 0L
      over <- 0L
      ball_in_over <- 0L
      bowler_over_count <- stats::setNames(integer(length(bowl_xi)), bowl_xi)
      current_bowler <- sample(bowl_xi, 1L)

      while (over < 20L && wickets < 10L && striker_pos <= length(bat_xi)) {
        if (ball_in_over == 0L) {
          # new over: pick bowler with remaining overs (<4)
          eligible <- names(bowler_over_count)[bowler_over_count < 4L]
          if (!length(eligible)) eligible <- bowl_xi
          current_bowler <- sample(eligible, 1L)
        }

        ball_in_over <- ball_in_over + 1L
        innings_balls <- innings_balls + 1L
        phase <- if (over < 6L) "powerplay" else if (over < 15L) "middle" else "death"

        batter <- bat_xi[[min(striker_pos, length(bat_xi))]]
        non_striker <- if (non_pos <= length(bat_xi)) bat_xi[[non_pos]] else NA_character_
        batter_is_home <- identical(batting_team, home_team)

        # Linear expected runs off the bat (matches DML partially-linear target)
        mu <- 0.95 +
          true_batter[[batter]] +
          true_bowler[[current_bowler]] +
          venue_effect[[venue]] +
          league_effect[[league]] +
          innings_effect[[as.character(inn)]] +
          phase_effect[[phase]] +
          home_effect * as.numeric(batter_is_home) +
          0.015 * (over - 9.5)

        if (inn == 2L && is.finite(target)) {
          req_rr <- max(0, (target + 1 - innings_runs) / max(1, (120 - innings_balls + 1) / 6))
          mu <- mu + 0.025 * (req_rr - 8)
        }
        mu <- max(0.08, min(mu, 2.8))

        # P(wicket) rises slightly in death and vs good bowlers
        p_wkt <- plogis(-3.2 - 1.2 * true_batter[[batter]] + 1.0 * true_bowler[[current_bowler]] +
          if (phase == "death") 0.25 else 0)
        is_wicket <- stats::runif(1) < p_wkt

        if (is_wicket) {
          bat_score <- 0L
          extras <- 0L
          how_out <- sample(c("caught", "bowled", "lbw", "run_out"), 1L, prob = c(0.55, 0.25, 0.12, 0.08))
          wickets <- wickets + 1L
        } else {
          how_out <- NA_character_
          # multinomial-ish runs: 0,1,2,3,4,6 with mean ~ mu
          probs <- .runs_probs(mu)
          bat_score <- sample(c(0L, 1L, 2L, 3L, 4L, 6L), 1L, prob = probs)
          extras <- if (stats::runif(1) < 0.04) sample(c(1L, 1L, 1L, 5L), 1L) else 0L
        }

        total_runs <- bat_score + extras
        innings_runs <- innings_runs + total_runs

        rows[[length(rows) + 1L]] <- data.table::data.table(
          match_id = match_id,
          series = league,
          season = season,
          venue = venue,
          day_night = day_night,
          home_team = home_team,
          innings = as.integer(inn),
          over = as.integer(over),
          ball_in_over = as.integer(ball_in_over),
          legal_ball_index = as.integer(innings_balls),
          phase = phase,
          batting_team = batting_team,
          bowling_team = bowling_team,
          striker_id = batter,
          non_striker_id = non_striker,
          striker_batting_position = as.integer(striker_pos),
          bowler_id = current_bowler,
          batter_is_home = batter_is_home,
          bat_score = as.integer(bat_score),
          extras = as.integer(extras),
          total_runs = as.integer(total_runs),
          is_wicket = is_wicket,
          how_out = how_out,
          cum_innings_runs = as.integer(innings_runs),
          cum_innings_wickets = as.integer(wickets),
          target = if (inn == 2L) as.integer(target) else NA_integer_
        )

        # strike rotation
        if (!is_wicket && bat_score %% 2L == 1L) {
          tmp <- striker_pos
          striker_pos <- non_pos
          non_pos <- tmp
        }
        if (is_wicket) {
          striker_pos <- max(striker_pos, non_pos) + 1L
          if (striker_pos == non_pos) striker_pos <- striker_pos + 1L
        }

        # end of over
        if (ball_in_over >= 6L) {
          bowler_over_count[[current_bowler]] <- bowler_over_count[[current_bowler]] + 1L
          ball_in_over <- 0L
          over <- over + 1L
          # swap ends
          tmp <- striker_pos
          striker_pos <- non_pos
          non_pos <- tmp
        }

        if (inn == 2L && is.finite(target) && innings_runs > target) {
          break
        }
      }

      if (inn == 1L) {
        target <- innings_runs
      }
    }
  }

  deliveries <- data.table::rbindlist(rows)
  meta <- data.table::rbindlist(match_meta)

  true_effects <- list(
    batter = true_batter,
    bowler = true_bowler,
    venue = venue_effect,
    league = league_effect,
    innings = innings_effect,
    phase = phase_effect,
    home = home_effect,
    star_batters = batters[star_ix],
    weak_batters = batters[weak_ix]
  )

  list(
    deliveries = deliveries,
    match_meta = meta,
    true_effects = true_effects,
    meta = list(
      format = "T20",
      n_matches = n_matches,
      n_batters = n_batters,
      n_bowlers = n_bowlers,
      n_venues = n_venues,
      seed = as.integer(seed),
      n_deliveries = nrow(deliveries),
      generated_at = as.character(Sys.time())
    )
  )
}

.runs_probs <- function(mu) {
  # Map continuous mean to discrete support {0,1,2,3,4,6}
  mu <- max(0.05, min(mu, 2.5))
  # Base shape then tilt toward boundaries as mu grows
  base <- c(0.38, 0.32, 0.12, 0.03, 0.10, 0.05)
  tilt <- c(-0.10, -0.02, 0.02, 0.01, 0.05, 0.04) * (mu - 0.9)
  p <- pmax(0.001, base + tilt)
  p / sum(p)
}

#' Write synthetic dataset to disk under data/.
save_synthetic_dataset <- function(obj, raw_dir = "data/raw", processed_dir = "data/processed") {
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)

  bbb_path <- file.path(raw_dir, "t20_ball_by_ball.csv")
  meta_path <- file.path(raw_dir, "t20_match_meta.csv")
  truth_path <- file.path(processed_dir, "true_effects.rds")
  meta_rds <- file.path(processed_dir, "generation_meta.rds")

  data.table::fwrite(obj$deliveries, bbb_path)
  data.table::fwrite(obj$match_meta, meta_path)
  saveRDS(obj$true_effects, truth_path)
  saveRDS(obj$meta, meta_rds)

  invisible(list(
    ball_by_ball = bbb_path,
    match_meta = meta_path,
    true_effects = truth_path,
    generation_meta = meta_rds
  ))
}
