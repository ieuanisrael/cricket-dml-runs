#' Build the ball-level analysis frame for player-effect DML.
#'
#' Outcome: `bat_score` on a delivery.
#' Treatment design: striker indicators (reference = most common striker).
#' Controls: context + bowler (not striker).
#'
#' @param deliveries data.table/data.frame of ball-by-ball rows.
#' @param min_balls Drop strikers with fewer than this many balls faced.
#' @return list with `frame`, `y`, `D` (sparse Matrix), `X` (sparse Matrix),
#'   `batter_levels`, `reference_batter`, `feature_info`.
prepare_player_dml_frame <- function(deliveries, min_balls = 80L) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Install Matrix", call. = FALSE)
  }

  dt <- data.table::as.data.table(deliveries)
  # Keep batting events with a striker (all rows are strikes in our generator)
  dt <- dt[!is.na(striker_id) & !is.na(bat_score)]

  # Real schema uses power_play; synthetic uses phase
  if (!"phase" %in% names(dt) && "power_play" %in% names(dt)) {
    dt[, phase := data.table::fifelse(
      as.integer(power_play) == 1L,
      "powerplay",
      data.table::fifelse(over >= 15L, "death", "middle")
    )]
  }
  if (!"phase" %in% names(dt) && "over" %in% names(dt)) {
    dt[, phase := data.table::fifelse(
      over < 6L,
      "powerplay",
      data.table::fifelse(over >= 15L, "death", "middle")
    )]
  }

  ball_n <- dt[, .N, by = striker_id]
  keep <- ball_n[N >= as.integer(min_balls)]$striker_id
  dt <- dt[striker_id %in% keep]

  # Reference: most balls faced (stable baseline)
  ref <- ball_n[striker_id %in% keep][which.max(N)]$striker_id

  if (!"season" %in% names(dt)) dt[, season := "unknown"]
  if (!"venue" %in% names(dt)) dt[, venue := "unknown"]
  if (!"batter_is_home" %in% names(dt)) dt[, batter_is_home := 0L]
  if (!"striker_batting_position" %in% names(dt)) {
    dt[, striker_batting_position := 1L]
  }

  dt[, `:=`(
    venue = factor(venue),
    season = factor(season),
    phase = factor(phase),
    innings = factor(innings),
    bowler_id = factor(bowler_id),
    batter_is_home = as.integer(batter_is_home),
    over_z = as.numeric(scale(as.numeric(over))),
    batting_position_z = as.numeric(scale(as.numeric(striker_batting_position)))
  )]

  y <- as.numeric(dt$bat_score)

  # Treatment matrix: batter dummies excluding reference
  batter_fac <- factor(dt$striker_id)
  batter_levels <- setdiff(levels(batter_fac), ref)
  D <- Matrix::sparse.model.matrix(~ 0 + striker_id, data = dt)
  keep_cols <- setdiff(colnames(D), paste0("striker_id", ref))
  # colnames are striker_idBatter_001 style
  ref_col <- paste0("striker_id", ref)
  D <- D[, setdiff(colnames(D), ref_col), drop = FALSE]
  colnames(D) <- sub("^striker_id", "", colnames(D))

  # Controls: everything that confounds batter assignment / scoring except batter
  X <- Matrix::sparse.model.matrix(
    ~ 0 + 
      #series + 
      venue + 
      #season + 
      phase + 
      #innings +
      bowler_id + 
      #day_night + 
      batter_is_home + 
      over_z + 
      batting_position_z,
    data = dt
  )

  list(
    frame = dt,
    y = y,
    D = D,
    X = X,
    batter_levels = colnames(D),
    reference_batter = ref,
    feature_info = list(
      n_obs = length(y),
      n_batters = ncol(D),
      n_controls = ncol(X),
      min_balls = as.integer(min_balls),
      mean_y = mean(y)
    )
  )
}
