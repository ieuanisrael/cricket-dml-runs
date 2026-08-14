#' Delivery-level Elo ratings for batters and bowlers.
#'
#' Each ball is a duel. Batter "score" in [0, 1] from runs / wicket; expected
#' score from Elo difference; both ratings update (zero-sum). Chronological
#' order: match_id, innings, over, ball_in_over.
#'
#' This is an outcome-derived **skill proxy**, not independent ground truth.
#'
#' @param deliveries Ball-by-ball data.table/data.frame.
#' @param k Elo K-factor per delivery (small; many balls).
#' @param base Starting rating.
#' @param scale Elo logistic scale (classic 400).
#' @return list with `batter`, `bowler` final ratings and `history` optional.
compute_elo_ratings <- function(
    deliveries,
    k = 8,
    base = 1500,
    scale = 400
) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }

  dt <- data.table::as.data.table(deliveries)
  req <- c("match_id", "innings", "over", "ball_in_over", "striker_id", "bowler_id", "bat_score")
  miss <- setdiff(req, names(dt))
  if (length(miss)) {
    stop("Missing columns for Elo: ", paste(miss, collapse = ", "), call. = FALSE)
  }
  if (!"is_wicket" %in% names(dt)) {
    dt[, is_wicket := FALSE]
  }

  data.table::setorder(dt, match_id, innings, over, ball_in_over)

  bat_r <- new.env(parent = emptyenv())
  bowl_r <- new.env(parent = emptyenv())
  bat_n <- new.env(parent = emptyenv())
  bowl_n <- new.env(parent = emptyenv())

  get_r <- function(env, id, base) {
    if (is.null(env[[id]])) env[[id]] <- base
    env[[id]]
  }
  bump_n <- function(env, id) {
    if (is.null(env[[id]])) env[[id]] <- 0L
    env[[id]] <- env[[id]] + 1L
  }

  n <- nrow(dt)
  elo_bat_pre <- numeric(n)
  elo_bowl_pre <- numeric(n)
  outcome <- numeric(n)
  expected <- numeric(n)

  for (i in seq_len(n)) {
    b <- as.character(dt$striker_id[[i]])
    w <- as.character(dt$bowler_id[[i]])
    rb <- get_r(bat_r, b, base)
    rw <- get_r(bowl_r, w, base)
    elo_bat_pre[[i]] <- rb
    elo_bowl_pre[[i]] <- rw

    s <- .elo_striker_score(dt$bat_score[[i]], isTRUE(dt$is_wicket[[i]]))
    e <- 1 / (1 + 10^((rw - rb) / scale))
    outcome[[i]] <- s
    expected[[i]] <- e

    bat_r[[b]] <- rb + k * (s - e)
    bowl_r[[w]] <- rw + k * (e - s)
    bump_n(bat_n, b)
    bump_n(bowl_n, w)
  }

  batter_ids <- ls(bat_r)
  bowler_ids <- ls(bowl_r)

  batter <- data.table::data.table(
    striker_id = batter_ids,
    elo = vapply(batter_ids, function(id) bat_r[[id]], numeric(1)),
    balls = vapply(batter_ids, function(id) as.integer(bat_n[[id]]), integer(1))
  )
  data.table::setorder(batter, -elo)

  bowler <- data.table::data.table(
    bowler_id = bowler_ids,
    elo = vapply(bowler_ids, function(id) bowl_r[[id]], numeric(1)),
    balls = vapply(bowler_ids, function(id) as.integer(bowl_n[[id]]), integer(1))
  )
  data.table::setorder(bowler, -elo)

  history <- data.table::copy(dt[, .(
    match_id, innings, over, ball_in_over, striker_id, bowler_id,
    bat_score, is_wicket
  )])
  history[, `:=`(
    elo_striker_pre = elo_bat_pre,
    elo_bowler_pre = elo_bowl_pre,
    elo_outcome = outcome,
    elo_expected = expected
  )]

  list(
    striker = batter,
    bowler = bowler,
    history = history,
    meta = list(k = k, base = base, scale = scale, n_deliveries = n)
  )
}

#' Map ball result to batter success in [0, 1] for Elo.
.elo_striker_score <- function(runs, is_wicket) {
  if (isTRUE(is_wicket)) {
    return(0)
  }
  runs <- as.integer(runs)
  if (is.na(runs) || runs <= 0L) return(0.40)
  if (runs == 1L) return(0.55)
  if (runs == 2L) return(0.65)
  if (runs == 3L) return(0.72)
  if (runs == 4L) return(0.82)
  if (runs >= 6L) return(0.92)
  0.60
}

#' Align batter Elo to DML reference contrast for comparison.
#'
#' @param elo_striker Output `$striker` from [compute_elo_ratings].
#' @param reference_batter Reference id used in DML.
#' @param min_balls Optional filter.
batter_elo_vs_ref <- function(elo_striker, reference_batter, min_balls = 0L) {
  dt <- data.table::as.data.table(elo_striker)
  if (min_balls > 0L) {
    dt <- dt[balls >= as.integer(min_balls)]
  }
  ref_elo <- dt[striker_id == reference_batter]$elo
  if (!length(ref_elo)) {
    stop("Reference batter missing from Elo table: ", reference_batter, call. = FALSE)
  }
  out <- dt[striker_id != reference_batter]
  out[, `:=`(
    elo_vs_ref = elo - ref_elo,
    reference_batter = reference_batter
  )]
  data.table::setorder(out, -elo_vs_ref)
  out
}

#' Compare batter Elo contrasts to DML effects.
compare_elo_vs_dml <- function(dml_estimates, elo_vs_ref) {
  dml <- data.table::as.data.table(dml_estimates)[, .(
    striker_id,
    balls_faced,
    dml_effect = effect_runs_per_ball,
    dml_ci_lo = ci_lo,
    dml_ci_hi = ci_hi,
    reference_batter
  )]
  elo <- data.table::as.data.table(elo_vs_ref)[, .(
    striker_id,
    elo,
    elo_vs_ref,
    elo_balls = balls
  )]
  tab <- merge(dml, elo, by = "striker_id", all = FALSE)
  # Standardize for a scale-free association check
  tab[, elo_vs_ref_z := as.numeric(scale(elo_vs_ref))]
  tab[, dml_z := as.numeric(scale(dml_effect))]
  data.table::setorder(tab, -dml_effect)

  list(
    table = tab,
    corr = stats::cor(tab$dml_effect, tab$elo_vs_ref),
    spearman = stats::cor(tab$dml_effect, tab$elo_vs_ref, method = "spearman"),
    corr_z = stats::cor(tab$dml_z, tab$elo_vs_ref_z)
  )
}

#' Scatter: Elo vs DML batter effects.
plot_elo_vs_dml <- function(comparison, out_path = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not installed; skipping plot")
    return(invisible(NULL))
  }
  dt <- comparison$table
  p <- ggplot2::ggplot(dt, ggplot2::aes(x = elo_vs_ref, y = dml_effect)) +
    ggplot2::geom_hline(yintercept = 0, colour = "grey80") +
    ggplot2::geom_vline(xintercept = 0, colour = "grey80") +
    ggplot2::geom_smooth(method = "lm", se = TRUE, colour = "#64748b", fill = "#cbd5e1", linewidth = 0.6) +
    ggplot2::geom_point(colour = "#1d4ed8", size = 2.2, alpha = 0.85) +
    ggplot2::labs(
      title = "Elo vs DML batter effects",
      subtitle = sprintf(
        "Pearson r = %.3f | Spearman = %.3f | Elo and DML both vs %s",
        comparison$corr,
        comparison$spearman,
        unique(dt$reference_batter)
      ),
      x = "Elo rating minus reference",
      y = "DML effect (runs per ball)"
    ) +
    ggplot2::theme_minimal(base_size = 12)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(out_path, p, width = 7, height = 6, dpi = 120)
  }
  invisible(p)
}

#' Forest of top DML batters with Elo contrast markers (scaled to DML axis).
plot_elo_on_dml_forest <- function(comparison, out_path = NULL, top_n = 30L) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not installed; skipping plot")
    return(invisible(NULL))
  }
  dt <- utils::head(comparison$table, as.integer(top_n))
  # Affine-map Elo contrasts onto DML scale for visual overlay only
  fit <- stats::lm(dml_effect ~ elo_vs_ref, data = comparison$table)
  dt[, elo_on_dml_scale := stats::predict(fit, newdata = dt)]
  dt[, striker_id := factor(striker_id, levels = rev(striker_id))]

  p <- ggplot2::ggplot(dt, ggplot2::aes(y = striker_id)) +
    ggplot2::geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    ggplot2::geom_pointrange(
      ggplot2::aes(x = dml_effect, xmin = dml_ci_lo, xmax = dml_ci_hi),
      colour = "#1d4ed8"
    ) +
    ggplot2::geom_point(
      ggplot2::aes(x = elo_on_dml_scale),
      colour = "#b45309",
      shape = 18,
      size = 2.6
    ) +
    ggplot2::labs(
      title = "DML batter effects with Elo proxy overlay",
      subtitle = paste0(
        "Blue = DML (±95% CI); amber = Elo contrast mapped to DML scale via linear fit; vs ",
        unique(dt$reference_batter)
      ),
      x = "Runs per ball (DML scale)",
      y = NULL
    ) +
    ggplot2::theme_minimal(base_size = 12)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(out_path, p, width = 8.5, height = max(4, 0.22 * nrow(dt)), dpi = 120)
  }
  invisible(p)
}
