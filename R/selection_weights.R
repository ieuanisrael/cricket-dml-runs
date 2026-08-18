#' Selection / exposure weights for unequal balls faced.
#'
#' Batters who survive longer contribute more deliveries. These weights
#' rebalance the sample.
#'
#' Methods:
#' - `kernel_exposure`: w(t) ∝ f_target(t) / f_hat(t) on balls-so-far within
#'   a striker–innings stay (Gaussian-kernel density; uniform target on 1…T_max).
#' - `innings_equal`: each striker–innings stay gets total weight 1.
#' - `none`: uniform weights.
#'
#' @param frame data.table from [prepare_player_dml_frame] (`$frame`).
#' @param method Weighting scheme.
#' @param t_max Cap for target support (kernel_exposure).
#' @param bw Bandwidth for Gaussian kernel density (NULL = bw.nrd0).
#' @param trim Truncate weights above this multiple of the median.
#' @return numeric weight vector (mean 1).
compute_selection_weights <- function(
    frame,
    method = c("kernel_exposure", "innings_equal", "none"),
    t_max = 30L,
    bw = NULL,
    trim = 10
) {
  method <- match.arg(method)
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }
  dt <- data.table::as.data.table(frame)
  n <- nrow(dt)
  balls_so_far <- .balls_so_far(dt)
  stay_id <- paste(dt$match_id, dt$innings, dt$striker_id, sep = "|")

  if (identical(method, "none")) {
    w <- rep(1, n)
  } else if (identical(method, "innings_equal")) {
    stay_n <- as.numeric(table(stay_id)[as.character(stay_id)])
    w <- 1 / stay_n
  } else {
    t_max <- as.integer(t_max)
    t <- pmin(as.numeric(balls_so_far), t_max)
    if (is.null(bw)) {
      bw <- stats::bw.nrd0(t)
      if (!is.finite(bw) || bw <= 0) bw <- 1
    }
    dens <- stats::density(t, bw = bw, from = 1, to = t_max, n = 512)
    f_hat <- stats::approx(dens$x, pmax(dens$y, .Machine$double.eps), xout = t)$y
    f_target <- 1 / t_max
    w <- f_target / f_hat
  }

  w[!is.finite(w) | w <= 0] <- stats::median(w[is.finite(w) & w > 0], na.rm = TRUE)
  if (!is.null(trim) && is.finite(trim) && trim > 0) {
    w <- pmin(w, trim * stats::median(w))
  }
  w <- w / mean(w)

  attr(w, "method") <- method
  attr(w, "balls_so_far") <- balls_so_far
  attr(w, "t_max") <- if (identical(method, "kernel_exposure")) as.integer(t_max) else NA_integer_
  attr(w, "bw") <- if (identical(method, "kernel_exposure")) bw else NA_real_
  w
}

.balls_so_far <- function(dt) {
  tmp <- data.table::as.data.table(dt)
  tmp[, .row_id := .I]
  ord_cols <- c("match_id", "innings", "striker_id")
  if (all(c("over", "ball_in_over") %in% names(tmp))) {
    ord_cols <- c(ord_cols, "over", "ball_in_over")
  } else if ("legal_ball_index" %in% names(tmp)) {
    ord_cols <- c(ord_cols, "legal_ball_index")
  }
  data.table::setorderv(tmp, ord_cols)
  tmp[, balls_so_far := seq_len(.N), by = .(match_id, innings, striker_id)]
  data.table::setorder(tmp, .row_id)
  as.integer(tmp$balls_so_far)
}

#' Diagnostic: mean weight vs balls-so-far.
plot_selection_weights <- function(weights, out_path = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not installed; skipping weight plot")
    return(invisible(NULL))
  }
  t <- attr(weights, "balls_so_far")
  if (is.null(t)) t <- seq_along(weights)
  dt <- data.table::data.table(balls_so_far = as.integer(t), weight = as.numeric(weights))
  agg <- dt[, .(mean_weight = mean(weight), n = .N), by = balls_so_far]
  data.table::setorder(agg, balls_so_far)

  p <- ggplot2::ggplot(agg, ggplot2::aes(x = balls_so_far, y = mean_weight)) +
    ggplot2::geom_hline(yintercept = 1, linetype = 2, colour = "grey50") +
    ggplot2::geom_line(colour = "#1d4ed8", linewidth = 0.8) +
    ggplot2::geom_point(ggplot2::aes(size = n), colour = "#1d4ed8", alpha = 0.7) +
    ggplot2::scale_size_area(max_size = 8) +
    ggplot2::labs(
      title = "Selection weights vs balls faced in stay",
      subtitle = paste0("method = ", attr(weights, "method")),
      x = "Balls so far (striker–innings)",
      y = "Mean weight (normalised to 1)",
      size = "Deliveries"
    ) +
    ggplot2::theme_minimal(base_size = 12)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(out_path, p, width = 7.5, height = 4.5, dpi = 120)
  }
  invisible(p)
}

#' Compare unweighted vs selection-weighted DML player effects.
compare_weighted_vs_unweighted <- function(unweighted_estimates, weighted_estimates) {
  u <- data.table::as.data.table(unweighted_estimates)[, .(
    striker_id,
    balls_faced,
    unweighted = effect_runs_per_ball,
    unweighted_ci_lo = ci_lo,
    unweighted_ci_hi = ci_hi,
    reference_batter
  )]
  w <- data.table::as.data.table(weighted_estimates)[, .(
    striker_id,
    weighted = effect_runs_per_ball,
    weighted_ci_lo = ci_lo,
    weighted_ci_hi = ci_hi
  )]
  tab <- merge(u, w, by = "striker_id", all = FALSE)
  tab[, `:=`(
    diff_weighted_minus_unweighted = weighted - unweighted,
    abs_diff = abs(weighted - unweighted)
  )]
  data.table::setorder(tab, -weighted)

  list(
    table = tab,
    corr = stats::cor(tab$unweighted, tab$weighted),
    spearman = stats::cor(tab$unweighted, tab$weighted, method = "spearman"),
    mae = mean(tab$abs_diff),
    rmse = sqrt(mean(tab$diff_weighted_minus_unweighted^2))
  )
}

#' Scatterplot of weighted vs unweighted DML effects.
plot_weighted_vs_unweighted <- function(comparison, out_path = NULL, weight_method = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not installed; skipping plot")
    return(invisible(NULL))
  }
  dt <- comparison$table
  lims <- range(c(dt$unweighted, dt$weighted), finite = TRUE)
  pad <- diff(lims) * 0.08
  if (!is.finite(pad) || pad == 0) pad <- 0.05
  lims <- c(lims[1] - pad, lims[2] + pad)

  subtitle <- sprintf(
    "Pearson r = %.3f | Spearman = %.3f",
    comparison$corr,
    comparison$spearman
  )
  if (!is.null(weight_method) && nzchar(weight_method)) {
    subtitle <- paste0(subtitle, " | weights = ", weight_method)
  }

  p <- ggplot2::ggplot(dt, ggplot2::aes(x = unweighted, y = weighted)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
    ggplot2::geom_hline(yintercept = 0, colour = "grey80") +
    ggplot2::geom_vline(xintercept = 0, colour = "grey80") +
    ggplot2::geom_smooth(method = "lm", se = TRUE, colour = "#64748b", fill = "#cbd5e1", linewidth = 0.6) +
    ggplot2::geom_point(colour = "#1d4ed8", size = 2.2, alpha = 0.85) +
    ggplot2::coord_equal(xlim = lims, ylim = lims) +
    ggplot2::labs(
      title = "Weighted vs unweighted DML batter effects",
      subtitle = subtitle,
      x = "Unweighted DML effect (runs per ball)",
      y = "Weighted DML effect (runs per ball)"
    ) +
    ggplot2::theme_minimal(base_size = 12)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(out_path, p, width = 7, height = 6, dpi = 120)
  }
  invisible(p)
}
