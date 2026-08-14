#' Naive / context-only runs above expectation (RAE) by batter.
#'
#' Fits E[Y | X] with the same control matrix as DML (no batter dummies),
#' then averages ball-level residuals by batter. Contrasts are relative to
#' the DML reference batter for comparability.
#'
#' @param prepared Output of [prepare_player_dml_frame].
#' @param n_folds Cross-fitting folds for the expectation model.
#' @param seed RNG seed for folds.
#' @return list with `estimates` (data.table) and `diagnostics`.
estimate_player_rae <- function(prepared, n_folds = 5L, seed = 42L) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Install glmnet: install.packages(\"glmnet\")", call. = FALSE)
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }

  y <- prepared$y
  X <- prepared$X
  n <- length(y)
  ref <- prepared$reference_batter

  set.seed(as.integer(seed))
  fold_id <- sample(rep(seq_len(n_folds), length.out = n))
  y_hat <- rep(NA_real_, n)

  for (k in seq_len(n_folds)) {
    test <- fold_id == k
    train <- !test
    fit <- glmnet::cv.glmnet(
      x = X[train, , drop = FALSE],
      y = y[train],
      family = "gaussian",
      alpha = 0.5,
      nfolds = 3L,
      standardize = TRUE
    )
    y_hat[test] <- as.numeric(predict(fit, newx = X[test, , drop = FALSE], s = "lambda.min"))
  }

  rae_ball <- y - y_hat
  dt <- data.table::data.table(
    striker_id = prepared$frame$striker_id,
    rae = rae_ball
  )
  by_batter <- dt[, .(
    balls_faced = .N,
    rae_mean = mean(rae),
    rae_se = stats::sd(rae) / sqrt(.N)
  ), by = striker_id]

  ref_rae <- by_batter[striker_id == ref]$rae_mean
  if (!length(ref_rae)) {
    stop("Reference batter not found in RAE table: ", ref, call. = FALSE)
  }

  by_batter[, `:=`(
    rae_vs_ref = rae_mean - ref_rae,
    reference_batter = ref,
    ci_lo = rae_mean - ref_rae - 1.96 * rae_se,
    ci_hi = rae_mean - ref_rae + 1.96 * rae_se
  )]
  # Drop reference row from contrast table (effect is definitionally 0)
  estimates <- by_batter[striker_id != ref]
  data.table::setorder(estimates, -rae_vs_ref)

  list(
    estimates = estimates,
    ball_rae = rae_ball,
    diagnostics = list(
      n_obs = n,
      n_folds = as.integer(n_folds),
      reference_batter = ref,
      mean_abs_rae = mean(abs(rae_ball)),
      seed = as.integer(seed)
    )
  )
}

#' Compare batter RAE contrasts to DML effects.
#'
#' @param dml_estimates data.table from [estimate_player_effects_dml].
#' @param rae_estimates data.table from [estimate_player_rae].
#' @return list with `table`, summary metrics, and optional plots via helpers.
compare_rae_vs_dml <- function(dml_estimates, rae_estimates) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }

  dml <- data.table::as.data.table(dml_estimates)[, .(
    striker_id,
    balls_faced,
    dml_effect = effect_runs_per_ball,
    dml_se = se,
    dml_ci_lo = ci_lo,
    dml_ci_hi = ci_hi,
    reference_batter
  )]
  rae <- data.table::as.data.table(rae_estimates)[, .(
    striker_id,
    rae_vs_ref,
    rae_mean,
    rae_se,
    rae_ci_lo = ci_lo,
    rae_ci_hi = ci_hi
  )]

  tab <- merge(dml, rae, by = "striker_id", all = FALSE)
  tab[, `:=`(
    diff_dml_minus_rae = dml_effect - rae_vs_ref,
    abs_diff = abs(dml_effect - rae_vs_ref)
  )]
  data.table::setorder(tab, -dml_effect)

  list(
    table = tab,
    corr = stats::cor(tab$dml_effect, tab$rae_vs_ref),
    spearman = stats::cor(tab$dml_effect, tab$rae_vs_ref, method = "spearman"),
    mae = mean(tab$abs_diff),
    rmse = sqrt(mean(tab$diff_dml_minus_rae^2))
  )
}

#' Scatter of RAE vs DML player effects.
plot_rae_vs_dml <- function(comparison, out_path = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not installed; skipping plot")
    return(invisible(NULL))
  }
  dt <- comparison$table
  lims <- range(c(dt$dml_effect, dt$rae_vs_ref), finite = TRUE)
  pad <- diff(lims) * 0.08
  lims <- c(lims[1] - pad, lims[2] + pad)

  p <- ggplot2::ggplot(dt, ggplot2::aes(x = rae_vs_ref, y = dml_effect)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
    ggplot2::geom_hline(yintercept = 0, colour = "grey80") +
    ggplot2::geom_vline(xintercept = 0, colour = "grey80") +
    ggplot2::geom_point(colour = "#1d4ed8", size = 2.2, alpha = 0.85) +
    ggplot2::coord_equal(xlim = lims, ylim = lims) +
    ggplot2::labs(
      title = "RAE vs DML batter effects",
      subtitle = sprintf(
        "Pearson r = %.3f | Spearman = %.3f | both vs %s",
        comparison$corr,
        comparison$spearman,
        unique(dt$reference_batter)
      ),
      x = "RAE (mean residual vs reference)",
      y = "DML effect (runs per ball)"
    ) +
    ggplot2::theme_minimal(base_size = 12)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(out_path, p, width = 7, height = 6, dpi = 120)
  }
  invisible(p)
}

#' Side-by-side forest of DML and RAE for top players by |DML|.
plot_rae_dml_forest <- function(comparison, out_path = NULL, top_n = 30L) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not installed; skipping plot")
    return(invisible(NULL))
  }
  dt <- utils::head(comparison$table, as.integer(top_n))
  long <- data.table::rbindlist(list(
    dt[, .(striker_id, method = "DML", effect = dml_effect, lo = dml_ci_lo, hi = dml_ci_hi)],
    dt[, .(striker_id, method = "RAE", effect = rae_vs_ref, lo = rae_ci_lo, hi = rae_ci_hi)]
  ))
  long[, striker_id := factor(striker_id, levels = rev(dt$striker_id))]
  long[, method := factor(method, levels = c("DML", "RAE"))]

  p <- ggplot2::ggplot(long, ggplot2::aes(x = effect, y = striker_id, colour = method)) +
    ggplot2::geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    ggplot2::geom_pointrange(
      ggplot2::aes(xmin = lo, xmax = hi),
      position = ggplot2::position_dodge(width = 0.55),
      linewidth = 0.45
    ) +
    ggplot2::scale_colour_manual(values = c(DML = "#1d4ed8", RAE = "#b45309")) +
    ggplot2::labs(
      title = "Batter effects: DML vs RAE",
      subtitle = paste0("Relative to ", unique(comparison$table$reference_batter)),
      x = "Runs per ball (vs reference)",
      y = NULL,
      colour = NULL
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(legend.position = "top")

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(out_path, p, width = 8.5, height = max(4.5, 0.28 * nrow(dt)), dpi = 120)
  }
  invisible(p)
}
