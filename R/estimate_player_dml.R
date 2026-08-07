#' Cross-fitted double / debiased ML for high-dimensional batter effects.
#'
#' Partially linear model: Y = D θ + g(X) + ε, with E[ε | D, X] = 0.
#' Nuisances ĝ(X) = E[Y | X] and m̂(X) = E[D | X] estimated by cv.glmnet
#' with K-fold cross-fitting, then OLS of residual Y on residual D.
#'
#' @param prepared Output of [prepare_player_dml_frame].
#' @param n_folds Cross-fitting folds.
#' @param seed RNG seed for folds.
#' @param cluster Optional cluster vector (e.g. match_id) for clustered SEs.
#' @return data.table of player effects with SE / CI, plus diagnostics list.
estimate_player_effects_dml <- function(
    prepared,
    n_folds = 5L,
    seed = 42L,
    cluster = NULL
) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Install glmnet: install.packages(\"glmnet\")", call. = FALSE)
  }
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Install Matrix", call. = FALSE)
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }

  y <- prepared$y
  D <- prepared$D
  X <- prepared$X
  n <- length(y)
  p <- ncol(D)

  if (is.null(cluster)) {
    cluster <- prepared$frame$match_id
  }
  cluster <- as.character(cluster)
  stopifnot(length(cluster) == n)

  set.seed(as.integer(seed))
  fold_id <- sample(rep(seq_len(n_folds), length.out = n))

  y_resid <- rep(NA_real_, n)
  D_resid <- Matrix::Matrix(0, nrow = n, ncol = p, sparse = TRUE)
  colnames(D_resid) <- colnames(D)

  for (k in seq_len(n_folds)) {
    test <- fold_id == k
    train <- !test

    # E[Y | X]
    fit_y <- glmnet::cv.glmnet(
      x = X[train, , drop = FALSE],
      y = y[train],
      family = "gaussian",
      alpha = 0.5,
      nfolds = 3L,
      standardize = TRUE
    )
    y_hat <- as.numeric(predict(fit_y, newx = X[test, , drop = FALSE], s = "lambda.min"))
    y_resid[test] <- y[test] - y_hat

    # E[D_j | X] for each batter column (gaussian PLR residualization)
    X_train <- X[train, , drop = FALSE]
    X_test <- X[test, , drop = FALSE]
    for (j in seq_len(p)) {
      d_j <- as.numeric(D[train, j])
      # Skip degenerate columns in this fold
      if (sd(d_j) < 1e-8) {
        D_resid[test, j] <- as.numeric(D[test, j]) - mean(d_j)
        next
      }
      fit_d <- glmnet::cv.glmnet(
        x = X_train,
        y = d_j,
        family = "gaussian",
        alpha = 0.5,
        nfolds = 3L,
        standardize = TRUE
      )
      d_hat <- as.numeric(predict(fit_d, newx = X_test, s = "lambda.min"))
      D_resid[test, j] <- as.numeric(D[test, j]) - d_hat
    }
  }

  # OLS on residuals (no intercept: D is relative to reference batter)
  DtD <- as.matrix(Matrix::crossprod(D_resid))
  Dty <- as.numeric(Matrix::crossprod(D_resid, y_resid))
  # ridge-stabilize if near-collinear
  ridge <- 1e-8 * mean(diag(DtD))
  theta <- as.numeric(solve(DtD + diag(ridge, p), Dty))
  names(theta) <- colnames(D)

  # Cluster-robust sandwich variance
  u <- as.numeric(y_resid - D_resid %*% theta)
  meat <- Matrix::Matrix(0, p, p, sparse = FALSE)
  unique_c <- unique(cluster)
  for (cl in unique_c) {
    ix <- which(cluster == cl)
    score <- Matrix::crossprod(D_resid[ix, , drop = FALSE], u[ix])
    meat <- meat + Matrix::tcrossprod(score)
  }
  bread <- solve(DtD + diag(ridge, p))
  V <- bread %*% as.matrix(meat) %*% bread
  se <- sqrt(pmax(diag(V), 0))

  z <- theta / se
  pval <- 2 * stats::pnorm(-abs(z))
  # BH-FDR across batters
  fdr <- stats::p.adjust(pval, method = "BH")

  est <- data.table::data.table(
    batter_id = names(theta),
    effect_runs_per_ball = as.numeric(theta),
    se = as.numeric(se),
    ci_lo = as.numeric(theta - 1.96 * se),
    ci_hi = as.numeric(theta + 1.96 * se),
    p_value = as.numeric(pval),
    fdr_q = as.numeric(fdr),
    reference_batter = prepared$reference_batter
  )
  data.table::setorder(est, -effect_runs_per_ball)

  balls <- prepared$frame[, .N, by = batter_id]
  data.table::setnames(balls, "N", "balls_faced")
  est <- balls[est, on = "batter_id"]

  list(
    estimates = est,
    diagnostics = list(
      n_obs = n,
      n_batters = p,
      n_folds = as.integer(n_folds),
      n_clusters = length(unique_c),
      reference_batter = prepared$reference_batter,
      mean_abs_y_resid = mean(abs(y_resid)),
      seed = as.integer(seed)
    ),
    residuals = list(y = y_resid) # D residuals large; omit by default
  )
}

#' Forest plot of estimated player effects.
plot_player_effects <- function(estimates, out_path = NULL, top_n = 30L) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not installed; skipping plot")
    return(invisible(NULL))
  }
  dt <- data.table::copy(estimates)
  dt <- utils::head(dt, as.integer(top_n))
  dt[, batter_id := factor(batter_id, levels = rev(batter_id))]

  p <- ggplot2::ggplot(dt, ggplot2::aes(x = effect_runs_per_ball, y = batter_id)) +
    ggplot2::geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    ggplot2::geom_pointrange(
      ggplot2::aes(xmin = ci_lo, xmax = ci_hi),
      colour = "#1d4ed8"
    ) +
    ggplot2::labs(
      title = "DML batter run effects (T20, per ball)",
      subtitle = paste0("Relative to ", unique(dt$reference_batter)),
      x = "Runs per ball (ATE-style coefficient)",
      y = NULL
    ) +
    ggplot2::theme_minimal(base_size = 12)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(out_path, p, width = 8, height = max(4, 0.22 * nrow(dt)), dpi = 120)
  }
  invisible(p)
}
