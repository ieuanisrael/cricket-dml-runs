#' Cross-fitted propensity that a striker is on strike given context X.
#'
#' For each delivery, the multi-valued treatment is which striker faces the ball.
#' This estimates e_j(X) = P(striker = j | X) with one-vs-rest elastic nets
#' and K-fold cross-fitting (out-of-fold predictions).
#'
#' "In the game at that moment" is operationalised as being the striker on
#' that delivery (given match context: venue, phase/power_play, bowler, etc.).
#'
#' @param prepared Output of [prepare_player_dml_frame].
#' @param n_folds Cross-fitting folds.
#' @param seed RNG seed.
#' @param include_reference If TRUE, also estimate propensity for the reference striker.
#' @return list with `by_striker`, `ball_level`, `diagnostics`.
estimate_striker_propensity <- function(
    prepared,
    n_folds = 5L,
    seed = 42L,
    include_reference = TRUE
) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Install glmnet", call. = FALSE)
  }
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Install Matrix", call. = FALSE)
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }

  X <- prepared$X
  D <- prepared$D
  n <- nrow(X)
  frame <- data.table::as.data.table(prepared$frame)
  ref <- prepared$reference_batter

  # Full striker dummy matrix including reference
  all_strikers <- sort(unique(as.character(frame$striker_id)))
  if (!include_reference) {
    all_strikers <- setdiff(all_strikers, ref)
  }

  D_full <- Matrix::sparse.model.matrix(~ 0 + striker_id, data = frame)
  colnames(D_full) <- sub("^striker_id", "", colnames(D_full))
  D_full <- D_full[, all_strikers, drop = FALSE]
  p <- ncol(D_full)

  set.seed(as.integer(seed))
  fold_id <- sample(rep(seq_len(n_folds), length.out = n))

  prop_hat <- Matrix::Matrix(0, nrow = n, ncol = p, sparse = FALSE)
  colnames(prop_hat) <- all_strikers

  for (k in seq_len(n_folds)) {
    test <- fold_id == k
    train <- !test
    X_train <- X[train, , drop = FALSE]
    X_test <- X[test, , drop = FALSE]

    for (j in seq_len(p)) {
      d_j <- as.numeric(D_full[train, j])
      if (stats::sd(d_j) < 1e-8) {
        prop_hat[test, j] <- mean(d_j)
        next
      }
      # Logistic propensity for being this striker
      fit <- tryCatch(
        glmnet::cv.glmnet(
          x = X_train,
          y = d_j,
          family = "binomial",
          alpha = 0.5,
          nfolds = min(3L, sum(train)),
          standardize = TRUE,
          type.measure = "deviance"
        ),
        error = function(e) NULL
      )
      if (is.null(fit)) {
        # Fallback: gaussian PLR residualization-style mean prediction
        fit_g <- glmnet::cv.glmnet(
          x = X_train, y = d_j, family = "gaussian",
          alpha = 0.5, nfolds = 3L, standardize = TRUE
        )
        prop_hat[test, j] <- pmin(1, pmax(0, as.numeric(
          predict(fit_g, newx = X_test, s = "lambda.min")
        )))
      } else {
        prop_hat[test, j] <- as.numeric(
          predict(fit, newx = X_test, s = "lambda.min", type = "response")
        )
      }
    }
  }

  # Renormalise each row toward a probability simplex (one-vs-rest can overshoot)
  row_sums <- as.numeric(Matrix::rowSums(prop_hat))
  row_sums[row_sums < 1e-8] <- 1
  prop_hat <- sweep(as.matrix(prop_hat), 1L, row_sums, "/")

  observed <- as.character(frame$striker_id)
  prop_observed <- prop_hat[cbind(seq_len(n), match(observed, all_strikers))]

  ball_level <- data.table::data.table(
    match_id = frame$match_id,
    innings = frame$innings,
    over = frame$over,
    striker_id = observed,
    propensity = prop_observed
  )
  if ("phase" %in% names(frame)) {
    ball_level[, phase := as.character(frame$phase)]
  } else if ("power_play" %in% names(frame)) {
    ball_level[, phase := ifelse(as.integer(frame$power_play) == 1L, "powerplay", "other")]
  }
  if ("striker_batting_position" %in% names(frame)) {
    ball_level[, striker_batting_position := frame$striker_batting_position]
  }

  by_striker <- data.table::data.table(
    striker_id = all_strikers,
    mean_propensity = colMeans(prop_hat),
    median_propensity = apply(prop_hat, 2L, stats::median),
    balls_faced = as.numeric(Matrix::colSums(D_full)),
    empirical_share = as.numeric(Matrix::colSums(D_full)) / n,
    is_reference = all_strikers == ref
  )
  # Mean propensity on balls they actually faced
  faced_mean <- ball_level[, .(mean_propensity_when_facing = mean(propensity)), by = striker_id]
  by_striker <- faced_mean[by_striker, on = "striker_id"]
  data.table::setorder(by_striker, -mean_propensity)

  list(
    by_striker = by_striker,
    ball_level = ball_level,
    propensity_matrix = prop_hat,
    strikers = all_strikers,
    diagnostics = list(
      n_obs = n,
      n_strikers = p,
      n_folds = as.integer(n_folds),
      reference_batter = ref,
      seed = as.integer(seed),
      mean_prop_observed = mean(prop_observed)
    )
  )
}
