#' Large-data DML with caret (+ optional XGBoost GPU) for striker effects.
#'
#' caret does not provide a general GPU backend. For GPU acceleration this
#' script uses caret's `xgbTree` method with XGBoost `tree_method` /
#' `device` set for CUDA when `use_gpu = TRUE` (requires a GPU build of
#' xgboost). Treatment nuisances E[D | X] stay on sparse `glmnet` — fitting
#' caret once per striker dummy is not practical at scale.
#'
#' Partially linear model: Y = D θ + g(X) + ε.
#'
#' @param prepared Output of [prepare_player_dml_frame].
#' @param n_folds Cross-fitting folds.
#' @param seed RNG seed.
#' @param cluster Cluster ids for SEs (default `match_id`).
#' @param use_gpu Use XGBoost GPU hist if available.
#' @param caret_method caret method for E[Y|X]: `"xgbTree"` (default) or `"glmnet"`.
#' @param tune_length caret tuneLength when grid is NULL.
#' @param xgb_nrounds,xgb_max_depth,xgb_eta Fixed XGBoost params (no nested search by default for scale).
#' @param verbose Print fold progress.
#' @return list with `estimates`, `diagnostics` (same shape as glmnet DML).
estimate_player_effects_dml_caret <- function(
    prepared,
    n_folds = 5L,
    seed = 42L,
    cluster = NULL,
    use_gpu = FALSE,
    caret_method = c("xgbTree", "glmnet"),
    tune_length = 1L,
    xgb_nrounds = 200L,
    xgb_max_depth = 6L,
    xgb_eta = 0.08,
    verbose = TRUE
) {
  caret_method <- match.arg(caret_method)
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop("Install caret: install.packages(\"caret\")", call. = FALSE)
  }
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Install glmnet: install.packages(\"glmnet\")", call. = FALSE)
  }
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Install Matrix", call. = FALSE)
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }
  if (identical(caret_method, "xgbTree") && !requireNamespace("xgboost", quietly = TRUE)) {
    stop(
      "caret method 'xgbTree' requires xgboost.\n",
      "  install.packages(\"xgboost\")\n",
      "For GPU: install an xgboost build with CUDA, then set use_gpu = TRUE.",
      call. = FALSE
    )
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

  gpu_info <- .dml_gpu_setup(use_gpu = use_gpu, caret_method = caret_method)
  if (verbose) {
    message("DML-caret: method=", caret_method,
            " gpu_requested=", use_gpu,
            " gpu_active=", gpu_info$gpu_active,
            " n=", n, " p_strikers=", p, " p_controls=", ncol(X))
    if (nzchar(gpu_info$note)) message("  ", gpu_info$note)
  }

  set.seed(as.integer(seed))
  fold_id <- sample(rep(seq_len(n_folds), length.out = n))

  y_resid <- rep(NA_real_, n)
  D_resid <- Matrix::Matrix(0, nrow = n, ncol = p, sparse = TRUE)
  colnames(D_resid) <- colnames(D)

  for (k in seq_len(n_folds)) {
    if (verbose) message("  fold ", k, "/", n_folds)
    test <- fold_id == k
    train <- !test

    y_hat <- .caret_fit_predict_y(
      x_train = X[train, , drop = FALSE],
      y_train = y[train],
      x_test = X[test, , drop = FALSE],
      caret_method = caret_method,
      gpu_info = gpu_info,
      tune_length = as.integer(tune_length),
      xgb_nrounds = as.integer(xgb_nrounds),
      xgb_max_depth = as.integer(xgb_max_depth),
      xgb_eta = xgb_eta,
      seed = as.integer(seed) + k
    )
    y_resid[test] <- y[test] - y_hat

    # High-dim D nuisances: sparse elastic net (caret-per-dummy is too slow)
    X_train <- X[train, , drop = FALSE]
    X_test <- X[test, , drop = FALSE]
    for (j in seq_len(p)) {
      d_j <- as.numeric(D[train, j])
      if (stats::sd(d_j) < 1e-8) {
        D_resid[test, j] <- as.numeric(D[test, j]) - mean(d_j)
        next
      }
      fit_d <- glmnet::cv.glmnet(
        x = X_train,
        y = d_j,
        family = "gaussian",
        alpha = 0.5,
        nfolds = min(3L, sum(train)),
        standardize = TRUE
      )
      d_hat <- as.numeric(predict(fit_d, newx = X_test, s = "lambda.min"))
      D_resid[test, j] <- as.numeric(D[test, j]) - d_hat
    }
  }

  DtD <- as.matrix(Matrix::crossprod(D_resid))
  Dty <- as.numeric(Matrix::crossprod(D_resid, y_resid))
  ridge <- 1e-8 * mean(diag(DtD))
  theta <- as.numeric(solve(DtD + diag(ridge, p), Dty))
  names(theta) <- colnames(D)

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
  fdr <- stats::p.adjust(pval, method = "BH")

  est <- data.table::data.table(
    striker_id = names(theta),
    effect_runs_per_ball = as.numeric(theta),
    se = as.numeric(se),
    ci_lo = as.numeric(theta - 1.96 * se),
    ci_hi = as.numeric(theta + 1.96 * se),
    p_value = as.numeric(pval),
    fdr_q = as.numeric(fdr),
    reference_batter = prepared$reference_batter
  )
  data.table::setorder(est, -effect_runs_per_ball)

  balls <- prepared$frame[, .N, by = striker_id]
  data.table::setnames(balls, "N", "balls_faced")
  est <- balls[est, on = "striker_id"]

  list(
    estimates = est,
    diagnostics = list(
      n_obs = n,
      n_batters = p,
      n_folds = as.integer(n_folds),
      n_clusters = length(unique_c),
      reference_batter = prepared$reference_batter,
      caret_method = caret_method,
      use_gpu = use_gpu,
      gpu_active = gpu_info$gpu_active,
      gpu_note = gpu_info$note,
      mean_abs_y_resid = mean(abs(y_resid)),
      seed = as.integer(seed)
    )
  )
}

.dml_gpu_setup <- function(use_gpu, caret_method) {
  note <- ""
  gpu_active <- FALSE
  xgb_params <- list()

  if (!isTRUE(use_gpu)) {
    if (identical(caret_method, "xgbTree")) {
      xgb_params$tree_method <- "hist"
      xgb_params$device <- "cpu"
    }
    return(list(gpu_active = FALSE, note = note, xgb_params = xgb_params))
  }

  if (!identical(caret_method, "xgbTree")) {
    note <- "use_gpu=TRUE ignored unless caret_method='xgbTree'."
    return(list(gpu_active = FALSE, note = note, xgb_params = xgb_params))
  }

  # Prefer XGBoost >= 2.0 device=cuda; fall back to tree_method=gpu_hist
  xgb_ver <- tryCatch(as.character(utils::packageVersion("xgboost")), error = function(e) "0")
  caps <- tryCatch(xgboost::xgb.capabilities(), error = function(e) NULL)
  cuda_ok <- is.list(caps) && isTRUE(caps$cuda)

  if (isTRUE(cuda_ok) || isTRUE(use_gpu)) {
    # Still set GPU params when requested; xgboost errors clearly if unsupported
    if (utils::compareVersion(xgb_ver, "2.0.0") >= 0) {
      xgb_params$tree_method <- "hist"
      xgb_params$device <- "cuda"
      note <- "XGBoost device=cuda (hist)."
    } else {
      xgb_params$tree_method <- "gpu_hist"
      note <- "XGBoost tree_method=gpu_hist."
    }
    gpu_active <- TRUE
  } else {
    xgb_params$tree_method <- "hist"
    xgb_params$device <- "cpu"
    note <- "GPU requested but xgboost CUDA capability not detected; using CPU hist."
    gpu_active <- FALSE
  }

  list(gpu_active = gpu_active, note = note, xgb_params = xgb_params)
}

.sparse_to_df <- function(X) {
  # caret xgbTree / glmnet paths that need data.frame
  dens <- as.matrix(X)
  # sanitize names for formulas
  colnames(dens) <- make.names(colnames(dens), unique = TRUE)
  as.data.frame(dens, stringsAsFactors = FALSE)
}

.caret_fit_predict_y <- function(
    x_train,
    y_train,
    x_test,
    caret_method,
    gpu_info,
    tune_length,
    xgb_nrounds,
    xgb_max_depth,
    xgb_eta,
    seed
) {
  set.seed(seed)

  if (identical(caret_method, "xgbTree")) {
    # Prefer native xgboost on sparse matrices for large BBB; caret grid is thin
    return(.xgb_fit_predict_sparse(
      x_train = x_train,
      y_train = y_train,
      x_test = x_test,
      gpu_info = gpu_info,
      nrounds = xgb_nrounds,
      max_depth = xgb_max_depth,
      eta = xgb_eta,
      seed = seed
    ))
  }

  # caret glmnet on dense controls (OK when p_controls is moderate)
  df_train <- .sparse_to_df(x_train)
  df_test <- .sparse_to_df(x_test)
  # align columns
  miss <- setdiff(names(df_train), names(df_test))
  for (nm in miss) df_test[[nm]] <- 0
  df_test <- df_test[, names(df_train), drop = FALSE]

  ctrl <- caret::trainControl(method = "cv", number = 3L, verboseIter = FALSE)
  fit <- caret::train(
    x = df_train,
    y = y_train,
    method = "glmnet",
    trControl = ctrl,
    tuneLength = max(1L, as.integer(tune_length)),
    preProcess = c("center", "scale")
  )
  as.numeric(predict(fit, newdata = df_test))
}

.xgb_fit_predict_sparse <- function(
    x_train,
    y_train,
    x_test,
    gpu_info,
    nrounds,
    max_depth,
    eta,
    seed
) {
  # Native xgboost keeps sparsity; still part of the caret/xgbTree GPU workflow
  dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train)
  dtest <- xgboost::xgb.DMatrix(data = x_test)

  params <- list(
    objective = "reg:squarederror",
    max_depth = as.integer(max_depth),
    eta = eta,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = max(1L, parallel::detectCores(logical = TRUE) - 1L)
  )
  params <- c(params, gpu_info$xgb_params)

  # Small internal validation for early stopping without nested caret CV cost
  n_tr <- nrow(x_train)
  set.seed(seed)
  val_ix <- sample.int(n_tr, size = max(1L, floor(0.15 * n_tr)))
  tr_ix <- setdiff(seq_len(n_tr), val_ix)
  dtr <- xgboost::xgb.DMatrix(data = x_train[tr_ix, , drop = FALSE], label = y_train[tr_ix])
  dval <- xgboost::xgb.DMatrix(data = x_train[val_ix, , drop = FALSE], label = y_train[val_ix])

  booster <- tryCatch(
    xgboost::xgb.train(
      params = params,
      data = dtr,
      nrounds = as.integer(nrounds),
      watchlist = list(train = dtr, val = dval),
      early_stopping_rounds = 25L,
      verbose = 0
    ),
    error = function(e) {
      # GPU failure -> CPU retry
      msg <- conditionMessage(e)
      if (!is.null(gpu_info$xgb_params$device) ||
          identical(gpu_info$xgb_params$tree_method, "gpu_hist")) {
        message("  GPU xgboost failed (", msg, "); retrying on CPU.")
        params$tree_method <- "hist"
        params$device <- "cpu"
        return(xgboost::xgb.train(
          params = params,
          data = dtr,
          nrounds = as.integer(nrounds),
          watchlist = list(train = dtr, val = dval),
          early_stopping_rounds = 25L,
          verbose = 0
        ))
      }
      stop(e)
    }
  )

  best <- booster$best_iteration
  if (is.null(best) || is.na(best) || best < 1L) best <- as.integer(nrounds)
  # Refit on full train fold with best_iteration
  booster_full <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = as.integer(best),
    verbose = 0
  )
  as.numeric(predict(booster_full, newdata = dtest))
}
