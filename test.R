#!/usr/bin/env Rscript
#' Double / debiased ML demo with mixed-type covariates and GPU nuisances.
#'
#' Partially linear model:
#'   Y = θ D + g(X) + ε
#' with cross-fitted neural estimates of E[Y|X] and E[D|X] on GPU (torch),
#' then OLS on residuals for θ.
#'
#' Covariates intentionally mix:
#'   - continuous numeric
#'   - binary
#'   - unordered categorical
#'   - ordered / ordinal
#'
#' Usage:
#'   Rscript test.R
#'   Rscript test.R --folds=5 --epochs=20 --n=2000 --require-gpu=true

suppressPackageStartupMessages({
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Install torch: install.packages(\"torch\"); torch::install_torch()", call. = FALSE)
  }
  library(torch)
})

args <- commandArgs(trailingOnly = TRUE)
parse_flag <- function(flag, default) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", flag, "="), "", hit[[1]])
}

n_folds <- as.integer(parse_flag("--folds", "5"))
epochs <- as.integer(parse_flag("--epochs", "20"))
batch_size <- as.integer(parse_flag("--batch-size", "128"))
lr <- as.numeric(parse_flag("--lr", "0.001"))
n_obs <- as.integer(parse_flag("--n", "2000"))
seed <- as.integer(parse_flag("--seed", "123"))
true_theta <- as.numeric(parse_flag("--theta", "1.5"))
require_gpu <- tolower(parse_flag("--require-gpu", "false")) %in% c("1", "true", "t", "yes")

set.seed(seed)
torch_manual_seed(seed)

# =============================================================================
# Device
# =============================================================================

resolve_device <- function(require_gpu = FALSE) {
  if (isTRUE(cuda_is_available())) {
    message(sprintf("CUDA available (%d device(s)). Using GPU 0.", cuda_device_count()))
    tryCatch({
      tmp <- torch_tensor(1, device = "cuda")
      rm(tmp)
      cuda_synchronize()
    }, error = function(e) {
      stop("CUDA alloc failed: ", conditionMessage(e), call. = FALSE)
    })
    return(torch_device("cuda"))
  }
  if (require_gpu) {
    stop("CUDA not available and --require-gpu=true was set.", call. = FALSE)
  }
  message("CUDA not available; training nuisances on CPU.")
  torch_device("cpu")
}

device <- resolve_device(require_gpu = require_gpu)

# =============================================================================
# Mixed-type data-generating process
# =============================================================================

#' Simulate a cricket-flavoured DML toy world with heterogeneous X.
#'
#' Returns a data.frame of raw mixed types plus encoded matrices and truth.
make_mixed_dml_data <- function(n = 2000L, theta = 1.5) {
  # --- Continuous numeric ---
  opp_strength <- rnorm(n, 0, 1)          # standardised opposition strength
  over_frac <- runif(n, 0, 1)             # progress through innings
  ball_speed <- rnorm(n, 135, 8)          # km/h-ish

  # --- Binary ---
  is_home <- rbinom(n, 1L, 0.55)
  day_night <- rbinom(n, 1L, 0.65)

  # --- Unordered categorical ---
  venue <- sample(c("Coastal", "Capital", "Highland", "Desert"), n, replace = TRUE,
                  prob = c(0.35, 0.30, 0.20, 0.15))
  phase <- sample(c("powerplay", "middle", "death"), n, replace = TRUE,
                  prob = c(0.30, 0.45, 0.25))
  bowler_hand <- sample(c("left", "right"), n, replace = TRUE, prob = c(0.28, 0.72))

  # --- Ordinal ---
  # batting position 1 (opener) ... 7 (lower); treat as ordered
  batting_position <- sample(1:7, n, replace = TRUE, prob = c(0.18, 0.16, 0.16, 0.14, 0.14, 0.12, 0.10))
  # pitch quality rating 1 (poor) ... 5 (excellent)
  pitch_rating <- sample(1:5, n, replace = TRUE, prob = c(0.10, 0.20, 0.35, 0.25, 0.10))

  raw <- data.frame(
    opp_strength = opp_strength,
    over_frac = over_frac,
    ball_speed = ball_speed,
    is_home = factor(is_home, levels = c(0, 1), labels = c("away", "home")),
    day_night = factor(day_night, levels = c(0, 1), labels = c("day", "night")),
    venue = factor(venue, levels = c("Coastal", "Capital", "Highland", "Desert")),
    phase = factor(phase, levels = c("powerplay", "middle", "death")),
    bowler_hand = factor(bowler_hand, levels = c("left", "right")),
    batting_position = factor(batting_position, levels = 1:7, ordered = TRUE),
    pitch_rating = factor(pitch_rating, levels = 1:5, ordered = TRUE),
    stringsAsFactors = FALSE
  )

  # Variable-type dictionary for documentation / checks
  var_types <- data.frame(
    variable = c(
      "opp_strength", "over_frac", "ball_speed",
      "is_home", "day_night",
      "venue", "phase", "bowler_hand",
      "batting_position", "pitch_rating"
    ),
    type = c(
      "numeric", "numeric", "numeric",
      "binary", "binary",
      "categorical", "categorical", "categorical",
      "ordinal", "ordinal"
    ),
    stringsAsFactors = FALSE
  )

  enc <- encode_mixed_features(raw)

  # Nonlinear confounding g(X) used in both propensity and outcome
  g <- 0.35 * enc$X_scaled[, "opp_strength"] +
    0.25 * sin(2 * pi * enc$X_scaled[, "over_frac"]) -
    0.15 * enc$X_scaled[, "ball_speed"] +
    0.20 * (raw$is_home == "home") +
    0.10 * (raw$phase == "death") -
    0.12 * (raw$venue == "Desert") +
    0.08 * as.numeric(raw$batting_position) / 7 +
    0.05 * as.numeric(raw$pitch_rating) / 5 +
    0.15 * enc$X_scaled[, "opp_strength"] * (raw$phase == "death")

  # Treatment propensity depends on X (selection into "aggressive intent" / matchup)
  e <- plogis(-0.2 + 0.6 * g + 0.3 * (raw$bowler_hand == "left"))
  D <- rbinom(n, 1L, e)

  # Outcome: partially linear — θ is the causal effect of D
  Y <- theta * D + g + rnorm(n, 0, 0.75)

  list(
    raw = raw,
    X = enc$X,                 # model matrix (numeric columns, dummies, ordinal codes)
    X_scaled = enc$X_scaled,
    feature_info = enc$feature_info,
    var_types = var_types,
    D = as.numeric(D),
    Y = as.numeric(Y),
    true_theta = theta,
    true_propensity = e,
    n = n
  )
}

#' Encode mixed-type data.frame -> numeric design matrix.
#'
#' - numeric: left as-is then column-scaled for the net
#' - binary / categorical: treatment contrasts (drop first level)
#' - ordinal: integer codes 1..L (monotonic numeric embedding)
encode_mixed_features <- function(df) {
  stopifnot(is.data.frame(df))
  pieces <- list()
  info <- list()

  for (nm in names(df)) {
    v <- df[[nm]]
    if (is.ordered(v)) {
      # Ordinal: integer score in [1, L]
      code <- as.numeric(v)
      mat <- matrix(code, ncol = 1L, dimnames = list(NULL, nm))
      pieces[[nm]] <- mat
      info[[nm]] <- list(type = "ordinal", levels = levels(v), cols = nm)
    } else if (is.factor(v) || is.character(v)) {
      v <- factor(v)
      # Unordered: full-rank dummy encoding without intercept column per factor
      mm <- stats::model.matrix(~ 0 + v)
      colnames(mm) <- paste0(nm, "=", gsub("^v", "", colnames(mm)))
      # Drop first level for identification (reference)
      ref <- levels(v)[[1]]
      drop_col <- paste0(nm, "=", ref)
      keep <- setdiff(colnames(mm), drop_col)
      mm <- mm[, keep, drop = FALSE]
      pieces[[nm]] <- mm
      info[[nm]] <- list(type = "categorical", levels = levels(v), reference = ref, cols = colnames(mm))
    } else {
      mat <- matrix(as.numeric(v), ncol = 1L, dimnames = list(NULL, nm))
      pieces[[nm]] <- mat
      info[[nm]] <- list(type = "numeric", cols = nm)
    }
  }

  X <- do.call(cbind, pieces)
  # Column-scale for neural nets (preserve column names)
  mu <- colMeans(X)
  sds <- apply(X, 2L, stats::sd)
  sds[!is.finite(sds) | sds < 1e-8] <- 1
  X_scaled <- sweep(sweep(X, 2L, mu, "-"), 2L, sds, "/")
  colnames(X_scaled) <- colnames(X)

  list(
    X = X,
    X_scaled = X_scaled,
    feature_info = info,
    center = mu,
    scale = sds
  )
}

# =============================================================================
# GPU regressor for nuisance functions E[· | X]
# =============================================================================

build_regressor <- function(input_dim, hidden_1 = 64L, hidden_2 = 32L) {
  nn_sequential(
    nn_linear(input_dim, hidden_1),
    nn_relu(),
    nn_dropout(0.15),
    nn_linear(hidden_1, hidden_2),
    nn_relu(),
    nn_linear(hidden_2, 1L)
  )
}

batch_iterator <- function(x, y, batch_size, device, shuffle = TRUE) {
  n <- nrow(x)
  order_idx <- if (shuffle) sample.int(n) else seq_len(n)
  starts <- seq(1L, n, by = batch_size)
  i <- 0L
  force(x); force(y); force(device)
  function() {
    i <<- i + 1L
    if (i > length(starts)) return(NULL)
    from <- starts[[i]]
    to <- min(from + batch_size - 1L, n)
    idx <- order_idx[from:to]
    list(
      x = torch_tensor(x[idx, , drop = FALSE], dtype = torch_float(), device = device),
      y = torch_tensor(matrix(y[idx], ncol = 1L), dtype = torch_float(), device = device)
    )
  }
}

train_regressor <- function(
    x_train,
    y_train,
    epochs = 20L,
    batch_size = 128L,
    lr = 1e-3,
    device,
    verbose = FALSE
) {
  model <- build_regressor(ncol(x_train))
  model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = lr)
  loss_fn <- nn_mse_loss()

  for (epoch in seq_len(epochs)) {
    model$train()
    next_batch <- batch_iterator(x_train, y_train, batch_size, device, shuffle = TRUE)
    epoch_loss <- 0
    n_batches <- 0L
    repeat {
      batch <- next_batch()
      if (is.null(batch)) break
      optimizer$zero_grad()
      pred <- model(batch$x)
      loss <- loss_fn(pred, batch$y)
      loss$backward()
      optimizer$step()
      epoch_loss <- epoch_loss + loss$item()
      n_batches <- n_batches + 1L
    }
    if (verbose) {
      cat(sprintf("  epoch %02d | mse=%.4f\n", epoch, epoch_loss / max(1L, n_batches)))
    }
  }
  if (device$type == "cuda") cuda_synchronize()
  model
}

predict_regressor <- function(model, x_new, device) {
  model$eval()
  with_no_grad({
    xt <- torch_tensor(x_new, dtype = torch_float(), device = device)
    as.numeric(as.array(model(xt)$to(device = "cpu")))
  })
}

# =============================================================================
# Cross-fitted Double ML (partially linear ATE)
# =============================================================================

#' Cross-fitted DML for binary treatment with neural nuisances on `device`.
estimate_dml_ate <- function(
    X,
    D,
    Y,
    n_folds = 5L,
    epochs = 20L,
    batch_size = 128L,
    lr = 1e-3,
    device,
    seed = 123L,
    verbose = TRUE
) {
  n <- nrow(X)
  set.seed(seed)
  fold_id <- sample(rep(seq_len(n_folds), length.out = n))

  y_hat <- rep(NA_real_, n)
  d_hat <- rep(NA_real_, n)

  if (verbose) {
    message(sprintf(
      "DML cross-fitting: folds=%d | n=%d | p=%d | device=%s | epochs=%d",
      n_folds, n, ncol(X), device$type, epochs
    ))
  }

  for (k in seq_len(n_folds)) {
    test <- fold_id == k
    train <- !test
    if (verbose) {
      cat(sprintf("\n=== Fold %d/%d (train=%d, test=%d) ===\n",
                  k, n_folds, sum(train), sum(test)))
    }

    if (verbose) cat("Fitting E[Y | X] ...\n")
    m_y <- train_regressor(
      x_train = X[train, , drop = FALSE],
      y_train = Y[train],
      epochs = epochs,
      batch_size = batch_size,
      lr = lr,
      device = device,
      verbose = verbose
    )
    y_hat[test] <- predict_regressor(m_y, X[test, , drop = FALSE], device)

    if (verbose) cat("Fitting E[D | X] ...\n")
    m_d <- train_regressor(
      x_train = X[train, , drop = FALSE],
      y_train = D[train],
      epochs = epochs,
      batch_size = batch_size,
      lr = lr,
      device = device,
      verbose = verbose
    )
    d_hat[test] <- predict_regressor(m_d, X[test, , drop = FALSE], device)
  }

  # Residual-on-residual OLS (ATE)
  y_resid <- Y - y_hat
  d_resid <- D - d_hat
  theta_hat <- sum(d_resid * y_resid) / sum(d_resid^2)
  u <- y_resid - theta_hat * d_resid

  # HC1-style robust SE
  n_eff <- n
  meat <- mean((d_resid^2) * (u^2))
  bread <- mean(d_resid^2)
  se <- sqrt(meat / (n_eff * bread^2))
  ci <- theta_hat + c(-1.96, 1.96) * se

  list(
    theta = theta_hat,
    se = se,
    ci_lo = ci[[1]],
    ci_hi = ci[[2]],
    y_resid = y_resid,
    d_resid = d_resid,
    y_hat = y_hat,
    d_hat = d_hat,
    fold_id = fold_id,
    diagnostics = list(
      n = n,
      p = ncol(X),
      n_folds = n_folds,
      device = device$type,
      mean_abs_y_resid = mean(abs(y_resid)),
      mean_abs_d_resid = mean(abs(d_resid)),
      corr_d_dhat = stats::cor(D, d_hat)
    )
  )
}

# =============================================================================
# Main
# =============================================================================

cat("============================================================\n")
cat("Double ML with mixed-type covariates (GPU neural nuisances)\n")
cat("============================================================\n")

dat <- make_mixed_dml_data(n = n_obs, theta = true_theta)

cat("\nVariable types in X:\n")
print(dat$var_types, row.names = FALSE)

cat(sprintf(
  "\nEncoded design: n=%d | p=%d columns after expanding categoricals/ordinals\n",
  dat$n, ncol(dat$X_scaled)
))
cat("Feature columns:\n")
cat(paste(" -", colnames(dat$X_scaled)), sep = "\n")
cat(sprintf("\nTrue θ = %.4f | mean(D)=%.3f | mean(Y)=%.3f\n",
            dat$true_theta, mean(dat$D), mean(dat$Y)))

fit <- estimate_dml_ate(
  X = dat$X_scaled,
  D = dat$D,
  Y = dat$Y,
  n_folds = n_folds,
  epochs = epochs,
  batch_size = batch_size,
  lr = lr,
  device = device,
  seed = seed,
  verbose = TRUE
)

cat("\n============================================================\n")
cat("DML results\n")
cat("============================================================\n")
cat(sprintf("Device:              %s\n", fit$diagnostics$device))
cat(sprintf("Folds:               %d\n", fit$diagnostics$n_folds))
cat(sprintf("True θ:              %.4f\n", dat$true_theta))
cat(sprintf("DML θ̂:              %.4f\n", fit$theta))
cat(sprintf("Robust SE:           %.4f\n", fit$se))
cat(sprintf("95%% CI:              [%.4f, %.4f]\n", fit$ci_lo, fit$ci_hi))
cat(sprintf("Error θ̂ − θ:         %.4f\n", fit$theta - dat$true_theta))
cat(sprintf("Mean |Y residual|:   %.4f\n", fit$diagnostics$mean_abs_y_resid))
cat(sprintf("Mean |D residual|:   %.4f\n", fit$diagnostics$mean_abs_d_resid))
cat(sprintf("Corr(D, D̂):          %.4f\n", fit$diagnostics$corr_d_dhat))

# Naive OLS of Y on D only (confounded) for contrast
naive <- stats::lm(dat$Y ~ dat$D)
naive_theta <- unname(stats::coef(naive)[["dat$D"]])
cat(sprintf("\nNaive OLS θ (confounded): %.4f | |bias|=%.4f\n",
            naive_theta, abs(naive_theta - dat$true_theta)))
cat(sprintf("DML |bias|:                 %.4f\n", abs(fit$theta - dat$true_theta)))

invisible(list(data = dat, fit = fit, naive_theta = naive_theta))
