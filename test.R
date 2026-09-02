#!/usr/bin/env Rscript
#' Double / debiased ML on ball-by-ball data with the real raw schema.
#'
#' Partially linear model:
#'   Y = θ D + g(X) + ε
#' Y = bat_score, D = batter_is_home (default; override with --treatment=),
#' X = mixed-type controls from the raw BBB columns.
#'
#' Cross-fitted GPU/CPU torch nets estimate E[Y|X] and E[D|X]; residual OLS
#' yields θ.
#'
#' Usage:
#'   Rscript test.R
#'   Rscript test.R --data=data/raw/real_bbb.csv --folds=5 --epochs=20
#'   Rscript test.R --n=2000 --treatment=power_play --require-gpu=true

suppressPackageStartupMessages({
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Install torch: install.packages(\"torch\"); torch::install_torch()", call. = FALSE)
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table: install.packages(\"data.table\")", call. = FALSE)
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
true_theta <- as.numeric(parse_flag("--theta", "0.35"))
data_path <- parse_flag("--data", "")
treatment_col <- parse_flag("--treatment", "batter_is_home")
require_gpu <- tolower(parse_flag("--require-gpu", "false")) %in% c("1", "true", "t", "yes")

set.seed(seed)
torch_manual_seed(seed)

# =============================================================================
# Real raw BBB schema (snake_case)
# =============================================================================

RAW_BBB_COLUMNS <- c(
  "match_id",
  "series",
  "venue",
  "season",
  "team_a_at_home",
  "team_a_id",
  "team_batting_id",
  "team_bowling_id",
  "home_team",
  "batter_is_home",
  "innings",
  "over",
  "ball_in_over",
  "legal_ball",
  "striker_id",
  "striker_name",
  "non_striker_id",
  "non_striker_name",
  "striker_hand_id",
  "striker_batting_position",
  "bowler_id",
  "bowler_name",
  "bowler_hand_id",
  "power_play",
  "bat_score",
  "cumulative_inning_extra_runs",
  "cumulative_inning_bat_score",
  "cumulative_inning_wickets",
  "free_hit",
  "striker_dismissed",
  "non_striker_dismissed",
  "how_out_id",
  "batter_dismissal"
)

# Type map for controls used in X (treatment & outcome handled separately)
CONTROL_VAR_TYPES <- data.frame(
  variable = c(
    "series", "venue", "season",
    "team_a_id", "team_batting_id", "team_bowling_id", "home_team",
    "team_a_at_home", "power_play", "free_hit", "legal_ball",
    "innings", "over", "ball_in_over",
    "striker_batting_position",
    "striker_hand_id", "bowler_hand_id",
    "bowler_id",
    "cumulative_inning_extra_runs", "cumulative_inning_bat_score",
    "cumulative_inning_wickets"
  ),
  type = c(
    "categorical", "categorical", "categorical",
    "categorical", "categorical", "categorical", "categorical",
    "binary", "binary", "binary", "binary",
    "ordinal", "numeric", "ordinal",
    "ordinal",
    "categorical", "categorical",
    "categorical",
    "numeric", "numeric",
    "ordinal"
  ),
  stringsAsFactors = FALSE
)

# Columns that must not enter X when estimating effects on bat_score
# (labels, ids used only for clustering, or same-ball outcomes)
EXCLUDE_FROM_X <- c(
  "match_id",
  "striker_id", "striker_name",
  "non_striker_id", "non_striker_name",
  "bowler_name",
  "bat_score",
  "striker_dismissed", "non_striker_dismissed",
  "how_out_id", "batter_dismissal"
)

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
# Simulate / load data in the real schema
# =============================================================================

#' Synthetic BBB rows matching RAW_BBB_COLUMNS, with known θ for batter_is_home.
simulate_real_schema_bbb <- function(n = 2000L, theta = 0.35) {
  n_matches <- max(20L, as.integer(n / 120))
  match_id <- sprintf("M%04d", sample.int(n_matches, n, replace = TRUE))
  series <- sample(c("Premier Smash", "Coastal T20", "Capital League"), n, TRUE,
                   prob = c(0.45, 0.35, 0.20))
  venue <- sample(sprintf("Venue_%02d", 1:8), n, TRUE)
  season <- sample(c("2023/24", "2024/25", "2025/26"), n, TRUE, prob = c(0.3, 0.4, 0.3))

  team_a_id <- sample(sprintf("Team_%02d", 1:12), n, TRUE)
  team_batting_id <- team_a_id
  # flip batting team half the time
  flip <- rbinom(n, 1L, 0.5) == 1L
  team_bowling_id <- sample(sprintf("Team_%02d", 1:12), n, TRUE)
  team_bowling_id[flip] <- team_a_id[flip]
  team_batting_id[flip] <- sample(sprintf("Team_%02d", 1:12), sum(flip), TRUE)

  team_a_at_home <- rbinom(n, 1L, 0.5)
  home_team <- ifelse(team_a_at_home == 1L, team_a_id, team_bowling_id)
  batter_is_home <- as.integer(team_batting_id == home_team)

  innings <- sample(1:2, n, TRUE)
  over <- sample(0:19, n, TRUE)
  ball_in_over <- sample(1:6, n, TRUE)
  legal_ball <- rbinom(n, 1L, 0.96)

  striker_id <- sprintf("Batter_%03d", sample.int(48L, n, TRUE))
  non_striker_id <- sprintf("Batter_%03d", sample.int(48L, n, TRUE))
  striker_name <- paste0("Player_", striker_id)
  non_striker_name <- paste0("Player_", non_striker_id)
  striker_hand_id <- sample(c("L", "R"), n, TRUE, prob = c(0.28, 0.72))
  striker_batting_position <- sample(1:8, n, TRUE, prob = c(0.16, 0.15, 0.14, 0.13, 0.12, 0.12, 0.10, 0.08))

  bowler_id <- sprintf("Bowler_%03d", sample.int(36L, n, TRUE))
  bowler_name <- paste0("Player_", bowler_id)
  bowler_hand_id <- sample(c("L", "R"), n, TRUE, prob = c(0.30, 0.70))

  power_play <- as.integer(over < 6L)
  free_hit <- rbinom(n, 1L, 0.02)

  # Confounding structure for known θ on batter_is_home
  g <- 0.15 * power_play +
    0.04 * over +
    0.08 * (striker_batting_position <= 3) +
    0.10 * (bowler_hand_id == "L") +
    0.05 * (striker_hand_id == "L") -
    0.03 * innings +
    0.12 * free_hit

  # Correlate home slightly with context so naive OLS is biased
  e <- plogis(-0.4 + 0.5 * g + 0.2 * power_play)
  # Keep simulated home indicator but tilt it via latent draw for DGP of Y
  # (batter_is_home stays as assigned above; θ multiplies that column)
  bat_score_mean <- pmax(0.05, 0.9 + theta * batter_is_home + g)
  bat_score <- vapply(bat_score_mean, function(mu) {
    probs <- pmax(0.001, c(0.38, 0.32, 0.12, 0.03, 0.10, 0.05) +
      c(-0.08, -0.02, 0.02, 0.01, 0.04, 0.03) * (mu - 0.9))
    sample(c(0L, 1L, 2L, 3L, 4L, 6L), 1L, prob = probs / sum(probs))
  }, integer(1))

  striker_dismissed <- as.integer(rbinom(n, 1L, plogis(-3.0 + 0.15 * over)))
  non_striker_dismissed <- integer(n)
  how_out_id <- ifelse(striker_dismissed == 1L,
                       sample(c("caught", "bowled", "lbw", "run_out"), n, TRUE,
                              prob = c(0.55, 0.25, 0.12, 0.08)),
                       NA_character_)
  batter_dismissal <- how_out_id

  cumulative_inning_bat_score <- as.integer(
    ave(bat_score, match_id, innings, FUN = cumsum)
  )
  cumulative_inning_extra_runs <- as.integer(
    ave(rbinom(n, 1L, 0.04), match_id, innings, FUN = cumsum)
  )
  cumulative_inning_wickets <- as.integer(
    ave(striker_dismissed, match_id, innings, FUN = cumsum)
  )

  data.table::data.table(
    match_id = match_id,
    series = series,
    venue = venue,
    season = season,
    team_a_at_home = as.integer(team_a_at_home),
    team_a_id = team_a_id,
    team_batting_id = team_batting_id,
    team_bowling_id = team_bowling_id,
    home_team = home_team,
    batter_is_home = as.integer(batter_is_home),
    innings = as.integer(innings),
    over = as.integer(over),
    ball_in_over = as.integer(ball_in_over),
    legal_ball = as.integer(legal_ball),
    striker_id = striker_id,
    striker_name = striker_name,
    non_striker_id = non_striker_id,
    non_striker_name = non_striker_name,
    striker_hand_id = striker_hand_id,
    striker_batting_position = as.integer(striker_batting_position),
    bowler_id = bowler_id,
    bowler_name = bowler_name,
    bowler_hand_id = bowler_hand_id,
    power_play = as.integer(power_play),
    bat_score = as.integer(bat_score),
    cumulative_inning_extra_runs = cumulative_inning_extra_runs,
    cumulative_inning_bat_score = cumulative_inning_bat_score,
    cumulative_inning_wickets = cumulative_inning_wickets,
    free_hit = as.integer(free_hit),
    striker_dismissed = striker_dismissed,
    non_striker_dismissed = non_striker_dismissed,
    how_out_id = how_out_id,
    batter_dismissal = batter_dismissal
  )
}

validate_raw_schema <- function(dt) {
  miss <- setdiff(RAW_BBB_COLUMNS, names(dt))
  if (length(miss)) {
    stop(
      "Raw BBB missing required columns:\n  ",
      paste(miss, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

load_real_schema_bbb <- function(path) {
  if (!file.exists(path)) {
    stop("Data file not found: ", path, call. = FALSE)
  }
  dt <- data.table::fread(path)
  # tolerate optional extras; require the declared schema
  validate_raw_schema(dt)
  dt
}

# =============================================================================
# Build DML matrices from real-schema BBB
# =============================================================================

coerce_control_types <- function(dt, var_types) {
  out <- data.table::as.data.table(dt)
  for (i in seq_len(nrow(var_types))) {
    nm <- var_types$variable[[i]]
    if (!nm %in% names(out)) next
    typ <- var_types$type[[i]]
    if (typ == "binary") {
      out[[nm]] <- as.integer(out[[nm]])
      out[[nm]] <- factor(out[[nm]], levels = sort(unique(out[[nm]])))
    } else if (typ == "categorical") {
      out[[nm]] <- factor(as.character(out[[nm]]))
    } else if (typ == "ordinal") {
      # ordered factor on sorted unique numeric/character levels
      vals <- out[[nm]]
      if (is.numeric(vals) || is.integer(vals)) {
        lev <- sort(unique(as.numeric(vals)))
        out[[nm]] <- factor(as.numeric(vals), levels = lev, ordered = TRUE)
      } else {
        lev <- sort(unique(as.character(vals)))
        out[[nm]] <- factor(as.character(vals), levels = lev, ordered = TRUE)
      }
    } else {
      out[[nm]] <- as.numeric(out[[nm]])
    }
  }
  out
}

#' Prepare Y, D, X from a real-schema BBB table.
#'
#' @param dt data.table/data.frame with RAW_BBB_COLUMNS
#' @param treatment_col binary treatment column name
#' @param true_theta known θ when data were simulated; NULL for real data
prepare_dml_from_bbb <- function(dt, treatment_col = "batter_is_home", true_theta = NULL) {
  dt <- data.table::as.data.table(dt)
  validate_raw_schema(dt)

  if (!treatment_col %in% names(dt)) {
    stop("Treatment column not found: ", treatment_col, call. = FALSE)
  }

  # Keep legal deliveries with observed score / treatment
  dt <- dt[!is.na(bat_score) & !is.na(get(treatment_col))]
  if ("legal_ball" %in% names(dt)) {
    dt <- dt[legal_ball == 1 | is.na(legal_ball)]
  }

  Y <- as.numeric(dt$bat_score)
  D <- as.numeric(dt[[treatment_col]])
  # coerce logical / factor treatment to 0/1
  if (is.logical(dt[[treatment_col]])) {
    D <- as.numeric(dt[[treatment_col]])
  } else if (is.factor(dt[[treatment_col]])) {
    D <- as.numeric(dt[[treatment_col]]) - 1
  }
  if (!all(D %in% c(0, 1))) {
    # allow 0/1 numeric already; otherwise binarize by != 0
    D <- as.numeric(D != 0)
  }

  # Controls: typed columns except treatment, exclusions
  ctrl_vars <- CONTROL_VAR_TYPES$variable
  ctrl_vars <- setdiff(ctrl_vars, c(treatment_col, EXCLUDE_FROM_X))
  ctrl_vars <- intersect(ctrl_vars, names(dt))
  var_types <- CONTROL_VAR_TYPES[CONTROL_VAR_TYPES$variable %in% ctrl_vars, , drop = FALSE]

  raw_x <- coerce_control_types(dt[, ..ctrl_vars], var_types)
  # convert to data.frame for encode_mixed_features
  raw_x <- as.data.frame(raw_x)

  enc <- encode_mixed_features(raw_x)

  list(
    raw = dt,
    X = enc$X,
    X_scaled = enc$X_scaled,
    feature_info = enc$feature_info,
    var_types = var_types,
    D = as.numeric(D),
    Y = as.numeric(Y),
    treatment_col = treatment_col,
    outcome_col = "bat_score",
    true_theta = true_theta,
    n = nrow(dt),
    match_id = as.character(dt$match_id)
  )
}

# =============================================================================
# Mixed-type encoder
# =============================================================================

#' Encode mixed-type data.frame -> numeric design matrix.
encode_mixed_features <- function(df) {
  stopifnot(is.data.frame(df))
  pieces <- list()
  info <- list()

  for (nm in names(df)) {
    v <- df[[nm]]
    if (is.ordered(v)) {
      code <- as.numeric(v)
      mat <- matrix(code, ncol = 1L, dimnames = list(NULL, nm))
      pieces[[nm]] <- mat
      info[[nm]] <- list(type = "ordinal", levels = levels(v), cols = nm)
    } else if (is.factor(v) || is.character(v)) {
      v <- factor(v)
      mm <- stats::model.matrix(~ 0 + v)
      colnames(mm) <- paste0(nm, "=", gsub("^v", "", colnames(mm)))
      ref <- levels(v)[[1]]
      drop_col <- paste0(nm, "=", ref)
      keep <- setdiff(colnames(mm), drop_col)
      if (!length(keep)) {
        # single-level factor: skip
        next
      }
      mm <- mm[, keep, drop = FALSE]
      # Cap ultra-high-cardinality categoricals (e.g. bowler_id) to top levels
      if (ncol(mm) > 40L) {
        freqs <- colSums(mm)
        keep_top <- names(sort(freqs, decreasing = TRUE))[seq_len(40L)]
        mm <- mm[, keep_top, drop = FALSE]
      }
      pieces[[nm]] <- mm
      info[[nm]] <- list(type = "categorical", levels = levels(v), reference = ref, cols = colnames(mm))
    } else {
      mat <- matrix(as.numeric(v), ncol = 1L, dimnames = list(NULL, nm))
      pieces[[nm]] <- mat
      info[[nm]] <- list(type = "numeric", cols = nm)
    }
  }

  X <- do.call(cbind, pieces)
  mu <- colMeans(X)
  sds <- apply(X, 2L, stats::sd)
  sds[!is.finite(sds) | sds < 1e-8] <- 1
  X_scaled <- sweep(sweep(X, 2L, mu, "-"), 2L, sds, "/")
  colnames(X_scaled) <- colnames(X)

  list(X = X, X_scaled = X_scaled, feature_info = info, center = mu, scale = sds)
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
    x_train, y_train,
    epochs = 20L, batch_size = 128L, lr = 1e-3,
    device, verbose = FALSE
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
# Cross-fitted Double ML
# =============================================================================

estimate_dml_ate <- function(
    X, D, Y,
    n_folds = 5L, epochs = 20L, batch_size = 128L, lr = 1e-3,
    device, seed = 123L, cluster = NULL, verbose = TRUE
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
      X[train, , drop = FALSE], Y[train],
      epochs = epochs, batch_size = batch_size, lr = lr,
      device = device, verbose = verbose
    )
    y_hat[test] <- predict_regressor(m_y, X[test, , drop = FALSE], device)

    if (verbose) cat("Fitting E[D | X] ...\n")
    m_d <- train_regressor(
      X[train, , drop = FALSE], D[train],
      epochs = epochs, batch_size = batch_size, lr = lr,
      device = device, verbose = verbose
    )
    d_hat[test] <- predict_regressor(m_d, X[test, , drop = FALSE], device)
  }

  y_resid <- Y - y_hat
  d_resid <- D - d_hat
  denom <- sum(d_resid^2)
  if (denom < 1e-12) stop("Degenerate D residuals; check treatment variation.", call. = FALSE)
  theta_hat <- sum(d_resid * y_resid) / denom
  u <- y_resid - theta_hat * d_resid

  # Match-clustered sandwich SE when cluster provided
  if (is.null(cluster)) {
    meat <- mean((d_resid^2) * (u^2))
    bread <- mean(d_resid^2)
    se <- sqrt(meat / (n * bread^2))
    n_clusters <- n
  } else {
    cluster <- as.character(cluster)
    score_sum <- 0
    bread <- sum(d_resid^2) / n
    for (cl in unique(cluster)) {
      ix <- which(cluster == cl)
      sc <- sum(d_resid[ix] * u[ix])
      score_sum <- score_sum + sc^2
    }
    n_clusters <- length(unique(cluster))
    meat <- score_sum / n_clusters
    se <- sqrt(meat / (n_clusters * bread^2))
  }
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
      n_clusters = n_clusters,
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
cat("Double ML on real BBB schema (mixed types, GPU nuisances)\n")
cat("============================================================\n")
cat("Schema columns:\n")
cat(paste(" -", RAW_BBB_COLUMNS), sep = "\n")
cat("\n")

if (nzchar(data_path)) {
  cat("Loading real data:", data_path, "\n")
  bbb <- load_real_schema_bbb(data_path)
  dat <- prepare_dml_from_bbb(bbb, treatment_col = treatment_col, true_theta = NULL)
} else {
  cat(sprintf("No --data given; simulating n=%d rows in real schema (θ=%.3f)\n",
              n_obs, true_theta))
  bbb <- simulate_real_schema_bbb(n = n_obs, theta = true_theta)
  dat <- prepare_dml_from_bbb(bbb, treatment_col = treatment_col, true_theta = true_theta)
}

cat(sprintf("\nOutcome Y: %s | Treatment D: %s\n", dat$outcome_col, dat$treatment_col))
cat("\nControl variable types in X:\n")
print(dat$var_types, row.names = FALSE)

cat(sprintf(
  "\nEncoded design: n=%d | p=%d columns after expanding categoricals/ordinals\n",
  dat$n, ncol(dat$X_scaled)
))
cat("Feature columns (first 30):\n")
show_cols <- colnames(dat$X_scaled)
if (length(show_cols) > 30L) show_cols <- c(utils::head(show_cols, 30L), "...")
cat(paste(" -", show_cols), sep = "\n")
cat(sprintf("\nmean(D)=%.3f | mean(Y)=%.3f\n", mean(dat$D), mean(dat$Y)))
if (!is.null(dat$true_theta)) {
  cat(sprintf("True θ (simulation only): %.4f\n", dat$true_theta))
}

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
  cluster = dat$match_id,
  verbose = TRUE
)

cat("\n============================================================\n")
cat("DML results\n")
cat("============================================================\n")
cat(sprintf("Device:              %s\n", fit$diagnostics$device))
cat(sprintf("Folds:               %d\n", fit$diagnostics$n_folds))
cat(sprintf("Clusters (matches):  %d\n", fit$diagnostics$n_clusters))
cat(sprintf("Treatment:           %s\n", dat$treatment_col))
cat(sprintf("Outcome:             %s\n", dat$outcome_col))
if (!is.null(dat$true_theta)) {
  cat(sprintf("True θ:              %.4f\n", dat$true_theta))
}
cat(sprintf("DML θ̂:              %.4f\n", fit$theta))
cat(sprintf("Cluster-robust SE:   %.4f\n", fit$se))
cat(sprintf("95%% CI:              [%.4f, %.4f]\n", fit$ci_lo, fit$ci_hi))
if (!is.null(dat$true_theta)) {
  cat(sprintf("Error θ̂ − θ:         %.4f\n", fit$theta - dat$true_theta))
}
cat(sprintf("Mean |Y residual|:   %.4f\n", fit$diagnostics$mean_abs_y_resid))
cat(sprintf("Mean |D residual|:   %.4f\n", fit$diagnostics$mean_abs_d_resid))
cat(sprintf("Corr(D, D̂):          %.4f\n", fit$diagnostics$corr_d_dhat))

naive <- stats::lm(dat$Y ~ dat$D)
naive_theta <- unname(stats::coef(naive)[[2]])
cat(sprintf("\nNaive OLS θ (confounded): %.4f\n", naive_theta))
if (!is.null(dat$true_theta)) {
  cat(sprintf("Naive |bias|:               %.4f\n", abs(naive_theta - dat$true_theta)))
  cat(sprintf("DML |bias|:                 %.4f\n", abs(fit$theta - dat$true_theta)))
}

invisible(list(data = dat, fit = fit, naive_theta = naive_theta, schema = RAW_BBB_COLUMNS))
