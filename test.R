#!/usr/bin/env Rscript
#' GPU neural-net training with k-fold cross-validation (torch).
#'
#' Usage:
#'   Rscript test.R
#'   Rscript test.R --folds=5 --epochs=30 --batch-size=128 --lr=0.001
#'
#' Requires: torch (with CUDA build for GPU). Falls back to CPU if CUDA is absent.

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
epochs <- as.integer(parse_flag("--epochs", "25"))
batch_size <- as.integer(parse_flag("--batch-size", "128"))
lr <- as.numeric(parse_flag("--lr", "0.001"))
n_obs <- as.integer(parse_flag("--n", "1500"))
seed <- as.integer(parse_flag("--seed", "123"))
require_gpu <- tolower(parse_flag("--require-gpu", "false")) %in% c("1", "true", "t", "yes")

set.seed(seed)
torch_manual_seed(seed)

# -----------------------------------------------------
# Device: prefer CUDA GPU
# -----------------------------------------------------

resolve_device <- function(require_gpu = FALSE) {
  cuda_ok <- isTRUE(cuda_is_available())
  if (cuda_ok) {
    n_gpu <- cuda_device_count()
    message(sprintf("CUDA available (%d device(s)). Using GPU 0.", n_gpu))
    # Warm-up allocation so failures surface early
    tryCatch(
      {
        tmp <- torch_tensor(1, device = "cuda")
        rm(tmp)
        cuda_synchronize()
      },
      error = function(e) {
        stop("CUDA reported available but tensor alloc failed: ", conditionMessage(e), call. = FALSE)
      }
    )
    return(torch_device("cuda"))
  }
  if (require_gpu) {
    stop("CUDA not available and --require-gpu=true was set.", call. = FALSE)
  }
  message("CUDA not available; training on CPU.")
  torch_device("cpu")
}

device <- resolve_device(require_gpu = require_gpu)

# -----------------------------------------------------
# Dataset (synthetic binary classification)
# Labels are 1/2 for torch R cross-entropy (1-based classes).
# -----------------------------------------------------

make_data <- function(n = 1200L, p = 25L, signal = 1.0) {
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  linear_score <- rowSums(x[, 1:8, drop = FALSE]) * signal + 0.5 * x[, 9]
  prob <- plogis(linear_score)
  y01 <- rbinom(n, size = 1L, prob = prob)
  list(
    x = scale(x),
    y = as.integer(y01 + 1L), # class ids in {1, 2}
    y01 = y01,
    n_features = p,
    n_classes = 2L
  )
}

# -----------------------------------------------------
# Mini-batch iterator (tensors created on device)
# -----------------------------------------------------

batch_iterator <- function(x, y, batch_size, device, shuffle = TRUE) {
  n <- nrow(x)
  order_idx <- if (shuffle) sample.int(n) else seq_len(n)
  starts <- seq(1L, n, by = batch_size)

  force(x)
  force(y)
  force(device)

  i <- 0L
  function() {
    i <<- i + 1L
    if (i > length(starts)) {
      return(NULL)
    }
    from <- starts[[i]]
    to <- min(from + batch_size - 1L, n)
    idx <- order_idx[from:to]
    list(
      x = torch_tensor(x[idx, , drop = FALSE], dtype = torch_float(), device = device),
      y = torch_tensor(y[idx], dtype = torch_long(), device = device)
    )
  }
}

# -----------------------------------------------------
# Model
# -----------------------------------------------------

build_model <- function(input_dim, n_classes = 2L, hidden_1 = 64L, hidden_2 = 32L) {
  nn_sequential(
    nn_linear(input_dim, hidden_1),
    nn_relu(),
    nn_dropout(0.2),
    nn_linear(hidden_1, hidden_2),
    nn_relu(),
    nn_dropout(0.1),
    nn_linear(hidden_2, n_classes)
  )
}

# -----------------------------------------------------
# Train / evaluate on a fixed device
# -----------------------------------------------------

train_model <- function(
    x_train,
    y_train,
    x_val = NULL,
    y_val = NULL,
    epochs = 25L,
    batch_size = 128L,
    lr = 1e-3,
    device,
    verbose = TRUE
) {
  model <- build_model(ncol(x_train), n_classes = length(unique(y_train)))
  model$to(device = device)

  optimizer <- optim_adam(model$parameters, lr = lr)
  loss_fn <- nn_cross_entropy_loss()

  history <- data.frame(
    epoch = integer(),
    train_loss = numeric(),
    val_loss = numeric(),
    val_accuracy = numeric()
  )

  for (epoch in seq_len(epochs)) {
    model$train()
    next_batch <- batch_iterator(x_train, y_train, batch_size, device, shuffle = TRUE)
    epoch_loss <- 0
    n_batches <- 0L

    repeat {
      batch <- next_batch()
      if (is.null(batch)) break

      optimizer$zero_grad()
      logits <- model(batch$x)
      loss <- loss_fn(logits, batch$y)
      loss$backward()
      optimizer$step()

      epoch_loss <- epoch_loss + loss$item()
      n_batches <- n_batches + 1L
    }

    train_loss <- epoch_loss / max(1L, n_batches)
    val_loss <- NA_real_
    val_acc <- NA_real_

    if (!is.null(x_val) && !is.null(y_val) && nrow(x_val) > 0) {
      metrics <- evaluate_model(model, x_val, y_val, device)
      val_loss <- metrics$loss
      val_acc <- metrics$accuracy
    }

    history <- rbind(
      history,
      data.frame(
        epoch = epoch,
        train_loss = train_loss,
        val_loss = val_loss,
        val_accuracy = val_acc
      )
    )

    if (verbose) {
      if (is.finite(val_acc)) {
        cat(sprintf(
          "Epoch %02d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f\n",
          epoch, train_loss, val_loss, val_acc
        ))
      } else {
        cat(sprintf("Epoch %02d | train_loss=%.4f\n", epoch, train_loss))
      }
    }
  }

  if (device$type == "cuda") {
    cuda_synchronize()
  }

  list(model = model, history = history)
}

evaluate_model <- function(model, x_eval, y_eval, device) {
  model$eval()
  with_no_grad({
    xt <- torch_tensor(x_eval, dtype = torch_float(), device = device)
    yt <- torch_tensor(y_eval, dtype = torch_long(), device = device)
    logits <- model(xt)
    loss <- nn_cross_entropy_loss()(logits, yt)$item()
    pred <- as.integer(as.array(logits$argmax(dim = 2L)$to(device = "cpu")))
    accuracy <- mean(pred == as.integer(y_eval))
    list(accuracy = accuracy, loss = loss, pred = pred)
  })
}

# -----------------------------------------------------
# K-fold cross-validation (each fold trains on GPU)
# -----------------------------------------------------

run_cv <- function(
    x,
    y,
    k = 5L,
    epochs = 25L,
    batch_size = 128L,
    lr = 1e-3,
    device
) {
  n <- nrow(x)
  folds <- sample(rep(seq_len(k), length.out = n))

  fold_accuracy <- numeric(k)
  fold_loss <- numeric(k)
  fold_models <- vector("list", k)

  message(sprintf(
    "\nStarting %d-fold CV on device=%s | n=%d | p=%d | epochs=%d | batch=%d",
    k, device$type, n, ncol(x), epochs, batch_size
  ))

  for (fold in seq_len(k)) {
    test_idx <- which(folds == fold)
    train_idx <- which(folds != fold)

    cat(sprintf("\n=== Fold %d/%d (train=%d, val=%d) ===\n",
                fold, k, length(train_idx), length(test_idx)))

    fit <- train_model(
      x_train = x[train_idx, , drop = FALSE],
      y_train = y[train_idx],
      x_val = x[test_idx, , drop = FALSE],
      y_val = y[test_idx],
      epochs = epochs,
      batch_size = batch_size,
      lr = lr,
      device = device,
      verbose = TRUE
    )

    metrics <- evaluate_model(
      fit$model,
      x[test_idx, , drop = FALSE],
      y[test_idx],
      device
    )
    fold_accuracy[[fold]] <- metrics$accuracy
    fold_loss[[fold]] <- metrics$loss
    fold_models[[fold]] <- fit$model

    cat(sprintf(
      "Fold %d held-out | loss=%.4f | accuracy=%.4f\n",
      fold, metrics$loss, metrics$accuracy
    ))
  }

  list(
    fold_accuracy = fold_accuracy,
    fold_loss = fold_loss,
    mean_accuracy = mean(fold_accuracy),
    sd_accuracy = stats::sd(fold_accuracy),
    mean_loss = mean(fold_loss),
    models = fold_models,
    folds = folds
  )
}

# -----------------------------------------------------
# Hold-out split + CV on train + final GPU refit
# -----------------------------------------------------

train_test_with_cv <- function(
    x,
    y,
    device,
    test_fraction = 0.2,
    cv_folds = 5L,
    epochs = 25L,
    batch_size = 128L,
    lr = 1e-3
) {
  n <- nrow(x)
  test_n <- max(1L, floor(n * test_fraction))
  test_idx <- sample.int(n, size = test_n)
  train_idx <- setdiff(seq_len(n), test_idx)

  cat(sprintf(
    "\nDevice: %s | Training rows: %d | Test rows: %d\n",
    device$type, length(train_idx), length(test_idx)
  ))

  cv_results <- run_cv(
    x = x[train_idx, , drop = FALSE],
    y = y[train_idx],
    k = cv_folds,
    epochs = epochs,
    batch_size = batch_size,
    lr = lr,
    device = device
  )

  cat("\n=== Final refit on full training set (GPU) ===\n")
  final_fit <- train_model(
    x_train = x[train_idx, , drop = FALSE],
    y_train = y[train_idx],
    x_val = x[test_idx, , drop = FALSE],
    y_val = y[test_idx],
    epochs = epochs,
    batch_size = batch_size,
    lr = lr,
    device = device,
    verbose = TRUE
  )

  final_metrics <- evaluate_model(
    final_fit$model,
    x[test_idx, , drop = FALSE],
    y[test_idx],
    device
  )

  list(
    cv = cv_results,
    final_test_accuracy = final_metrics$accuracy,
    final_test_loss = final_metrics$loss,
    model = final_fit$model,
    history = final_fit$history,
    test_idx = test_idx,
    train_idx = train_idx,
    device = device$type
  )
}

# -----------------------------------------------------
# Main
# -----------------------------------------------------

cat("============================\n")
cat("GPU training + cross-validation\n")
cat("============================\n")

sim_data <- make_data(n = n_obs, p = 30L, signal = 1.2)

results <- train_test_with_cv(
  x = sim_data$x,
  y = sim_data$y,
  device = device,
  test_fraction = 0.2,
  cv_folds = n_folds,
  epochs = epochs,
  batch_size = batch_size,
  lr = lr
)

cat("\n============================\n")
cat("Cross-validation summary\n")
cat("============================\n")
cat(sprintf("Device: %s\n", results$device))
cat(sprintf("Fold accuracies: %s\n", paste(sprintf("%.4f", results$cv$fold_accuracy), collapse = ", ")))
cat(sprintf("Mean CV accuracy: %.4f ± %.4f\n", results$cv$mean_accuracy, results$cv$sd_accuracy))
cat(sprintf("Mean CV loss: %.4f\n", results$cv$mean_loss))

cat("\n============================\n")
cat("Final hold-out test metrics\n")
cat("============================\n")
cat(sprintf("Test accuracy: %.4f\n", results$final_test_accuracy))
cat(sprintf("Test loss: %.4f\n", results$final_test_loss))

# Leave results in the global env when sourced interactively
invisible(results)
