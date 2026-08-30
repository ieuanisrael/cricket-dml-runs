library(torch)

set.seed(123)

# -----------------------------------------------------
# Device configuration
# -----------------------------------------------------

device <- if (cuda_is_available()) {
  message("Using CUDA GPU for training.")
  torch_device("cuda")
} else {
  message("CUDA not available; using CPU.")
  torch_device("cpu")
}

# -----------------------------------------------------
# Simulated binary-classification dataset
# -----------------------------------------------------

make_data <- function(n = 1200L, p = 25L, signal = 1.0) {
  x <- matrix(rnorm(n * p), nrow = n)
  linear_score <- rowSums(x[, 1:8]) * signal + 0.5 * x[, 9]
  prob <- plogis(linear_score)
  y <- rbinom(n, size = 1L, prob = prob)
  list(x = x, y = y)
}

# -----------------------------------------------------
# Model definition
# -----------------------------------------------------

build_model <- function(input_dim, hidden_1 = 64L, hidden_2 = 32L) {
  nn_sequential(
    nn_linear(input_dim, hidden_1),
    nn_relu(),
    nn_dropout(0.2),
    nn_linear(hidden_1, hidden_2),
    nn_relu(),
    nn_linear(hidden_2, 2L)
  )
}

# -----------------------------------------------------
# Training utilities
# -----------------------------------------------------

train_one_epoch <- function(model, x_batch, y_batch, optimizer, loss_fn, device) {
  model$train()
  optimizer$zero_grad()

  logits <- model(x_batch$to(device = device))
  loss <- loss_fn(logits, y_batch$to(device = device))

  loss$backward()
  optimizer$step()

  loss$item()
}

train_model <- function(
    x_train,
    y_train,
    epochs = 25,
    batch_size = 128,
    lr = 0.001,
    verbose = TRUE) {

  model <- build_model(ncol(x_train))
  model$to(device = device)

  optimizer <- optim_adam(model$parameters, lr = lr)
  loss_fn <- nn_cross_entropy_loss()

  n <- nrow(x_train)
  for (epoch in seq_len(epochs)) {
    perm <- sample.int(n)
    epoch_loss <- 0

    for (i in seq(1, n, by = batch_size)) {
      idx <- perm[i:min(i + batch_size - 1L, n)]

      xb <- torch_tensor(x_train[idx, , drop = FALSE], dtype = torch_float())
      yb <- torch_tensor(y_train[idx], dtype = torch_long())

      epoch_loss <- epoch_loss + train_one_epoch(
        model = model,
        x_batch = xb,
        y_batch = yb,
        optimizer = optimizer,
        loss_fn = loss_fn,
        device = device
      )
    }

    if (verbose) {
      cat(sprintf("Epoch %02d | Avg loss: %.4f\n", epoch, epoch_loss / ceiling(n / batch_size)))
    }
  }

  model
}

evaluate_model <- function(model, x_eval, y_eval, device) {
  model$eval()

  with_no_grad({
    x_eval_tensor <- torch_tensor(x_eval, dtype = torch_float())$to(device = device)
    logits <- model(x_eval_tensor)

    pred_labels <- as.integer(as.array(logits$argmax(dim = 2L)$cpu()))
    probs <- as.array(logits$softmax(dim = 2L)$cpu())

    accuracy <- mean(pred_labels == as.integer(y_eval))
    loss <- nn_cross_entropy_loss()(logits, torch_tensor(y_eval, dtype = torch_long())$to(device = device))$item()

    list(accuracy = accuracy, loss = loss, probabilities = probs)
  })
}

# -----------------------------------------------------
# K-fold cross-validation
# -----------------------------------------------------

run_cv <- function(x, y, k = 5L, epochs = 25, batch_size = 128, lr = 0.001) {
  n <- length(y)
  folds <- sample(rep(seq_len(k), length.out = n))
  cv_results <- numeric(k)

  for (fold in seq_len(k)) {
    test_idx <- which(folds == fold)
    train_idx <- which(folds != fold)

    cat(sprintf("\n=== Fold %d/%d ===\n", fold, k))

    model <- train_model(
      x_train = x[train_idx, , drop = FALSE],
      y_train = y[train_idx],
      epochs = epochs,
      batch_size = batch_size,
      lr = lr,
      verbose = FALSE
    )

    metrics <- evaluate_model(model, x[test_idx, , drop = FALSE], y[test_idx], device)
    cv_results[fold] <- metrics$accuracy

    cat(sprintf("Fold accuracy: %.4f\n", metrics$accuracy))
  }

  list(
    fold_accuracy = cv_results,
    mean_accuracy = mean(cv_results),
    sd_accuracy = sd(cv_results)
  )
}

# -----------------------------------------------------
# Train / test split with CV on training set
# -----------------------------------------------------

train_test_with_cv <- function(
    x,
    y,
    test_fraction = 0.2,
    cv_folds = 5L,
    epochs = 25,
    batch_size = 128,
    lr = 0.001) {

  n <- nrow(x)
  test_n <- max(1L, floor(n * test_fraction))
  test_idx <- sample.int(n, size = test_n)
  train_idx <- setdiff(seq_len(n), test_idx)

  cat("\nTraining rows:", length(train_idx), "| Test rows:", length(test_idx), "\n")

  cv_results <- run_cv(
    x = x[train_idx, , drop = FALSE],
    y = y[train_idx],
    k = cv_folds,
    epochs = epochs,
    batch_size = batch_size,
    lr = lr
  )

  final_model <- train_model(
    x_train = x[train_idx, , drop = FALSE],
    y_train = y[train_idx],
    epochs = epochs,
    batch_size = batch_size,
    lr = lr,
    verbose = FALSE
  )

  final_metrics <- evaluate_model(final_model, x[test_idx, , drop = FALSE], y[test_idx], device)

  list(
    cv = cv_results,
    final_test_accuracy = final_metrics$accuracy,
    final_test_loss = final_metrics$loss,
    model = final_model,
    test_idx = test_idx,
    train_idx = train_idx
  )
}

# -----------------------------------------------------
# Run the pipeline
# -----------------------------------------------------

sim_data <- make_data(n = 1500L, p = 30L, signal = 1.2)

results <- train_test_with_cv(
  x = sim_data$x,
  y = sim_data$y,
  test_fraction = 0.2,
  cv_folds = 5L,
  epochs = 25,
  batch_size = 128,
  lr = 0.001
)

cat("\n============================\n")
cat("Cross-validation summary\n")
cat("============================\n")
cat(sprintf("Fold accuracies: %s\n", paste(sprintf("%.4f", results$cv$fold_accuracy), collapse = ", ")))
cat(sprintf("Mean CV accuracy: %.4f\n", results$cv$mean_accuracy))
cat(sprintf("CV SD: %.4f\n", results$cv$sd_accuracy))

cat("\n============================\n")
cat("Final hold-out test metrics\n")
cat("============================\n")
cat(sprintf("Test accuracy: %.4f\n", results$final_test_accuracy))
cat(sprintf("Test loss: %.4f\n", results$final_test_loss))