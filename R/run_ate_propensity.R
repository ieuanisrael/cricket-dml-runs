#!/usr/bin/env Rscript
#' Two-model pipeline: striker run ATEs (DML) + on-strike propensity.
#'
#' Model 1 — Average treatment effect of each striker on bat_score
#'            (partially linear DML vs a reference striker).
#' Model 2 — Propensity that a player is on strike given context X
#'            (cross-fitted P(striker = j | X)).
#'
#' Usage:
#'   Rscript R/run_ate_propensity.R
#'   Rscript R/run_ate_propensity.R --data=data/raw/t20_ball_by_ball.csv --out=outputs/ate_propensity
#'   Rscript R/run_ate_propensity.R --data=/path/to/real_bbb.csv --min-balls=80 --folds=5

root <- Sys.getenv("CRICKET_DML_ROOT", unset = "")
if (!nzchar(root)) {
  args_all <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args_all, value = TRUE)
  if (length(file_arg)) {
    root <- dirname(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))))
  } else {
    root <- getwd()
  }
}
setwd(root)

source(file.path(root, "R/generate_synthetic_bbb.R"))
source(file.path(root, "R/prepare_analysis_frame.R"))
source(file.path(root, "R/selection_weights.R"))
source(file.path(root, "R/estimate_player_dml.R"))
source(file.path(root, "R/estimate_striker_propensity.R"))
source(file.path(root, "R/plot_ate_propensity.R"))

args <- commandArgs(trailingOnly = TRUE)
parse_flag <- function(flag, default) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", flag, "="), "", hit[[1]])
}

data_path <- parse_flag("--data", "")
n_matches <- as.integer(parse_flag("--matches", "40"))
min_balls <- as.integer(parse_flag("--min-balls", "40"))
n_folds <- as.integer(parse_flag("--folds", "3"))
seed <- as.integer(parse_flag("--seed", "42"))
weight_method <- parse_flag("--weights", "kernel_exposure")
t_max <- as.integer(parse_flag("--t-max", "30"))
out_dir <- parse_flag("--out", "outputs/ate_propensity")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Data ----
if (nzchar(data_path)) {
  message("==> Loading BBB: ", data_path)
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }
  deliveries <- data.table::fread(data_path)
} else {
  message("==> No --data; generating synthetic T20 BBB (matches=", n_matches, ")")
  sim <- generate_synthetic_t20_bbb(
    n_matches = n_matches,
    n_batters = 48L,
    n_bowlers = 36L,
    n_venues = 8L,
    seed = seed
  )
  deliveries <- sim$deliveries
  save_synthetic_dataset(sim)
}
message("    n=", nrow(deliveries))

message("==> Preparing analysis frame (min_balls=", min_balls, ")")
prepared <- prepare_player_dml_frame(deliveries, min_balls = min_balls)
message(
  "    n_obs=", prepared$feature_info$n_obs,
  " strikers=", prepared$feature_info$n_batters,
  " controls=", prepared$feature_info$n_controls,
  " ref=", prepared$reference_batter
)

# ---- Selection weights (optional; for ATE model) ----
message("==> Selection weights (", weight_method, ")")
sel_w <- compute_selection_weights(prepared$frame, method = weight_method, t_max = t_max)

# ---- Model 1: striker ATEs on runs ----
message("==> Model 1: DML striker ATEs on bat_score (folds=", n_folds, ")")
ate_fit <- estimate_player_effects_dml(
  prepared,
  n_folds = n_folds,
  seed = seed,
  cluster = prepared$frame$match_id,
  weights = sel_w
)
ate_path <- file.path(out_dir, "striker_ate_dml.csv")
data.table::fwrite(ate_fit$estimates, ate_path)

# ---- Model 2: on-strike propensity ----
message("==> Model 2: striker on-strike propensity P(j | X) (folds=", n_folds, ")")
prop_fit <- estimate_striker_propensity(
  prepared,
  n_folds = n_folds,
  seed = seed,
  include_reference = TRUE
)
prop_path <- file.path(out_dir, "striker_propensity.csv")
ball_prop_path <- file.path(out_dir, "ball_propensity_observed.csv")
data.table::fwrite(prop_fit$by_striker, prop_path)
data.table::fwrite(prop_fit$ball_level, ball_prop_path)

# ---- Combined table ----
combined <- merge(
  ate_fit$estimates[, .(
    striker_id, balls_faced,
    ate_runs_per_ball = effect_runs_per_ball,
    ate_se = se, ate_ci_lo = ci_lo, ate_ci_hi = ci_hi,
    ate_p_value = p_value, ate_fdr_q = fdr_q,
    reference_batter
  )],
  prop_fit$by_striker[, .(
    striker_id, mean_propensity, empirical_share,
    mean_propensity_when_facing, is_reference
  )],
  by = "striker_id",
  all = TRUE
)
data.table::setorder(combined, -ate_runs_per_ball)
combined_path <- file.path(out_dir, "striker_ate_and_propensity.csv")
data.table::fwrite(combined, combined_path)

# ---- Graphs ----
message("==> Writing ATE + propensity plots")
plot_info <- plot_ate_and_propensity(
  ate_estimates = ate_fit$estimates,
  propensity = prop_fit,
  out_dir = out_dir,
  top_n = 30L
)
data.table::fwrite(
  plot_info$ate_propensity_table,
  file.path(out_dir, "ate_vs_propensity_scatter_data.csv")
)

manifest <- c(
  paste0("generated_at: ", Sys.time()),
  paste0("root: ", root),
  paste0("data: ", if (nzchar(data_path)) data_path else "synthetic"),
  paste0("n_deliveries: ", nrow(deliveries)),
  paste0("n_obs: ", prepared$feature_info$n_obs),
  paste0("n_strikers_ate: ", prepared$feature_info$n_batters),
  paste0("min_balls: ", min_balls),
  paste0("n_folds: ", n_folds),
  paste0("seed: ", seed),
  paste0("weight_method: ", weight_method),
  paste0("reference_batter: ", prepared$reference_batter),
  paste0("model1_ate: ", ate_path),
  paste0("model2_propensity: ", prop_path),
  paste0("combined: ", combined_path),
  paste0("plots: ", plot_info$plots_dir)
)
writeLines(manifest, file.path(out_dir, "run_manifest.txt"))

message("==> Done")
message("    Model 1 (ATE):        ", ate_path)
message("    Model 2 (propensity): ", prop_path)
message("    Combined table:       ", combined_path)
message("    Plots:                ", normalizePath(plot_info$plots_dir))
message("    Outputs:              ", normalizePath(out_dir))
