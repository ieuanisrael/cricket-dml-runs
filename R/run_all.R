#!/usr/bin/env Rscript
#' End-to-end: synthetic T20 BBB -> DML (weighted vs unweighted) -> RAE / Elo.

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
source(file.path(root, "R/compare_rae_dml.R"))
source(file.path(root, "R/elo_ratings.R"))

args <- commandArgs(trailingOnly = TRUE)
parse_flag <- function(flag, default) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", flag, "="), "", hit[[1]])
}

n_matches <- as.integer(parse_flag("--matches", "60"))
min_balls <- as.integer(parse_flag("--min-balls", "60"))
n_folds <- as.integer(parse_flag("--folds", "5"))
seed <- as.integer(parse_flag("--seed", "42"))
weight_method <- parse_flag("--weights", "kernel_exposure")
t_max <- as.integer(parse_flag("--t-max", "30"))
out_dir <- parse_flag("--out", "outputs/run_default")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("==> Generating synthetic T20 ball-by-ball (matches=", n_matches, ")")
sim <- generate_synthetic_t20_bbb(
  n_matches = n_matches,
  n_batters = 48L,
  n_bowlers = 36L,
  n_venues = 8L,
  seed = seed
)
paths <- save_synthetic_dataset(sim)
message("    deliveries: ", paths$ball_by_ball, " (n=", nrow(sim$deliveries), ")")

message("==> Preparing DML analysis frame (min_balls=", min_balls, ")")
prepared <- prepare_player_dml_frame(sim$deliveries, min_balls = min_balls)
message(
  "    n_obs=", prepared$feature_info$n_obs,
  " batters=", prepared$feature_info$n_batters,
  " controls=", prepared$feature_info$n_controls,
  " ref=", prepared$reference_batter
)

message("==> Unweighted DML (folds=", n_folds, ")")
fit_unw <- estimate_player_effects_dml(
  prepared,
  n_folds = n_folds,
  seed = seed,
  cluster = prepared$frame$match_id,
  weights = NULL
)
data.table::fwrite(fit_unw$estimates, file.path(out_dir, "player_effects_dml_unweighted.csv"))

message("==> Selection weights (method=", weight_method, ")")
sel_w <- compute_selection_weights(
  prepared$frame,
  method = weight_method,
  t_max = t_max
)
plot_selection_weights(sel_w, out_path = file.path(out_dir, "plots/selection_weights.png"))
message("    mean=", round(mean(sel_w), 3), " max=", round(max(sel_w), 3))

message("==> Weighted DML (folds=", n_folds, ")")
fit <- estimate_player_effects_dml(
  prepared,
  n_folds = n_folds,
  seed = seed,
  cluster = prepared$frame$match_id,
  weights = sel_w
)
est_path <- file.path(out_dir, "player_effects_dml.csv")
data.table::fwrite(fit$estimates, est_path)

plot_player_effects(
  fit$estimates,
  out_path = file.path(out_dir, "plots/player_effects_forest.png"),
  top_n = 30L
)

message("==> Weighted vs unweighted DML scatter")
wcmp <- compare_weighted_vs_unweighted(fit_unw$estimates, fit$estimates)
data.table::fwrite(wcmp$table, file.path(out_dir, "weighted_vs_unweighted.csv"))
plot_weighted_vs_unweighted(
  wcmp,
  out_path = file.path(out_dir, "plots/weighted_vs_unweighted_scatter.png"),
  weight_method = weight_method
)
message("    Pearson r = ", round(wcmp$corr, 4))

message("==> Estimating context-only RAE by batter")
rae <- estimate_player_rae(prepared, n_folds = n_folds, seed = seed)
data.table::fwrite(rae$estimates, file.path(out_dir, "player_effects_rae.csv"))

message("==> Comparing RAE vs DML (weighted)")
cmp <- compare_rae_vs_dml(fit$estimates, rae$estimates)
data.table::fwrite(cmp$table, file.path(out_dir, "rae_vs_dml.csv"))
plot_rae_vs_dml(cmp, out_path = file.path(out_dir, "plots/rae_vs_dml_scatter.png"))
plot_rae_dml_forest(cmp, out_path = file.path(out_dir, "plots/rae_vs_dml_forest.png"), top_n = 30L)

message("==> Computing delivery-level Elo (striker & bowler skill proxy)")
elo <- compute_elo_ratings(sim$deliveries, k = 8, base = 1500, scale = 400)
data.table::fwrite(elo$striker, file.path(out_dir, "elo_striker.csv"))
data.table::fwrite(elo$bowler, file.path(out_dir, "elo_bowler.csv"))
elo_bat <- batter_elo_vs_ref(
  elo$striker,
  reference_batter = prepared$reference_batter,
  min_balls = min_balls
)
data.table::fwrite(elo_bat, file.path(out_dir, "elo_striker_vs_ref.csv"))

message("==> Comparing Elo vs DML (weighted)")
elo_cmp <- compare_elo_vs_dml(fit$estimates, elo_bat)
data.table::fwrite(elo_cmp$table, file.path(out_dir, "elo_vs_dml.csv"))
plot_elo_vs_dml(elo_cmp, out_path = file.path(out_dir, "plots/elo_vs_dml_scatter.png"))
plot_elo_on_dml_forest(
  elo_cmp,
  out_path = file.path(out_dir, "plots/player_effects_forest_elo.png"),
  top_n = 30L
)

manifest <- c(
  paste0("generated_at: ", Sys.time()),
  paste0("root: ", root),
  paste0("n_matches: ", n_matches),
  paste0("n_deliveries: ", nrow(sim$deliveries)),
  paste0("min_balls: ", min_balls),
  paste0("n_folds: ", n_folds),
  paste0("seed: ", seed),
  paste0("weight_method: ", weight_method),
  paste0("weight_t_max: ", t_max),
  paste0("weight_max: ", round(max(sel_w), 5)),
  paste0("weighted_vs_unweighted_pearson: ", round(wcmp$corr, 5)),
  paste0("weighted_vs_unweighted_spearman: ", round(wcmp$spearman, 5)),
  paste0("reference_batter: ", prepared$reference_batter),
  paste0("rae_dml_pearson: ", round(cmp$corr, 5)),
  paste0("rae_dml_spearman: ", round(cmp$spearman, 5)),
  paste0("elo_dml_pearson: ", round(elo_cmp$corr, 5)),
  paste0("elo_dml_spearman: ", round(elo_cmp$spearman, 5)),
  paste0("estimates_dml_weighted: ", est_path),
  paste0("estimates_dml_unweighted: ", file.path(out_dir, "player_effects_dml_unweighted.csv")),
  paste0("comparison_weighted: ", file.path(out_dir, "weighted_vs_unweighted.csv"))
)
writeLines(manifest, file.path(out_dir, "run_manifest.txt"))

message("==> Done")
message("    Weighted vs unweighted r: ", round(wcmp$corr, 4))
message("    Outputs: ", normalizePath(out_dir))
