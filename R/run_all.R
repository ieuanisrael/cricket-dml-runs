#!/usr/bin/env Rscript
#' End-to-end: synthetic T20 BBB -> DML batter effects -> RAE vs DML.

root <- Sys.getenv("CRICKET_DML_ROOT", unset = "")
if (!nzchar(root)) {
  # Prefer script location when run via Rscript
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
source(file.path(root, "R/estimate_player_dml.R"))
source(file.path(root, "R/compare_rae_dml.R"))

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

message("==> Estimating player effects via cross-fitted DML (folds=", n_folds, ")")
fit <- estimate_player_effects_dml(
  prepared,
  n_folds = n_folds,
  seed = seed,
  cluster = prepared$frame$match_id
)

est_path <- file.path(out_dir, "player_effects_dml.csv")
data.table::fwrite(fit$estimates, est_path)

plot_player_effects(
  fit$estimates,
  out_path = file.path(out_dir, "plots/player_effects_forest.png"),
  top_n = 30L
)

message("==> Estimating context-only RAE by batter")
rae <- estimate_player_rae(prepared, n_folds = n_folds, seed = seed)
data.table::fwrite(rae$estimates, file.path(out_dir, "player_effects_rae.csv"))

message("==> Comparing RAE vs DML")
cmp <- compare_rae_vs_dml(fit$estimates, rae$estimates)
data.table::fwrite(cmp$table, file.path(out_dir, "rae_vs_dml.csv"))
plot_rae_vs_dml(cmp, out_path = file.path(out_dir, "plots/rae_vs_dml_scatter.png"))
plot_rae_dml_forest(cmp, out_path = file.path(out_dir, "plots/rae_vs_dml_forest.png"), top_n = 30L)

manifest <- c(
  paste0("generated_at: ", Sys.time()),
  paste0("root: ", root),
  paste0("n_matches: ", n_matches),
  paste0("n_deliveries: ", nrow(sim$deliveries)),
  paste0("min_balls: ", min_balls),
  paste0("n_folds: ", n_folds),
  paste0("seed: ", seed),
  paste0("reference_batter: ", prepared$reference_batter),
  paste0("rae_dml_pearson: ", round(cmp$corr, 5)),
  paste0("rae_dml_spearman: ", round(cmp$spearman, 5)),
  paste0("rae_dml_rmse: ", round(cmp$rmse, 5)),
  paste0("rae_dml_mae: ", round(cmp$mae, 5)),
  paste0("estimates_dml: ", est_path),
  paste0("estimates_rae: ", file.path(out_dir, "player_effects_rae.csv")),
  paste0("comparison: ", file.path(out_dir, "rae_vs_dml.csv"))
)
writeLines(manifest, file.path(out_dir, "run_manifest.txt"))

message("==> Done")
message("    RAE vs DML Pearson:  ", round(cmp$corr, 4))
message("    RAE vs DML Spearman: ", round(cmp$spearman, 4))
message("    Outputs: ", normalizePath(out_dir))
