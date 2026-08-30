#!/usr/bin/env Rscript
#' Run caret / XGBoost(-GPU) DML on synthetic or real BBB data.
#'
#' Examples:
#'   Rscript R/run_dml_caret_gpu.R --data=data/raw/t20_ball_by_ball.csv --gpu=false
#'   Rscript R/run_dml_caret_gpu.R --data=/path/to/real_bbb.csv --gpu=true --folds=5
#'
#' GPU notes:
#'   - caret itself is not a GPU framework.
#'   - This runner uses XGBoost via the caret/xgbTree stack with device=cuda
#'     (or tree_method=gpu_hist) when --gpu=true and a CUDA build is installed.

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

source(file.path(root, "R/prepare_analysis_frame.R"))
source(file.path(root, "R/estimate_player_dml.R")) # plot_player_effects
source(file.path(root, "R/estimate_player_dml_caret.R"))

args <- commandArgs(trailingOnly = TRUE)
parse_flag <- function(flag, default) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", flag, "="), "", hit[[1]])
}
parse_bool <- function(flag, default = FALSE) {
  raw <- tolower(parse_flag(flag, if (default) "true" else "false"))
  raw %in% c("1", "true", "t", "yes", "y")
}

data_path <- parse_flag("--data", "data/raw/real_bbl_data.csv")
min_balls <- as.integer(parse_flag("--min-balls", "180"))
n_folds <- as.integer(parse_flag("--folds", "5"))
seed <- as.integer(parse_flag("--seed", "42"))
use_gpu <- parse_bool("--gpu", TRUE)
caret_method <- parse_flag("--method", "xgbTree")
xgb_nrounds <- as.integer(parse_flag("--nrounds", "200"))
out_dir <- parse_flag("--out", "outputs/run_caret_gpu")

if (!file.exists(data_path)) {
  stop("Data file not found: ", data_path, call. = FALSE)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("==> Reading BBB: ", data_path)
deliveries <- data.table::fread(data_path)
message("    n=", nrow(deliveries), " cols=", ncol(deliveries))

req <- c("striker_id", "bat_score", "bowler_id", "venue", "phase", "over",
         "striker_batting_position", "batter_is_home", "match_id")
miss <- setdiff(req, names(deliveries))
if (length(miss)) {
  stop("Missing required columns: ", paste(miss, collapse = ", "), call. = FALSE)
}

message("==> Preparing analysis frame (min_balls=", min_balls, ")")
prepared <- prepare_player_dml_frame(deliveries, min_balls = min_balls)
message(
  "    n_obs=", prepared$feature_info$n_obs,
  " strikers=", prepared$feature_info$n_batters,
  " controls=", prepared$feature_info$n_controls,
  " ref=", prepared$reference_batter
)

message("==> DML with caret/XGBoost (gpu=", use_gpu, ", method=", caret_method, ")")
fit <- estimate_player_effects_dml_caret(
  prepared,
  n_folds = n_folds,
  seed = seed,
  cluster = prepared$frame$match_id,
  use_gpu = use_gpu,
  caret_method = caret_method,
  xgb_nrounds = xgb_nrounds,
  verbose = TRUE
)

est_path <- file.path(out_dir, "player_effects_dml_caret.csv")
data.table::fwrite(fit$estimates, est_path)
saveRDS(fit$diagnostics, file.path(out_dir, "diagnostics_caret.rds"))

plot_player_effects(
  fit$estimates,
  out_path = file.path(out_dir, "plots/player_effects_forest_caret.png"),
  top_n = 40L
)

manifest <- c(
  paste0("generated_at: ", Sys.time()),
  paste0("data: ", normalizePath(data_path)),
  paste0("n_rows_raw: ", nrow(deliveries)),
  paste0("n_obs: ", fit$diagnostics$n_obs),
  paste0("n_strikers: ", fit$diagnostics$n_batters),
  paste0("min_balls: ", min_balls),
  paste0("n_folds: ", n_folds),
  paste0("seed: ", seed),
  paste0("caret_method: ", fit$diagnostics$caret_method),
  paste0("use_gpu: ", use_gpu),
  paste0("gpu_active: ", fit$diagnostics$gpu_active),
  paste0("gpu_note: ", fit$diagnostics$gpu_note),
  paste0("xgb_nrounds: ", xgb_nrounds),
  paste0("reference_batter: ", prepared$reference_batter),
  paste0("estimates: ", est_path)
)
writeLines(manifest, file.path(out_dir, "run_manifest.txt"))

message("==> Done")
message("    gpu_active: ", fit$diagnostics$gpu_active)
message("    Outputs: ", normalizePath(out_dir))
