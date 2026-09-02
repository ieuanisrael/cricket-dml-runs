#' Plots for striker ATE (run effects) and on-strike propensity models.
#'
#' @param ate_estimates data.table from [estimate_player_effects_dml].
#' @param propensity Output of [estimate_striker_propensity].
#' @param out_dir Directory for PNG outputs.
#' @param top_n How many strikers to show in ranked plots.
plot_ate_and_propensity <- function(
    ate_estimates,
    propensity,
    out_dir,
    top_n = 30L
) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Install ggplot2", call. = FALSE)
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table", call. = FALSE)
  }
  dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

  ate <- data.table::as.data.table(ate_estimates)
  prop <- data.table::as.data.table(propensity$by_striker)
  ball <- data.table::as.data.table(propensity$ball_level)

  # ---- 1. ATE forest ----
  ate_top <- utils::head(ate, as.integer(top_n))
  ate_top[, striker_id := factor(striker_id, levels = rev(striker_id))]
  p_ate <- ggplot2::ggplot(ate_top, ggplot2::aes(x = effect_runs_per_ball, y = striker_id)) +
    ggplot2::geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    ggplot2::geom_pointrange(
      ggplot2::aes(xmin = ci_lo, xmax = ci_hi),
      colour = "#1d4ed8"
    ) +
    ggplot2::labs(
      title = "Model 1: Striker average treatment effects on runs",
      subtitle = paste0("DML runs/ball vs ", unique(ate$reference_batter)),
      x = "ATE (runs per ball)",
      y = NULL
    ) +
    ggplot2::theme_minimal(base_size = 12)
  ggplot2::ggsave(
    file.path(out_dir, "plots/ate_striker_forest.png"),
    p_ate,
    width = 8,
    height = max(4, 0.22 * nrow(ate_top)),
    dpi = 120
  )

  # ---- 2. Propensity ranking ----
  prop_plot <- utils::head(prop, as.integer(top_n))
  prop_plot[, striker_id := factor(striker_id, levels = rev(striker_id))]
  p_prop <- ggplot2::ggplot(prop_plot, ggplot2::aes(x = mean_propensity, y = striker_id)) +
    ggplot2::geom_col(fill = "#0f766e", width = 0.7) +
    ggplot2::geom_point(
      ggplot2::aes(x = empirical_share),
      colour = "#b45309",
      size = 2
    ) +
    ggplot2::labs(
      title = "Model 2: Propensity to be on strike given context",
      subtitle = "Teal bars = mean ê_j(X); amber points = empirical share of balls",
      x = "P(striker = j | X)",
      y = NULL
    ) +
    ggplot2::theme_minimal(base_size = 12)
  ggplot2::ggsave(
    file.path(out_dir, "plots/propensity_striker_bars.png"),
    p_prop,
    width = 8,
    height = max(4, 0.22 * nrow(prop_plot)),
    dpi = 120
  )

  # ---- 3. ATE vs propensity scatter ----
  merged <- merge(
    ate[, .(striker_id, ate = effect_runs_per_ball, ate_se = se, balls_faced)],
    prop[, .(striker_id, mean_propensity, empirical_share)],
    by = "striker_id",
    all = FALSE
  )
  p_scatter <- ggplot2::ggplot(merged, ggplot2::aes(x = mean_propensity, y = ate)) +
    ggplot2::geom_hline(yintercept = 0, colour = "grey80") +
    ggplot2::geom_vline(xintercept = mean(merged$mean_propensity), linetype = 3, colour = "grey70") +
    ggplot2::geom_smooth(method = "lm", se = TRUE, colour = "#64748b", fill = "#cbd5e1", linewidth = 0.6) +
    ggplot2::geom_point(ggplot2::aes(size = balls_faced), colour = "#1d4ed8", alpha = 0.8) +
    ggplot2::scale_size_area(max_size = 10) +
    ggplot2::labs(
      title = "Striker ATE vs on-strike propensity",
      subtitle = sprintf(
        "Pearson r = %.3f | size = balls faced",
        stats::cor(merged$ate, merged$mean_propensity)
      ),
      x = "Mean propensity P(on strike | X)",
      y = "DML ATE (runs per ball)",
      size = "Balls"
    ) +
    ggplot2::theme_minimal(base_size = 12)
  ggplot2::ggsave(
    file.path(out_dir, "plots/ate_vs_propensity_scatter.png"),
    p_scatter,
    width = 7.5,
    height = 6,
    dpi = 120
  )

  # ---- 4. Observed-striker propensity by over (selection into game state) ----
  if ("over" %in% names(ball)) {
    by_over <- ball[, .(
      mean_propensity = mean(propensity),
      n = .N
    ), by = over]
    data.table::setorder(by_over, over)
    p_over <- ggplot2::ggplot(by_over, ggplot2::aes(x = over, y = mean_propensity)) +
      ggplot2::geom_line(colour = "#0f766e", linewidth = 0.9) +
      ggplot2::geom_point(ggplot2::aes(size = n), colour = "#0f766e", alpha = 0.7) +
      ggplot2::scale_size_area(max_size = 8) +
      ggplot2::labs(
        title = "Propensity of the on-strike batter by over",
        subtitle = "Mean ê(X) for the observed striker — higher means more predictable assignment",
        x = "Over",
        y = "Mean propensity of observed striker",
        size = "Deliveries"
      ) +
      ggplot2::theme_minimal(base_size = 12)
    ggplot2::ggsave(
      file.path(out_dir, "plots/propensity_by_over.png"),
      p_over,
      width = 7.5,
      height = 4.5,
      dpi = 120
    )
  }

  # ---- 5. Overlap / common support for top strikers ----
  top_ids <- utils::head(merged[order(-abs(ate))]$striker_id, 8L)
  # Ball-level propensities for those strikers from matrix if available
  if (!is.null(propensity$propensity_matrix) && !is.null(propensity$strikers)) {
    pm <- propensity$propensity_matrix
    long <- data.table::rbindlist(lapply(as.character(top_ids), function(id) {
      j <- match(id, propensity$strikers)
      if (is.na(j)) return(NULL)
      data.table::data.table(striker_id = id, propensity = as.numeric(pm[, j]))
    }))
    if (nrow(long)) {
      p_overlap <- ggplot2::ggplot(long, ggplot2::aes(x = propensity, fill = striker_id)) +
        ggplot2::geom_density(alpha = 0.35, colour = NA) +
        ggplot2::labs(
          title = "Propensity overlap across top |ATE| strikers",
          subtitle = "Density of ê_j(X) over all deliveries (common support check)",
          x = "P(striker = j | X)",
          y = "Density",
          fill = "Striker"
        ) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(legend.position = "bottom")
      ggplot2::ggsave(
        file.path(out_dir, "plots/propensity_overlap_top.png"),
        p_overlap,
        width = 8,
        height = 5.5,
        dpi = 120
      )
    }
  }

  invisible(list(
    ate_propensity_table = merged,
    plots_dir = file.path(out_dir, "plots")
  ))
}
