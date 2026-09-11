"""Average potential outcomes via DoubleMLAPOS on T20 ball-by-ball data.

Three specifications are available.

``--treatment match_state`` (default) estimates expected runs per ball for each
striker, anchored on the situation they faced: the bowler, the over and phase,
the wickets already down, and the required run rate. Fixture identifiers are
deliberately excluded, because together they pin down a single match in which
only one batter was on strike.

``--treatment position`` estimates the effect of a place in the batting order,
controlling for striker identity.

``--treatment striker`` reproduces the original striker-identity design for
comparison; its overlap diagnostics exist to show why it does not work.

Note on interpretation: anchoring on match state makes the estimand a
*state-conditional scoring rate* -- expected runs on a delivery given this
situation -- and not a batter's total contribution to an innings. It is also
conditional on the batter having survived to be at the crease in that state.

Two fits are performed. The unclustered fit supports ``causal_contrast()`` and
``sensitivity_analysis()``; the clustered fit gives standard errors that treat
deliveries within a match as dependent. DoubleML cannot do both at once --
``concat()`` and ``bootstrap()`` both refuse clustered frameworks -- so the
runner reports the ratio between the two sets of standard errors, which is the
factor by which the contrast inference is optimistic.

Usage:
    python run_apos.py --treatment position --data data/raw/t20_ball_by_ball.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

import doubleml as dml  # noqa: E402

from cricket_dml.frame import (  # noqa: E402
    AnalysisFrame,
    build_frame,
    load_deliveries,
    position_variation,
)

# With position as the treatment, omitting striker identity is the natural
# benchmark: it measures how far the estimates move when the control for player
# skill is removed, which is exactly the skill-driven selection into the order.
DEFAULT_BENCHMARKS = {
    "striker": ["striker_batting_position"],
    "position": ["striker_id"],
    # Omitting the bowler tests how much of a batter's apparent effect is really
    # the quality of bowling they happened to face.
    "match_state": ["bowler_id"],
    "match_state_fine": ["bowler_id"],
}


def make_learners(frame: AnalysisFrame, seed: int) -> tuple[object, object]:
    """Gradient-boosted nuisance learners that treat coded columns as categorical."""
    categorical = frame.categorical_indices or None
    outcome = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        min_samples_leaf=20,
        categorical_features=categorical,
        random_state=seed,
    )
    propensity = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        min_samples_leaf=20,
        categorical_features=categorical,
        random_state=seed,
    )
    return outcome, propensity


def fit_apos(
    frame: AnalysisFrame,
    n_folds: int,
    seed: int,
    trimming_threshold: float,
    normalize_ipw: bool,
) -> dml.DoubleMLAPOS:
    """Fit average potential outcomes for every striker in the comparison set.

    DoubleMLAPOS does not accept clustered data (it is a sampling mixin rather
    than a DoubleML subclass and lacks the per-cluster fold machinery), so the
    scores here treat deliveries as independent. See `pairwise_irm` for the
    match-clustered counterpart.
    """
    outcome, propensity = make_learners(frame, seed)
    data = dml.DoubleMLData(
        frame.data[[frame.y_col, frame.d_col, *frame.x_cols]].copy(),
        y_col=frame.y_col,
        d_cols=frame.d_col,
        x_cols=frame.x_cols,
    )
    np.random.seed(seed)
    model = dml.DoubleMLAPOS(
        data,
        ml_g=outcome,
        ml_m=propensity,
        treatment_levels=frame.treatment_levels,
        n_folds=n_folds,
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
    )
    model.fit()
    return model


def pairwise_irm(
    frame: AnalysisFrame,
    treated_level: int,
    control_level: int,
    cluster: bool,
    n_folds: int,
    seed: int,
    trimming_threshold: float,
    normalize_ipw: bool,
) -> dml.DoubleMLIRM:
    """ATE of one striker versus another on the deliveries those two faced.

    This is the estimand with the mildest overlap demand: the population is
    restricted to balls faced by the pair, so the propensity is bounded away
    from zero by construction. Unlike APOS, IRM supports clustered data, which
    is what makes the standard-error comparison possible.
    """
    subset = frame.data.loc[frame.data[frame.d_col].isin([treated_level, control_level])].copy()
    subset[frame.d_col] = (subset[frame.d_col] == treated_level).astype(int)

    columns = [frame.y_col, frame.d_col, *frame.x_cols]
    if cluster:
        columns.append(frame.cluster_col)
        subset[frame.cluster_col] = pd.Categorical(subset[frame.cluster_col]).codes
    payload = subset[columns].reset_index(drop=True)

    data = dml.DoubleMLData(
        payload,
        y_col=frame.y_col,
        d_cols=frame.d_col,
        x_cols=frame.x_cols,
        cluster_cols=frame.cluster_col if cluster else None,
    )
    outcome, propensity = make_learners(frame, seed)
    np.random.seed(seed)
    model = dml.DoubleMLIRM(
        data,
        ml_g=outcome,
        ml_m=propensity,
        n_folds=n_folds,
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
    )
    model.fit()
    return model


def overlap_diagnostics(
    model: dml.DoubleMLAPOS, frame: AnalysisFrame, trimming_threshold: float
) -> pd.DataFrame:
    """Per-striker propensity summary: the empirical test of the overlap claim.

    DoubleML exposes propensities *after* trimming, so a count of values below
    the threshold would always be zero. What is informative is the mass sitting
    exactly on the clip boundary: those deliveries carry no comparison
    information, and the average potential outcome over them is extrapolation
    from the outcome model rather than a contrast between observed batters.
    """
    balls = frame.data[frame.d_col].value_counts()
    tolerance = 1e-9
    rows = []
    for index, level in enumerate(model.treatment_levels):
        propensity = np.asarray(model.modellist[index].predictions["ml_m"]).ravel()
        clipped_low = float((propensity <= trimming_threshold + tolerance).mean())
        clipped_high = float((propensity >= 1.0 - trimming_threshold - tolerance).mean())
        rows.append(
            {
                "striker": frame.label(level),
                "balls_faced": int(balls.get(level, 0)),
                "propensity_mean": propensity.mean(),
                "propensity_p05": np.quantile(propensity, 0.05),
                "propensity_p95": np.quantile(propensity, 0.95),
                "share_clipped_low": clipped_low,
                "share_clipped_high": clipped_high,
                "share_on_boundary": clipped_low + clipped_high,
                "max_ipw_weight": float(1.0 / max(propensity.min(), 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def summarize(model: dml.DoubleMLAPOS, frame: AnalysisFrame) -> pd.DataFrame:
    summary = model.summary.copy()
    summary.index = [frame.label(level) for level in model.treatment_levels]
    summary.index.name = frame.treatment
    return summary


def all_pairwise_contrasts(model: dml.DoubleMLAPOS, frame: AnalysisFrame) -> pd.DataFrame:
    """Every unique striker-vs-striker contrast on a common baseline."""
    contrast = model.causal_contrast(reference_levels=model.treatment_levels)
    table = contrast.summary.copy()
    relabelled = []
    for name in table.index:
        left, right = str(name).split(" vs ")
        relabelled.append(f"{frame.label(int(float(left)))} vs {frame.label(int(float(right)))}")
    table.index = relabelled
    table.index.name = "contrast"
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/t20_ball_by_ball.csv")
    parser.add_argument("--out", default=None, help="Defaults to outputs/apos_<treatment>.")
    parser.add_argument(
        "--treatment",
        choices=["match_state", "match_state_fine", "position", "striker"],
        default="match_state",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Strikers by balls faced.")
    parser.add_argument("--strikers", nargs="*", default=None, help="Restrict to these strikers.")
    parser.add_argument("--min-balls", type=int, default=80)
    parser.add_argument("--min-level-balls", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trimming-threshold", type=float, default=1e-2)
    parser.add_argument("--normalize-ipw", action="store_true")
    parser.add_argument(
        "--benchmark",
        nargs="*",
        default=None,
        help="Observed confounders to omit when calibrating sensitivity parameters.",
    )
    parser.add_argument("--cf-y", type=float, default=0.03, help="Fallback outcome confounding.")
    parser.add_argument("--cf-d", type=float, default=0.03, help="Fallback treatment confounding.")
    parser.add_argument(
        "--drop-controls",
        nargs="*",
        default=[],
        help="Controls to exclude. Match-identifying controls (venue, series, season, "
        "bowler_id) make striker identity nearly deterministic and destroy overlap.",
    )
    parser.add_argument("--skip-cluster", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out or f"outputs/apos_{args.treatment}")
    out_dir.mkdir(parents=True, exist_ok=True)

    deliveries = load_deliveries(args.data)
    frame = build_frame(
        deliveries,
        treatment=args.treatment,
        strikers=args.strikers,
        top_n=None if args.strikers else args.top_n,
        min_balls=args.min_balls,
        min_level_balls=args.min_level_balls,
        drop_controls=args.drop_controls,
    )
    print(
        f"Treatment '{args.treatment}': {len(frame.treatment_levels)} levels "
        f"({', '.join(frame.label(level) for level in frame.treatment_levels)}), "
        f"{len(frame.data):,} deliveries"
    )
    print(f"Controls: {', '.join(frame.x_cols)}")
    print(f"Excluded as mediators or post-treatment: {', '.join(frame.excluded_mediators)}")
    print(f"Mean propensity implied by design: {1 / len(frame.treatment_levels):.4f}")

    if args.treatment == "position" and "striker_batting_position" in deliveries.columns:
        variation = position_variation(deliveries)
        multi = int((variation["n_positions"] > 1).sum())
        print(
            f"Within-player variation: {multi} of {len(variation)} strikers batted at "
            f"more than one position (median {variation['n_positions'].median():.0f} positions)."
        )

    model = fit_apos(
        frame,
        n_folds=args.folds,
        seed=args.seed,
        trimming_threshold=args.trimming_threshold,
        normalize_ipw=args.normalize_ipw,
    )

    apos = summarize(model, frame)
    overlap = overlap_diagnostics(model, frame, args.trimming_threshold)
    contrasts = all_pairwise_contrasts(model, frame)

    print("\n=== Average potential outcomes (runs per ball) ===")
    print(apos.to_string())
    print("\n=== Overlap diagnostics ===")
    print(overlap.to_string(index=False))
    print(f"\n=== Pairwise contrasts ({len(contrasts)} pairs) ===")
    print(contrasts.to_string())

    apos.to_csv(out_dir / "striker_apo.csv")
    overlap.to_csv(out_dir / "overlap_diagnostics.csv", index=False)
    contrasts.to_csv(out_dir / "pairwise_contrasts.csv")

    manifest: dict[str, object] = {
        "data": args.data,
        "treatment": args.treatment,
        "n_deliveries": int(len(frame.data)),
        "levels": [frame.label(level) for level in frame.treatment_levels],
        "controls": frame.x_cols,
        "excluded_mediators": frame.excluded_mediators,
        "folds": args.folds,
        "trimming_threshold": args.trimming_threshold,
        "normalize_ipw": bool(args.normalize_ipw),
        "seed": args.seed,
    }

    requested = args.benchmark if args.benchmark is not None else DEFAULT_BENCHMARKS[args.treatment]
    benchmark_set = [column for column in requested if column in frame.x_cols]
    if benchmark_set:
        print(f"\n=== Sensitivity benchmarked on {benchmark_set} ===")
        benchmark = model.sensitivity_benchmark(benchmarking_set=benchmark_set)
        benchmark.index = [frame.label(level) for level in model.treatment_levels]
        print(benchmark.to_string())
        benchmark.to_csv(out_dir / "sensitivity_benchmark.csv")

        raw_cf_y = float(np.nanmax(benchmark["cf_y"].to_numpy()))
        raw_cf_d = float(np.nanmax(benchmark["cf_d"].to_numpy()))
        # cf_d must lie in [0, 1). A benchmarked value at the boundary means the
        # short model's Riesz representer variance collapsed, which happens when
        # the benchmarking variable is a weak confounder or the sample is small.
        # Falling back to the documented default keeps the bounds interpretable.
        degenerate = raw_cf_y >= 1.0 or raw_cf_d >= 1.0
        if degenerate:
            cf_y, cf_d = args.cf_y, args.cf_d
            print(
                f"  benchmark returned cf_y={raw_cf_y:.3f}, cf_d={raw_cf_d:.3f}, which is "
                f"degenerate; falling back to cf_y={cf_y}, cf_d={cf_d}."
            )
        else:
            cf_y, cf_d = raw_cf_y, raw_cf_d
    else:
        cf_y, cf_d, degenerate = args.cf_y, args.cf_d, False

    model.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=1.0)
    print(f"\n=== Sensitivity bounds at cf_y={cf_y:.4f}, cf_d={cf_d:.4f}, rho=1 ===")
    print(model.sensitivity_summary)
    print(
        "RV (%) is the confounding strength, as a share of residual variance in both\n"
        "runs and striker assignment, that would pull an estimate to zero; RVa (%)\n"
        "additionally accounts for sampling uncertainty."
    )
    manifest["sensitivity"] = {
        "cf_y": cf_y,
        "cf_d": cf_d,
        "rho": 1.0,
        "benchmark_degenerate": bool(degenerate),
    }
    Path(out_dir / "sensitivity_summary.txt").write_text(model.sensitivity_summary)

    if not args.skip_cluster:
        # The two busiest strikers give the best-powered pair for this check.
        balls = frame.data[frame.d_col].value_counts()
        treated_level, control_level = (int(level) for level in balls.index[:2])
        pair = f"{frame.label(treated_level)} vs {frame.label(control_level)}"
        print(f"\n=== Pairwise IRM ATE, {pair} (match clustering check) ===")

        irm_kwargs = {
            "treated_level": treated_level,
            "control_level": control_level,
            "n_folds": args.folds,
            "seed": args.seed,
            "trimming_threshold": args.trimming_threshold,
            "normalize_ipw": args.normalize_ipw,
        }
        iid_irm = pairwise_irm(frame, cluster=False, **irm_kwargs)
        clustered_irm = pairwise_irm(frame, cluster=True, **irm_kwargs)

        def boundary_mass(model: dml.DoubleMLIRM) -> float:
            propensity = np.asarray(model.predictions["ml_m"]).ravel()
            tolerance = args.trimming_threshold + 1e-9
            return float(
                ((propensity <= tolerance) | (propensity >= 1.0 - tolerance)).mean()
            )

        comparison = pd.DataFrame(
            {
                "coef": [iid_irm.coef[0], clustered_irm.coef[0]],
                "std_err": [iid_irm.se[0], clustered_irm.se[0]],
                "share_on_boundary": [boundary_mass(iid_irm), boundary_mass(clustered_irm)],
            },
            index=["independent_balls", "match_clustered"],
        )
        print(comparison.to_string())

        # Clustered fits split folds on matches. Two things can break them: too
        # few matches to spread across folds, or a treatment that never varies
        # within a match. Both show up as a point estimate that moves far more
        # than sampling noise would allow, and neither yields a usable ratio.
        # Clustering should widen the interval, not relocate the estimate. Movement
        # is therefore judged in units of the unclustered standard error: beyond a
        # few of those, the two fits disagree by more than sampling noise permits
        # and the ratio of their standard errors is meaningless.
        n_clusters = frame.data[frame.cluster_col].nunique()
        moved = abs(clustered_irm.coef[0] - iid_irm.coef[0])
        moved_in_se = moved / max(iid_irm.se[0], 1e-12)
        too_few_clusters = n_clusters < 10 * args.folds

        if moved_in_se > 3.0 or too_few_clusters:
            if too_few_clusters:
                cause = (
                    f"only {n_clusters} matches are available for {args.folds} folds, "
                    f"leaving roughly {n_clusters // args.folds} matches per fold"
                )
            else:
                cause = (
                    "the treatment barely varies within a match, so held-out "
                    "propensities collapse to the trimming bounds"
                )
            print(
                f"\nThe clustered fit is not usable: the point estimate moved by "
                f"{moved:.2f} ({moved_in_se:.1f} unclustered standard errors), whereas "
                f"clustering should only widen the interval. Here {cause}. Treat this as "
                f"a failed fit, and re-run the check on an archive with many matches."
            )
            manifest["cluster_check"] = {
                "pair": pair,
                "status": "degenerate",
                "n_clusters": int(n_clusters),
                "moved_in_se": moved_in_se,
            }
        else:
            inflation = float(clustered_irm.se[0] / iid_irm.se[0])
            print(
                f"\nMatch clustering multiplies the standard error by {inflation:.2f}. "
                f"The APOS contrasts above treat deliveries as independent, so their "
                f"intervals are too narrow by roughly this factor."
            )
            manifest["cluster_check"] = {"pair": pair, "se_inflation": inflation}
        comparison.to_csv(out_dir / "clustered_se_comparison.csv")

    Path(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote results to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
