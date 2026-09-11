"""Ball-level analysis frames for DoubleML estimation on T20 ball-by-ball data.

The unit is a legal delivery and the outcome is runs off the bat. Two treatment
specifications are supported, and the difference between them is the whole
identification story:

``striker``
    Treatment is striker identity. Overlap is hopeless once the controls
    identify a match, because a match determines who was available to bat.
    Retained for comparison against ``R/prepare_analysis_frame.R``.

``position``
    Treatment is the striker's place in the batting order. Because the same
    player bats at different positions across matches, every level has
    substantial propensity, and striker identity becomes a *control* for skill
    rather than the object of interest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Y_COL = "bat_score"
D_COL = "treatment"
CLUSTER_COL = "match_id"

# HistGradientBoosting supports at most 255 categories per categorical feature.
MAX_CATEGORY_LEVELS = 255

# These raw columns are contemporaneous with the delivery being modelled:
# `cum_innings_runs` and `cum_innings_wickets` both *include* the current ball,
# so using them as controls would feed the outcome into the nuisance models.
# The `match_state` anchor uses lagged derivations of them instead, under the
# distinct names produced by `add_match_state`.
NEVER_CONTROLS = (
    "cum_innings_runs",
    "cum_innings_wickets",
    "target",
    "total_runs",
    "extras",
    "is_wicket",
    "how_out",
    "non_striker_id",
    "legal_ball_index",
    "ball_in_over",
)

# T20 innings length in legal deliveries.
BALLS_PER_INNINGS = 120


def add_match_state(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Attach lagged match-state columns describing the situation *before* a ball.

    Every column here is shifted within the striker's innings so that it
    reflects only completed deliveries. `cum_innings_runs` and
    `cum_innings_wickets` in the raw schema include the ball in question, so
    using them unshifted would leak the outcome.

    Produces:
        `wickets_down`    wickets lost before this delivery
        `runs_before`     innings runs before this delivery
        `balls_remaining` legal deliveries left in the innings, inclusive
        `req_run_rate`    runs per over still required (chases only, else 0)
        `is_chasing`      1 in the innings with a target
        `current_run_rate` runs per over scored so far
    """
    frame = deliveries.copy()
    order = [CLUSTER_COL, "innings"]
    sort_keys = [*order]
    for candidate in ("legal_ball_index", "over", "ball_in_over"):
        if candidate in frame.columns:
            sort_keys.append(candidate)
    frame = frame.sort_values(sort_keys).copy()
    grouped = frame.groupby(order, sort=False)

    for source, target in (
        ("cum_innings_wickets", "wickets_down"),
        ("cum_innings_runs", "runs_before"),
    ):
        if source in frame.columns:
            values = pd.to_numeric(frame[source], errors="coerce")
            frame[target] = values.groupby([frame[c] for c in order]).shift(1).fillna(0.0)
        else:
            frame[target] = 0.0

    if "legal_ball_index" in frame.columns:
        balls_done = pd.to_numeric(frame["legal_ball_index"], errors="coerce") - 1
    else:
        balls_done = grouped.cumcount()
    balls_done = balls_done.clip(lower=0)
    frame["balls_remaining"] = (BALLS_PER_INNINGS - balls_done).clip(lower=1)
    frame["current_run_rate"] = np.where(
        balls_done > 0, frame["runs_before"] / (balls_done / 6.0), 0.0
    )

    if "target" in frame.columns:
        target_runs = pd.to_numeric(frame["target"], errors="coerce")
        frame["is_chasing"] = target_runs.notna().astype(int)
        still_needed = (target_runs - frame["runs_before"]).clip(lower=0)
        frame["req_run_rate"] = np.where(
            target_runs.notna(), still_needed / (frame["balls_remaining"] / 6.0), 0.0
        )
    else:
        frame["is_chasing"] = 0
        frame["req_run_rate"] = 0.0

    # Coarse bands of the same state. Conditioning on the raw continuous values
    # makes each (bowler, over, wickets, rate) cell nearly unique, so the
    # propensity degenerates for want of anyone to compare against. Bands keep
    # the cricketing meaning while leaving several batters per cell.
    frame["wickets_band"] = pd.cut(
        frame["wickets_down"], bins=[-0.5, 2.5, 5.5, 10.5], labels=["0-2", "3-5", "6+"]
    ).astype(str)
    frame["rrr_band"] = np.where(
        frame["is_chasing"] == 0,
        "not_chasing",
        pd.cut(
            frame["req_run_rate"],
            bins=[-0.01, 6, 9, 12, 1e9],
            labels=["under_6", "6_to_9", "9_to_12", "over_12"],
        ).astype(str),
    )

    return frame.sort_index()


@dataclass(frozen=True)
class TreatmentSpec:
    """Which column is the treatment, and what may legitimately control for it."""

    column: str
    categorical_controls: tuple[str, ...]
    numeric_controls: tuple[str, ...]
    label_prefix: str = ""
    needs_match_state: bool = False


TREATMENT_SPECS: dict[str, TreatmentSpec] = {
    "striker": TreatmentSpec(
        column="striker_id",
        categorical_controls=("venue", "phase", "bowler_id", "innings", "season", "series", "day_night"),
        numeric_controls=("over", "striker_batting_position", "batter_is_home"),
    ),
    "position": TreatmentSpec(
        column="striker_batting_position",
        # `over`, `phase` and `bowler_id` are deliberately absent. Batting
        # position determines when a batter arrives and therefore which overs
        # and bowlers they face, so those columns are mediators: controlling
        # for them would strip out the main pathway by which position matters
        # and leave a narrow direct effect instead of the total effect.
        categorical_controls=(
            "striker_id",
            "batting_team",
            "bowling_team",
            "venue",
            "innings",
            "season",
            "series",
            "day_night",
        ),
        numeric_controls=("batter_is_home",),
        label_prefix="position_",
    ),
    "match_state": TreatmentSpec(
        column="striker_id",
        # Anchored on the situation a batter faced rather than on the fixture.
        # Venue, series, season, teams and day/night are all absent on purpose:
        # together they fingerprint a single match, and within one match only
        # one batter was on strike, which is what destroys overlap. A bowler,
        # by contrast, faces batters from every opposing side across a season,
        # so conditioning on the bowler links batters instead of isolating them.
        categorical_controls=("bowler_id", "phase", "wickets_band", "rrr_band"),
        numeric_controls=(),
        needs_match_state=True,
    ),
    "match_state_fine": TreatmentSpec(
        # Same anchor on the raw continuous state. Retained so the cost of
        # fine-grained conditioning can be measured rather than assumed.
        column="striker_id",
        categorical_controls=("bowler_id", "phase"),
        numeric_controls=(
            "over",
            "wickets_down",
            "req_run_rate",
            "is_chasing",
            "current_run_rate",
            "balls_remaining",
        ),
        needs_match_state=True,
    ),
}


@dataclass
class AnalysisFrame:
    """Numeric frame plus the metadata DoubleML and the learners need."""

    data: pd.DataFrame
    x_cols: list[str]
    level_labels: dict[int, str]
    treatment: str
    categorical_indices: list[int] = field(default_factory=list)
    excluded_mediators: list[str] = field(default_factory=list)
    y_col: str = Y_COL
    d_col: str = D_COL
    cluster_col: str = CLUSTER_COL

    @property
    def treatment_levels(self) -> list[int]:
        return sorted(self.data[self.d_col].unique().tolist())

    def label(self, level: int) -> str:
        return self.level_labels[int(level)]


def load_deliveries(path: str) -> pd.DataFrame:
    """Read a ball-by-ball CSV, tolerating either the synthetic or real schema."""
    deliveries = pd.read_csv(path, low_memory=False)
    missing = {Y_COL, "striker_id"} - set(deliveries.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    return deliveries


def _derive_phase(deliveries: pd.DataFrame) -> pd.Series:
    """Powerplay / middle / death, from `phase`, `power_play`, or `over`."""
    if "phase" in deliveries.columns:
        return deliveries["phase"].astype(str)
    if "over" not in deliveries.columns:
        return pd.Series("unknown", index=deliveries.index)

    over = pd.to_numeric(deliveries["over"], errors="coerce")
    if "power_play" in deliveries.columns:
        is_powerplay = pd.to_numeric(deliveries["power_play"], errors="coerce").fillna(0) == 1
    else:
        is_powerplay = over < 6
    return pd.Series(
        np.where(is_powerplay, "powerplay", np.where(over >= 15, "death", "middle")),
        index=deliveries.index,
    )


def _collapse_rare(values: pd.Series, max_levels: int) -> pd.Series:
    """Keep the most frequent levels, pooling the remainder into `other`."""
    if values.nunique() <= max_levels:
        return values
    keep = values.value_counts().index[: max_levels - 1]
    return values.where(values.isin(keep), other="other")


def position_variation(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Distinct batting positions per player, counted over striker-innings stays.

    Overlap for the position treatment rests on within-player variation: if
    every player always batted in the same slot, striker identity would
    determine the treatment and conditioning on it would leave nothing to
    compare.
    """
    stays = deliveries.groupby(
        [CLUSTER_COL, "innings", "striker_id"], as_index=False
    )["striker_batting_position"].first()
    summary = stays.groupby("striker_id")["striker_batting_position"].agg(
        n_positions="nunique", n_stays="count"
    )
    return summary.sort_values("n_stays", ascending=False)


def build_frame(
    deliveries: pd.DataFrame,
    treatment: str = "striker",
    strikers: list[str] | None = None,
    top_n: int | None = None,
    min_balls: int = 80,
    min_level_balls: int = 50,
    drop_controls: list[str] | None = None,
) -> AnalysisFrame:
    """Build the estimation frame for a chosen treatment specification.

    Args:
        deliveries: Ball-by-ball rows.
        treatment: Key into ``TREATMENT_SPECS``.
        strikers: Restrict the sample to these ``striker_id`` values.
        top_n: If ``strikers`` is None, keep this many strikers by balls faced.
        min_balls: Drop strikers with fewer deliveries than this.
        min_level_balls: Drop treatment levels with fewer deliveries than this.
        drop_controls: Further control columns to leave out of ``X``.
    """
    if treatment not in TREATMENT_SPECS:
        raise ValueError(f"treatment must be one of {sorted(TREATMENT_SPECS)}; got {treatment!r}")
    spec = TREATMENT_SPECS[treatment]
    excluded = set(drop_controls or ())

    if spec.column not in deliveries.columns:
        raise ValueError(f"Treatment column {spec.column!r} is not present in the data.")

    frame = deliveries.loc[deliveries["striker_id"].notna() & deliveries[Y_COL].notna()].copy()
    frame = frame.loc[frame[spec.column].notna()]
    frame["phase"] = _derive_phase(frame)
    if spec.needs_match_state:
        frame = add_match_state(frame)

    balls = frame["striker_id"].value_counts()
    eligible = balls.loc[balls >= min_balls]
    if eligible.empty:
        raise ValueError(f"No striker faced at least {min_balls} balls (max was {int(balls.max())}).")

    if strikers is not None:
        unknown = sorted(set(strikers) - set(eligible.index))
        if unknown:
            raise ValueError(f"Striker(s) not present with >= {min_balls} balls: {unknown}")
        keep_strikers = list(strikers)
    elif top_n is not None:
        keep_strikers = eligible.index[:top_n].tolist()
    else:
        keep_strikers = eligible.index.tolist()

    frame = frame.loc[frame["striker_id"].isin(keep_strikers)].copy()

    # Keep only treatment levels with enough support to estimate a nuisance on.
    level_counts = frame[spec.column].value_counts()
    keep_levels = level_counts.loc[level_counts >= min_level_balls].index
    frame = frame.loc[frame[spec.column].isin(keep_levels)].copy()
    if frame[spec.column].nunique() < 2:
        raise ValueError(
            f"Fewer than two {spec.column} levels have {min_level_balls}+ deliveries."
        )

    # Integer codes: DoubleML treats the treatment column as numeric, and
    # treatment_levels must be comparable to the values stored in `d`.
    codes = pd.Categorical(frame[spec.column], categories=sorted(frame[spec.column].unique()))
    level_labels = {i: f"{spec.label_prefix}{name}" for i, name in enumerate(codes.categories)}

    model = pd.DataFrame(index=frame.index)
    model[Y_COL] = pd.to_numeric(frame[Y_COL], errors="coerce").astype(float)
    model[D_COL] = codes.codes.astype(int)
    model[CLUSTER_COL] = frame[CLUSTER_COL].astype(str) if CLUSTER_COL in frame else "unknown"

    x_cols: list[str] = []
    categorical_indices: list[int] = []

    for column in spec.categorical_controls:
        if column not in frame.columns or column in excluded or column in NEVER_CONTROLS:
            continue
        values = _collapse_rare(frame[column].astype(str), MAX_CATEGORY_LEVELS)
        if values.nunique() < 2:
            continue
        model[column] = pd.Categorical(values).codes.astype(int)
        categorical_indices.append(len(x_cols))
        x_cols.append(column)

    for column in spec.numeric_controls:
        if column not in frame.columns or column in excluded or column in NEVER_CONTROLS:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0 or values.nunique() < 2:
            continue
        model[column] = values.fillna(values.median()).astype(float)
        x_cols.append(column)

    if not x_cols:
        raise ValueError("No usable controls were found in the delivery schema.")

    mediators = [
        column
        for column in ("over", "phase", "bowler_id", *NEVER_CONTROLS)
        if column in deliveries.columns and column not in x_cols
    ]

    return AnalysisFrame(
        data=model.dropna(subset=[Y_COL]).reset_index(drop=True),
        x_cols=x_cols,
        level_labels=level_labels,
        treatment=treatment,
        categorical_indices=categorical_indices,
        excluded_mediators=mediators,
    )
