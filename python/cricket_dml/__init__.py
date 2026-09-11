"""Analysis-frame construction for DoubleML estimation on T20 ball-by-ball data.

Re-exports the public surface of ``cricket_dml.frame`` so callers can import
from the package directly:

    from cricket_dml import build_frame, load_deliveries

Declaring the package explicitly also removes the dependency on
namespace-package resolution, which only held while ``run_apos.py`` was the
sole entry point and put the parent directory on ``sys.path`` itself.
"""

from .frame import (
    BALLS_PER_INNINGS,
    CLUSTER_COL,
    D_COL,
    MAX_CATEGORY_LEVELS,
    NEVER_CONTROLS,
    TREATMENT_SPECS,
    Y_COL,
    AnalysisFrame,
    TreatmentSpec,
    add_match_state,
    build_frame,
    load_deliveries,
    position_variation,
)

__all__ = [
    "Y_COL",
    "D_COL",
    "CLUSTER_COL",
    "NEVER_CONTROLS",
    "MAX_CATEGORY_LEVELS",
    "BALLS_PER_INNINGS",
    "TreatmentSpec",
    "TREATMENT_SPECS",
    "AnalysisFrame",
    "load_deliveries",
    "add_match_state",
    "build_frame",
    "position_variation",
]
