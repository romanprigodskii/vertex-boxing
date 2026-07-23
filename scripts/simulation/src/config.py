"""Static config for the boxing model — version, temporal split, params.

Differences from the vertexmma model this is ported from:
  * THREE-outcome target (win / draw / loss). Boxing draws are frequent enough
    that a binary head biases both the ratings and the market comparison.
  * NO per-round punch features (CompuBox is proprietary) — so the online
    opponent-adjusted attack/defense skill ratings are dropped. The signal is
    partially recovered from method-of-result, opponent-adjusted KO-for/
    KO-against, and rounds-per-bout.
  * Success is judged on CLOSING-LINE VALUE on the competitive subset
    (market-implied 30–70 %), NOT accuracy — the ~90–95 % favorite base rate
    on regional cards makes accuracy meaningless. See scripts/run_killtest.py.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
DATA_DIR = PACKAGE_ROOT / "data"
ARTIFACTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

MODEL_VERSION = "v0.1.0"

# Temporal split anchors — updated as data accrues. Bouts before TRAIN_END →
# train, [TRAIN_END, VAL_END) → validation, ≥ VAL_END → held-out test.
TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"

# Multiclass LightGBM (win_a / draw / win_b). Tune once real data lands.
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": ["multi_logloss"],
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
}
LGB_NUM_ROUNDS = 2000
LGB_EARLY_STOPPING_ROUNDS = 100

# The kill-test only scores bouts the market thought were competitive.
COMPETITIVE_PROB_LOW = 0.30
COMPETITIVE_PROB_HIGH = 0.70

# Confidence bands from max class prob (display only).
CONFIDENCE_BANDS = (("low", 0.0, 0.45), ("medium", 0.45, 0.60), ("high", 0.60, 1.0))


def confidence_label(top_prob: float) -> str:
    for label, lo, hi in CONFIDENCE_BANDS:
        if lo <= top_prob < hi:
            return label
    return "high"
