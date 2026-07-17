"""Two-tower session-to-device uplift distillation (reproducibility artifact).

Public API for building the synthetic world, preparing features, training the
two-tower uplift model and NN baselines, and evaluating with AUCC / normalized
Qini.  See ``demos/`` for the four paper arguments reproduced end-to-end.
"""

from .aggregation import aggregate_to_device, broadcast_device_scores
from .diagnostics import balance_report
from .dgp import (
    DGPConfig,
    SyntheticData,
    generate,
    stratified_zscore,
)
from .evaluation import compute_aucc, normalized_qini, uplift_curve
from .features import Encoders, Tensors, fit_encoders, transform
from .model import (
    BaselineConfig,
    DragonNet,
    TARNet,
    TwoTowerConfig,
    TwoTowerUpliftModel,
)
from .trainer import (
    TrainConfig,
    predict_device_uplift_two_tower,
    predict_session_uplift_two_tower,
    predict_uplift_baseline,
    train_baseline,
    train_two_tower,
)

__all__ = [
    "DGPConfig",
    "SyntheticData",
    "generate",
    "stratified_zscore",
    "Encoders",
    "Tensors",
    "fit_encoders",
    "transform",
    "TwoTowerConfig",
    "TwoTowerUpliftModel",
    "BaselineConfig",
    "TARNet",
    "DragonNet",
    "TrainConfig",
    "train_two_tower",
    "train_baseline",
    "predict_session_uplift_two_tower",
    "predict_device_uplift_two_tower",
    "predict_uplift_baseline",
    "compute_aucc",
    "normalized_qini",
    "uplift_curve",
    "aggregate_to_device",
    "broadcast_device_scores",
    "balance_report",
]
