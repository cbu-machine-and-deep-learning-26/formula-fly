"""Training loops and the shared evaluation harness.

Everything exported here is torch-free, so the base install and the evaluation harness can
import the package. The PPO loop itself (:mod:`fly_driver.training.train`,
:mod:`fly_driver.training.ppo`) needs torch and is imported by its full module path.
"""

from fly_driver.training.config import (
    DEFAULT_SEEDS,
    KNOWN_BRAINS,
    KNOWN_ENV_TYPES,
    KNOWN_EYE_TYPES,
    EnvConfig,
    EyeConfig,
    LoggingConfig,
    PeriodicEvalConfig,
    PolicyConfig,
    PPOConfig,
    TrainConfig,
    WandbConfig,
)
from fly_driver.training.evaluation import (
    CLEAN_CONDITION,
    ConditionSummary,
    EpisodeRecord,
    EvalConfig,
    EvalReport,
    evaluate,
    seed_everything,
)
from fly_driver.training.loggers import (
    CsvLogger,
    RunLogger,
    TrainingEpisode,
    UpdateRecord,
    WandbLogger,
)
from fly_driver.training.perturbations import PERTURBATIONS, PerturbationSpec
from fly_driver.training.video import available_backend, open_video

__all__ = [
    "CLEAN_CONDITION",
    "DEFAULT_SEEDS",
    "KNOWN_BRAINS",
    "KNOWN_ENV_TYPES",
    "KNOWN_EYE_TYPES",
    "PERTURBATIONS",
    "ConditionSummary",
    "CsvLogger",
    "EnvConfig",
    "EpisodeRecord",
    "EvalConfig",
    "EvalReport",
    "EyeConfig",
    "LoggingConfig",
    "PPOConfig",
    "PeriodicEvalConfig",
    "PerturbationSpec",
    "PolicyConfig",
    "RunLogger",
    "TrainConfig",
    "TrainingEpisode",
    "UpdateRecord",
    "WandbConfig",
    "WandbLogger",
    "available_backend",
    "evaluate",
    "open_video",
    "seed_everything",
]
