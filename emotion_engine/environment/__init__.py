"""Multi-agent simulation environment."""

from emotion_engine.environment.base_env import BaseEmotionEnv
from emotion_engine.environment.scenarios.caretaking import CaretakingScenario
from emotion_engine.environment.scenarios.crisis import CrisisScenario

__all__ = [
    "BaseEmotionEnv",
    "CaretakingScenario",
    "CrisisScenario",
]
