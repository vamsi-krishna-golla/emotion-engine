"""Training infrastructure."""

from emotion_engine.training.trainer import Trainer
from emotion_engine.training.curriculum import CurriculumScheduler, CurriculumStage, create_simple_curriculum

__all__ = [
    "Trainer",
    "CurriculumScheduler",
    "CurriculumStage",
    "create_simple_curriculum",
]
