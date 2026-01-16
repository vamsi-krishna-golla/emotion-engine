"""Training infrastructure."""

from emotion_engine.training.trainer import Trainer
from emotion_engine.training.curriculum import CurriculumScheduler

__all__ = ["Trainer", "CurriculumScheduler"]
