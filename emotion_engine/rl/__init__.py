"""Reinforcement learning framework."""

from emotion_engine.rl.agent import EmotionalAgent
from emotion_engine.rl.ppo import PPO, PPOConfig
from emotion_engine.rl.reward_shaping import RewardShaper, RewardComponents, CurriculumRewardScheduler
from emotion_engine.rl.replay_buffer import RolloutBuffer

__all__ = [
    "EmotionalAgent",
    "PPO",
    "PPOConfig",
    "RewardShaper",
    "RewardComponents",
    "CurriculumRewardScheduler",
    "RolloutBuffer",
]
