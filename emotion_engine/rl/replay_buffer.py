"""
Replay Buffer for storing and sampling experiences.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Experience:
    """Single experience tuple."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    log_prob: float
    value: float
    emotion_state: np.ndarray


class RolloutBuffer:
    """
    Buffer for collecting trajectories for on-policy algorithms like PPO.

    Stores complete episodes and computes advantages using GAE.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        emotion_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            observation_dim: Dimension of observations
            action_dim: Dimension of actions
            emotion_dim: Dimension of emotion state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.emotion_dim = emotion_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.reset()

    def reset(self):
        """Reset buffer to empty state."""
        self.observations = np.zeros((self.buffer_size, self.observation_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.bool_)
        self.emotion_states = np.zeros((self.buffer_size, self.emotion_dim), dtype=np.float32)

        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        emotion_state: np.ndarray,
    ):
        """Add a transition to the buffer."""
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        self.emotion_states[self.pos] = emotion_state

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(self, last_value: float):
        """
        Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for the last state (for bootstrapping)
        """
        n_steps = self.buffer_size if self.full else self.pos

        last_gae_lam = 0
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

        self.returns = self.advantages + self.values

    def get(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer as PyTorch tensors.

        Args:
            batch_size: If provided, return random batches instead of full buffer

        Returns:
            Dictionary of tensors
        """
        n_steps = self.buffer_size if self.full else self.pos

        if batch_size is None:
            indices = np.arange(n_steps)
        else:
            indices = np.random.randint(0, n_steps, size=batch_size)

        return {
            'observations': torch.from_numpy(self.observations[indices]),
            'actions': torch.from_numpy(self.actions[indices]),
            'old_log_probs': torch.from_numpy(self.log_probs[indices]),
            'advantages': torch.from_numpy(self.advantages[indices]),
            'returns': torch.from_numpy(self.returns[indices]),
            'emotion_states': torch.from_numpy(self.emotion_states[indices]),
        }

    def size(self) -> int:
        """Return current size of buffer."""
        return self.buffer_size if self.full else self.pos
