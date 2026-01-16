"""
Proximal Policy Optimization (PPO) Algorithm

Implements PPO with clipped objective for stable policy updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import numpy as np

from emotion_engine.rl.replay_buffer import RolloutBuffer


class PPO:
    """
    Proximal Policy Optimization algorithm.

    Features:
    - Clipped surrogate objective
    - Value function loss
    - Entropy bonus for exploration
    - Gradient clipping for stability
    """

    def __init__(
        self,
        policy_network: nn.Module,
        value_network: nn.Module,
        learning_rate: float = 3e-4,
        n_epochs: int = 10,
        batch_size: int = 64,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize PPO.

        Args:
            policy_network: Actor network
            value_network: Critic network
            learning_rate: Learning rate for optimizer
            n_epochs: Number of epochs per update
            batch_size: Batch size for training
            clip_range: PPO clip range (epsilon)
            vf_coef: Value function loss coefficient
            ent_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.policy = policy_network
        self.value_net = value_network

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Optimizer for both policy and value networks
        params = list(self.policy.parameters()) + list(self.value_net.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def update(
        self,
        rollout_buffer: RolloutBuffer,
        emotion_encoder: nn.Module,
    ) -> Dict[str, float]:
        """
        Update policy and value network using collected rollouts.

        Args:
            rollout_buffer: Buffer containing trajectories
            emotion_encoder: Emotion encoder network

        Returns:
            Dictionary of training metrics
        """
        # Compute returns and advantages
        # For now, use last value as 0 (can be improved with bootstrapping)
        rollout_buffer.compute_returns_and_advantages(last_value=0.0)

        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kl_divs = []

        # Multiple epochs over the same data
        for epoch in range(self.n_epochs):
            # Get data in batches
            data = rollout_buffer.get(batch_size=self.batch_size)

            observations = data['observations']
            actions = data['actions']
            old_log_probs = data['old_log_probs']
            advantages = data['advantages']
            returns = data['returns']
            emotion_states = data['emotion_states']

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Encode observations to emotion features
            # Note: This is simplified - in practice, need to split observations
            # For now, assume observations are already emotion features
            emotion_features = observations  # Placeholder

            # Dummy relationship context (should come from agent state)
            relationship_context = torch.zeros(observations.shape[0], 64)

            # Evaluate actions with current policy
            log_probs, entropy = self.policy.evaluate_actions(
                emotion_features,
                emotion_states,
                relationship_context,
                actions
            )

            # Get value estimates
            values = self.value_net(emotion_features, emotion_states).squeeze(-1)

            # Policy loss (clipped surrogate objective)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped value function)
            value_loss = 0.5 * ((values - returns) ** 2).mean()

            # Entropy loss (for exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_net.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()

            # Logging
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())

            # Approximate KL divergence
            with torch.no_grad():
                approx_kl_div = ((ratio - 1) - torch.log(ratio)).mean()
                approx_kl_divs.append(approx_kl_div.item())

        # Return average metrics
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'approx_kl': np.mean(approx_kl_divs),
        }

    def save(self, path: str):
        """Save networks and optimizer state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load networks and optimizer state."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class PPOConfig:
    """Configuration for PPO algorithm."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        """Initialize PPO configuration."""
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PPOConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'vf_coef': self.vf_coef,
            'ent_coef': self.ent_coef,
            'max_grad_norm': self.max_grad_norm,
        }
