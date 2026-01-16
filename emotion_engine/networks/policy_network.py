"""
Policy Network (Actor)

Emotion-conditioned policy for action selection.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Actor network that outputs action distribution conditioned on emotion state.

    For continuous action spaces, outputs mean and log_std.
    """

    def__init__(
        self,
        emotion_features_dim: int = 256,
        emotion_state_dim: int = 16,
        relationship_dim: int = 64,
        action_dim: int = 4,
        hidden_dim: int = 256,
        continuous: bool = True,
    ):
        """
        Initialize policy network.

        Args:
            emotion_features_dim: Dimension of emotion encoder output
            emotion_state_dim: Dimension of emotion state vector
            relationship_dim: Dimension of relationship context
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            continuous: Whether action space is continuous
        """
        super().__init__()

        self.action_dim = action_dim
        self.continuous = continuous

        # Input combines emotion features, emotion state, and relationship context
        input_dim = emotion_features_dim + emotion_state_dim + relationship_dim

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        if continuous:
            # Continuous actions: output mean and log_std
            self.mean_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh(),  # Bound actions to [-1, 1]
            )

            self.log_std_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )
        else:
            # Discrete actions: output logits
            self.action_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )

    def forward(
        self,
        emotion_features: torch.Tensor,
        emotion_state: torch.Tensor,
        relationship_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            emotion_features: (batch, emotion_features_dim)
            emotion_state: (batch, emotion_state_dim)
            relationship_context: (batch, relationship_dim)

        Returns:
            If continuous:
                - Action means: (batch, action_dim)
                - Action log_stds: (batch, action_dim)
            If discrete:
                - Action logits: (batch, action_dim)
                - None
        """
        # Concatenate inputs
        x = torch.cat([emotion_features, emotion_state, relationship_context], dim=-1)

        # Shared processing
        features = self.shared(x)

        if self.continuous:
            # Output mean and log_std
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            return mean, log_std
        else:
            # Output logits
            logits = self.action_head(features)
            return logits, None

    def get_action_distribution(
        self,
        emotion_features: torch.Tensor,
        emotion_state: torch.Tensor,
        relationship_context: torch.Tensor,
    ):
        """
        Get action distribution for sampling.

        Returns:
            torch.distributions.Distribution
        """
        if self.continuous:
            mean, log_std = self.forward(emotion_features, emotion_state, relationship_context)
            std = torch.exp(log_std)
            return Normal(mean, std)
        else:
            logits, _ = self.forward(emotion_features, emotion_state, relationship_context)
            return Categorical(logits=logits)

    def sample_action(
        self,
        emotion_features: torch.Tensor,
        emotion_state: torch.Tensor,
        relationship_context: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            emotion_features, emotion_state, relationship_context: As in forward()
            deterministic: If True, return mean (continuous) or argmax (discrete)

        Returns:
            - Actions: (batch, action_dim)
            - Log probabilities: (batch,)
        """
        dist = self.get_action_distribution(emotion_features, emotion_state, relationship_context)

        if deterministic:
            if self.continuous:
                actions = dist.mean
            else:
                actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)

        if self.continuous:
            # Sum log probs across action dimensions
            log_probs = log_probs.sum(dim=-1)

        return actions, log_probs

    def evaluate_actions(
        self,
        emotion_features: torch.Tensor,
        emotion_state: torch.Tensor,
        relationship_context: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.

        Args:
            emotion_features, emotion_state, relationship_context: As in forward()
            actions: Actions to evaluate (batch, action_dim)

        Returns:
            - Log probabilities: (batch,)
            - Entropy: (batch,)
        """
        dist = self.get_action_distribution(emotion_features, emotion_state, relationship_context)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        if self.continuous:
            log_probs = log_probs.sum(dim=-1)
            entropy = entropy.sum(dim=-1)

        return log_probs, entropy
