"""
Emotion Encoder Network

Encodes raw observations into emotion-relevant features.
Multi-headed architecture processes self-state, social context, and environment separately.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class EmotionEncoder(nn.Module):
    """
    Encodes observations into emotion feature representations.

    Architecture:
    - Self-state branch: Processes agent's own state
    - Social branch: Processes other agents' states with attention
    - Environment branch: Processes environmental features
    - Fusion layer: Combines all branches into unified feature vector
    """

    def __init__(
        self,
        self_state_dim: int = 32,
        social_dim: int = 64,
        env_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 256,
        max_agents: int = 10,
    ):
        """
        Initialize emotion encoder.

        Args:
            self_state_dim: Dimension of self-state observations
            social_dim: Dimension of per-agent social observations
            env_dim: Dimension of environment observations
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            max_agents: Maximum number of other agents to process
        """
        super().__init__()

        self.self_state_dim = self_state_dim
        self.social_dim = social_dim
        self.env_dim = env_dim
        self.output_dim = output_dim
        self.max_agents = max_agents

        # Self-state branch (MLP)
        self.self_encoder = nn.Sequential(
            nn.Linear(self_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Social branch (processes multiple agents)
        self.social_encoder = nn.Sequential(
            nn.Linear(social_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Simple attention mechanism for social context
        self.social_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Environment branch
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Fusion layer
        fusion_input_dim = 256 + 128 + 128  # self + social + env
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        self_state: torch.Tensor,
        social_context: torch.Tensor,
        environment: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            self_state: (batch_size, self_state_dim)
            social_context: (batch_size, max_agents, social_dim)
            environment: (batch_size, env_dim)

        Returns:
            Emotion features: (batch_size, output_dim)
        """
        batch_size = self_state.shape[0]

        # Process self-state
        self_features = self.self_encoder(self_state)  # (batch, 256)

        # Process social context with attention
        # social_context: (batch, max_agents, social_dim)
        social_encoded = self.social_encoder(social_context)  # (batch, max_agents, 128)

        # Compute attention weights
        attention_logits = self.social_attention(social_encoded)  # (batch, max_agents, 1)
        attention_weights = torch.softmax(attention_logits, dim=1)  # (batch, max_agents, 1)

        # Weighted sum of social features
        social_features = (social_encoded * attention_weights).sum(dim=1)  # (batch, 128)

        # Process environment
        env_features = self.env_encoder(environment)  # (batch, 128)

        # Fuse all features
        combined = torch.cat([self_features, social_features, env_features], dim=-1)
        emotion_features = self.fusion(combined)  # (batch, output_dim)

        return emotion_features

    def get_attention_weights(
        self,
        social_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for social context (for visualization).

        Args:
            social_context: (batch_size, max_agents, social_dim)

        Returns:
            Attention weights: (batch_size, max_agents)
        """
        social_encoded = self.social_encoder(social_context)
        attention_logits = self.social_attention(social_encoded)
        attention_weights = torch.softmax(attention_logits, dim=1)
        return attention_weights.squeeze(-1)


class SimplifiedEmotionEncoder(nn.Module):
    """
    Simplified emotion encoder for testing and baseline comparisons.

    Single MLP that processes concatenated observations.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        """
        Initialize simplified encoder.

        Args:
            obs_dim: Total observation dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: (batch_size, obs_dim)

        Returns:
            Features: (batch_size, output_dim)
        """
        return self.encoder(obs)
