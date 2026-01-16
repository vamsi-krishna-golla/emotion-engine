"""
Value Network (Critic)

Estimates state value for policy optimization.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Critic network that estimates state value conditioned on emotion state.
    """

    def __init__(
        self,
        emotion_features_dim: int = 256,
        emotion_state_dim: int = 16,
        hidden_dim: int = 256,
    ):
        """
        Initialize value network.

        Args:
            emotion_features_dim: Dimension of emotion encoder output
            emotion_state_dim: Dimension of emotion state vector
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        input_dim = emotion_features_dim + emotion_state_dim

        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        emotion_features: torch.Tensor,
        emotion_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            emotion_features: (batch, emotion_features_dim)
            emotion_state: (batch, emotion_state_dim)

        Returns:
            State values: (batch, 1)
        """
        x = torch.cat([emotion_features, emotion_state], dim=-1)
        value = self.value_net(x)
        return value


class DualValueNetwork(nn.Module):
    """
    Dual value network that estimates both extrinsic and intrinsic value.

    Useful for separating task rewards from emotional rewards.
    """

    def __init__(
        self,
        emotion_features_dim: int = 256,
        emotion_state_dim: int = 16,
        hidden_dim: int = 256,
    ):
        """
        Initialize dual value network.

        Args:
            emotion_features_dim: Dimension of emotion encoder output
            emotion_state_dim: Dimension of emotion state vector
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        input_dim = emotion_features_dim + emotion_state_dim

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Extrinsic value head (task rewards)
        self.extrinsic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Intrinsic value head (emotional rewards)
        self.intrinsic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        emotion_features: torch.Tensor,
        emotion_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            emotion_features: (batch, emotion_features_dim)
            emotion_state: (batch, emotion_state_dim)

        Returns:
            - Extrinsic value: (batch, 1)
            - Intrinsic value: (batch, 1)
        """
        x = torch.cat([emotion_features, emotion_state], dim=-1)
        features = self.shared(x)

        extrinsic_value = self.extrinsic_head(features)
        intrinsic_value = self.intrinsic_head(features)

        return extrinsic_value, intrinsic_value

    def get_total_value(
        self,
        emotion_features: torch.Tensor,
        emotion_state: torch.Tensor,
        extrinsic_weight: float = 1.0,
        intrinsic_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Get weighted total value.

        Args:
            emotion_features, emotion_state: As in forward()
            extrinsic_weight: Weight for extrinsic value
            intrinsic_weight: Weight for intrinsic value

        Returns:
            Total value: (batch, 1)
        """
        extrinsic, intrinsic = self.forward(emotion_features, emotion_state)
        return extrinsic * extrinsic_weight + intrinsic * intrinsic_weight
