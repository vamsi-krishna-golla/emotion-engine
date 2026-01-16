"""
Social Attention Module

Attention mechanism for focusing on relevant agents in multi-agent scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SocialAttention(nn.Module):
    """
    Multi-head attention mechanism for social context.

    Allows agents to focus on specific other agents based on their emotional state.
    """

    def __init__(
        self,
        query_dim: int = 64,
        key_dim: int = 64,
        value_dim: int = 64,
        output_dim: int = 256,
        num_heads: int = 4,
    ):
        """
        Initialize social attention.

        Args:
            query_dim: Dimension of query (self state)
            key_dim: Dimension of keys (other agents)
            value_dim: Dimension of values (other agents)
            output_dim: Output dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        # Query, key, value projections
        self.query_proj = nn.Linear(query_dim, output_dim)
        self.key_proj = nn.Linear(key_dim, output_dim)
        self.value_proj = nn.Linear(value_dim, output_dim)

        # Output projection
        self.out_proj = nn.Linear(output_dim, output_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            query: Self state (batch_size, query_dim)
            keys: Other agents' states (batch_size, num_agents, key_dim)
            values: Other agents' states (batch_size, num_agents, value_dim)
            mask: Optional mask for invalid agents (batch_size, num_agents)

        Returns:
            - Attended context: (batch_size, output_dim)
            - Attention weights: (batch_size, num_agents)
        """
        batch_size, num_agents, _ = keys.shape

        # Project query, keys, values
        Q = self.query_proj(query)  # (batch, output_dim)
        K = self.key_proj(keys)  # (batch, num_agents, output_dim)
        V = self.value_proj(values)  # (batch, num_agents, output_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, 1, head_dim)
        K = K.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, agents, head_dim)
        V = V.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, agents, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, 1, agents)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, agents)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch, heads, 1, agents)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (batch, heads, 1, head_dim)

        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1)  # (batch, output_dim)
        output = self.out_proj(attended)  # (batch, output_dim)

        # Layer norm
        output = self.layer_norm(output)

        # Average attention weights across heads for interpretability
        attention_weights = attention_weights.mean(dim=1).squeeze(2)  # (batch, agents)

        return output, attention_weights


class EmpathyAttention(nn.Module):
    """
    Specialized attention for empathy-based focus.

    Focuses more on agents that are in distress or need help.
    """

    def __init__(
        self,
        emotion_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 256,
    ):
        """
        Initialize empathy attention.

        Args:
            emotion_dim: Dimension of emotion state
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()

        # Encode self emotion state
        self.self_encoder = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
        )

        # Encode other agents' emotion states
        self.other_encoder = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
        )

        # Compute empathy-based attention
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        self_emotion: torch.Tensor,
        other_emotions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute empathy-based attention.

        Args:
            self_emotion: (batch_size, emotion_dim)
            other_emotions: (batch_size, num_agents, emotion_dim)
            mask: Optional mask (batch_size, num_agents)

        Returns:
            - Attended context: (batch_size, output_dim)
            - Attention weights: (batch_size, num_agents)
        """
        batch_size, num_agents, _ = other_emotions.shape

        # Encode emotions
        self_encoded = self.self_encoder(self_emotion)  # (batch, hidden)
        other_encoded = self.other_encoder(other_emotions)  # (batch, agents, hidden)

        # Expand self_encoded for concatenation
        self_expanded = self_encoded.unsqueeze(1).expand(-1, num_agents, -1)  # (batch, agents, hidden)

        # Concatenate self and other
        combined = torch.cat([self_expanded, other_encoded], dim=-1)  # (batch, agents, 2*hidden)

        # Compute attention scores
        attention_logits = self.attention_net(combined).squeeze(-1)  # (batch, agents)

        # Apply mask if provided
        if mask is not None:
            attention_logits = attention_logits.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(attention_logits, dim=-1)  # (batch, agents)

        # Apply attention to other_encoded
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (batch, agents, 1)
        attended = (other_encoded * attention_weights_expanded).sum(dim=1)  # (batch, hidden)

        # Project to output dimension
        output = self.output_proj(attended)  # (batch, output_dim)

        return output, attention_weights
