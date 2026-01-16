"""
Emotion Composer Network

Learns to compose primitive emotions into complex emotions using attention mechanisms.
"""

import torch
import torch.nn as nn
from typing import Dict


class EmotionComposer(nn.Module):
    """
    Composes primitive emotions into complex emotions using transformer architecture.

    Learns which primitive emotions combine to form complex emotions like maternal love.
    """

    def __init__(
        self,
        num_primitives: int = 10,
        num_complex: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
    ):
        """
        Initialize emotion composer.

        Args:
            num_primitives: Number of primitive emotions
            num_complex: Number of complex emotions to learn
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()

        self.num_primitives = num_primitives
        self.num_complex = num_complex
        self.hidden_dim = hidden_dim

        # Embed primitive emotions to hidden dimension
        self.primitive_embedding = nn.Linear(num_primitives, hidden_dim)

        # Transformer layers to learn emotion relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output heads for each complex emotion
        self.complex_emotion_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_complex)
        ])

        # Learnable query tokens for each complex emotion
        self.complex_queries = nn.Parameter(torch.randn(1, num_complex, hidden_dim))

    def forward(self, primitive_emotions: torch.Tensor) -> torch.Tensor:
        """
        Compose complex emotions from primitives.

        Args:
            primitive_emotions: (batch_size, num_primitives)

        Returns:
            Complex emotions: (batch_size, num_complex)
        """
        batch_size = primitive_emotions.shape[0]

        # Embed primitives
        embedded = self.primitive_embedding(primitive_emotions)  # (batch, hidden_dim)
        embedded = embedded.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Expand complex queries for batch
        queries = self.complex_queries.expand(batch_size, -1, -1)  # (batch, num_complex, hidden_dim)

        # Concatenate queries and primitive embeddings
        transformer_input = torch.cat([queries, embedded], dim=1)  # (batch, num_complex+1, hidden_dim)

        # Process through transformer
        transformer_output = self.transformer(transformer_input)  # (batch, num_complex+1, hidden_dim)

        # Extract complex emotion representations (first num_complex tokens)
        complex_representations = transformer_output[:, :self.num_complex, :]  # (batch, num_complex, hidden_dim)

        # Compute complex emotion intensities
        complex_emotions = []
        for i, head in enumerate(self.complex_emotion_heads):
            emotion_intensity = head(complex_representations[:, i, :])  # (batch, 1)
            complex_emotions.append(emotion_intensity)

        complex_emotions = torch.cat(complex_emotions, dim=-1)  # (batch, num_complex)

        return complex_emotions

    def get_composition_attention(
        self,
        primitive_emotions: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights showing how primitives compose into complex emotions.

        Args:
            primitive_emotions: (batch_size, num_primitives)

        Returns:
            Attention weights: (batch_size, num_complex, num_primitives)
        """
        # This would require modifying the transformer to return attention weights
        # For now, return a placeholder
        batch_size = primitive_emotions.shape[0]
        return torch.ones(batch_size, self.num_complex, self.num_primitives) / self.num_primitives


class SimpleEmotionComposer(nn.Module):
    """
    Simple rule-based emotion composer for baseline comparisons.

    Uses fixed weighted combinations of primitive emotions.
    """

    def __init__(self, num_primitives: int = 10, num_complex: int = 4):
        super().__init__()

        self.num_primitives = num_primitives
        self.num_complex = num_complex

        # Fixed composition weights (can be learned)
        # These are based on psychological models
        self.register_buffer("composition_weights", self._initialize_weights())

    def _initialize_weights(self) -> torch.Tensor:
        """
        Initialize composition weights based on domain knowledge.

        Rows are complex emotions, columns are primitive emotions.
        Order: attachment, empathy, fear, joy, anger, curiosity, trust, protective_instinct, altruism, distress
        Complex: maternal_love, compassion, grief, devotion
        """
        weights = torch.zeros(self.num_complex, self.num_primitives)

        # Maternal love = attachment + empathy + protective_instinct + altruism
        weights[0, [0, 1, 7, 8]] = torch.tensor([0.35, 0.25, 0.25, 0.15])

        # Compassion = empathy + altruism + trust
        weights[1, [1, 6, 8]] = torch.tensor([0.40, 0.30, 0.30])

        # Grief = attachment + distress (when attachment is lost)
        weights[2, [0, 9]] = torch.tensor([0.50, 0.50])

        # Devotion = attachment + trust + altruism
        weights[3, [0, 6, 8]] = torch.tensor([0.40, 0.35, 0.25])

        return weights

    def forward(self, primitive_emotions: torch.Tensor) -> torch.Tensor:
        """
        Compose complex emotions using fixed weights.

        Args:
            primitive_emotions: (batch_size, num_primitives)

        Returns:
            Complex emotions: (batch_size, num_complex)
        """
        # Matrix multiplication: (batch, primitives) @ (primitives, complex).T
        complex_emotions = torch.matmul(primitive_emotions, self.composition_weights.T)
        return torch.clamp(complex_emotions, 0.0, 1.0)
