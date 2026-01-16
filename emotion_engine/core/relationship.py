"""
Relationship Module

Tracks relationships and attachment bonds between agents.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import time


@dataclass
class InteractionMemory:
    """Record of a single interaction between agents."""

    timestamp: float
    interaction_type: str  # "help", "harm", "share", "protect", etc.
    outcome: float  # Positive/negative value
    emotion_state: Optional[np.ndarray] = None  # Agent's emotions during interaction


@dataclass
class Relationship:
    """
    Represents a relationship between two agents.

    Tracks attachment strength, interaction history, and bond dynamics.
    """

    agent_id: int
    other_agent_id: int
    attachment_level: float = 0.1  # Attachment strength (0-1)
    trust_level: float = 0.1  # Trust level (0-1)
    interaction_count: int = 0
    last_interaction_time: float = field(default_factory=time.time)
    memory: List[InteractionMemory] = field(default_factory=list)
    memory_capacity: int = 100  # Max interactions to remember

    def update_attachment(
        self,
        interaction_type: str,
        outcome: float,
        decay_rate: float = 0.001,
        delta_time: float = 1.0
    ):
        """
        Update attachment level based on interaction.

        Args:
            interaction_type: Type of interaction
            outcome: Positive/negative outcome value
            decay_rate: Rate of attachment decay over time
            delta_time: Time since last interaction
        """
        # Apply temporal decay first
        time_elapsed = delta_time
        self.attachment_level *= (1.0 - decay_rate * time_elapsed)

        # Interaction type weights for attachment
        interaction_weights = {
            "help": 0.15,
            "share": 0.20,
            "protect": 0.25,
            "proximity": 0.05,
            "cooperation": 0.15,
            "harm": -0.20,
            "abandon": -0.15,
        }

        weight = interaction_weights.get(interaction_type, 0.05)

        # Update attachment (positive outcomes strengthen, negative weaken)
        delta_attachment = weight * outcome

        # Attachments are easier to form early, harder to strengthen when already strong
        if delta_attachment > 0:
            delta_attachment *= (1.0 - self.attachment_level)

        self.attachment_level += delta_attachment
        self.attachment_level = np.clip(self.attachment_level, 0.0, 1.0)

        # Update trust similarly
        if outcome > 0:
            self.trust_level += 0.1 * outcome * (1.0 - self.trust_level)
        else:
            self.trust_level += 0.2 * outcome  # Trust decreases faster

        self.trust_level = np.clip(self.trust_level, 0.0, 1.0)

        # Update interaction count and time
        self.interaction_count += 1
        self.last_interaction_time = time.time()

    def add_memory(self, interaction_type: str, outcome: float, emotion_state: Optional[np.ndarray] = None):
        """
        Add an interaction to episodic memory.

        Args:
            interaction_type: Type of interaction
            outcome: Outcome value
            emotion_state: Agent's emotion state during interaction
        """
        memory = InteractionMemory(
            timestamp=time.time(),
            interaction_type=interaction_type,
            outcome=outcome,
            emotion_state=emotion_state
        )

        self.memory.append(memory)

        # Prune old memories if capacity exceeded
        if len(self.memory) > self.memory_capacity:
            self.memory = self.memory[-self.memory_capacity:]

    def get_recent_interactions(self, n: int = 10) -> List[InteractionMemory]:
        """Get the n most recent interactions."""
        return self.memory[-n:]

    def get_interaction_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of interaction history.

        Returns:
            Dictionary with summary stats
        """
        if not self.memory:
            return {
                "total_interactions": 0,
                "positive_ratio": 0.0,
                "avg_outcome": 0.0,
            }

        outcomes = [m.outcome for m in self.memory]
        positive_count = sum(1 for o in outcomes if o > 0)

        return {
            "total_interactions": len(self.memory),
            "positive_ratio": positive_count / len(self.memory),
            "avg_outcome": np.mean(outcomes),
            "recent_outcome": np.mean([m.outcome for m in self.memory[-10:]]) if self.memory else 0.0,
        }

    def decay_over_time(self, current_time: float, decay_rate: float = 0.001):
        """
        Apply temporal decay to attachment when no interactions occur.

        Args:
            current_time: Current timestamp
            decay_rate: Rate of decay per time unit
        """
        time_since_last = current_time - self.last_interaction_time
        self.attachment_level *= (1.0 - decay_rate * time_since_last)
        self.attachment_level = max(0.0, self.attachment_level)

    def __repr__(self) -> str:
        return (
            f"Relationship(agent={self.agent_id}, other={self.other_agent_id}, "
            f"attachment={self.attachment_level:.2f}, trust={self.trust_level:.2f}, "
            f"interactions={self.interaction_count})"
        )


class RelationshipManager:
    """
    Manages all relationships for an agent.

    Handles multiple relationships and provides convenient access methods.
    """

    def __init__(self, agent_id: int, decay_rate: float = 0.001):
        """
        Initialize relationship manager.

        Args:
            agent_id: ID of the agent whose relationships we're managing
            decay_rate: Default decay rate for attachments
        """
        self.agent_id = agent_id
        self.decay_rate = decay_rate
        self.relationships: Dict[int, Relationship] = {}

    def get_or_create_relationship(self, other_agent_id: int) -> Relationship:
        """
        Get existing relationship or create a new one.

        Args:
            other_agent_id: ID of the other agent

        Returns:
            Relationship object
        """
        if other_agent_id not in self.relationships:
            self.relationships[other_agent_id] = Relationship(
                agent_id=self.agent_id,
                other_agent_id=other_agent_id
            )

        return self.relationships[other_agent_id]

    def update_relationship(
        self,
        other_agent_id: int,
        interaction_type: str,
        outcome: float,
        emotion_state: Optional[np.ndarray] = None,
        delta_time: float = 1.0
    ):
        """
        Update relationship based on interaction.

        Args:
            other_agent_id: ID of the other agent
            interaction_type: Type of interaction
            outcome: Outcome value
            emotion_state: Agent's emotion state during interaction
            delta_time: Time since last update
        """
        relationship = self.get_or_create_relationship(other_agent_id)
        relationship.update_attachment(interaction_type, outcome, self.decay_rate, delta_time)
        relationship.add_memory(interaction_type, outcome, emotion_state)

    def get_attachment_level(self, other_agent_id: int) -> float:
        """Get attachment level to another agent."""
        if other_agent_id not in self.relationships:
            return 0.0
        return self.relationships[other_agent_id].attachment_level

    def get_trust_level(self, other_agent_id: int) -> float:
        """Get trust level toward another agent."""
        if other_agent_id not in self.relationships:
            return 0.0
        return self.relationships[other_agent_id].trust_level

    def get_strongest_attachment(self) -> Tuple[Optional[int], float]:
        """
        Get the agent we're most attached to.

        Returns:
            Tuple of (agent_id, attachment_level)
        """
        if not self.relationships:
            return None, 0.0

        strongest = max(
            self.relationships.items(),
            key=lambda x: x[1].attachment_level
        )

        return strongest[0], strongest[1].attachment_level

    def get_all_attachments(self) -> Dict[int, float]:
        """
        Get all attachment levels.

        Returns:
            Dictionary mapping agent_id -> attachment_level
        """
        return {
            agent_id: rel.attachment_level
            for agent_id, rel in self.relationships.items()
        }

    def decay_all_relationships(self, current_time: float):
        """Apply temporal decay to all relationships."""
        for relationship in self.relationships.values():
            relationship.decay_over_time(current_time, self.decay_rate)

    def get_relationship_summary(self) -> Dict[str, any]:
        """
        Get summary of all relationships.

        Returns:
            Dictionary with summary statistics
        """
        if not self.relationships:
            return {
                "num_relationships": 0,
                "strongest_attachment": 0.0,
                "avg_attachment": 0.0,
                "total_interactions": 0,
            }

        attachments = [rel.attachment_level for rel in self.relationships.values()]
        interactions = [rel.interaction_count for rel in self.relationships.values()]

        return {
            "num_relationships": len(self.relationships),
            "strongest_attachment": max(attachments),
            "avg_attachment": np.mean(attachments),
            "total_interactions": sum(interactions),
        }

    def __repr__(self) -> str:
        summary = self.get_relationship_summary()
        return (
            f"RelationshipManager(agent={self.agent_id}, "
            f"relationships={summary['num_relationships']}, "
            f"strongest_attachment={summary['strongest_attachment']:.2f})"
        )
