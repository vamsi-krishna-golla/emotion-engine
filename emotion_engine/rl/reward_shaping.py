"""
Reward Shaping Module

Implements intrinsic emotional rewards that encourage prosocial behaviors and self-sacrifice.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Container for different reward components."""
    extrinsic: float = 0.0  # Task rewards (survival, offspring survival)
    empathy: float = 0.0  # Reward for helping others
    attachment: float = 0.0  # Reward for maintaining bonds
    altruism: float = 0.0  # Reward for self-sacrifice
    protective: float = 0.0  # Reward for protecting vulnerable agents
    social: float = 0.0  # Reward for social interactions

    def total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted total reward."""
        if weights is None:
            weights = {
                'extrinsic': 1.0,
                'empathy': 0.5,
                'attachment': 0.3,
                'altruism': 0.4,
                'protective': 0.4,
                'social': 0.2,
            }

        return (
            self.extrinsic * weights['extrinsic'] +
            self.empathy * weights['empathy'] +
            self.attachment * weights['attachment'] +
            self.altruism * weights['altruism'] +
            self.protective * weights['protective'] +
            self.social * weights['social']
        )


class RewardShaper:
    """
    Shapes rewards to encourage emotional and prosocial behaviors.

    Combines extrinsic task rewards with intrinsic emotional rewards.
    """

    def __init__(
        self,
        empathy_weight: float = 0.5,
        attachment_weight: float = 0.3,
        altruism_weight: float = 0.4,
        protective_weight: float = 0.4,
        social_weight: float = 0.2,
        altruism_threshold: float = 1.5,
    ):
        """
        Initialize reward shaper.

        Args:
            empathy_weight: Weight for empathy rewards
            attachment_weight: Weight for attachment rewards
            altruism_weight: Weight for altruism rewards
            protective_weight: Weight for protective rewards
            social_weight: Weight for social interaction rewards
            altruism_threshold: Minimum benefit/cost ratio for altruism reward
        """
        self.weights = {
            'extrinsic': 1.0,
            'empathy': empathy_weight,
            'attachment': attachment_weight,
            'altruism': altruism_weight,
            'protective': protective_weight,
            'social': social_weight,
        }
        self.altruism_threshold = altruism_threshold

    def compute_reward(
        self,
        agent_id: int,
        extrinsic_reward: float,
        emotion_state: np.ndarray,
        agent_states: Dict[int, Dict],
        attachments: Dict[int, float],
        action_info: Dict,
    ) -> RewardComponents:
        """
        Compute total reward with emotional shaping.

        Args:
            agent_id: ID of the agent
            extrinsic_reward: Base task reward
            emotion_state: Agent's current emotion state vector
            agent_states: Dictionary of other agents' states
            attachments: Dictionary of attachment levels to other agents
            action_info: Information about the action taken

        Returns:
            RewardComponents with all reward components
        """
        rewards = RewardComponents(extrinsic=extrinsic_reward)

        # Extract emotion values (assuming standard order)
        # Order: attachment, empathy, fear, joy, anger, curiosity, trust,
        #        protective_instinct, altruism, distress, [complex emotions...]
        empathy_level = emotion_state[1] if len(emotion_state) > 1 else 0.0
        altruism_level = emotion_state[8] if len(emotion_state) > 8 else 0.0
        protective_level = emotion_state[7] if len(emotion_state) > 7 else 0.0

        # Compute empathy reward
        rewards.empathy = self._compute_empathy_reward(
            empathy_level, agent_states, attachments
        )

        # Compute attachment reward
        rewards.attachment = self._compute_attachment_reward(attachments)

        # Compute altruism reward
        rewards.altruism = self._compute_altruism_reward(
            altruism_level, action_info, agent_states, attachments
        )

        # Compute protective reward
        rewards.protective = self._compute_protective_reward(
            protective_level, action_info, agent_states, attachments
        )

        # Compute social reward
        rewards.social = self._compute_social_reward(action_info, attachments)

        return rewards

    def _compute_empathy_reward(
        self,
        empathy_level: float,
        agent_states: Dict[int, Dict],
        attachments: Dict[int, float],
    ) -> float:
        """
        Reward for empathetic responses to others' states.

        Higher reward when helping agents we're attached to.
        """
        if not agent_states or not attachments:
            return 0.0

        empathy_reward = 0.0

        for other_id, other_state in agent_states.items():
            if other_id not in attachments:
                continue

            attachment_level = attachments[other_id]

            # Get other agent's wellbeing change (if available)
            wellbeing_delta = other_state.get('wellbeing_delta', 0.0)

            # Reward proportional to empathy, attachment, and other's wellbeing improvement
            if wellbeing_delta > 0:
                empathy_reward += empathy_level * attachment_level * wellbeing_delta

        return empathy_reward

    def _compute_attachment_reward(self, attachments: Dict[int, float]) -> float:
        """
        Reward for maintaining attachment bonds.

        Small positive reward for having strong attachments.
        """
        if not attachments:
            return 0.0

        # Reward is proportional to squared attachment (encourages strong bonds)
        attachment_reward = sum(att ** 2 for att in attachments.values()) * 0.1

        return attachment_reward

    def _compute_altruism_reward(
        self,
        altruism_level: float,
        action_info: Dict,
        agent_states: Dict[int, Dict],
        attachments: Dict[int, float],
    ) -> float:
        """
        Reward for self-sacrifice that helps others.

        Only rewards when benefit to others exceeds cost to self.
        """
        self_cost = action_info.get('self_cost', 0.0)

        if self_cost <= 0:
            return 0.0  # No sacrifice, no altruism reward

        total_other_benefit = 0.0

        for other_id, other_state in agent_states.items():
            if other_id not in attachments:
                continue

            attachment_level = attachments[other_id]
            other_benefit = other_state.get('benefit_received', 0.0)

            # Weight benefit by attachment
            total_other_benefit += other_benefit * attachment_level

        # Only reward if benefit/cost ratio exceeds threshold
        if total_other_benefit > self_cost * self.altruism_threshold:
            # Reward scales with altruism level and net benefit
            altruism_reward = altruism_level * (total_other_benefit - self_cost)
            return altruism_reward

        return 0.0

    def _compute_protective_reward(
        self,
        protective_level: float,
        action_info: Dict,
        agent_states: Dict[int, Dict],
        attachments: Dict[int, float],
    ) -> float:
        """
        Reward for protecting vulnerable agents.

        Higher reward for protecting agents we're attached to.
        """
        protected_agents = action_info.get('protected_agents', [])

        if not protected_agents:
            return 0.0

        protective_reward = 0.0

        for other_id in protected_agents:
            if other_id not in attachments:
                continue

            attachment_level = attachments[other_id]

            # Check if agent was vulnerable
            if other_id in agent_states:
                vulnerability = agent_states[other_id].get('vulnerability', 0.5)
                threat_level = agent_states[other_id].get('threat_level', 0.0)

                # Reward scales with protective instinct, attachment, vulnerability, and threat
                protective_reward += (
                    protective_level * attachment_level * vulnerability * threat_level
                )

        return protective_reward

    def _compute_social_reward(
        self,
        action_info: Dict,
        attachments: Dict[int, float],
    ) -> float:
        """
        Reward for positive social interactions.

        Encourages proximity and cooperation with attached agents.
        """
        social_reward = 0.0

        # Reward for proximity to attached agents
        nearby_agents = action_info.get('nearby_agents', [])
        for other_id in nearby_agents:
            if other_id in attachments:
                social_reward += attachments[other_id] * 0.05

        # Reward for cooperation
        cooperated_with = action_info.get('cooperated_with', [])
        for other_id in cooperated_with:
            if other_id in attachments:
                social_reward += attachments[other_id] * 0.1

        return social_reward

    def get_total_reward(self, reward_components: RewardComponents) -> float:
        """Get weighted total reward from components."""
        return reward_components.total(self.weights)

    def update_weights(self, new_weights: Dict[str, float]):
        """Update reward weights (for curriculum learning)."""
        self.weights.update(new_weights)


class CurriculumRewardScheduler:
    """
    Schedules reward weights over training to implement curriculum learning.

    Gradually increases weight of complex emotional rewards.
    """

    def __init__(
        self,
        initial_weights: Dict[str, float],
        final_weights: Dict[str, float],
        transition_steps: int = 1_000_000,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            initial_weights: Starting reward weights
            final_weights: Target reward weights
            transition_steps: Steps to transition from initial to final
        """
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.transition_steps = transition_steps
        self.current_step = 0

    def get_current_weights(self) -> Dict[str, float]:
        """Get current reward weights based on training progress."""
        if self.current_step >= self.transition_steps:
            return self.final_weights.copy()

        # Linear interpolation
        progress = self.current_step / self.transition_steps

        current_weights = {}
        for key in self.initial_weights:
            initial = self.initial_weights[key]
            final = self.final_weights.get(key, initial)
            current_weights[key] = initial + (final - initial) * progress

        return current_weights

    def step(self):
        """Advance one step in the curriculum."""
        self.current_step += 1


def create_default_reward_shaper() -> RewardShaper:
    """Create reward shaper with default settings."""
    return RewardShaper(
        empathy_weight=0.5,
        attachment_weight=0.3,
        altruism_weight=0.4,
        protective_weight=0.4,
        social_weight=0.2,
        altruism_threshold=1.5,
    )


def create_curriculum_scheduler() -> CurriculumRewardScheduler:
    """
    Create curriculum scheduler that gradually emphasizes emotional rewards.

    Stage 1: Focus on survival and basic attachment
    Stage 2: Introduce empathy and prosocial rewards
    Stage 3: Full emotional reward system
    """
    initial_weights = {
        'extrinsic': 1.0,
        'empathy': 0.1,
        'attachment': 0.2,
        'altruism': 0.0,
        'protective': 0.1,
        'social': 0.1,
    }

    final_weights = {
        'extrinsic': 1.0,
        'empathy': 0.5,
        'attachment': 0.3,
        'altruism': 0.4,
        'protective': 0.4,
        'social': 0.2,
    }

    return CurriculumRewardScheduler(
        initial_weights=initial_weights,
        final_weights=final_weights,
        transition_steps=2_000_000,
    )
