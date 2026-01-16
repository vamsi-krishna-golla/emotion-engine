"""
Emotional Agent

Integrates emotion system with RL for emotion-conditioned decision making.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

from emotion_engine.core.emotion_state import EmotionState, create_baseline_emotion_state
from emotion_engine.core.emotion_dynamics import EmotionDynamics
from emotion_engine.core.relationship import RelationshipManager
from emotion_engine.networks.emotion_encoder import EmotionEncoder
from emotion_engine.networks.emotion_composer import EmotionComposer
from emotion_engine.networks.policy_network import PolicyNetwork
from emotion_engine.networks.value_network import ValueNetwork
from emotion_engine.rl.reward_shaping import RewardShaper


class EmotionalAgent:
    """
    RL agent with integrated emotion system.

    Core loop: observe → update emotions → select action → receive reward
    """

    def __init__(
        self,
        agent_id: int,
        observation_space_dim: int,
        action_space_dim: int,
        emotion_features_dim: int = 256,
        hidden_dim: int = 256,
        device: str = 'cpu',
    ):
        """
        Initialize emotional agent.

        Args:
            agent_id: Unique agent ID
            observation_space_dim: Dimension of observation space
            action_space_dim: Dimension of action space
            emotion_features_dim: Dimension of emotion feature vector
            hidden_dim: Hidden layer dimension
            device: Device to run networks on ('cpu' or 'cuda')
        """
        self.agent_id = agent_id
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.device = device

        # Emotion system
        self.emotion_state = create_baseline_emotion_state()
        self.emotion_dynamics = EmotionDynamics(decay_rate=0.01)
        self.relationship_manager = RelationshipManager(agent_id=agent_id)

        # Neural networks
        self.emotion_encoder = EmotionEncoder(
            self_state_dim=32,
            social_dim=64,
            env_dim=32,
            output_dim=emotion_features_dim,
        ).to(device)

        self.emotion_composer = EmotionComposer(
            num_primitives=10,
            num_complex=4,
        ).to(device)

        self.policy = PolicyNetwork(
            emotion_features_dim=emotion_features_dim,
            emotion_state_dim=16,  # 10 primitives + 4 complex + 2 (valence, arousal)
            relationship_dim=64,
            action_dim=action_space_dim,
            hidden_dim=hidden_dim,
            continuous=True,
        ).to(device)

        self.value_net = ValueNetwork(
            emotion_features_dim=emotion_features_dim,
            emotion_state_dim=16,
            hidden_dim=hidden_dim,
        ).to(device)

        # Reward shaping
        self.reward_shaper = RewardShaper()

        # Training mode
        self.training = True

    def set_training(self, mode: bool):
        """Set training mode."""
        self.training = mode
        self.emotion_encoder.train(mode)
        self.emotion_composer.train(mode)
        self.policy.train(mode)
        self.value_net.train(mode)

    def observe(
        self,
        observation: np.ndarray,
        other_agents_states: Dict[int, Dict],
    ) -> torch.Tensor:
        """
        Process observation and encode to emotion features.

        Args:
            observation: Raw observation from environment
            other_agents_states: States of other agents

        Returns:
            Emotion feature vector
        """
        # Parse observation into components
        # This is simplified - actual parsing depends on observation structure
        obs_tensor = torch.from_numpy(observation).float().to(self.device).unsqueeze(0)

        # For now, split observation into components (placeholder logic)
        obs_dim = observation.shape[0]
        third = obs_dim // 3

        self_state = obs_tensor[:, :third]
        social_context = obs_tensor[:, third:2*third].unsqueeze(1)  # Add agent dimension
        environment = obs_tensor[:, 2*third:]

        # Pad if needed
        if self_state.shape[1] < 32:
            self_state = torch.cat([
                self_state,
                torch.zeros(1, 32 - self_state.shape[1]).to(self.device)
            ], dim=1)

        if social_context.shape[2] < 64:
            social_context = torch.cat([
                social_context,
                torch.zeros(1, 1, 64 - social_context.shape[2]).to(self.device)
            ], dim=2)

        if environment.shape[1] < 32:
            environment = torch.cat([
                environment,
                torch.zeros(1, 32 - environment.shape[1]).to(self.device)
            ], dim=1)

        # Encode observation to emotion features
        emotion_features = self.emotion_encoder(self_state, social_context, environment)

        return emotion_features

    def update_emotions(
        self,
        emotion_stimulation: Dict[str, float],
        delta_time: float = 1.0,
    ):
        """
        Update emotion state based on stimulation.

        Args:
            emotion_stimulation: Dictionary of emotion changes
            delta_time: Time elapsed
        """
        self.emotion_state = self.emotion_dynamics.step(
            self.emotion_state,
            emotion_stimulation,
            delta_time
        )

        # Update complex emotions using neural composer
        with torch.no_grad():
            primitive_vector = torch.from_numpy(
                np.array([self.emotion_state.primitives[e] for e in self.emotion_state.primitives])
            ).float().unsqueeze(0).to(self.device)

            complex_emotions = self.emotion_composer(primitive_vector).squeeze(0).cpu().numpy()

            # Update complex emotion state
            complex_names = list(self.emotion_state.complex_emotions.keys())
            for i, name in enumerate(complex_names):
                self.emotion_state.complex_emotions[name] = float(complex_emotions[i])

    def select_action(
        self,
        emotion_features: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action based on current emotional state.

        Args:
            emotion_features: Encoded emotion features
            deterministic: If True, select mean action (no sampling)

        Returns:
            - Action: (action_dim,)
            - Log probability: scalar
            - Value estimate: scalar
        """
        # Get emotion state vector
        emotion_state_vector = torch.from_numpy(
            self.emotion_state.to_vector()
        ).float().unsqueeze(0).to(self.device)

        # Get relationship context (simplified - use strongest attachment)
        attachments = self.relationship_manager.get_all_attachments()
        if attachments:
            max_attachment = max(attachments.values())
        else:
            max_attachment = 0.0

        relationship_context = torch.zeros(1, 64).to(self.device)
        relationship_context[0, 0] = max_attachment  # Encode attachment in first dim

        # Sample action from policy
        with torch.no_grad() if not self.training else torch.enable_grad():
            action, log_prob = self.policy.sample_action(
                emotion_features,
                emotion_state_vector,
                relationship_context,
                deterministic=deterministic
            )

            value = self.value_net(emotion_features, emotion_state_vector)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item()
        )

    def compute_reward(
        self,
        extrinsic_reward: float,
        agent_states: Dict[int, Dict],
        action_info: Dict,
    ) -> float:
        """
        Compute total reward with emotional shaping.

        Args:
            extrinsic_reward: Base task reward
            agent_states: States of other agents
            action_info: Information about action taken

        Returns:
            Total shaped reward
        """
        emotion_state_vector = self.emotion_state.to_vector()
        attachments = self.relationship_manager.get_all_attachments()

        reward_components = self.reward_shaper.compute_reward(
            agent_id=self.agent_id,
            extrinsic_reward=extrinsic_reward,
            emotion_state=emotion_state_vector,
            agent_states=agent_states,
            attachments=attachments,
            action_info=action_info,
        )

        return self.reward_shaper.get_total_reward(reward_components)

    def update_relationships(
        self,
        other_agent_id: int,
        interaction_type: str,
        outcome: float,
        delta_time: float = 1.0,
    ):
        """
        Update relationship with another agent.

        Args:
            other_agent_id: ID of other agent
            interaction_type: Type of interaction
            outcome: Outcome value (positive/negative)
            delta_time: Time since last interaction
        """
        emotion_vector = self.emotion_state.to_vector()

        self.relationship_manager.update_relationship(
            other_agent_id=other_agent_id,
            interaction_type=interaction_type,
            outcome=outcome,
            emotion_state=emotion_vector,
            delta_time=delta_time,
        )

    def step(
        self,
        observation: np.ndarray,
        other_agents_states: Dict[int, Dict],
        emotion_stimulation: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Full agent step: observe → update emotions → select action.

        Args:
            observation: Environment observation
            other_agents_states: States of other agents
            emotion_stimulation: Optional emotion changes from events

        Returns:
            - Action
            - Log probability
            - Value estimate
        """
        # Update emotions if stimulation provided
        if emotion_stimulation:
            self.update_emotions(emotion_stimulation, delta_time=1.0)

        # Encode observation
        emotion_features = self.observe(observation, other_agents_states)

        # Select action
        action, log_prob, value = self.select_action(emotion_features)

        return action, log_prob, value

    def get_emotion_summary(self) -> Dict[str, float]:
        """Get summary of current emotional state."""
        dominant_prim, prim_intensity = self.emotion_state.get_dominant_emotion("primitive")
        dominant_comp, comp_intensity = self.emotion_state.get_dominant_emotion("complex")

        return {
            'dominant_primitive': dominant_prim,
            'dominant_primitive_intensity': prim_intensity,
            'dominant_complex': dominant_comp,
            'dominant_complex_intensity': comp_intensity,
            'valence': self.emotion_state.valence,
            'arousal': self.emotion_state.arousal,
            **self.emotion_state.primitives,
            **{f'complex_{k}': v for k, v in self.emotion_state.complex_emotions.items()},
        }

    def save(self, path: str):
        """Save agent networks."""
        torch.save({
            'emotion_encoder': self.emotion_encoder.state_dict(),
            'emotion_composer': self.emotion_composer.state_dict(),
            'policy': self.policy.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.emotion_encoder.load_state_dict(checkpoint['emotion_encoder'])
        self.emotion_composer.load_state_dict(checkpoint['emotion_composer'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.value_net.load_state_dict(checkpoint['value_net'])
