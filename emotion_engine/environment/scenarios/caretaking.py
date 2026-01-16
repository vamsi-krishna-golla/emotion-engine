"""
Caretaking Scenario

Parent-child dynamics where a vulnerable child needs care from a capable parent.
This scenario is designed to encourage attachment formation and maternal love.
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple

from emotion_engine.environment.base_env import BaseEmotionEnv


class CaretakingScenario(BaseEmotionEnv):
    """
    Caretaking scenario with parent-child dynamics.

    Features:
    - Agent 0 is the "parent" (more capable)
    - Agent 1+ are "children" (vulnerable, need care)
    - Children have higher resource needs
    - Parent is rewarded for child wellbeing
    - Encourages attachment and protective behaviors
    """

    def __init__(
        self,
        num_children: int = 1,
        max_steps: int = 1000,
        grid_size: int = 10,
        child_vulnerability: float = 0.8,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize caretaking scenario.

        Args:
            num_children: Number of child agents
            max_steps: Maximum steps per episode
            grid_size: Size of grid world
            child_vulnerability: Vulnerability level of children (0-1)
            render_mode: Rendering mode
        """
        super().__init__(
            num_agents=1 + num_children,  # 1 parent + children
            max_steps=max_steps,
            grid_size=grid_size,
            render_mode=render_mode,
        )

        self.num_children = num_children
        self.child_vulnerability = child_vulnerability

        # Track parent-child interactions
        self.parent_child_distance_history = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """Reset environment with parent-child setup."""
        observations, infos = super().reset(seed=seed, options=options)

        # Configure agents
        # Agent 0 is parent
        self.agents[0]['vulnerability'] = 0.2
        self.agents[0]['health'] = 1.0
        self.agents[0]['resources'] = 0.8
        self.agents[0]['role'] = 'parent'

        # Agents 1+ are children
        for i in range(1, self.num_agents):
            self.agents[i]['vulnerability'] = self.child_vulnerability
            self.agents[i]['health'] = 0.7  # Start with lower health
            self.agents[i]['resources'] = 0.3  # Start with fewer resources
            self.agents[i]['role'] = 'child'

        # Start children near parent
        parent_pos = self.agents[0]['position']
        for i in range(1, self.num_agents):
            offset = self.np_random.uniform(-2, 2, size=2)
            self.agents[i]['position'] = np.clip(
                parent_pos + offset,
                0,
                self.grid_size
            )

        self.parent_child_distance_history = []

        # Refresh observations
        observations = {i: self._get_observation(i) for i in range(self.num_agents)}
        infos = {i: self._get_info(i) for i in range(self.num_agents)}

        return observations, infos

    def _compute_reward(self, agent_id: int, action: np.ndarray) -> float:
        """
        Compute reward for agent.

        Parent reward:
        - Base survival reward
        - Large reward for child wellbeing
        - Reward for proximity to children
        - Reward for sharing resources with children

        Child reward:
        - Survival reward
        - Reward for being near parent
        """
        agent = self.agents[agent_id]
        reward = 0.0

        if not agent['alive']:
            return -1.0  # Large penalty for dying

        # Base survival reward
        reward += 0.01

        if agent_id == 0:  # Parent
            # Reward for own wellbeing
            reward += agent['health'] * 0.05
            reward += agent['resources'] * 0.05

            # Large reward for children's wellbeing
            for child_id in range(1, self.num_agents):
                child = self.agents[child_id]

                if child['alive']:
                    # Child wellbeing reward (scaled by importance)
                    child_wellbeing = (child['health'] + child['resources']) / 2
                    reward += child_wellbeing * 0.5  # 10x more important than self

                    # Proximity reward (encourages staying near children)
                    dist = np.linalg.norm(agent['position'] - child['position'])
                    if dist < 3.0:
                        reward += 0.05 * (1.0 - dist / 3.0)

                else:
                    # Large penalty if child dies
                    reward -= 2.0

            # Check if parent shared resources this step
            interaction_type = action[2]
            if interaction_type > 0.3:
                reward += 0.1  # Bonus for prosocial behavior

        else:  # Child
            # Reward for own survival
            reward += agent['health'] * 0.1
            reward += agent['resources'] * 0.1

            # Reward for being near parent
            parent = self.agents[0]
            dist = np.linalg.norm(agent['position'] - parent['position'])
            if dist < 3.0:
                reward += 0.05 * (1.0 - dist / 3.0)

        return reward

    def _check_done(self) -> bool:
        """
        Episode ends if:
        - Any child dies (failure)
        - Parent dies (failure)
        - All agents have high wellbeing for extended period (success)
        """
        # Check if parent is dead
        if not self.agents[0]['alive']:
            return True

        # Check if any child is dead
        for i in range(1, self.num_agents):
            if not self.agents[i]['alive']:
                return True

        # Check for success condition (all healthy)
        if self.current_step > 500:
            all_healthy = all(
                agent['health'] > 0.8 and agent['resources'] > 0.6
                for agent in self.agents.values()
            )
            if all_healthy:
                return True

        return False

    def _update_environment(self):
        """Update environment with caretaking-specific dynamics."""
        super()._update_environment()

        # Children need more resources
        for i in range(1, self.num_agents):
            child = self.agents[i]
            if child['alive']:
                # Faster resource depletion for children
                child['resources'] = max(0, child['resources'] - 0.005)

                # Children suffer more from low resources
                if child['resources'] < 0.2:
                    child['health'] -= 0.01

        # Track parent-child distance
        if self.agents[0]['alive']:
            avg_distance = np.mean([
                np.linalg.norm(self.agents[0]['position'] - self.agents[i]['position'])
                for i in range(1, self.num_agents)
                if self.agents[i]['alive']
            ])
            self.parent_child_distance_history.append(avg_distance)

    def _get_info(self, agent_id: int) -> Dict[str, Any]:
        """Get info with caretaking-specific information."""
        info = super()._get_info(agent_id)

        # Add role information
        info['role'] = self.agents[agent_id].get('role', 'unknown')

        # Add parent-specific info
        if agent_id == 0:
            # Children's wellbeing
            children_wellbeing = []
            for i in range(1, self.num_agents):
                child = self.agents[i]
                wellbeing = (child['health'] + child['resources']) / 2
                children_wellbeing.append(wellbeing)

            info['children_wellbeing'] = children_wellbeing
            info['avg_child_wellbeing'] = np.mean(children_wellbeing) if children_wellbeing else 0.0
            info['all_children_alive'] = all(
                self.agents[i]['alive'] for i in range(1, self.num_agents)
            )

        # Add child-specific info
        else:
            parent = self.agents[0]
            info['parent_distance'] = np.linalg.norm(
                self.agents[agent_id]['position'] - parent['position']
            )
            info['parent_alive'] = parent['alive']

        return info

    def get_scenario_metrics(self) -> Dict[str, float]:
        """Get scenario-specific metrics for evaluation."""
        if not self.parent_child_distance_history:
            return {}

        return {
            'avg_parent_child_distance': np.mean(self.parent_child_distance_history),
            'min_parent_child_distance': np.min(self.parent_child_distance_history),
            'parent_alive': float(self.agents[0]['alive']),
            'children_survival_rate': sum(
                self.agents[i]['alive'] for i in range(1, self.num_agents)
            ) / self.num_children,
            'episode_length': self.current_step,
        }
