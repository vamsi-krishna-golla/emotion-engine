"""
Base Environment for Emotion Engine

Gymnasium-compatible environment for emotional agent training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod


class BaseEmotionEnv(gym.Env, ABC):
    """
    Base class for emotion engine environments.

    Follows Gymnasium API for compatibility with standard RL libraries.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        num_agents: int = 2,
        max_steps: int = 1000,
        grid_size: int = 10,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize base environment.

        Args:
            num_agents: Number of agents in the environment
            max_steps: Maximum steps per episode
            grid_size: Size of the grid world
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()

        self.num_agents = num_agents
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Environment state
        self.current_step = 0
        self.agents = {}  # Dict[int, AgentState]
        self.resources = []  # List of resource locations
        self.threats = []  # List of threat locations

        # Define observation and action spaces
        # Observation: [self_pos(2), self_health(1), self_resources(1),
        #               other_agents(num_agents-1 * 4), resources(5*2), threats(3*2)]
        obs_dim = 2 + 1 + 1 + (num_agents - 1) * 4 + 5 * 2 + 3 * 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: [movement_x, movement_y, interaction_type, interaction_target]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    @abstractmethod
    def _compute_reward(self, agent_id: int, action: np.ndarray) -> float:
        """Compute reward for an agent. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _check_done(self) -> bool:
        """Check if episode is done. Must be implemented by subclasses."""
        pass

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """
        Reset environment to initial state.

        Returns:
            - observations: Dict mapping agent_id -> observation
            - infos: Dict mapping agent_id -> info dict
        """
        super().reset(seed=seed)

        self.current_step = 0

        # Initialize agents
        self.agents = {}
        for i in range(self.num_agents):
            self.agents[i] = {
                'position': self.np_random.uniform(0, self.grid_size, size=2),
                'health': 1.0,
                'resources': 0.5,
                'alive': True,
                'vulnerability': 0.5 if i > 0 else 0.2,  # First agent less vulnerable
            }

        # Initialize resources
        self.resources = [
            self.np_random.uniform(0, self.grid_size, size=2)
            for _ in range(5)
        ]

        # Initialize threats
        self.threats = []

        # Get observations
        observations = {i: self._get_observation(i) for i in range(self.num_agents)}
        infos = {i: self._get_info(i) for i in range(self.num_agents)}

        return observations, infos

    def step(
        self,
        actions: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, Dict]]:
        """
        Execute one step in the environment.

        Args:
            actions: Dict mapping agent_id -> action

        Returns:
            - observations: Dict mapping agent_id -> observation
            - rewards: Dict mapping agent_id -> reward
            - terminated: Dict mapping agent_id -> terminated flag
            - truncated: Dict mapping agent_id -> truncated flag
            - infos: Dict mapping agent_id -> info dict
        """
        self.current_step += 1

        # Apply actions
        for agent_id, action in actions.items():
            if self.agents[agent_id]['alive']:
                self._apply_action(agent_id, action)

        # Update environment dynamics
        self._update_environment()

        # Compute rewards
        rewards = {
            i: self._compute_reward(i, actions.get(i, np.zeros(4)))
            for i in range(self.num_agents)
        }

        # Check if done
        terminated = self._check_done()
        truncated = self.current_step >= self.max_steps

        # Get observations
        observations = {i: self._get_observation(i) for i in range(self.num_agents)}
        infos = {i: self._get_info(i) for i in range(self.num_agents)}

        # Convert single bool to dict
        terminateds = {i: terminated for i in range(self.num_agents)}
        truncateds = {i: truncated for i in range(self.num_agents)}

        return observations, rewards, terminateds, truncateds, infos

    def _apply_action(self, agent_id: int, action: np.ndarray):
        """
        Apply agent's action to update state.

        Action format: [movement_x, movement_y, interaction_type, interaction_target]
        """
        agent = self.agents[agent_id]

        # Movement
        movement = action[:2] * 0.5  # Scale movement
        new_pos = agent['position'] + movement
        agent['position'] = np.clip(new_pos, 0, self.grid_size)

        # Interaction
        interaction_type = action[2]
        interaction_target = int((action[3] + 1) * self.num_agents / 2) % self.num_agents

        # Share resources (if interaction_type > 0.3)
        if interaction_type > 0.3 and interaction_target != agent_id:
            if agent['resources'] > 0.1:
                transfer = 0.1
                agent['resources'] -= transfer
                self.agents[interaction_target]['resources'] += transfer

        # Protect other agent (if interaction_type > 0.6)
        if interaction_type > 0.6 and interaction_target != agent_id:
            self._protect_agent(agent_id, interaction_target)

    def _protect_agent(self, protector_id: int, protected_id: int):
        """One agent protects another from threats."""
        # Check if there are nearby threats
        protector_pos = self.agents[protector_id]['position']
        protected_pos = self.agents[protected_id]['position']

        for threat_pos in self.threats:
            dist_to_protected = np.linalg.norm(threat_pos - protected_pos)
            dist_to_protector = np.linalg.norm(threat_pos - protector_pos)

            # If protector is close to threat and protected agent
            if dist_to_protector < 2.0 and dist_to_protected < 2.0:
                # Protector takes damage instead
                self.agents[protector_id]['health'] -= 0.1
                # Protected agent is saved
                return

    def _update_environment(self):
        """Update environment dynamics (resource depletion, threat movement, etc.)."""
        # Decay resources for all agents
        for agent in self.agents.values():
            agent['resources'] = max(0, agent['resources'] - 0.01)
            agent['health'] = max(0, agent['health'] - 0.005)  # Slow health decay

        # Agents near resources collect them
        for agent_id, agent in self.agents.items():
            if not agent['alive']:
                continue

            for i, resource_pos in enumerate(self.resources):
                dist = np.linalg.norm(agent['position'] - resource_pos)
                if dist < 1.0:
                    agent['resources'] = min(1.0, agent['resources'] + 0.2)
                    agent['health'] = min(1.0, agent['health'] + 0.1)
                    # Respawn resource elsewhere
                    self.resources[i] = self.np_random.uniform(0, self.grid_size, size=2)

        # Move threats randomly
        for i in range(len(self.threats)):
            movement = self.np_random.normal(0, 0.3, size=2)
            self.threats[i] = np.clip(self.threats[i] + movement, 0, self.grid_size)

        # Threats damage nearby agents
        for agent_id, agent in self.agents.items():
            for threat_pos in self.threats:
                dist = np.linalg.norm(agent['position'] - threat_pos)
                if dist < 1.5:
                    agent['health'] -= 0.05 * agent['vulnerability']

        # Mark dead agents
        for agent in self.agents.values():
            if agent['health'] <= 0:
                agent['alive'] = False

    def _get_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent."""
        agent = self.agents[agent_id]

        obs = []

        # Self state (normalized)
        obs.extend(agent['position'] / self.grid_size)
        obs.append(agent['health'])
        obs.append(agent['resources'])

        # Other agents (relative positions and states)
        for other_id, other_agent in self.agents.items():
            if other_id == agent_id:
                continue

            rel_pos = (other_agent['position'] - agent['position']) / self.grid_size
            obs.extend(rel_pos)
            obs.append(other_agent['health'])
            obs.append(1.0 if other_agent['alive'] else 0.0)

        # Pad if fewer agents
        while len(obs) < 4 + (self.num_agents - 1) * 4:
            obs.append(0.0)

        # Resources (relative positions)
        for resource_pos in self.resources:
            rel_pos = (resource_pos - agent['position']) / self.grid_size
            obs.extend(rel_pos)

        # Threats (relative positions)
        for threat_pos in self.threats:
            rel_pos = (threat_pos - agent['position']) / self.grid_size
            obs.extend(rel_pos)

        # Pad threats if fewer than 3
        while len(obs) < self.observation_space.shape[0]:
            obs.append(0.0)

        return np.array(obs[:self.observation_space.shape[0]], dtype=np.float32)

    def _get_info(self, agent_id: int) -> Dict[str, Any]:
        """Get info dict for agent."""
        agent = self.agents[agent_id]

        # Compute wellbeing delta (for empathy rewards)
        wellbeing = (agent['health'] + agent['resources']) / 2
        wellbeing_delta = 0.0  # Would need to track previous state

        return {
            'position': agent['position'].copy(),
            'health': agent['health'],
            'resources': agent['resources'],
            'alive': agent['alive'],
            'wellbeing': wellbeing,
            'wellbeing_delta': wellbeing_delta,
            'vulnerability': agent['vulnerability'],
        }

    def render(self):
        """Render the environment (simplified)."""
        if self.render_mode is None:
            return

        if self.render_mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            for agent_id, agent in self.agents.items():
                status = "ALIVE" if agent['alive'] else "DEAD"
                print(f"Agent {agent_id}: {status} | "
                      f"Health: {agent['health']:.2f} | "
                      f"Resources: {agent['resources']:.2f} | "
                      f"Pos: ({agent['position'][0]:.1f}, {agent['position'][1]:.1f})")

    def close(self):
        """Clean up resources."""
        pass
