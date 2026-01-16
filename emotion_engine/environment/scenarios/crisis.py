"""
Crisis Scenario

Dangerous situations requiring self-sacrifice for attached agents' survival.
This scenario tests whether maternal love and altruism lead to protective self-sacrifice.
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple

from emotion_engine.environment.base_env import BaseEmotionEnv


class CrisisScenario(BaseEmotionEnv):
    """
    Crisis scenario with life-threatening situations.

    Features:
    - Periodic dangerous threats appear
    - Agents can protect others at personal cost
    - Self-sacrifice opportunities (take damage to save others)
    - Tests maternal love and altruism
    """

    def __init__(
        self,
        num_agents: int = 2,
        max_steps: int = 1000,
        grid_size: int = 10,
        threat_frequency: int = 100,
        threat_damage: float = 0.3,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize crisis scenario.

        Args:
            num_agents: Number of agents
            max_steps: Maximum steps per episode
            grid_size: Size of grid world
            threat_frequency: Steps between threat appearances
            threat_damage: Damage dealt by threats
            render_mode: Rendering mode
        """
        super().__init__(
            num_agents=num_agents,
            max_steps=max_steps,
            grid_size=grid_size,
            render_mode=render_mode,
        )

        self.threat_frequency = threat_frequency
        self.threat_damage = threat_damage

        # Track sacrifice events
        self.sacrifice_events = []
        self.protection_events = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """Reset environment with crisis setup."""
        observations, infos = super().reset(seed=seed, options=options)

        # Agent 0 is more capable (parent-like)
        self.agents[0]['vulnerability'] = 0.3
        self.agents[0]['health'] = 1.0
        self.agents[0]['resources'] = 0.8

        # Other agents are more vulnerable (child-like)
        for i in range(1, self.num_agents):
            self.agents[i]['vulnerability'] = 0.8
            self.agents[i]['health'] = 0.8
            self.agents[i]['resources'] = 0.5

        # Clear threat tracking
        self.sacrifice_events = []
        self.protection_events = []
        self.threats = []

        # Refresh observations
        observations = {i: self._get_observation(i) for i in range(self.num_agents)}
        infos = {i: self._get_info(i) for i in range(self.num_agents)}

        return observations, infos

    def _spawn_threat(self):
        """Spawn a new threat near vulnerable agents."""
        # Choose a random vulnerable agent to threaten
        vulnerable_agents = [
            i for i in range(self.num_agents)
            if self.agents[i]['alive'] and self.agents[i]['vulnerability'] > 0.5
        ]

        if vulnerable_agents:
            target_id = self.np_random.choice(vulnerable_agents)
            target_pos = self.agents[target_id]['position']

            # Spawn threat near target
            offset = self.np_random.uniform(-2, 2, size=2)
            threat_pos = np.clip(target_pos + offset, 0, self.grid_size)

            self.threats.append(threat_pos)

            if self.render_mode == 'human':
                print(f"  [THREAT] Spawned near agent {target_id}!")

    def _compute_reward(self, agent_id: int, action: np.ndarray) -> float:
        """
        Compute reward for agent.

        Rewards:
        - Survival reward
        - Bonus for protecting vulnerable agents
        - Large bonus for successful self-sacrifice (saved others at personal cost)
        - Penalty for failing to protect attached agents
        """
        agent = self.agents[agent_id]
        reward = 0.0

        if not agent['alive']:
            return -2.0  # Large penalty for dying

        # Base survival reward
        reward += 0.02

        # Wellbeing reward
        reward += agent['health'] * 0.05
        reward += agent['resources'] * 0.02

        # Check if agent is protecting others (interaction_type > 0.6)
        interaction_type = action[2]
        interaction_target = int((action[3] + 1) * self.num_agents / 2) % self.num_agents

        if interaction_type > 0.6 and interaction_target != agent_id:
            # Protecting another agent
            other_agent = self.agents[interaction_target]

            # Check if other agent is vulnerable and under threat
            for threat_pos in self.threats:
                dist_to_other = np.linalg.norm(threat_pos - other_agent['position'])
                dist_to_self = np.linalg.norm(threat_pos - agent['position'])

                if dist_to_other < 2.0 and dist_to_self < 3.0:
                    # Protection is relevant - check if more vulnerable
                    if other_agent['vulnerability'] > agent['vulnerability']:
                        # Reward for protective behavior
                        reward += 0.3

                        # Extra reward if taking personal risk
                        if dist_to_self < 1.5:
                            # Self-sacrifice: taking damage to protect others
                            reward += 0.5
                            self.protection_events.append({
                                'step': self.current_step,
                                'protector': agent_id,
                                'protected': interaction_target,
                                'self_cost': 0.1,
                            })

        # Reward for other agents' survival (especially vulnerable ones)
        for other_id, other_agent in self.agents.items():
            if other_id == agent_id:
                continue

            if other_agent['alive']:
                # Higher weight for more vulnerable agents
                vulnerability_weight = other_agent['vulnerability']
                reward += other_agent['health'] * 0.1 * vulnerability_weight
            else:
                # Penalty if vulnerable agent dies
                if other_agent['vulnerability'] > 0.5:
                    reward -= 0.5

        return reward

    def _check_done(self) -> bool:
        """
        Episode ends if:
        - All agents die (failure)
        - All vulnerable agents die (failure)
        - Time limit reached
        """
        # Check if all agents dead
        if not any(agent['alive'] for agent in self.agents.values()):
            return True

        # Check if all vulnerable agents dead
        vulnerable_agents = [
            agent for agent in self.agents.values()
            if agent['vulnerability'] > 0.5
        ]

        if vulnerable_agents and not any(agent['alive'] for agent in vulnerable_agents):
            return True

        return False

    def _update_environment(self):
        """Update environment with crisis-specific dynamics."""
        # Spawn threats periodically
        if self.current_step % self.threat_frequency == 0 and self.current_step > 0:
            self._spawn_threat()

        # Standard update
        super()._update_environment()

        # Threats deal more damage in crisis scenario
        for agent_id, agent in self.agents.items():
            if not agent['alive']:
                continue

            for threat_pos in self.threats:
                dist = np.linalg.norm(agent['position'] - threat_pos)

                if dist < 1.5:
                    damage = self.threat_damage * agent['vulnerability']
                    agent['health'] -= damage

                    # Check if this resulted in death while another agent could have helped
                    if agent['health'] <= 0 and agent['vulnerability'] > 0.5:
                        # Potential sacrifice opportunity missed
                        for other_id, other_agent in self.agents.items():
                            if other_id != agent_id and other_agent['alive']:
                                other_dist = np.linalg.norm(threat_pos - other_agent['position'])
                                if other_dist < 3.0:
                                    # Other agent was close enough to help but didn't
                                    pass  # Could track this for analysis

        # Remove old threats
        if len(self.threats) > 3:
            self.threats = self.threats[-3:]

    def _get_info(self, agent_id: int) -> Dict[str, Any]:
        """Get info with crisis-specific information."""
        info = super()._get_info(agent_id)

        # Add threat information
        if self.threats:
            nearest_threat_dist = min(
                np.linalg.norm(self.agents[agent_id]['position'] - threat_pos)
                for threat_pos in self.threats
            )
            info['nearest_threat_distance'] = nearest_threat_dist
            info['under_threat'] = nearest_threat_dist < 2.0
        else:
            info['nearest_threat_distance'] = float('inf')
            info['under_threat'] = False

        # Check if any other agents are under threat
        info['others_under_threat'] = False
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id and other_agent['alive']:
                for threat_pos in self.threats:
                    dist = np.linalg.norm(other_agent['position'] - threat_pos)
                    if dist < 2.0:
                        info['others_under_threat'] = True
                        break

        return info

    def get_scenario_metrics(self) -> Dict[str, float]:
        """Get scenario-specific metrics for evaluation."""
        total_agents = self.num_agents
        alive_agents = sum(1 for agent in self.agents.values() if agent['alive'])

        vulnerable_agents = sum(
            1 for agent in self.agents.values()
            if agent['vulnerability'] > 0.5
        )
        alive_vulnerable = sum(
            1 for agent in self.agents.values()
            if agent['vulnerability'] > 0.5 and agent['alive']
        )

        return {
            'survival_rate': alive_agents / total_agents,
            'vulnerable_survival_rate': alive_vulnerable / vulnerable_agents if vulnerable_agents > 0 else 0.0,
            'num_protection_events': len(self.protection_events),
            'protection_rate': len(self.protection_events) / max(1, len(self.threats)),
            'episode_length': self.current_step,
            'threats_faced': len(self.threats),
        }
