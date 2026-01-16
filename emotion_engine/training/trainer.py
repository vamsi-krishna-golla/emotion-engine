"""
Training Infrastructure

Main trainer class for training emotional agents with PPO.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time

from emotion_engine.rl.agent import EmotionalAgent
from emotion_engine.rl.ppo import PPO, PPOConfig
from emotion_engine.rl.replay_buffer import RolloutBuffer
from emotion_engine.rl.reward_shaping import RewardShaper, create_curriculum_scheduler
from emotion_engine.environment.base_env import BaseEmotionEnv


class Trainer:
    """
    Trainer for emotional agents using PPO.

    Handles:
    - Rollout collection from environment
    - PPO updates
    - Logging and checkpointing
    - Curriculum learning
    """

    def __init__(
        self,
        env: BaseEmotionEnv,
        agents: Dict[int, EmotionalAgent],
        ppo_config: PPOConfig,
        reward_shaper: RewardShaper,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
        save_interval: int = 1000,
        device: str = 'cpu',
    ):
        """
        Initialize trainer.

        Args:
            env: Training environment
            agents: Dictionary of emotional agents
            ppo_config: PPO configuration
            reward_shaper: Reward shaper for intrinsic rewards
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Steps between logging
            save_interval: Steps between checkpoint saves
            device: Device for training
        """
        self.env = env
        self.agents = agents
        self.num_agents = len(agents)
        self.reward_shaper = reward_shaper
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval
        self.save_interval = save_interval

        # Create PPO optimizers for each agent
        self.ppo_optimizers = {}
        for agent_id, agent in agents.items():
            self.ppo_optimizers[agent_id] = PPO(
                policy_network=agent.policy,
                value_network=agent.value_net,
                learning_rate=ppo_config.learning_rate,
                n_epochs=ppo_config.n_epochs,
                batch_size=ppo_config.batch_size,
                clip_range=ppo_config.clip_range,
                vf_coef=ppo_config.vf_coef,
                ent_coef=ppo_config.ent_coef,
                max_grad_norm=ppo_config.max_grad_norm,
                gamma=ppo_config.gamma,
                gae_lambda=ppo_config.gae_lambda,
            )

        # Create rollout buffers
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        emotion_dim = 16  # 10 primitives + 4 complex + 2 (valence, arousal)

        self.rollout_buffers = {}
        for agent_id in agents.keys():
            self.rollout_buffers[agent_id] = RolloutBuffer(
                buffer_size=ppo_config.n_steps,
                observation_dim=obs_dim,
                action_dim=action_dim,
                emotion_dim=emotion_dim,
                gamma=ppo_config.gamma,
                gae_lambda=ppo_config.gae_lambda,
            )

        # Training metrics
        self.global_step = 0
        self.episode_count = 0
        self.training_metrics = []

    def collect_rollouts(self, n_steps: int) -> Dict[str, Any]:
        """
        Collect rollouts from environment.

        Args:
            n_steps: Number of steps to collect

        Returns:
            Dictionary of rollout statistics
        """
        # Reset buffers
        for buffer in self.rollout_buffers.values():
            buffer.reset()

        # Reset environment
        observations, infos = self.env.reset()

        episode_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        episode_lengths = {agent_id: 0 for agent_id in self.agents.keys()}

        for step in range(n_steps):
            # Get actions from all agents
            actions = {}
            log_probs = {}
            values = {}

            for agent_id, agent in self.agents.items():
                # Get emotion features
                emotion_features = agent.observe(observations[agent_id], {})

                # Select action
                action, log_prob, value = agent.select_action(
                    emotion_features,
                    deterministic=False
                )

                actions[agent_id] = action
                log_probs[agent_id] = log_prob
                values[agent_id] = value

            # Step environment
            next_observations, rewards, terminateds, truncateds, next_infos = self.env.step(actions)

            # Compute shaped rewards
            shaped_rewards = {}
            for agent_id in self.agents.keys():
                # Get other agents' states for reward shaping
                other_states = {
                    other_id: next_infos[other_id]
                    for other_id in self.agents.keys()
                    if other_id != agent_id
                }

                # Action info for reward computation
                action_info = {
                    'self_cost': 0.0,  # Would compute from action
                    'protected_agents': [],
                    'nearby_agents': [],
                }

                # Compute shaped reward
                shaped_reward = self.agents[agent_id].compute_reward(
                    extrinsic_reward=rewards[agent_id],
                    agent_states=other_states,
                    action_info=action_info,
                )

                shaped_rewards[agent_id] = shaped_reward

            # Store transitions
            for agent_id, agent in self.agents.items():
                done = terminateds[agent_id] or truncateds[agent_id]

                self.rollout_buffers[agent_id].add(
                    observation=observations[agent_id],
                    action=actions[agent_id],
                    reward=shaped_rewards[agent_id],
                    value=values[agent_id],
                    log_prob=log_probs[agent_id],
                    done=done,
                    emotion_state=agent.emotion_state.to_vector(),
                )

                episode_rewards[agent_id] += shaped_rewards[agent_id]
                episode_lengths[agent_id] += 1

            # Update observations
            observations = next_observations
            infos = next_infos

            # Check if episode done
            if any(terminateds.values()) or any(truncateds.values()):
                self.episode_count += 1
                observations, infos = self.env.reset()

                # Log episode stats
                if self.episode_count % self.log_interval == 0:
                    avg_reward = np.mean(list(episode_rewards.values()))
                    avg_length = np.mean(list(episode_lengths.values()))
                    print(f"  Episode {self.episode_count}: "
                          f"Avg Reward={avg_reward:.3f}, "
                          f"Avg Length={avg_length:.1f}")

                episode_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
                episode_lengths = {agent_id: 0 for agent_id in self.agents.keys()}

        # Compute returns and advantages
        for agent_id, agent in self.agents.items():
            # Get last value for bootstrapping
            with torch.no_grad():
                emotion_features = agent.observe(observations[agent_id], {})
                emotion_state_vector = torch.from_numpy(
                    agent.emotion_state.to_vector()
                ).float().unsqueeze(0).to(self.device)

                last_value = agent.value_net(emotion_features, emotion_state_vector).item()

            self.rollout_buffers[agent_id].compute_returns_and_advantages(last_value)

        return {
            'episode_count': self.episode_count,
            'avg_episode_reward': np.mean(list(episode_rewards.values())),
            'avg_episode_length': np.mean(list(episode_lengths.values())),
        }

    def update(self) -> Dict[str, float]:
        """
        Update agents using collected rollouts.

        Returns:
            Dictionary of training metrics
        """
        all_metrics = {}

        for agent_id, agent in self.agents.items():
            # Get data from rollout buffer
            data = self.rollout_buffers[agent_id].get(batch_size=self.ppo_optimizers[agent_id].batch_size)

            observations = data['observations']
            actions = data['actions']
            old_log_probs = data['old_log_probs']
            advantages = data['advantages']
            returns = data['returns']
            emotion_states = data['emotion_states']

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Multiple epochs over the same data
            policy_losses = []
            value_losses = []
            entropy_losses = []

            for epoch in range(self.ppo_optimizers[agent_id].n_epochs):
                # Encode observations to emotion features (batch processing)
                with torch.no_grad():
                    # Simple encoding: use observations directly as emotion features for now
                    # In production, would use agent.emotion_encoder properly
                    emotion_features = observations[:, :256] if observations.shape[1] >= 256 else torch.nn.functional.pad(
                        observations, (0, max(0, 256 - observations.shape[1]))
                    )

                # Dummy relationship context
                relationship_context = torch.zeros(observations.shape[0], 64)

                # Evaluate actions with current policy
                log_probs, entropy = agent.policy.evaluate_actions(
                    emotion_features,
                    emotion_states,
                    relationship_context,
                    actions
                )

                # Get value estimates
                values = agent.value_net(emotion_features, emotion_states).squeeze(-1)

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_optimizers[agent_id].clip_range,
                    1.0 + self.ppo_optimizers[agent_id].clip_range
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((values - returns) ** 2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.ppo_optimizers[agent_id].vf_coef * value_loss +
                    self.ppo_optimizers[agent_id].ent_coef * entropy_loss
                )

                # Optimization step
                self.ppo_optimizers[agent_id].optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.policy.parameters()) + list(agent.value_net.parameters()),
                    self.ppo_optimizers[agent_id].max_grad_norm
                )
                self.ppo_optimizers[agent_id].optimizer.step()

                # Logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

            # Store metrics with agent prefix
            all_metrics[f'agent_{agent_id}_policy_loss'] = np.mean(policy_losses)
            all_metrics[f'agent_{agent_id}_value_loss'] = np.mean(value_losses)
            all_metrics[f'agent_{agent_id}_entropy_loss'] = np.mean(entropy_losses)

        return all_metrics

    def train(
        self,
        total_steps: int,
        n_steps_per_update: int = 2048,
    ) -> List[Dict[str, Any]]:
        """
        Main training loop.

        Args:
            total_steps: Total training steps
            n_steps_per_update: Steps to collect before each update

        Returns:
            List of training metrics
        """
        print(f"\n{'='*70}")
        print(f" Training Emotional Agents")
        print(f"{'='*70}")
        print(f"Total steps: {total_steps:,}")
        print(f"Steps per update: {n_steps_per_update:,}")
        print(f"Num agents: {self.num_agents}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        # Set agents to training mode
        for agent in self.agents.values():
            agent.set_training(True)

        start_time = time.time()
        metrics_history = []

        while self.global_step < total_steps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts(n_steps_per_update)

            # Update agents
            update_metrics = self.update()

            # Update global step
            self.global_step += n_steps_per_update

            # Combine metrics
            metrics = {
                'global_step': self.global_step,
                'episode_count': rollout_stats['episode_count'],
                **update_metrics,
            }

            metrics_history.append(metrics)

            # Log progress
            if self.global_step % (self.log_interval * n_steps_per_update) == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed

                print(f"\nStep {self.global_step:,} / {total_steps:,} "
                      f"({100*self.global_step/total_steps:.1f}%)")
                print(f"  Steps/sec: {steps_per_sec:.1f}")
                print(f"  Episodes: {rollout_stats['episode_count']}")

                # Print agent emotions
                for agent_id, agent in self.agents.items():
                    summary = agent.get_emotion_summary()
                    print(f"  Agent {agent_id}: "
                          f"Maternal Love={summary.get('complex_maternal_love', 0):.3f}, "
                          f"Attachment={summary.get('attachment', 0):.3f}")

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step}.pt")

        # Final save
        self.save_checkpoint("final_checkpoint.pt")

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f" Training Complete!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Total steps: {self.global_step:,}")
        print(f"Total episodes: {self.episode_count}")
        print(f"{'='*70}\n")

        return metrics_history

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'agents': {
                agent_id: {
                    'emotion_encoder': agent.emotion_encoder.state_dict(),
                    'emotion_composer': agent.emotion_composer.state_dict(),
                    'policy': agent.policy.state_dict(),
                    'value_net': agent.value_net.state_dict(),
                }
                for agent_id, agent in self.agents.items()
            },
            'ppo_optimizers': {
                agent_id: ppo.optimizer.state_dict()
                for agent_id, ppo in self.ppo_optimizers.items()
            },
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']

        for agent_id, agent in self.agents.items():
            agent_checkpoint = checkpoint['agents'][agent_id]
            agent.emotion_encoder.load_state_dict(agent_checkpoint['emotion_encoder'])
            agent.emotion_composer.load_state_dict(agent_checkpoint['emotion_composer'])
            agent.policy.load_state_dict(agent_checkpoint['policy'])
            agent.value_net.load_state_dict(agent_checkpoint['value_net'])

        for agent_id, ppo in self.ppo_optimizers.items():
            ppo.optimizer.load_state_dict(checkpoint['ppo_optimizers'][agent_id])

        print(f"Loaded checkpoint from step {self.global_step:,}")
