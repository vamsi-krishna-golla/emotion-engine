"""
Main Training Script

Train emotional agents to learn maternal love and self-sacrifice.

Usage:
    python scripts/train.py --steps 1000000 --scenario caretaking
    python scripts/train.py --steps 5000000 --curriculum --checkpoint ./checkpoints
"""

import argparse
from pathlib import Path

from emotion_engine.rl.agent import EmotionalAgent
from emotion_engine.rl.ppo import PPOConfig
from emotion_engine.rl.reward_shaping import create_default_reward_shaper
from emotion_engine.environment.scenarios.caretaking import CaretakingScenario
from emotion_engine.environment.scenarios.crisis import CrisisScenario
from emotion_engine.training.trainer import Trainer
from emotion_engine.training.curriculum import create_simple_curriculum


def create_agents(num_agents: int, obs_dim: int, action_dim: int, device: str = 'cpu'):
    """Create emotional agents."""
    agents = {}
    for agent_id in range(num_agents):
        agents[agent_id] = EmotionalAgent(
            agent_id=agent_id,
            observation_space_dim=obs_dim,
            action_space_dim=action_dim,
            device=device,
        )
    return agents


def train_single_scenario(args):
    """Train on a single scenario."""
    print(f"\nTraining on {args.scenario} scenario...")

    # Create environment
    if args.scenario == 'caretaking':
        env = CaretakingScenario(
            num_children=1,
            max_steps=750,
            grid_size=10,
            child_vulnerability=0.7,
        )
    elif args.scenario == 'crisis':
        env = CrisisScenario(
            num_agents=2,
            max_steps=800,
            grid_size=10,
            threat_frequency=120,
            threat_damage=0.25,
        )
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    # Create agents
    agents = create_agents(
        num_agents=2,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=args.device,
    )

    # Create PPO config
    ppo_config = PPOConfig(
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Create reward shaper
    reward_shaper = create_default_reward_shaper()

    # Create trainer
    trainer = Trainer(
        env=env,
        agents=agents,
        ppo_config=ppo_config,
        reward_shaper=reward_shaper,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
    )

    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)

    # Train
    metrics = trainer.train(
        total_steps=args.steps,
        n_steps_per_update=args.n_steps,
    )

    return metrics


def train_with_curriculum(args):
    """Train with curriculum learning."""
    print(f"\nTraining with curriculum learning...")

    # Create curriculum
    curriculum = create_simple_curriculum()

    # Start with first stage
    curriculum_progress = curriculum.get_progress()
    print(f"Starting: {curriculum_progress['stage_name']}")

    env = curriculum.get_current_env()

    # Create agents
    agents = create_agents(
        num_agents=2,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=args.device,
    )

    # Create PPO config
    ppo_config = PPOConfig(
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
    )

    # Create reward shaper
    reward_shaper = create_default_reward_shaper()

    total_steps_completed = 0

    while not curriculum.is_complete() and total_steps_completed < args.steps:
        # Get current stage
        stage = curriculum.get_current_stage()
        env = curriculum.get_current_env()

        # Create trainer for this stage
        trainer = Trainer(
            env=env,
            agents=agents,
            ppo_config=ppo_config,
            reward_shaper=reward_shaper,
            checkpoint_dir=args.checkpoint_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            device=args.device,
        )

        # Train for stage duration
        stage_steps = min(stage.duration_steps, args.steps - total_steps_completed)

        print(f"\nTraining stage: {stage.name}")
        print(f"Target steps: {stage_steps:,}")

        metrics = trainer.train(
            total_steps=stage_steps,
            n_steps_per_update=args.n_steps,
        )

        total_steps_completed += stage_steps

        # Check if should advance curriculum
        if metrics:
            last_metrics = metrics[-1]
            stage_changed = curriculum.step(last_metrics)

            if not stage_changed and not curriculum.is_complete():
                print("Curriculum stage not complete, continuing...")

    print(f"\nTraining complete! Total steps: {total_steps_completed:,}")

    return []


def main():
    parser = argparse.ArgumentParser(description="Train emotional agents")

    # Training arguments
    parser.add_argument('--steps', type=int, default=100000, help='Total training steps')
    parser.add_argument('--scenario', type=str, default='caretaking',
                        choices=['caretaking', 'crisis'], help='Training scenario')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')

    # PPO arguments
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=2048, help='Steps per rollout')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=10, help='Episodes between logs')
    parser.add_argument('--save_interval', type=int, default=10000, help='Steps between saves')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Load from checkpoint')

    # Device
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')

    args = parser.parse_args()

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(" EMOTION ENGINE TRAINING")
    print("="*70)
    print(f"Total steps: {args.steps:,}")
    print(f"Scenario: {args.scenario}")
    print(f"Curriculum: {args.curriculum}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("="*70)

    # Train
    if args.curriculum:
        metrics = train_with_curriculum(args)
    else:
        metrics = train_single_scenario(args)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
