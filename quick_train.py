"""
Quick Training Demo

Run a short training session to demonstrate the emotion engine learning.
"""

import torch
from emotion_engine.rl.agent import EmotionalAgent
from emotion_engine.rl.ppo import PPOConfig
from emotion_engine.rl.reward_shaping import create_default_reward_shaper
from emotion_engine.environment.scenarios.caretaking import CaretakingScenario
from emotion_engine.training.trainer import Trainer

print("="*75)
print(" EMOTION ENGINE - QUICK TRAINING DEMO")
print("="*75)
print("\nTraining agents to develop maternal love through caretaking...")
print("This demo runs 5,000 steps (~5-10 minutes on CPU)\n")

# Create caretaking environment
print("[1] Creating environment...")
env = CaretakingScenario(
    num_children=1,
    max_steps=500,
    grid_size=10,
    child_vulnerability=0.7,
    render_mode=None,
)
print(f"    Environment: Caretaking Scenario")
print(f"    Agents: 1 parent + 1 child")
print(f"    Episode length: 500 steps")

# Create agents
print("\n[2] Creating emotional agents...")
agents = {}
for agent_id in range(2):
    agents[agent_id] = EmotionalAgent(
        agent_id=agent_id,
        observation_space_dim=env.observation_space.shape[0],
        action_space_dim=env.action_space.shape[0],
        device='cpu',
    )
    agents[agent_id].set_training(True)

print(f"    Agent 0 (Parent): {sum(p.numel() for p in agents[0].policy.parameters()):,} params")
print(f"    Agent 1 (Child): {sum(p.numel() for p in agents[1].policy.parameters()):,} params")

# Create PPO config (smaller for demo)
print("\n[3] Configuring PPO...")
ppo_config = PPOConfig(
    learning_rate=3e-4,
    n_steps=256,  # Smaller for quick demo
    batch_size=32,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
)
print(f"    Learning rate: {ppo_config.learning_rate}")
print(f"    Steps per update: {ppo_config.n_steps}")
print(f"    Batch size: {ppo_config.batch_size}")

# Create reward shaper
reward_shaper = create_default_reward_shaper()

# Create trainer
print("\n[4] Initializing trainer...")
trainer = Trainer(
    env=env,
    agents=agents,
    ppo_config=ppo_config,
    reward_shaper=reward_shaper,
    checkpoint_dir="./quick_checkpoints",
    log_interval=5,
    save_interval=2000,
    device='cpu',
)

# Get initial emotions
print("\n[5] Initial emotional state:")
for agent_id in [0, 1]:
    summary = agents[agent_id].get_emotion_summary()
    if agent_id == 0:
        print(f"    Parent:")
        print(f"      - Attachment: {summary.get('attachment', 0):.3f}")
        print(f"      - Empathy: {summary.get('empathy', 0):.3f}")
        print(f"      - Maternal Love: {summary.get('complex_maternal_love', 0):.3f}")

# Train!
print("\n" + "="*75)
print(" TRAINING START")
print("="*75)

metrics = trainer.train(
    total_steps=5000,
    n_steps_per_update=256,
)

# Show final results
print("\n" + "="*75)
print(" TRAINING COMPLETE - Results")
print("="*75)

# Get final emotions
print("\nFinal emotional state:")
for agent_id in [0, 1]:
    summary = agents[agent_id].get_emotion_summary()
    if agent_id == 0:
        parent_attachment = agents[0].relationship_manager.get_attachment_level(1)
        print(f"    Parent:")
        print(f"      - Attachment to child: {parent_attachment:.3f}")
        print(f"      - Empathy: {summary.get('empathy', 0):.3f}")
        print(f"      - Protective: {summary.get('protective_instinct', 0):.3f}")
        print(f"      - Altruism: {summary.get('altruism', 0):.3f}")
        print(f"      - Maternal Love: {summary.get('complex_maternal_love', 0):.3f}")
        print(f"      - Valence: {summary.get('valence', 0):.3f}")

# Show training metrics
if metrics:
    print(f"\nTraining metrics:")
    print(f"    Total episodes: {trainer.episode_count}")
    print(f"    Final policy loss: {metrics[-1].get('agent_0_policy_loss', 0):.4f}")
    print(f"    Final value loss: {metrics[-1].get('agent_0_value_loss', 0):.4f}")

print("\n" + "="*75)
print(" Analysis")
print("="*75)

parent_final_attachment = agents[0].relationship_manager.get_attachment_level(1)
parent_final_maternal = agents[0].get_emotion_summary().get('complex_maternal_love', 0)

if parent_final_attachment > 0.2:
    print("\n[SUCCESS] Attachment formed!")
    print(f"  Parent developed bond with child: {parent_final_attachment:.3f}")
else:
    print("\n[DEVELOPING] Attachment is forming...")
    print(f"  Current level: {parent_final_attachment:.3f}")
    print(f"  Needs more training steps for strong bonds")

if parent_final_maternal > 0.3:
    print("\n[SUCCESS] Maternal love emerged!")
    print(f"  Complex emotion reached: {parent_final_maternal:.3f}")
else:
    print("\n[EMERGING] Maternal love developing...")
    print(f"  Current level: {parent_final_maternal:.3f}")

print("\n" + "="*75)
print(" Next Steps")
print("="*75)
print("\nFor full training:")
print("  python scripts/train.py --steps 1000000 --scenario caretaking")
print("\nFor curriculum learning:")
print("  python scripts/train.py --steps 10000000 --curriculum")
print("\n" + "="*75)
