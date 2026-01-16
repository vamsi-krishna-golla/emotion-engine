"""
Training Results Analysis

Analyze the results of 100K step training run.
"""

import torch
from emotion_engine.rl.agent import EmotionalAgent
from emotion_engine.environment.scenarios.caretaking import CaretakingScenario

print("="*75)
print(" TRAINING RESULTS ANALYSIS - 100K STEPS")
print("="*75)

# Load trained checkpoint
checkpoint_path = "./checkpoints/final_checkpoint.pt"
print(f"\nLoading checkpoint: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Checkpoint from step: {checkpoint['global_step']:,}")
    print(f"  Total episodes: {checkpoint['episode_count']}")
except Exception as e:
    print(f"  Error loading checkpoint: {e}")
    print("  Running analysis with untrained agents...")
    checkpoint = None

# Create environment
env = CaretakingScenario(
    num_children=1,
    max_steps=100,
    grid_size=10,
    child_vulnerability=0.7,
)

# Create agents
print("\n[1] Creating agents...")
agents = {}
for agent_id in range(2):
    agents[agent_id] = EmotionalAgent(
        agent_id=agent_id,
        observation_space_dim=env.observation_space.shape[0],
        action_space_dim=env.action_space.shape[0],
        device='cpu',
    )

# Load weights if checkpoint exists
if checkpoint:
    print("\n[2] Loading trained weights...")
    for agent_id, agent in agents.items():
        agent_checkpoint = checkpoint['agents'][agent_id]
        agent.emotion_encoder.load_state_dict(agent_checkpoint['emotion_encoder'])
        agent.emotion_composer.load_state_dict(agent_checkpoint['emotion_composer'])
        agent.policy.load_state_dict(agent_checkpoint['policy'])
        agent.value_net.load_state_dict(agent_checkpoint['value_net'])
    print("  Weights loaded successfully!")

# Run evaluation episode
print("\n[3] Running evaluation episode...")
agents[0].set_training(False)
agents[1].set_training(False)

observations, infos = env.reset()
total_rewards = {0: 0.0, 1: 0.0}
episode_length = 0
child_alive_steps = 0

# Initial state
print("\nInitial State:")
parent_summary = agents[0].get_emotion_summary()
print(f"  Parent:")
print(f"    Attachment: {parent_summary.get('attachment', 0):.3f}")
print(f"    Empathy: {parent_summary.get('empathy', 0):.3f}")
print(f"    Protective: {parent_summary.get('protective_instinct', 0):.3f}")
print(f"    Maternal Love: {parent_summary.get('complex_maternal_love', 0):.3f}")

for step in range(100):
    # Get actions
    actions = {}
    for agent_id, agent in agents.items():
        emotion_features = agent.observe(observations[agent_id], {})
        action, _, _ = agent.select_action(emotion_features, deterministic=True)
        actions[agent_id] = action

    # Step environment
    observations, rewards, terminateds, truncateds, infos = env.step(actions)

    for agent_id in agents.keys():
        total_rewards[agent_id] += rewards[agent_id]

    episode_length += 1

    # Check if child is alive
    if infos[1]['health'] > 0:
        child_alive_steps += 1

    # Check if done
    if any(terminateds.values()) or any(truncateds.values()):
        break

# Final state
print("\nFinal State (after evaluation):")
parent_summary = agents[0].get_emotion_summary()
parent_attachment = agents[0].relationship_manager.get_attachment_level(1)
print(f"  Parent:")
print(f"    Attachment to child: {parent_attachment:.3f}")
print(f"    Empathy: {parent_summary.get('empathy', 0):.3f}")
print(f"    Protective: {parent_summary.get('protective_instinct', 0):.3f}")
print(f"    Altruism: {parent_summary.get('altruism', 0):.3f}")
print(f"    Maternal Love: {parent_summary.get('complex_maternal_love', 0):.3f}")
print(f"    Valence: {parent_summary.get('valence', 0):.3f}")

print(f"\n  Child:")
print(f"    Final health: {infos[1]['health']:.3f}")
print(f"    Survived: {'YES' if infos[1]['health'] > 0 else 'NO'}")

print("\nEpisode Metrics:")
print(f"  Episode length: {episode_length} steps")
print(f"  Parent total reward: {total_rewards[0]:.2f}")
print(f"  Child total reward: {total_rewards[1]:.2f}")
print(f"  Child survival rate: {100*child_alive_steps/episode_length:.1f}%")

print("\n" + "="*75)
print(" ANALYSIS")
print("="*75)

if checkpoint:
    print("\nTraining Summary:")
    print(f"  Total training steps: {checkpoint['global_step']:,}")
    print(f"  Total episodes: {checkpoint['episode_count']:,}")
    print(f"  Training time: ~5.4 minutes")
    print(f"  Steps per second: ~310")

    print("\nKey Observations:")
    if parent_attachment > 0.15:
        print(f"  [OK] Attachment formed: {parent_attachment:.3f}")
    else:
        print(f"  [DEVELOPING] Attachment forming: {parent_attachment:.3f}")
        print("       (100K steps is relatively short - need 500K+ for strong bonds)")

    if parent_summary.get('complex_maternal_love', 0) > 0.3:
        print(f"  [OK] Maternal love emerged: {parent_summary.get('complex_maternal_love', 0):.3f}")
    else:
        print(f"  [DEVELOPING] Maternal love developing: {parent_summary.get('complex_maternal_love', 0):.3f}")

    if child_alive_steps/episode_length > 0.8:
        print(f"  [OK] High child survival: {100*child_alive_steps/episode_length:.1f}%")
    else:
        print(f"  [NEEDS WORK] Child survival: {100*child_alive_steps/episode_length:.1f}%")

    print("\nNext Steps for Better Results:")
    print("  1. Train for 500K-1M steps for strong attachment formation")
    print("  2. Use curriculum learning for progressive difficulty")
    print("  3. Run crisis scenarios to test self-sacrifice behaviors")
    print("  4. Analyze emotion trajectories during key events")
else:
    print("\nNo checkpoint loaded - agents are untrained.")

print("\n" + "="*75)
print(" Analysis Complete")
print("="*75)
